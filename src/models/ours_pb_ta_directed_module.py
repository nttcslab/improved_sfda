import os
import pdb
import math
from typing import Any, Dict, List, Tuple
import numpy as np

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy 
from src.models.components.classifier import Classifier
from src.models.basic_module import BasicLitModule
from src.models.components.utils import WeightScheduler

from src.models.components.faster_autoaugment import policy

from copy import deepcopy
from PIL import Image

from scipy.spatial.distance import cdist

from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm as colormap
import seaborn as sns

class OursPBTADirectedLitModule(BasicLitModule):
    def __init__(self, net: Classifier,
                       optimizer_args: dict, 
                       scheduler_args: dict, 
                       augment_args: dict,
                       aug_optimizer_args: dict,
                       pre_augment: torch.nn.Module,
                       post_augment: torch.nn.Module,
                       normalizer: torch.nn.Module,
                       train_acc: Accuracy,
                       val_acc: Accuracy,
                       test_acc: Accuracy,
                       pretrained_path: str,
                       lambda_neg: float=1.0,
                       lambda_neg_2: float=None,
                       lambda_aug: float=1.0,
                       lambda_strong: float=0.5,
                       aug_loss_type: str="nonsatent",
                       net_momentum: float=0.99,
                       interval_epoch: int=1,
                       warmup_epoch: int=0,
                       init_aug: str="random",
                       init_aug_prob: float=0.9,
                       num_chunks: int=4,
                       K: int=1,
                       M: int=5,
                       MM: int=5,
                       output_dir: str=None,
                       output_validation: bool=False,
                       optimize_params: str="back_bottle_head"):
        
        super().__init__(net, optimizer_args, scheduler_args, train_acc, val_acc, test_acc, label_smoothing=False, output_dir=output_dir)
        
        self.net.load_state_dict(torch.load(pretrained_path))
        self.net_teacher = deepcopy(self.net)

        self.lambda_neg = lambda_neg
        self.lambda_neg_2 = lambda_neg_2 if lambda_neg_2 is not None else lambda_neg
        self.lambda_aug = lambda_aug
        self.lambda_strong = lambda_strong
        self.net_momentum = net_momentum
        self.interval_epoch = interval_epoch
        self.warmup_epoch = warmup_epoch

        self.optimize_params = optimize_params
        self.scheduler_args = scheduler_args

        self.off_teacher_gradient()

        self.pre_augment = pre_augment
        self.post_augment = post_augment
        self.normalizer = normalizer
        if init_aug == "random":
            self.trainable_augment = policy.Policy.faster_auto_augment_policy(**augment_args)
        elif init_aug == "autoaug":
            self.trainable_augment = policy.Policy.init_w_auto_augment_policy(init_aug_prob, num_chunks)
        else:
            raise ValueError(init_aug)
        self.aug_optimizer_args = aug_optimizer_args
        self.aug_loss_type = aug_loss_type

        self.K = K
        self.M = M
        self.MM = MM

        self.output_validation = output_validation
        self.validation_outputs = []

        self.automatic_optimization = False

    def on_train_start(self):
        super().on_train_start()

        num_samples = len(self.trainer.datamodule)

        self.feat_bank = torch.zeros((num_samples, 256))
        self.score_bank = torch.zeros((num_samples, self.net.num_classes))

        self.net.eval()

        with torch.no_grad():
            for x, _, idx in self.trainer.datamodule.train_dataloader():
                x = x.to(self.device)
                x_w = self.pre_augment(x)
                x_w = self.post_augment(x_w)
                x_w = self.normalizer(x_w)
                out, feat = self.forward(x_w)
                feat_norm = F.normalize(feat)
                out = F.softmax(out, dim=1)

                self.feat_bank[idx] = feat_norm.detach().clone().cpu()
                self.score_bank[idx] = out.mean(0).detach().clone().cpu()

        self.net.train()


    def momentum_update(self):
        for param_s, param_t in zip(self.net.parameters(), self.net_teacher.parameters()):
            param_t.data = param_t.data * self.net_momentum + param_s.data * (1 - self.net_momentum)


    def off_teacher_gradient(self):
        # not calculate gradient for net_teacher
        for param in self.net_teacher.parameters():
            param.requires_grad = False


    def predict_weak(self, x):
        self.stop_bn_track_running_stats()

        with torch.no_grad():
            # weak aug
            x_w = self.pre_augment(x)
            x_w = self.post_augment(x_w)
            x_w = self.normalizer(x_w)

            logits_w_s, _ = self.net(x_w)
            logits_w_t, _ = self.net_teacher(x_w)

            prob_w_s = F.softmax(logits_w_s)
            prob_w_t = F.softmax(logits_w_t)

        self.activate_bn_track_running_stats()

        return prob_w_s.detach(), prob_w_t.detach()


    def predict_augment_student(self, x):
        self.stop_bn_track_running_stats()

        # strong aug
        x_s = self.pre_augment(x)
        x_s = self.trainable_augment(x_s)
        x_s = self.post_augment(x_s)
        x_s = self.normalizer(x_s)

        logits_s_s, _ = self.net(x_s)
        prob_s_s = F.softmax(logits_s_s, dim=1)

        self.activate_bn_track_running_stats()

        return prob_s_s


    def predict_augment_teacher(self, x):

        self.stop_bn_track_running_stats()

        # strong aug
        x_s = self.pre_augment(x)
        x_s = self.trainable_augment(x_s)
        x_s = self.post_augment(x_s)
        x_s = self.normalizer(x_s)

        logits_s_t, _ = self.net_teacher(x_s)
        prob_s_t = F.softmax(logits_s_t, dim=1)

        self.activate_bn_track_running_stats()

        return prob_s_t


    def stop_bn_track_running_stats(self):
        for m in self.net.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.track_running_stats = False


    def activate_bn_track_running_stats(self):
        for m in self.net.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.track_running_stats = True


    def save_images(self, x_origin, x_augmented, epoch):
        save_directory = os.path.join(self.output_dir, "augmented_samples", f"epoch_{epoch}")
        os.makedirs(save_directory, exist_ok=True)

        for idx, x in enumerate(x_origin):
            x_numpy = (x.cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(x_numpy.transpose(1,2,0)).resize((64,64)).save(os.path.join(save_directory, f"original_idx_{idx}.png"))

        for idx, x in enumerate(x_augmented):
            x_numpy = (x.cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(x_numpy.transpose(1,2,0)).resize((64,64)).save(os.path.join(save_directory, f"augmented_idx_{idx}.png"))


    def update_model(self, x, idx):
        if self.current_epoch > self.trainer.max_epochs * 0.25:
            self.M = self.MM

        optim, _ = self.optimizers() # type: ignore
        scheduler, _ = self.lr_schedulers() # type: ignore

        # optim grad
        optim.zero_grad()

        # step 1: calculate model loss with weak augment

        # weak aug
        x_p = self.pre_augment(x)
        x_w = self.post_augment(x_p)
        x_wn = self.normalizer(x_w)

        logits_w, feat_w = self.net(x_wn)
        prob_w = F.softmax(logits_w, dim=1)

        with torch.no_grad():
            feat_w = F.normalize(feat_w).detach().clone().cpu()

            self.feat_bank[idx] = feat_w.detach().clone().cpu()
            self.score_bank[idx] = prob_w.detach().clone().cpu()

            distance = feat_w @ self.feat_bank.T
            top_distances, idx_mNN = torch.topk(distance, dim=-1, largest=True, k=self.M)
            idx_mNN_exc_self = idx_mNN[:,1:]
            prob_memory = self.score_bank[idx_mNN_exc_self].mean(1).detach().to(self.device)

            self.log("train/distance_with_memorybank", top_distances.mean(), on_step=False, on_epoch=True, prog_bar=True)

        loss_pos_w = - (prob_w * prob_memory.to(self.device)).sum(1).mean()
        coeff_w = (1 + loss_pos_w).detach()

        B = prob_w.size(0)
        mask = torch.ones((B, B))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag # 自分自身との類似度は除くためのマスク
        dot_neg_w = prob_w @ prob_w.T
        loss_neg_w = (dot_neg_w * mask.to(self.device)).sum(-1).mean()
        loss_w = (loss_pos_w + loss_neg_w * coeff_w * self.lambda_neg) * (1 - self.lambda_strong)
        self.manual_backward(loss_w)

        self.log("train/loss_pos_w", loss_pos_w, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss_neg_w", loss_neg_w, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss_w", loss_w, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/coeff_w", coeff_w, on_step=False, on_epoch=True, prog_bar=False)

        # step 2: calculate directed model loss with strong agument

        self.stop_bn_track_running_stats()

        loss_pos_s_log = 0.0
        loss_neg_s_log = 0.0
        loss_s_log = 0.0
        coeff_s_log = 0.0

        for k in range(self.K):
            # augment with trainable augmentation
            with torch.no_grad():
                x_s = self.trainable_augment(x_w)
                x_sn = self.normalizer(x_s)
            
            logits_s, feat_s = self.net(x_sn)
            feat_s = F.normalize(feat_s)
            prob_s = F.softmax(logits_s, dim=1)
            
            loss_pos_s = -(prob_s * prob_memory.to(self.device)).sum(1).mean()

            coeff_s = (1 + loss_pos_s).detach()
            coeff_s_log += coeff_s.item() / self.K

            B = prob_w.size(0)
            mask = torch.ones((B, B))
            diag_num = torch.diag(mask)
            mask_diag = torch.diag_embed(diag_num)
            mask = mask - mask_diag
            dot_neg_s = prob_s @ prob_w.T.detach()
            loss_neg_s = (dot_neg_s * mask.to(self.device)).sum(-1).mean() * 2.0
            loss_s = (loss_pos_s + loss_neg_s * coeff_s * self.lambda_neg_2) * self.lambda_strong / self.K
            self.manual_backward(loss_s)

            loss_pos_s_log += loss_pos_s.item() / self.K
            loss_neg_s_log += loss_neg_s.item() / self.K
            loss_s_log += loss_s.item()
        
        self.log("train/loss_pos_s", loss_pos_s_log, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss_neg_s", loss_neg_s_log, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss_s", loss_s_log, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/coeff_s", coeff_s_log, on_step=False, on_epoch=True, prog_bar=False)

        self.activate_bn_track_running_stats()

        # update model with both weak and strong loss
        optim.step()

        # step scheduler, net_teacher
        scheduler.step()
        self.momentum_update()


    def update_augmentation(self, x):
        _, aug_optim = self.optimizers()
        _, aug_scheduler = self.lr_schedulers()

        # update data augmentation
        aug_optim.zero_grad()

        # predict for weak
        prob_w_s, prob_w_t = self.predict_weak(x)

        prob_s_s = self.predict_augment_student(x)
        if self.aug_loss_type == "dot":
            loss_student = - (prob_s_s * prob_w_s).sum(1).mean()
        elif self.aug_loss_type == "entropy":
            loss_student = - (prob_s_s * torch.log(prob_s_s + 1e-8)).sum(1).mean()
        elif self.aug_loss_type == "nonsatent":
            loss_student = (prob_s_s * torch.log(1.0 - prob_s_s + 1e-8)).sum(1).mean()
        else:
            raise ValueError(self.aug_loss_type)
        self.manual_backward(- loss_student)
        self.log("train/loss_student", loss_student, on_step=False, on_epoch=True, prog_bar=False)

        prob_s_t = self.predict_augment_teacher(x)
        if self.aug_loss_type == "dot":
            loss_teacher = - (prob_s_t * prob_w_t).sum(1).mean()
        elif self.aug_loss_type == "entropy" or self.aug_loss_type == "nonsatent":
            loss_teacher = - (prob_s_t * torch.log(prob_s_t + 1e-8)).sum(1).mean()
        else:
            raise ValueError(self.aug_loss_type)
        self.manual_backward(loss_teacher * self.lambda_aug)
        self.log("train/loss_teacher", loss_teacher, on_step=False, on_epoch=True, prog_bar=False)

        aug_optim.step()

        # step scheduler
        aug_scheduler.step()


    def training_step(self, batch: Any, batch_idx: int):
        x, _, idx = batch

        if self.current_epoch < self.warmup_epoch or (self.current_epoch + 1 - self.warmup_epoch) % self.interval_epoch == 0:
            self.trainable_augment.train()
            self.update_augmentation(x)
        else:
            self.trainable_augment.eval()
            self.update_model(x, idx)

        self.previous_x = x


    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        with torch.no_grad():
            # strong aug
            x_origin = self.pre_augment(self.previous_x)
            x_augmented = self.trainable_augment(x_origin)
            
            self.save_images(x_origin, x_augmented, self.current_epoch)


    def validation_step(self, batch: Any, batch_idx: int):
        x, targets, _ = batch
        logits, feats = self.forward(x)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        ret = {"loss": loss, "preds": preds, "targets": targets, "feats": feats}
        self.validation_outputs.append(ret)

        return {"loss": loss, "preds": preds, "targets": targets, "feats": feats}


    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

        if not self.trainer.sanity_checking and self.output_validation:
            confusion_matrix_dir = os.path.join(self.output_dir, "confusion_matrix")
            tsne_plot_dir = os.path.join(self.output_dir, "tsne_plot")
            os.makedirs(confusion_matrix_dir, exist_ok=True)
            os.makedirs(tsne_plot_dir, exist_ok=True)

            preds = []
            targets = []
            feats = []
            
            for output in self.validation_outputs:
                preds.append(output['preds'].cpu().numpy())
                targets.append(output['targets'].cpu().numpy())
                feats.append(output['feats'].cpu().numpy())
            
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
            feats = np.concatenate(feats, axis=0)

            # confusion matrixプロット
            cm = confusion_matrix(targets, preds, normalize="true") * 100
            plt.figure(figsize=(10,10))
            sns.heatmap(cm, annot=True, cmap="Blues", fmt=".1f", annot_kws={"fontsize": 6})
            plt.savefig(os.path.join(confusion_matrix_dir, f"epoch_{self.current_epoch}.pdf"))
            plt.clf()
            plt.close()

            # tsneプロット
            if not hasattr(self, "tsne_index"):
                if len(feats) > 2000:
                    self.tsne_index = np.random.permutation(np.arange(0, len(feats)))[:2000]
                else:
                    self.tsne_index = np.arange(0, len(feats))
            
            feat_selected = feats[self.tsne_index]
            targets_selected = targets[self.tsne_index]
            preds_selected = preds[self.tsne_index]

            tsne = TSNE(n_components=2, random_state=42)
            feat_tsne = tsne.fit_transform(feat_selected)

            labels = np.unique(targets_selected)
            cmap = colormap.get_cmap('viridis', len(labels))
            colors = [cmap(label) for label in labels]

            # 正解ラベルで色分け
            for label, color in zip(labels, colors):
                indicies = np.where(targets_selected == label)
                plt.scatter(feat_tsne[indicies, 0], feat_tsne[indicies, 1], s=3, color=color, label=label)
            plt.legend()
            plt.savefig(os.path.join(tsne_plot_dir, f"epoch_{self.current_epoch}.pdf"))
            plt.clf()
            plt.close()

            # 予測ラベルで色分け
            for label, color in zip(labels, colors):
                indicies = np.where(preds_selected == label)
                plt.scatter(feat_tsne[indicies, 0], feat_tsne[indicies, 1], s=3, color=color, label=label)
            plt.legend()
            plt.savefig(os.path.join(tsne_plot_dir, f"preds_epoch_{self.current_epoch}.pdf"))
            plt.clf()
            plt.close()

        self.validation_outputs.clear()


    def configure_optimizers(self):
        
        # optimizer for model
        for k, v in self.net.named_parameters():
            v.requires_grad = False

        total_params = []

        for net_name in self.optimize_params.split('_'):
            if net_name == 'back': # backboneのパラメータ
                params = []
                for k, v in self.net.backbone.named_parameters():
                    v.requires_grad = True
                    params.append(v)
                total_params.append({"params": params, "lr": 0.1 * self.optimizer_args.lr})
            
            elif net_name == 'backbn': # backboneのbnパラメータのみ
                params = []
                for k, v in self.net.backbone.named_parameters():
                    if 'bn' in k:
                        v.requires_grad = True
                        params.append(v)
                total_params.append({"params": params, "lr": 0.1 * self.optimizer_args.lr})
            
            elif net_name == 'bottle': # bottleneckのパラメータ
                params = []
                for k, v in self.net.bottleneck.named_parameters():
                    v.requires_grad = True
                    params.append(v)
                total_params.append({"params": params, "lr": 1.0 * self.optimizer_args.lr})

            elif net_name == 'bottlebn': # bottleneckのbnパラメータのみ
                params = []
                for k, v in self.net.bottleneck.named_parameters():
                    if 'bn' in k:
                        v.requires_grad = True
                        params.append(v)
                total_params.append({"params": params, "lr": 1.0 * self.optimizer_args.lr})

            elif net_name == 'head': # headのパラメータ
                params = []
                for k, v in self.net.head.named_parameters():
                    v.requires_grad = True
                    params.append(v)
                total_params.append({"params": params, "lr": 1.0 * self.optimizer_args.lr})

        optim = torch.optim.SGD(
            params=total_params, **self.optimizer_args
        )

        self.max_iter = int(self.trainer.max_epochs * self.trainer.datamodule.trainloader_size * (self.interval_epoch - 1) / self.interval_epoch)

        if self.scheduler_args is None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda x: 1) # dummy scheduler
        else:
            if self.scheduler_args.type == "exponential":
                scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda x: (1 + self.scheduler_args.gamma * x / self.max_iter) ** (- self.scheduler_args.power))
            elif self.scheduler_args.type == "multi_step":
                steps = [int(self.max_iter * step/self.trainer.max_epochs) for step in self.scheduler_args.steps]
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=steps, gamma=self.scheduler_args.gamma)
            else:
                raise ValueError(self.scheduler_args)

        # optimizer for augment
        aug_optim = torch.optim.AdamW(self.trainable_augment.parameters(), **self.aug_optimizer_args)

        self.aug_max_iter = int(self.trainer.max_epochs * self.trainer.datamodule.trainloader_size / self.interval_epoch)

        if self.scheduler_args is None:
            aug_scheduler = torch.optim.lr_scheduler.LambdaLR(aug_optim, lambda x: 1) # dummy scheduler
        else:
            if self.scheduler_args.type == "exponential":
                aug_scheduler = torch.optim.lr_scheduler.LambdaLR(aug_optim, lambda x: (1 + self.scheduler_args.gamma * x / self.aug_max_iter) ** (- self.scheduler_args.power))
            elif self.scheduler_args.type == "multi_step":
                steps = [int(self.aug_max_iter * step/self.trainer.max_epochs) for step in self.scheduler_args.steps]
                aug_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=steps, gamma=self.scheduler_args.gamma)
            else:
                raise ValueError(self.scheduler_args)

        return [optim, aug_optim], [scheduler, aug_scheduler]
