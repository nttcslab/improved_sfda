import os
import pdb
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

class OursPBTALitModule(BasicLitModule):
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
                       lambda_aug: float=1.0,
                       aug_loss_type: str="dot",
                       net_momentum: float=0.99,
                       interval_epoch: int=1,
                       warmup_epoch: int=0,
                       init_aug: str="random",
                       init_aug_prob: float=0.9,
                       num_chunks: int=4,
                       K: int=1,
                       output_dir: str=None,
                       optimize_params: str="back_bottle_head"):
        
        super().__init__(net, optimizer_args, scheduler_args, train_acc, val_acc, test_acc, label_smoothing=False, output_dir=output_dir)
        
        self.net.load_state_dict(torch.load(pretrained_path))
        
        self.lambda_neg = lambda_neg
        self.lambda_aug = lambda_aug
        self.net_momentum = net_momentum
        self.interval_epoch = interval_epoch
        self.warmup_epoch = warmup_epoch

        self.optimize_params = optimize_params
        self.scheduler_args = scheduler_args

        self.net_teacher = deepcopy(self.net)
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

        self.automatic_optimization = False


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

    def loss_for_model(self, x, prob_w_s):
        with torch.no_grad():
            # strong aug
            x_s_list = []
            for k in range(self.K): # データ拡張を変えつつK回サンプルする
                x_s = self.pre_augment(x)
                x_s = self.trainable_augment(x_s)
                x_s = self.post_augment(x_s)
                x_s = self.normalizer(x_s)
                x_s_list.append(x_s)
            
            B, ch, H, W = x_s_list[0].size()
            x_s = torch.stack(x_s_list) 
            x_s = x_s.reshape(self.K*B, ch, H, W) # バッチに展開

        ### calc pos loss
        logits_s_s, _ = self.net(x_s)
        prob_s_s = F.softmax(logits_s_s, dim=1)
        KxB, C = prob_s_s.size()
        assert KxB == self.K*B
        prob_s_s = prob_s_s.reshape(self.K, B, C).mean(0) # K回の平均をとる。
        loss_pos = -(prob_s_s * prob_w_s).sum(1).mean() # consistency between strong and weak augmentation for the same student network

        coeff = (1 + loss_pos.detach())

        ### calc neg loss
        B = prob_s_s.size(0)
        mask = torch.ones((B, B))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag # 自分自身との類似度は除くためのマスク
        dot_neg = prob_s_s @ prob_s_s.T
        loss_neg = (dot_neg * mask.to(self.device)).sum(-1).mean()

        return loss_pos, loss_neg, coeff
    
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

    def update_model(self, x):
        optim, _ = self.optimizers() # type: ignore
        scheduler, _ = self.lr_schedulers() # type: ignore

        self.stop_bn_track_running_stats()

        with torch.no_grad():
            # weak aug
            x_w = self.pre_augment(x)
            x_w = self.post_augment(x_w)
            x_w = self.normalizer(x_w)

            logits_w_s, _ = self.net(x_w)

            prob_w_s = F.softmax(logits_w_s)

        self.activate_bn_track_running_stats()

        # update model
        loss_pos, loss_neg, coeff = self.loss_for_model(x, prob_w_s)
        loss_model = loss_pos + loss_neg * coeff * self.lambda_neg
        # logging
        self.log("train/loss_model", loss_model, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_pos", loss_pos, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss_neg", loss_neg, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/coeff", coeff, on_step=False, on_epoch=True, prog_bar=False)
        optim.zero_grad()
        self.manual_backward(loss_model)
        # loss_model.backward()
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
        x, _, _ = batch

        if self.current_epoch < self.warmup_epoch or (self.current_epoch + 1 - self.warmup_epoch) % self.interval_epoch == 0:
            self.trainable_augment.train()
            self.update_augmentation(x)
        else:
            self.trainable_augment.eval()
            self.update_model(x)

        self.previous_x = x

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        with torch.no_grad():
            # strong aug
            x_origin = self.pre_augment(self.previous_x)
            x_augmented = self.trainable_augment(x_origin)
            
            self.save_images(x_origin, x_augmented, self.current_epoch)

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

        if self.scheduler_args is None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda x: 1) # dummy scheduler
        else:
            max_iter = self.trainer.max_epochs * self.trainer.datamodule.trainloader_size

            if self.scheduler_args.type == "exponential":
                scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda x: (1 + self.scheduler_args.gamma * x / max_iter) ** (- self.scheduler_args.power))
            elif self.scheduler_args.type == "multi_step":
                steps = [int(max_iter * step/self.trainer.max_epochs) for step in self.scheduler_args.steps]
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=steps, gamma=self.scheduler_args.gamma)
            else:
                raise ValueError(self.scheduler_args)

        # optimizer for augment
        aug_optim = torch.optim.AdamW(self.trainable_augment.parameters(), **self.aug_optimizer_args)
        scheduler_optim = torch.optim.lr_scheduler.LambdaLR(aug_optim, lambda x: 1) # dummy scheduler

        return [optim, aug_optim], [scheduler, scheduler_optim]
