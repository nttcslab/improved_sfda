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
                       net_teacher: Classifier, 
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
                       lambda_pos: float=0.0,
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
                       M: int=5,
                       use_memory_bank: bool=False,
                       pos_loss_mode: str="dot",
                       lambda_sw_ss: float=0.9,
                       coeff_type: str="dot",
                       coeff_beta: float=0.0,
                       # 試験実装用
                       use_div_decay: bool=True,
                       use_TeSLA_aug_update: bool=False,

                       logging_outputs: bool=False,

                       output_dir: str=None,
                       optimize_params: str="back_bottle_head"):
        
        super().__init__(net, optimizer_args, scheduler_args, train_acc, val_acc, test_acc, label_smoothing=False, output_dir=output_dir)
        self.net_teacher = net_teacher
        
        self.net.load_state_dict(torch.load(pretrained_path))
        self.net_teacher.load_state_dict(torch.load(pretrained_path))
        
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.lambda_aug = lambda_aug
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
        self.pos_loss_mode = pos_loss_mode
        self.coeff_type = coeff_type
        self.threshold = 0.95
        self.lambda_sw_ss = lambda_sw_ss
        self.use_memory_bank = use_memory_bank

        self.coeff_iter = 0
        self.coeff_beta = coeff_beta

        self.use_div_decay = use_div_decay
        self.use_TeSLA_aug_update = use_TeSLA_aug_update

        if self.use_TeSLA_aug_update:
            # Set normalization hooker if model has normalization layers
            self.current_norm_inputs = {}

            # For removing hooks
            self.hook_handlers = {}
            self.register_norm_hooks()

        self.automatic_optimization = False

        self.logging_outputs = logging_outputs
        self.epoch_output_buffer = []

    def on_train_start(self):
        super().on_train_start()

        if self.use_memory_bank:
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


    @torch.no_grad()
    def calculate_weight(self, prob_w, prob_s):
        pred_s = torch.argmax(prob_s, -1)
        pred_w = torch.argmax(prob_w, -1)
        agree = (pred_s == torch.unsqueeze(pred_w, dim=1)).sum(1)
        agree_plus_one = agree + 1.0
        return agree_plus_one / agree_plus_one.sum()


    @torch.no_grad()
    def calculate_coeff(self, prob_s, prob_w):
        if self.coeff_type == "dot":
            return 1 - (prob_s * prob_w).sum(1).mean()
        elif self.coeff_type == "binary":
            pred_s = torch.argmax(prob_s, dim=1)
            pred_w = torch.argmax(prob_w, dim=1)
            consistency = (pred_s == pred_w).sum() / len(pred_s)
            return 1 - consistency
        elif self.coeff_type == "soft_binary":
            prob_max_s, pred_s = torch.max(prob_s, dim=1)
            prob_max_w, pred_w = torch.max(prob_w, dim=1)
            consistency = torch.where(pred_s == pred_w, (prob_max_s + prob_max_w) / 2, torch.zeros(len(pred_s), device=self.device)).mean()
            return 1 - consistency
        elif self.coeff_type == "predefined":
            self.coeff_iter += 1
            return (1 + 10 * self.coeff_iter / self.max_iter) ** (- self.coeff_beta)
        else:
            raise ValueError(self.coeff_type)

    def loss_for_model(self, x, idx):
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

            x_w = self.pre_augment(x)
            x_w = self.post_augment(x_w)
            x_w = self.normalizer(x_w)

            out_w, feat_w = self.forward(x_w)
            prob_w_s = F.softmax(out_w, dim=1)

            if self.use_memory_bank:
                feat_w = F.normalize(feat_w).detach().clone().cpu()
                
                self.feat_bank[idx] = feat_w.detach().clone().cpu()
                self.score_bank[idx] = prob_w_s.detach().clone().cpu()

                distance = feat_w @ self.feat_bank.T
                _, idx_mNN = torch.topk(distance, dim=-1, largest=True, k=self.M + 1)
                idx_mNN_exc_self = idx_mNN[:,1:]
                prob_w_s = self.score_bank[idx_mNN_exc_self].mean(1).detach().to(self.device)

        ### calc pos loss weak
        out_w, feat_w = self.forward(x_w)
        softmax_out_weak = F.softmax(out_w, dim=1)

        ### calc pos loss
        logits_s_s, _ = self.net(x_s)
        softmax_out_strong_K = F.softmax(logits_s_s, dim=1)
        KxB, C = softmax_out_strong_K.size()
        assert KxB == self.K*B
        softmax_out_strong = softmax_out_strong_K.reshape(self.K, B, C).mean(0) # K回の平均をとる。

        prob_s_s = softmax_out_strong * self.lambda_pos + softmax_out_weak * (1 - self.lambda_pos)

        loss_pos = - (prob_s_s * prob_w_s).sum(1).mean()

        # if self.pos_loss_mode=="weightinig":
        #     weights = self.calculate_weight(prob_w_s, prob_s_s.reshape(self.K, B, C).permute(1,0,2))
        #     loss_pos = (-(prob_s_s_mean * prob_w_s).sum(1) * weights.detach()).sum()
        # elif self.pos_loss_mode=="thresholding":
        #     mask = (torch.max(prob_w_s, dim=1).values >= self.threshold).float()
        #     pseudo_labels = torch.argmax(prob_w_s, dim=1)
        #     ce_loss = F.cross_entropy(logits_s_s, pseudo_labels.repeat(self.K), reduce=False).reshape(self.K, B).mean(0) # cross entropy loss
        #     dot_loss = -(prob_s_s_mean * prob_w_s).sum(1) # dot loss
        #     loss_pos = ((ce_loss * mask).sum() + (dot_loss * (1 - mask)).sum()) / B
        # elif self.pos_loss_mode=="sw_ss":
        #     assert self.K > 1
        #     loss_pos_sw = -(prob_s_s_mean * prob_w_s).sum(1).mean()
        #     prob_s_s_reshape = prob_s_s.reshape(self.K, B, C)
        #     loss_pos_ss = None
        #     for i in range(self.K):
        #         for j in range(i + 1, self.K):
        #             if loss_pos_ss is None: # 初回
        #                 loss_pos_ss = -(prob_s_s_reshape[i] * prob_s_s_reshape[j]).sum(1).mean()
        #             else:
        #                 loss_pos_ss += -(prob_s_s_reshape[i] * prob_s_s_reshape[j]).sum(1).mean()
        #     loss_pos_ss = loss_pos_ss * 2 / (self.K * (self.K - 1))
        #     loss_pos = self.lambda_sw_ss * loss_pos_sw + (1 - self.lambda_sw_ss) * loss_pos_ss

        # else:
        #     loss_pos = -(prob_s_s_mean * prob_w_s).sum(1).mean() # consistency between strong and weak augmentation for the same student network

        if self.use_div_decay:
            coeff = self.calculate_coeff(prob_s_s.detach(), prob_w_s.detach())
        else:
            coeff = 1.0

        ### calc neg loss
        B = prob_s_s.size(0)
        mask = torch.ones((B, B))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag # 自分自身との類似度は除くためのマスク
        dot_neg = prob_s_s @ prob_s_s.T
        loss_neg = (dot_neg * mask.to(self.device)).sum(-1).mean()

        return loss_pos, loss_neg, coeff, softmax_out_strong_K.reshape(self.K, B, C).permute(1,0,2), prob_w_s
    
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
        optim, _ = self.optimizers() # type: ignore
        scheduler, _ = self.lr_schedulers() # type: ignore

        # update model
        loss_pos, loss_neg, coeff, prob_s_s, prob_w_s = self.loss_for_model(x, idx)
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

        return prob_s_s.detach(), prob_w_s.detach()

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

    ### implementations for tesla update

    def norm_hook(self, idx):
        def hook(module, input, output):
            input = input[0]
            self.current_norm_inputs[idx] = [input.mean(dim=(-2, -1)), input.var(dim=(-2, -1))]
        return hook


    def register_norm_hooks(self):
        idx = 0
        for m in self.net_teacher.modules():
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.SyncBatchNorm) or isinstance(m, torch.nn.LayerNorm):
                self.hook_handlers[idx] = m.register_forward_hook(self.norm_hook(idx))
                idx += 1

    def compute_norm_stat_loss(self, pre_norm_feats_curr, pre_norm_feats_target):
        loss = torch.zeros(pre_norm_feats_curr[0][0].size(0)).cuda()

        for i in range(len(pre_norm_feats_curr)):
            target_mean = pre_norm_feats_target[i][0].detach()
            curr_mean = pre_norm_feats_curr[i][0]

            loss += (curr_mean - target_mean).pow(2).mean(dim=-1)
        return loss / len(pre_norm_feats_curr)

    def update_augmentation_TeSLA(self, x):
        
        with torch.no_grad():
            x_w = self.pre_augment(x)
            x_w = self.post_augment(x_w)
            x_w = self.normalizer(x_w)

            self.net_teacher(x_w)

            original_norm_stats = {}
            for k, v in self.current_norm_inputs.items():
                original_norm_stats[k] = [v[0].detach(), v[1].detach()]

        _, aug_optim = self.optimizers()
        _, aug_scheduler = self.lr_schedulers()

        # update data augmentation
        aug_optim.zero_grad()

        # strong aug
        x_s = self.pre_augment(x)
        x_s = self.trainable_augment(x_s)
        x_s = self.post_augment(x_s)
        x_s = self.normalizer(x_s)

        logits_s_t, _ = self.net_teacher(x_s)
        prob_s_t = F.softmax(logits_s_t, dim=1)

        loss_ent = torch.sum(prob_s_t * torch.log(prob_s_t + 1e-8), dim=-1).mean()

        currn_norm_stats = self.current_norm_inputs
        loss_norm = self.compute_norm_stat_loss(currn_norm_stats, original_norm_stats).mean()

        loss = loss_ent + loss_norm * self.lambda_aug
        loss.backward()
        aug_optim.step()

        self.log("train/loss_aug_ent", loss_ent, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss_aug_norm", loss_norm, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss_aug", loss, on_step=False, on_epoch=True, prog_bar=False)

        aug_scheduler.step()

    ### --------------------------------

    def training_step(self, batch: Any, batch_idx: int):
        x, y_analysis_only, idx = batch

        if self.current_epoch < self.warmup_epoch or (self.current_epoch + 1 - self.warmup_epoch) % self.interval_epoch == 0:
            self.trainable_augment.train()
            if self.use_TeSLA_aug_update:
                self.update_augmentation_TeSLA(x)
            else:
                self.update_augmentation(x)
        else:
            self.trainable_augment.eval()
            prob_s_s, prob_s_w = self.update_model(x, idx)
            self.epoch_output_buffer.append([prob_s_s.cpu().numpy(), prob_s_w.cpu().numpy(), y_analysis_only.cpu().numpy()])

        self.previous_x = x

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()

        if self.logging_outputs:

            with torch.no_grad():
                # strong aug
                x_origin = self.pre_augment(self.previous_x)
                x_augmented = self.trainable_augment(x_origin)
                
                self.save_images(x_origin, x_augmented, self.current_epoch)

            if self.epoch_output_buffer != []:
                prob_s_s_list, prob_s_w_list, y_list = [], [], []
                for prob_s_s, prob_s_w, y  in self.epoch_output_buffer:
                    prob_s_s_list.append(prob_s_s)
                    prob_s_w_list.append(prob_s_w)
                    y_list.append(y)
                
                output_dict = {'prob_s_s': np.concatenate(prob_s_s_list), 'prob_s_w': np.concatenate(prob_s_w_list), 'y': np.concatenate(y_list)}

                save_directory = os.path.join(self.output_dir, "train_outputs")
                os.makedirs(save_directory, exist_ok=True)

                np.save(os.path.join(save_directory, f"outputs_epoch_{self.current_epoch}.npy"), output_dict, allow_pickle=True)

                self.epoch_output_buffer.clear()

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

        self.max_iter = self.trainer.max_epochs * self.trainer.datamodule.trainloader_size * (self.interval_epoch / (self.interval_epoch + 1))

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
        scheduler_optim = torch.optim.lr_scheduler.LambdaLR(aug_optim, lambda x: 1) # dummy scheduler

        return [optim, aug_optim], [scheduler, scheduler_optim]
