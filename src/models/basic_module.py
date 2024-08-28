from logging.config import dictConfig
import os
import pdb
from typing import Any, List

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.classifier import Classifier


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1).to(inputs.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss


class BasicLitModule(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: Classifier,
        optimizer_args: dict,
        scheduler_args: dict,
        train_acc: Accuracy,
        val_acc: Accuracy,
        test_acc: Accuracy,
        save_path_net_state_dict: str = None,
        label_smoothing: bool = True,
        smoothing_epsilon: float = 0.1,
        output_dir: str = None,
        save_model_every_epoch: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args

        self.net = net

        # loss function
        self.criterion = CrossEntropyLabelSmooth(num_classes=self.net.num_classes, epsilon=smoothing_epsilon) if label_smoothing else torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = train_acc
        self.val_acc = val_acc
        self.test_acc = test_acc

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.save_path_net_state_dict = save_path_net_state_dict
        self.save_model_every_epoch = save_model_every_epoch
        self.output_dir = output_dir

        if self.save_path_net_state_dict is not None:
            os.makedirs(os.path.dirname(self.save_path_net_state_dict), exist_ok=True)

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        x, y, _ = batch
        logits, _ = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

        if self.save_path_net_state_dict is not None and self.save_model_every_epoch:
            save_path_epoch = self.save_path_net_state_dict.replace(".pth", f"_epoch_{self.current_epoch}.pth")
            torch.save(self.net.state_dict(), save_path_epoch)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch_end(self):
        if self.output_dir is not None:
            torch.save(self.net.state_dict(), os.path.join(self.output_dir, "best_model.pth"))

        if self.save_path_net_state_dict is not None:
            torch.save(self.net.state_dict(), self.save_path_net_state_dict)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        params = [
            {"params": self.net.backbone.parameters(), "lr": 0.1 * self.optimizer_args.lr},
            {"params": self.net.bottleneck.parameters(), "lr": 1.0 * self.optimizer_args.lr},
            {"params": self.net.head.parameters(), "lr": 1.0 * self.optimizer_args.lr}
        ]

        optim = torch.optim.SGD(
            params=params, **self.optimizer_args
        )

        if self.scheduler_args is None:
            return optim
        
        max_iter = self.trainer.max_epochs * self.trainer.datamodule.trainloader_size

        if self.scheduler_args.type == "exponential":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda x: (1 + self.scheduler_args.gamma * x / max_iter) ** (- self.scheduler_args.power))
        elif self.scheduler_args.type == "multi_step":
            steps = [int(max_iter * step/self.trainer.max_epochs) for step in self.scheduler_args.steps]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=steps, gamma=self.scheduler_args.gamma)
        else:
            raise ValueError()

        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }