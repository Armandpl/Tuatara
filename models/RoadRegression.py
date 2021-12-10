import pytorch_lightning as pl
import torchmetrics

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision
from kornia import tensor_to_image
from kornia.augmentation import (ColorJitter,
                                 RandomErasing, RandomGrayscale,
                                 RandomInvert, RandomMotionBlur)

import wandb

config = dict(
    architecture="resnet34",
    pretrained=True,
    learning_rate=1e-4,
    loss="mse_loss"
)


class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter

        self.transforms = nn.Sequential(
            RandomErasing(p=0.1),
            RandomGrayscale(p=0.1),
            RandomInvert(p=0.1),
            RandomMotionBlur(3, 35., 0.5, p=0.1),
        )

        self.jitter = ColorJitter(0.5, 0.5, 0.5, 0.5)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        x_out = self.transforms(x)  # BxCxHxW
        if self._apply_color_jitter:
            x_out = self.jitter(x_out)
        return x_out


class RoadRegression(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.config = config

        # setting up metrics
        metrics = torchmetrics.MetricCollection([
            torchmetrics.MeanSquaredError(),
            torchmetrics.MeanAbsoluteError()
        ])
        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

        self.model = self.build_model()
        self.transform = DataAugmentation()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images_aug = self.transform(images)
        preds = self.forward(images_aug)
        loss = getattr(F, self.config["loss"])(preds, targets)

        metrics = self.train_metrics(preds, targets)
        self.log_dict(metrics, on_step=True, on_epoch=False)

        wandb_logger = self.logger.experiment
        batch_grid = self.make_grid(images_aug)
        images = wandb.Image(batch_grid, caption="Batch example")
        wandb_logger.log({"examples": images})

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.forward(images)

        metrics = self.valid_metrics(preds, targets)
        self.log_dict(metrics, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.forward(images)

        metrics = self.test_metrics(preds, targets)
        self.log_dict(metrics, on_step=False, on_epoch=True)

        return (images, preds, targets)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.config["learning_rate"])
        return optimizer

    # TODO We shouldn't use a separate function for this.
    def build_model(self):
        model_raw = torchvision.models.__dict__[self.config["architecture"]]
        model = model_raw(pretrained=self.config["pretrained"])
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model

    def make_grid(self, data):
        return tensor_to_image(torchvision.utils.make_grid(data, nrow=8))
