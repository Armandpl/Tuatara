import pytorch_lightning as pl
import torchmetrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

config = dict(
    architecture="resnet34",
    pretrained=True,
    learning_rate=1e-4,
    loss="mse_loss"
)


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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.forward(images)
        loss = getattr(F, self.config["loss"])(preds, targets)

        metrics = self.train_metrics(preds, targets)
        self.log_dict(metrics, on_step=True, on_epoch=False)

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
