import os
from typing import Optional

import pytorch_lightning as pl

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.xy_dataset import XYDataset
from utils.utils import torch2cv2, show_label

import wandb


class RoadDataModule(pl.LightningDataModule):

    def __init__(self, dataset_artifact: str, batch_size):
        super().__init__()
        self.dataset_artifact = dataset_artifact
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in dataloaders
        train_pth, val_pth, test_pth = [os.path.join(self.artifact_dir, split)
                                        for split in ["train", "val", "test"]]

        if stage == 'fit' or stage is None:
            self.train = XYDataset(train_pth, train=True)
            self.val = XYDataset(val_pth, train=False)

            self.dims = tuple(self.train[0][0].size())

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test = XYDataset(test_pth, train=False)

            self.dims = tuple(self.test[0][0].size())

    def train_dataloader(self):
        return self.make_loader(self.train, True)

    def val_dataloader(self):
        return self.make_loader(self.val, False)

    def test_dataloader(self):
        return self.make_loader(self.test, False)

    def make_loader(self, dataset, shuffle):
        return DataLoader(dataset=dataset,
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          pin_memory=True, num_workers=8)

    def test_epoch_end(self, test_step_outputs):
        images, predictions, targets = self.concat_test_outputs(
                                        test_step_outputs)

        # compute loss for each image of the test set
        losses = F.mse_loss(predictions, targets, reduction='none')

        test_table = self.create_table(images, predictions, targets, losses)

        wandb.log({"test/predictions": test_table})

    def create_table(self, images, predictions, targets, losses):
        # display preds and targets on images
        images_with_preds = []
        for idx, image in enumerate(images):
            img = torch2cv2(image)

            # show ground truth and prediction on the image
            img = show_label(img, targets[idx])
            img = show_label(img, predictions[idx], (0, 0, 255))

            images_with_preds.append(img)

        # create a WandB table
        my_data = [
            [wandb.Image(img), pred, target, loss.sum()]
            for img, pred, target, loss
            in zip(images_with_preds, predictions, targets, losses)
        ]

        columns = ["image", "prediction", "target", "loss"]
        table = wandb.Table(data=my_data, columns=columns)

        return table

    def concat_test_outputs(self, test_step_outputs):
        """
        Concatenate the output of the test step
        so that we can easily iterate on it and
        compute the loss for each item in one go.
        """
        images, predictions, targets = test_step_outputs[0]
        for i in range(1, len(test_step_outputs)):
            imgs, preds, targs = test_step_outputs[i]

            images = torch.cat((images, imgs), dim=0)
            predictions = torch.cat((predictions, preds), dim=0)
            targets = torch.cat((targets, targs), dim=0)

        return images, predictions, targets

    def prepare_data(self):
        artifact = wandb.use_artifact(self.dataset_artifact)
        self.artifact_dir = artifact.download()
