import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torch

import wandb

from models import RoadRegression
from data import RoadData

models = {
    "RoadRegression": RoadRegression.RoadRegression
}


def main(config):

    with wandb.init(project="racecar", config=config,
                    job_type="train", entity="wandb") as run:

        config = run.config

        dm = RoadData.RoadDataModule(config.dataset, config.batch_size)
        model = models[config.model_name]()

        wandb_logger = WandbLogger()

        trainer = pl.Trainer(
            logger=wandb_logger,
            gpus=1,
            max_epochs=config.epochs,
            log_every_n_steps=1
        )

        trainer.fit(model, dm)

        # finally we log the model to wandb.
        torch.save(model.model.state_dict(), "model.pth")
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file('model.pth')
        run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project Tuatara - '
                                                 'Train a model on a dataset '
                                                 'to win robocars competition')
    parser.add_argument('model_name', choices=['RoadRegression'],
                        help='Choose a model to train')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--dataset', type=str, default="mix_ready:latest",
                        help='Dataset to train on')
    args = parser.parse_args()

    main(args)
