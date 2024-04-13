"""Example config to define a model for MNIST digit inference."""

import logging
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import Metric

from example_project import transforms as t
from example_project.callbacks import MnistWriter
from example_project.datasets import PredictDataset, TrainDataset
from torch_lab.modules import LabModule, TrainLabModule
from torch_lab.transforms import TransformCompose

logger = logging.getLogger(__name__)


## Model code
class ToyModel(nn.Sequential):
    """Example model"""

    def __init__(self, n_channels, n_features, n_classes):
        self.n_channels = n_channels
        self.n_features = n_features
        self.n_classes = n_classes

        super().__init__(
            nn.Sequential(
                nn.Conv2d(n_channels, n_features, 3),
                nn.BatchNorm2d(n_features),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.AvgPool2d(2),
                nn.Conv2d(n_features, n_classes, 3),
                nn.BatchNorm2d(n_classes),
                nn.ReLU(),
            ),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        x = super().forward(x)
        x = x.squeeze(-1).squeeze(-1)
        return x


## Common constructors
def model_config(config: Dict[str, Any]) -> nn.Module:
    n_channels = 1
    n_classes = 10
    return ToyModel(
        n_channels,
        config["n_features"],
        n_classes,
    )


def transform_config(config: Dict[str, Any]) -> nn.Module:
    return TransformCompose(
        t.Scale(1 / 255),
        lambda x, md: (x.astype("float32"), md),
    )


def num_workers(config: Dict[str, Any]) -> int:
    return min(
        config.get("num_workers", os.cpu_count() or 0),
        os.cpu_count() or 0,
    )


## Train constructors
def metrics(config: Dict[str, Any]) -> Dict[str, Metric]:
    return {
        # "mse": MeanSquaredError(),
    }


## Configs
def train_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Download train images with bash:
        $ wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -o path/to/output
        $ wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -o path/to/output

    Or with python:
        # pip install mnist
        >>> import mnist
        >>> mnist.download_file("train-images-idx3-ubyte.gz", "path/to/output")
        >>> mnist.download_file("train-labels-idx1-ubyte.gz", "path/to/output")
    """
    augmentation = t.RandomFlip(0.4)

    train_dataset = TrainDataset(
        (
            config["data"],
            config["annotations"],
        ),
        slice(None, 48000),
        transform=augmentation + transform_config(config),
    )
    val_dataset = TrainDataset(
        (
            config["data"],
            config["annotations"],
        ),
        slice(48000, None),
        transform=transform_config(config),
    )

    return dict(
        module=TrainLabModule(
            model_config(config),
            loss_function=nn.BCEWithLogitsLoss(),
            metrics=metrics(config),
            optimizer_config={
                "optimizer": optim.AdamW,
                "optimizer_kwargs": dict(
                    lr=config["learning_rate"],
                    weight_decay=config["weight_decay"],
                ),
                "scheduler": optim.lr_scheduler.MultiStepLR,
                "scheduler_kwargs": dict(
                    milestones=config["milestones"],
                    gamma=config["gamma"],
                ),
                "monitor": config["monitor"],
            },
        ),
        train_dataloaders=DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            num_workers=num_workers(config),
            shuffle=True,
        ),
        val_dataloaders=DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            num_workers=num_workers(config),
            shuffle=False,
        ),
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor=config["monitor"],
                min_delta=0.0001,
                patience=config["patience"],
                verbose=True,
                mode="min",
            ),
        ],
    )


def infer_config(
    config: Dict[str, Any],
    weights_path: str,
    test_images_path: str,
) -> Dict[str, Any]:
    """
    Download test images with bash:
        $ wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -o path/to/output

    Or with python:
        # pip install mnist
        >>> import mnist
        >>> mnist.download_file("t10k-images-idx3-ubyte.gz", "path/to/output")
    """
    test_images_path = Path(test_images_path).expanduser()

    predict_dataset = PredictDataset(
        test_images_path,
        transform=transform_config(config),
    )

    return dict(
        module=LabModule(
            model_config(config),
            transform=lambda y_pred, md: (y_pred.argmax(1).cpu().numpy(), md),
        ),
        predict_dataloaders=DataLoader(
            predict_dataset,
            batch_size=config["batch_size"],
            num_workers=num_workers(config),
            shuffle=False,
        ),
        callbacks=[
            MnistWriter("./results.csv"),
        ],
    )
