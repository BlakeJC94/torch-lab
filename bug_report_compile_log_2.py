import logging
import os
from math import ceil
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Optional, List, Dict

import mnist
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, Tensor
from torchmetrics import Accuracy, Metric
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss as Loss
from torch.utils.data import (
    Dataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    Sampler,
)
from torch.utils.data import TensorDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Comment out to stop suppressing chatty dynamo logs
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)

COMPILE = True

pl.seed_everything(0, workers=True)


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        num_workers: int = os.cpu_count(),
        batch_size_train: int = 16,
        batch_size_test: int = 16,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_workers = num_workers
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.save_hyperparameters("num_workers", "batch_size_train", "batch_size_test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            sampler=RandomSampler(self.train_dataset),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_test,
            num_workers=self.num_workers,
            sampler=SequentialSampler(self.val_dataset),
        )


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 10)
        self.val_loss_values = list()

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("loss/train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("loss/validate", loss)
        self.val_loss_values.append((batch_idx, loss))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)



train_images = torch.as_tensor(mnist.train_images()).float() / 255
train_labels = torch.as_tensor(mnist.train_labels()).long()
val_images = torch.as_tensor(mnist.test_images()).float() / 255
val_labels = torch.as_tensor(mnist.test_labels()).long()

data_module = MyDataModule(
    train_dataset=TensorDataset(train_images, train_labels),
    val_dataset=TensorDataset(val_images, val_labels),
    num_workers=8,
    batch_size_train=256,
    batch_size_test=256,
)

module = LitModel()

callbacks = [
    EarlyStopping(
        monitor="loss/validate",
        min_delta=0.001,
        patience=3,
        verbose=True,
        mode="min",
    ),
    ModelCheckpoint(
        monitor="loss/validate",
    ),
]

save_dir = Path("./artifacts/bugreport")
save_dir.mkdir(exist_ok=True, parents=True)
logs_path = save_dir / "logs"
tensorboard_logger = TensorBoardLogger(
    save_dir=str(logs_path),
    name="",
    default_hp_metric=False,
)

trainer = pl.Trainer(
    logger=tensorboard_logger,
    callbacks=callbacks,
    # num_sanity_val_steps=0,
)

if COMPILE:
    module = torch.compile(module)

trainer.fit(module, data_module)

print(f"{trainer.callback_metrics=}")
print(f"{module.val_loss_values}")
