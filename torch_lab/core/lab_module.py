import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Literal
from copy import deepcopy

import pytorch_lightning as pl
import torch
from clearml import Logger
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.nn.modules.loss import _Loss as Loss
from torchmetrics import Metric

OptimizerKwargs = Dict[str, Any]
SchedulerKwargs = Dict[str, Any]
Stage = Literal["train", "val", "test", "predict"]

logger = logging.getLogger(__name__)

# TODO Add optimiser/scheduler kwargs as hparams


class LabModule(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_function: Loss,
        optimizer_config: Optional[Tuple[Optimizer, OptimizerKwargs]] = None,
        scheduler_config: Optional[Tuple[LRScheduler, SchedulerKwargs]] = None,
        hyperparams_ignore: Optional[List[str]] = None,
    ):
        super().__init__()
        self.model = model
        self.loss_function = loss_function

        # Validate optimizer and scheduler configuration
        for component in [optimizer_config, scheduler_config]:
            if component is not None:
                assert len(component) == 2
                component_class, component_kwargs = component
                assert callable(component_class)
                assert isinstance(component_kwargs, dict)

        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        self.hyperparams_ignore = hyperparams_ignore or []

        self.save_hyperparameters(
            ignore=[
                "loss_function",
                "model",
                *self.hyperparams_ignore,
            ],
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        output = {}

        if self.optimizer_config is not None:
            optmizer_class, optmizer_kwargs = self.optimizer_config
            output["optimizer"] = optmizer_class(self.parameters(), **optmizer_kwargs)

        if "optimizer" in output and self.scheduler_config is not None:
            scheduler_class, scheduler_kwargs = self.scheduler_config
            output["lr_scheduler"] = scheduler_class(
                output["optimizer"], **scheduler_kwargs
            )

        return output

    def training_step(self, batch, _batch_idx) -> Dict[str, Tensor]:
        x, y, *_md_batch = batch
        y_hat = self.model(x)
        loss = self.calculate_loss(y_hat, y)
        loss = self.log_loss(loss, batch_size=len(y_hat))
        return {"loss": loss, "preds": y_hat, "target": y}

    def validation_step(self, batch, _batch_idx) -> Dict[str, Tensor]:
        x, y, *_md_batch = batch
        y_hat = self.model(x)
        loss = self.calculate_loss(y_hat, y)
        loss = self.log_loss(loss, batch_size=len(y_hat))
        return {"loss": loss, "preds": y_hat, "target": y}

    def test_step(self, batch, _batch_idx) -> Dict[str, Tensor]:
        x, y, *_md_batch = batch
        y_hat = self.model(x)
        loss = self.calculate_loss(y_hat, y)
        loss = self.log_loss(loss, batch_size=len(y_hat))
        return {"loss": loss, "preds": y_hat, "target": y}

    def calculate_loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        loss = self.loss_function(y_hat, y)
        return loss

    def log_loss(self, loss, batch_size):
        stage = self.trainer.state.stage

        if isinstance(loss, dict):
            for k, v in loss.items():
                self.log(
                    f"loss/{k} ({stage})",
                    v,
                    batch_size=batch_size,
                )
            loss = torch.stack(list(loss.values())).mean()

        self.log(
            f"loss ({stage})",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
