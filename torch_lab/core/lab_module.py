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


class TrainMixin:
    def training_step(self, batch, _batch_idx) -> Dict[str, Tensor]:
        x, y, *_md_batch = batch
        y_hat = self.model(x)
        loss = self.calculate_loss(y_hat, y)
        return {"loss": loss, "preds": y_hat, "target": y}

    def on_train_batch_end(self, output: Dict[str, Tensor], _batch, _batch_idx):
        self.update_metrics(output["preds"], output["target"])

    def on_train_epoch_end(self):
        self.compute_metrics()


class ValMixin:
    def validation_step(self, batch, _batch_idx) -> Dict[str, Tensor]:
        x, y, *_md_batch = batch
        y_hat = self.model(x)
        loss = self.calculate_loss(y_hat, y)
        return {"loss": loss, "preds": y_hat, "target": y}

    def on_validation_batch_end(self, output: Dict[str, Tensor], _batch, _batch_idx):
        self.update_metrics(output["preds"], output["target"])
        # TODO Figure out outputs and output metrics

    def on_validation_epoch_end(self):
        self.compute_metrics()

class TestMixin:
    def test_step(self, batch, _batch_idx) -> Dict[str, Tensor]:
        x, y, *_md_batch = batch
        y_hat = self.model(x)
        loss = self.calculate_loss(y_hat, y)
        return {"loss": loss, "preds": y_hat, "target": y}

    def on_test_batch_end(self, output: Dict[str, Tensor], _batch, _batch_idx):
        self.update_metrics(output["preds"], output["target"])
        # TODO Figure out outputs and output metrics

    def on_test_epoch_end(self):
        self.compute_metrics()



class LabModule(TrainMixin, ValMixin, TestMixin, pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_function: Loss,
        optimizer_config: Optional[Tuple[Optimizer, OptimizerKwargs]] = None,
        scheduler_config: Optional[Tuple[LRScheduler, SchedulerKwargs]] = None,
        output_transforms: Optional[Callable] = None,
        metrics: Optional[Dict[str, Metric]] = None,
        # output_metrics: Optional[Dict[str, Metric]] = None,
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

        self.output_transforms = output_transforms

        self.metrics_train = metrics
        self.metrics_sanity_check = deepcopy(metrics)
        self.metrics_validate = deepcopy(metrics)
        self.metrics_test = deepcopy(metrics)

        # self.output_metrics_validate = output_metrics
        # self.output_metrics_test = deepcopy(output_metrics)

        self.hyperparams_ignore = hyperparams_ignore
        if hyperparams_ignore is None:
            hyperparams_ignore = []

        self.save_hyperparameters(
            ignore=["loss_function", "metrics", "model"] + hyperparams_ignore
        )

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

    def _log_result(self, result, name, batch_size):
        stage = self.trainer.state.stage
        if result is not None:
            key = f"{name}/{stage}"
            # if name == "loss" and (stage == "sanity_check" or stage == 'validate'):
            #     logger.info(f"TRACE: {key=}, {result=}")
            self.log(key, result, batch_size=batch_size)

    @torch.no_grad()
    def update_metrics(self, y_hat: torch.Tensor, y: torch.Tensor):
        stage = self.trainer.state.stage
        metrics = getattr(self, f"metrics_{stage}")

        for metric_name, metric in metrics.items():
            metric.update(y_hat, y)

            if getattr(metric, "compute_on_batch", True):
                result = metric.compute() if isinstance(metric, Metric) else metric
                self._log_result(result, metric_name, batch_size=len(y_hat))

    @torch.no_grad()
    def compute_metrics(self):
        stage = self.trainer.state.stage
        metrics = getattr(self, f"metrics_{stage}")

        for metric_name, metric in metrics.items():
            result = metric.compute() if isinstance(metric, Metric) else metric
            self._log_result(result, metric_name, batch_size=None)

            if (
                hasattr(metric, "plot")
                and (clearml_logger := Logger.current_logger()) is not None
            ):
                clearml_logger.report_plotly(
                    f"{metric_name} ({stage})",
                    stage,
                    metric.plot(),
                    iteration=self.current_epoch,
                )

    def calculate_loss(self, y_hat: Tensor, y: Tensor) -> Tensor:
        loss = self.loss_function(y_hat, y)
        self._log_result(loss, "loss", batch_size=len(y_hat))
        return loss
