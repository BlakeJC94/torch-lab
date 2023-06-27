from typing import Dict, List, Optional, Tuple, Any, Callable, Literal
from copy import deepcopy

import pytorch_lightning as pl
import torch
from clearml import Logger
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.nn.modules.loss import _Loss as Loss
from torchmetrics import Metric

OptimizerKwargs = Dict[str, Any]
SchedulerKwargs = Dict[str, Any]
Stage = Literal["train", "val", "test", "predict"]


class TrainMixin:
    def training_step(self, batch, _batch_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y, *_md_batch = batch
        y_hat = self.model(x)
        return y_hat, y

    def training_step_end(
        self,
        step_output: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y_hat, y = step_output
        loss = self.calculate_loss(y_hat, y, "train")
        self.update_metrics(y_hat, y, "train")
        return {"loss": loss}

    def training_epoch_end(self, _outputs: List[Any]):
        self.compute_and_log_metrics("train")


class ValMixin:
    def validation_step(self, batch, _batch_idx):
        x, y, *md_batch = batch
        y_hat = self.model(x)
        return y_hat, y, md_batch

    def validation_step_end(self, step_output):
        y_hat, y, md_batch = step_output
        loss = self.calculate_loss(y_hat, y, "val")
        self.update_metrics(y_hat, y, "val")
        step_output = {"loss": loss, "predictions": y_hat}
        if md_batch:
            md_batch = md_batch[0]
            out_batch = self.calculate_outputs(y_hat, md_batch)
            self.update_output_metrics(out_batch, md_batch, "val")
            step_output["outputs"] = out_batch
            step_output["metadata"] = md_batch
        return step_output

    def validation_epoch_end(self, outputs: List[Any]):
        self.compute_metrics("val")
        self.compute_output_metrics("val")


class LabModule(TrainMixin, ValMixin, pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_function: Loss,
        optimizer_config: Optional[Tuple[Optimizer, OptimizerKwargs]] = None,
        scheduler_config: Optional[Tuple[LRScheduler, SchedulerKwargs]] = None,
        output_transforms: Optional[Callable] = None,
        metrics: Optional[Dict[str, Metric]] = None,
        output_metrics: Optional[Dict[str, Metric]] = None,
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

        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        self.test_metrics = deepcopy(metrics)

        self.val_output_metrics = output_metrics
        self.test_output_metrics = deepcopy(output_metrics)

        self.hyperparams_ignore = hyperparams_ignore
        if hyperparams_ignore is None:
            hyperparams_ignore = []
        self.save_hyperparameters(
            ignore=["loss_function", "metrics", "model"] + hyperparams_ignore
        )

    def configure_optimizers(self):
        optimizer = None
        if self.optimizer_config is not None:
            optmizer_class, optmizer_kwargs = self.optimizer_config
            optimizer = optmizer_class(self.parameters(), **optmizer_kwargs)

        scheduler = None
        if optimizer is not None and self.scheduler_config is not None:
            scheduler_class, scheduler_kwargs = self.scheduler_config
            scheduler = scheduler_class(scheduler_class, **scheduler_kwargs)

        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def _log_metric(self, metric, name, stage, batch_size):
        result = metric.compute() if isinstance(metric, Metric) else metric
        if isinstance(result, dict):
            # There's a log_dict method to use?
            for k, v in result.items():
                self.log(f"{name} ({stage})/{k}", v, batch_size=batch_size)
        elif result is not None:
            self.log(f"{name}/{stage}", result, batch_size=batch_size)

    @torch.no_grad()
    def update_metrics(self, y_hat: torch.Tensor, y: torch.Tensor, stage: Stage):
        metrics = getattr(self, f"{stage}_metrics")
        for metric_name, metric in metrics.items():
            metric.update(y_hat, y)
            if getattr(metric, "compute_on_batch", True):
                self._log_metric(metric, metric_name, stage, batch_size=len(y_hat))

    @torch.no_grad()
    def compute_metrics(self, y_hat: torch.Tensor, y: torch.Tensor, stage: Stage):
        metrics = getattr(self, f"{stage}_metrics")
        for metric_name, metric in metrics.items():
            self._log_metric(metric, metric_name, stage)
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
