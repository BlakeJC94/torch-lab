from copy import deepcopy
from typing import Any, Callable, Dict, Optional, TypeAlias

import matplotlib
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from plotly import graph_objects as go
from torch import nn
from torch.optim import Optimizer, lr_scheduler
from torchmetrics import Metric

try:
    from clearml import Logger
except ImportError:
    Logger = None

LRScheduler: TypeAlias = lr_scheduler._LRScheduler


class LabModule(pl.LightningModule):
    """Wrapper class for Pytorch modules for use with DataLoaders wrapped around implementations of
    BaseDatasets.

    Models are implemented as a single attribute, and the checkpoint will save the state dict with
    key names that will work natively with the model outside the pl.module class.
    """

    def __init__(
        self,
        model: nn.Module,
        output_transform: Optional[Callable] = None,
    ):
        """Initialise LabModule.

        Args:
            model: PyTorch module to call in the forward method.
            output_transform: Transform to apply to outputs in the predict methods. Must take
                (data_batch, metadata_batch) tuple and output another (output_batch, metadata_batch)
                tuple.
        """
        super().__init__()
        self.model = model
        self.output_transform = output_transform

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def on_save_checkpoint(self, checkpoint):
        for key in list(checkpoint["state_dict"].keys()):
            value = checkpoint["state_dict"].pop(key)
            key_new = key.removeprefix("model.")
            checkpoint["state_dict"][key_new] = value

    def on_load_checkpoint(self, checkpoint):
        for key in list(checkpoint["state_dict"].keys()):
            value = checkpoint["state_dict"].pop(key)
            key_new = f"model.{key}"
            checkpoint["state_dict"][key_new] = value

    def apply_transform_to_batch(self, y_pred, md):
        out = None
        if self.output_transform is not None:
            out, md = self.output_transform(y_pred, md)
        return out, md

    ## Predict methods
    def predict_step(self, batch, batch_idx, _dataloader_idx=0):
        x, md = batch
        y_pred = self(x)
        out, md = self.apply_transform_to_batch(y_pred, md)
        return {"md": md, "y_pred": y_pred, "out": out}


class TrainLabModule(LabModule):
    """Wrapper class for Pytorch modules for use with DataLoaders wrapped around implementations of
    BaseDatasets.

    Models are implemented as a single attribute, and the checkpoint will save the state dict with
    key names that will work natively with the model outside the pl.module class.

    Training-specific attributes are implemented in this variant:
        - The loss function is a callable that maps batches (y_pred, y) to a float
        - The optimizer config is a dict the following keys:
            - "optimizer" (type from torch.optim)
            - "optimizer_kwargs" (optional, dict),
            - "scheduler" (optional, type from torch.optim.lr_scheduler)
            - "scheduler_kwargs" (optional, dict),
            - "monitor": (optional, str)
        - The metrics are passed as a dict mapping names to nn.Modules with an update/compute
              methods (e.g. torchmetrics).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_function: Callable,
        optimizer_config: Dict[str, Optimizer | LRScheduler | Dict[str, Any]],
        metrics: Optional[Dict[str, Metric]] = None,
        output_transform: Optional[Callable] = None,
    ):
        """Initialise a TrainLabModule.

        Args:
            model: PyTorch module to call in the forward method.
            loss_function: Loss function to call for training and validation batches.
            optimizer_config: Configuration for optimizer, and also an optional scheduler. Dict must
                have keys 'optimizer' (maps to class in torch.optim), 'optimizer_kwargs'. If
                using a scheduler, dict should also contain 'scheduler' (maps to class in
                torch.optim.lr_schedulers), 'scheduler_kwargs', and 'monitor'.
            metrics: A dict of {metric_name: function}. Functions should accept 2 args:
                predictions and labels, and return a scalar number.
            output_transform: Transform to apply to outputs in the validation and test methods.
                Must take (data_batch, metadata_batch) tuple and output another (output_batch,
                metadata_batch) tuple.
        """
        super().__init__(model, output_transform=output_transform)
        self.loss_function = loss_function
        self.optimizer_config = optimizer_config

        metrics = metrics or {}
        self.metrics = nn.ModuleDict(
            {
                f"{k}_metrics": nn.ModuleDict(deepcopy(metrics))
                for k in ["train", "sanity_check", "validate", "test", "predict"]
            }
        )

        self.save_hyperparameters(
            ignore=[
                "loss_function",
                "optimizer_factory",
                "scheduler_factory",
                "metrics",
                "model",
            ],
        )

    def configure_optimizers(self) -> Dict[str, Optimizer | LRScheduler]:
        """Return your favourite optimizer."""
        out = {}

        optimizer_class = self.optimizer_config["optimizer"]
        out["optimizer"] = optimizer_class(
            filter(lambda p: p.requires_grad, self.parameters()),
            **self.optimizer_config.get("optimizer_kwargs", {}),
        )

        if (scheduler_class := self.optimizer_config.get("scheduler")) is not None:
            out["lr_scheduler"] = {
                "scheduler": scheduler_class(
                    out["optimizer"],
                    **self.optimizer_config.get("scheduler_kwargs", {}),
                ),
                "monitor": self.optimizer_config["monitor"],
            }

        return out

    ## Metric and loss logging methods
    def loss_calculate_and_log(self, y_pred: Any, md: Any) -> torch.Tensor:
        y = md["y"]
        stage = self.get_stage()
        loss = self.loss_function(y_pred, y)
        if isinstance(loss, dict):
            self._log_metric(loss, "loss", stage, batch_size=len(y_pred))
            loss = torch.stack(list(loss.values())).mean()
        self.log(
            f"loss/{stage}",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(y_pred),
        )
        return loss

    @torch.no_grad()
    def metrics_update_and_log(self, y_pred: Any, md: Any) -> None:
        """Calculate and log metrics for a batch of predictions against target labels."""
        stage = self.get_stage()

        metrics = getattr(self.metrics, f"{stage}_metrics", {})
        for metric_name, metric in metrics.items():
            try:
                metric.update(y_pred, md)
            except Exception as err:
                raise ValueError(
                    f"Error when updating metric '{metric_name}': {str(err)}"
                ) from err
            if getattr(metric, "compute_on_batch", True):
                self._log_metric(metric, metric_name, stage, batch_size=len(y_pred))

    @torch.no_grad()
    def metrics_compute_and_log(self):
        stage = self.get_stage()
        metrics = getattr(self.metrics, f"{stage}_metrics", {})
        for metric_name, metric in metrics.items():
            self._log_metric(metric, metric_name, stage, epoch=True)
            metric.reset()

    def _log_metric(self, metric, name, stage, batch_size=None, epoch=False):
        try:
            result = metric.compute() if isinstance(metric, Metric) else metric
        except Exception as err:
            raise ValueError(
                f"Error when computing metric '{name}': {str(err)}"
            ) from err

        # Metrics used purely for side-effects (e.g., plotting) can return None and won't be logged
        # If metrics return a dict of results, train/val metrics are separated into different plots
        if isinstance(result, dict):
            for k, v in result.items():
                self.log(f"{name} ({stage})/{k}", v, batch_size=batch_size)
        elif result is not None:
            self.log(f"{name}/{stage}", result, batch_size=batch_size)

        if (
            epoch
            and hasattr(metric, "plot")
            and (Logger is not None)
            and (clearml_logger := Logger.current_logger()) is not None
        ):
            try:
                plot = metric.plot()
            except NotImplementedError:
                plot = None
            except Exception as err:
                raise ValueError(
                    f"Error when plotting metric '{name}': {str(err)}"
                ) from err
            if isinstance(plot, go.Figure):
                clearml_logger.report_plotly(
                    f"{name} ({stage})",
                    stage,
                    iteration=self.current_epoch,
                    figure=plot,
                )
            elif isinstance(plot, tuple):
                fig, _ax = plot
                clearml_logger.report_matplotlib(
                    f"{name} ({stage})",
                    stage,
                    iteration=self.current_epoch,
                    figure=fig,
                )
                plt.close(fig)

    def get_stage(self) -> str:
        return self.trainer.state.stage.value

    ## Train methods
    def training_step(self, batch, batch_idx, _dataloader_idx=0):
        x, md = batch
        y_pred = self(x)
        loss = self.loss_calculate_and_log(y_pred, md)
        self.metrics_update_and_log(y_pred, md)
        return {"loss": loss, "md": md, "y_pred": y_pred}

    def on_train_epoch_end(self):
        self.metrics_compute_and_log()

    ## Val methods
    def validation_step(self, batch, batch_idx, _dataloader_idx=0):
        x, md = batch
        y_pred = self(x)
        loss = self.loss_calculate_and_log(y_pred, md)
        self.metrics_update_and_log(y_pred, md)
        out, md = self.apply_transform_to_batch(y_pred, md)
        return {"loss": loss, "md": md, "y_pred": y_pred, "out": out}

    def on_validation_epoch_end(self):
        self.metrics_compute_and_log()

    ## Test methods
    def test_step(self, batch, batch_idx, _dataloader_idx=0):
        x, md = batch
        y_pred = self(x)
        loss = self.loss_calculate_and_log(y_pred, md)
        self.metrics_update_and_log(y_pred, md)
        out, md = self.apply_transform_to_batch(y_pred, md)
        return {"loss": loss, "md": md, "y_pred": y_pred, "out": out}

    def on_test_epoch_end(self):
        self.metrics_compute_and_log()
