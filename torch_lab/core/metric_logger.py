import logging
from copy import deepcopy

import torch
from clearml import Logger
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class LabMetricLogger(Callback):
    def __init__(self, metrics):
        metrics = torch.nn.ModuleDict(metrics or {})
        self.metrics_train = metrics
        self.metrics_validate = deepcopy(metrics)
        self.metrics_test = deepcopy(metrics)

    def unpack_outputs(self, outputs):
        if isinstance(outputs, dict):
            return outputs["y_hat"], outputs["y"]
        _, y_hat, y = outputs
        return y_hat, y

    @torch.no_grad()
    def update_metrics(self, y_hat, y, trainer, pl_module):
        stage = trainer.state.stage
        metrics = getattr(self, f"metrics_{stage}")
        batch_size = len(y_hat)

        for metric_name, metric in metrics.items():
            metric.update(y_hat, y)
            if getattr(metric, "compute_on_batch", True):
                result = metric.compute()
                self._log_metric(metric_name, result, trainer, batch_size=batch_size)

    @torch.no_grad()
    def compute_metrics(self, trainer, pl_module):
        stage = trainer.state.stage
        metrics = getattr(self, f"metrics_{stage}")

        for metric_name, metric in metrics.items():
            result = metric.compute()
            self._log_metric(metric_name, result, trainer)

            if (
                hasattr(metric, "plot")
                and (clearml_logger := Logger.current_logger()) is not None
            ):
                clearml_logger.report_plotly(
                    f"{metric_name} ({stage})",
                    stage,
                    metric.plot(),
                    iteration=trainer.current_epoch,
                )

    def _log_metric(self, name, value, trainer, **kwargs):
        stage = trainer.state.stage
        if not isinstance(value, dict):
            key = f"{name} ({stage})"
            self.log(key, value, **kwargs)
        else:
            for k, v in value.items():
                key = f"{name}/{k} ({stage})"
                self.log(key, v, **kwargs)

    ## Train methods
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        y_hat, y = self.unpack_outputs(outputs)
        self.update_metrics(y_hat, y, trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        self.compute_metrics(trainer, pl_module)

    ## Validation methods
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        y_hat, y = self.unpack_outputs(outputs)
        self.update_metrics(y_hat, y, trainer, pl_module)
        # TODO Figure out outputs and output metrics

    def on_validation_epoch_end(self, trainer, pl_module):
        self.compute_metrics(trainer, pl_module)

    ## Test methods
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        y_hat, y = self.unpack_outputs(outputs)
        self.update_metrics(y_hat, y, trainer, pl_module)
        # TODO Figure out outputs and output metrics

    def on_test_epoch_end(self, trainer, pl_module):
        self.compute_metrics(trainer, pl_module)
