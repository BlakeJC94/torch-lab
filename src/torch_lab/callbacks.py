"""Callbacks implemented for tasks."""

import json
import logging
import time
from pathlib import Path

import pytorch_lightning as pl
import torch

try:
    from clearml import OutputModel, Task
except ImportError:
    Task, OutputModel = None, None

logger = logging.getLogger(__name__)


class EpochProgress(pl.Callback):
    """Dead simple callback to print a message when an epoch completes (a quieter alternative to the
    progress bar).
    """

    @staticmethod
    def num_batches(val):
        if isinstance(val, list) and len(val) == 1:
            return val[0]
        return val

    def on_train_start(self, trainer, module):
        logger.info(
            f"Starting training with {self.num_batches(trainer.num_training_batches)} batches"
        )

    def on_validation_start(self, trainer, module):
        logger.info(
            f"Starting validation with {self.num_batches(trainer.num_val_batches)} batches"
        )

    def on_train_epoch_end(self, trainer, module):
        logger.info(f"Finished epoch {module.current_epoch + 1:04}")


class NanMonitor(pl.Callback):
    """Raise if any Nans are encountered"""

    def check(self, batch_idx, batch, outputs=None):
        outputs = outputs or {}
        to_check = {
            "x": batch[0],
            "y": batch[1]["y"],
            "y_pred": outputs.get("y_pred"),
            "loss": outputs.get("loss"),
            "out": outputs.get("out"),
        }
        for k, v in to_check.items():
            if v is None:
                continue
            v = torch.isnan(v)
            if v.ndim > 1:
                v = v.flatten(1).any(1)
            if v.any():
                nan_idxs_str = ""
                if v.ndim > 0:
                    nan_idxs = [i for i, b in enumerate(v) if b]
                    nan_idxs_str = ", ".join([str(idx) for idx in nan_idxs[:5]])
                    if len(nan_idxs) > 5:
                        nan_idxs_str += f", ... [{len(nan_idxs)}]"
                raise ArithmeticError(
                    f"Encountered NaN in '{k}' for batch {batch_idx} (samples {nan_idxs_str})"
                )

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.check(batch_idx, batch, outputs)

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.check(batch_idx, batch, outputs)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.check(batch_idx, batch, outputs)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.check(batch_idx, batch, outputs)


class ClearMLModelCheckpoint(pl.callbacks.ModelCheckpoint):
    """Base ModelCheckpoint with ClearML Hooks."""

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        task = Task.current_task()
        if task:
            self.upload_weights_to_task(task, trainer, filepath)

    @staticmethod
    def upload_weights_to_task(task, trainer, filepath):
        metrics = {
            "time": time.time(),
            "epoch": trainer.current_epoch
            ** {k: v.cpu().item() for k, v in trainer.callback_metrics.items()},
        }

        output_model = OutputModel(
            task=task,
            name=task.name,
            config_text=json.dumps(metrics, indent=2),
        )
        output_model.connect(task=task)

        output_model.update_weights(
            weights_filename=str(filepath),
            iteration=trainer.global_step,
            auto_delete_file=False,
        )
        output_model.wait_for_uploads()

    def on_exception(self, trainer, pl_module, exception):
        logger.warning(
            f"Encountered '{str(exception)}', attempting to save weights to ClearML"
        )

        task = Task.current_task()
        if not task:
            logger.info("No active ClearML task detected, exiting.")
            return

        for filepath in [self.best_model_path, self.last_model_path]:
            if Path(filepath).is_file():
                try:
                    self.upload_weights_to_task(task, trainer, filepath)
                except Exception as err:
                    logger.error(f"Couldn't upload '{filepath}': {str(err)}")
