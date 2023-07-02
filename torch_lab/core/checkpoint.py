import json
import time
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from clearml import Task, OutputModel


class ClearmlModelCheckpoint(ModelCheckpoint):
    def __init__(self, task: Task, monitor: str):
        super().__init__(
            monitor=monitor,
            filename="best_model",
            auto_insert_metric_name=False,
            save_last=True,
        )
        self.task = task

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        if not self.task:
            return
        metrics = {"time": time.time(), "epoch": trainer.current_epoch}
        metrics.update({k: v.cpu().item() for k, v in trainer.callback_metrics.items()})
        output_model = OutputModel(
            task=self.task,
            name=Path(filepath).stem,
            config_text=json.dumps(metrics, indent=2),
        )
        output_model.connect(task=self.task)
        output_model.update_weights(
            weights_filename=filepath,
            iteration=trainer.global_step,
            auto_delete_file=False,
        )

