"""Logger used for managing tasks on ClearML."""

import logging
import re
from pathlib import Path
from typing import Any, Dict

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from torch_lab.paths import get_task_dir_name

try:
    from clearml import Task
except ImportError:
    Task = None

logger = logging.getLogger(__name__)


class ClearMlLogger(TensorBoardLogger):
    def __init__(
        self,
        hparams: Dict[str, Any],
        config_path: str | Path,
        save_dir: str | Path,
        **kwargs: Any,
    ):
        save_dir = Path(save_dir)

        self.setup_task(hparams, config_path, save_dir.name)

        save_dir = save_dir.parent / f"{save_dir.name}-{self.task.id}"
        save_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(
            save_dir=save_dir,
            name="",
            version=None,
            log_graph=False,
            default_hp_metric=False,
            prefix="",
            sub_dir=None,
            **kwargs,
        )

    def setup_task(self, hparams, config_path, task_name) -> Task:
        logger.info(f"Task name: {task_name}")

        # Start ClearML
        task_init_kwargs = {
            "continue_last_task": False,
            "reuse_last_task_id": False,
            "auto_connect_frameworks": {
                "matplotlib": True,
                "pytorch": False,
                "tensorboard": True,
            },
            **hparams.get("task", {}),
            "task_name": task_name,
        }
        task = Task.init(**task_init_kwargs)

        ckpt_params = hparams.get("checkpoint", {})
        checkpoint_task_id = ckpt_params.get("checkpoint_task_id")
        if checkpoint_task_id:
            task.set_parent(checkpoint_task_id)

        # Connect configurations
        task.connect_configuration(config_path, "config")
        task.connect(hparams, "hparams")

        self.task = task
