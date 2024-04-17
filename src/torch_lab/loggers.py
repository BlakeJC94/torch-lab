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
        task_name: str,
        root_dir: str | Path,
        **kwargs: Any,
    ):
        root_dir = Path(root_dir)

        task = self.setup_task(hparams, config_path, task_name)

        save_dir = root_dir / f"{get_task_dir_name(task)}/logs"
        save_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(
            save_dir=str(save_dir),
            name="",
            version=None,
            log_graph=False,
            default_hp_metric=False,
            prefix="",
            sub_dir=None,
            **kwargs,
        )

        self.task = task

    @property
    def version(self):
        prev_tasks = Task.get_tasks(
            project_name=self.project_name,
            task_name=f"^{self.task_name}",
        )
        max_task_v = -1
        for t in prev_tasks:
            version_suffix = t.name.split("-", 2)[-1]
            version_search = re.search(r"\d+", version_suffix)
            if version_search is None:
                continue
            version = int(version_search.group(0))
            max_task_v = max(max_task_v, version)
        return str(max_task_v + 1)

    def setup_task(self, hparams, config_path, task_name) -> Task:
        task_init_kwargs = hparams.get("task", {})

        project_name = task_init_kwargs.get("project_name", "unnamed")
        task_base_name, task_stem_name = task_name.split("-", 1)
        for k, v in {
            "project name": project_name,
            "task base name": task_base_name,
            "task stem name": task_stem_name,
        }.items():
            if "-" in v:
                raise ValueError(f"The character '-' is forbidden in the {k} ('{v}')")

        # Increment version of task
        self.project_name = project_name
        self.task_name = task_name
        task_name = "-v".join([task_name, self.version])
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
            **task_init_kwargs,
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
        return task
