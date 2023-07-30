import logging
from pathlib import Path
from os import PathLike
from typing import Optional, Union

import pytorch_lightning as pl
from clearml import Task
from pytorch_lightning.loggers import TensorBoardLogger

from torch_lab.core.config import Config
from torch_lab.core.checkpoint import ClearmlModelCheckpoint
from torch_lab.core.metric_logger import LabMetricLogger
from torch_lab.core.utils import (
    verify_task_id,
    get_checkpoint_from_task,
    check_existing_tasks,
    init_task,
    connect_hparams_to_task,
    get_config_from_task,
)
from .paths import import_model_config, ARTIFACTS_DIR

logger = logging.getLogger(__name__)
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)

CLEARML_CONFIG_PATH = Path.home() / "clearml.conf"


def train(
    config_path: Optional[Union[PathLike, Config]] = None,
    config: Optional[Config] = None,
    task_id: Optional[str] = None,
    offline: bool = False,
    overwrite: bool = False,
    ckpt_monitor: str = "loss/validate",
    ckpt_name: str = "last",
    **trainer_kwargs,
):
    cli_params = locals()
    pl.seed_everything(0, workers=True)

    if not CLEARML_CONFIG_PATH.exists():
        raise FileNotFoundError("~/clearml.conf not found.")

    config = import_model_config(config_path) if config_path else config
    task_id = verify_task_id(task_id) if task_id else None
    if not (config or task_id):
        raise Exception("Must provide config and/or experiment_id.")

    # Get ckpt for finetune/continue
    ckpt_path, prior_task = None, None
    if task_id:
        prior_task = Task.get_task(task_id)
        ckpt_dir = ARTIFACTS_DIR / f"{config.project_name}/{config.task_name}"
        ckpt_path = get_checkpoint_from_task(prior_task, ckpt_name, ckpt_dir)

    # Setup ClearML
    task = None
    if config:  # New or FineTune
        if not offline:
            check_existing_tasks(config, overwrite)
            task = init_task(config)
            task = connect_hparams_to_task(task, config, config_path, cli_params)
    elif prior_task:  # Cont
        config = get_config_from_task(prior_task)
        if not offline:
            task = init_task(config, prior_task_id=task_id)

    tensorboard_logger = TensorBoardLogger(
        save_dir=ARTIFACTS_DIR,
        name=f"{config.project_name}/{config.task_name}",
        default_hp_metric=False,
    )

    callbacks = config.callbacks.get("train", [])
    callbacks.append(LabMetricLogger(metrics=config.metrics))
    if not offline:
        callbacks.append(ClearmlModelCheckpoint(task=task, monitor=ckpt_monitor))

    trainer = pl.Trainer(
        logger=tensorboard_logger,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    trainer.fit(
        config.module,
        config.data_module,
        ckpt_path=ckpt_path,
    )
