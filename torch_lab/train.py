import logging
import re
from pathlib import Path
from os import PathLike
from typing import Optional, Union

import pytorch_lightning as pl
from clearml import Task
from pytorch_lightning.loggers import TensorBoardLogger

from torch_lab.core.config import Config

from .paths import import_model_config, ARTIFACTS_DIR

# TODO
# task = Task.init(project_name="my project", task_name="my task")

logger = logging.getLogger(__name__)
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)

CLEARML_CONFIG_PATH = Path.home() / "clearml.conf"


def get_prior_experiments(config: Config):
    return Task.get_tasks(task_name=rf"^{config.task_name}$")  # Uses regex to search

def verify_task_id(task_id: str):
    if re.match(r"^[a-f0-9]{32}$", task_id) is None or not Task.get_task(task_id):
        raise Exception("Invalid experiment_id")

def setup_new_clearml_task(config, overwrite):
    prior_experiments = get_prior_experiments(config)
    if len(prior_experiments) > 0:
        if not overwrite:
            raise FileExistsError()
        for exp in prior_experiments:
            logger.info(f"Deleting experiment {exp.name}, ID: {exp.id}")
            exp.delete(
                delete_artifacts_and_models=True,
                skip_models_used_by_other_tasks=True,
                raise_on_error=True,
            )
    task = Task.init(project_name=config.project_name, task_name=config.task_name)
    # TODO log parameters to ClearML
    return task

# def get_experiment_artifacts(task_id):
#     model_config, task_path = None, None, None
#     task = Task.get_task(task_id=task_id)
#     task_path = ARTIFACTS_DIR / task.get_project_name() / f"{task.name}.{task.id}"

#     # temp_config_path = download_config_from_clearml(experiment, save_dir)
#     # model_config = import_model_config(temp_config_path)

#     # weights_path = download_weights_from_clearml(
#     #     experiment, best, task_path / "model_weights"
#     # )

#     ...

def parse_config_and_experiment_id(config, task_id):
    if not (config or task_id):
        raise Exception("Must provide config and/or experiment_id.")
    if task_id:
        verify_task_id(task_id)
    if config:
        config = import_model_config(config)
    return config, task_id

def train(
    config: Optional[Union[PathLike, Config]] = None,
    task_id: Optional[str] = None,
    offline: bool = False,
    overwrite: bool = False,
    **kwargs,
):
    pl.seed_everything(0, workers=True)
    config, task_id = parse_config_and_experiment_id(config, task_id)

    ckpt_path = None
    if config:
        if not offline:
            setup_new_clearml_task(config, overwrite)
        if task_id:
            raise NotImplementedError()
            task.set_parent(task_id)
            # TODO ckpt_path = setup_finetune_clearml_task(config, task_id, overwrite)
    elif not config and task_id:  # Continue
        raise NotImplementedError()
        # TODO Get config from experiment_id
        # TODO Get checkpoint
    else:
        raise Exception("NEVER REACHED")

    # IDEA: Could submit stats to ClearML using on_checkpoint hook?

    save_dir = ARTIFACTS_DIR / config.project_name / config.task_name
    save_dir.mkdir(exist_ok=True, parents=True)
    logs_path = save_dir / "logs"

    tensorboard_logger = TensorBoardLogger(
        save_dir=str(logs_path),
        name="",
        default_hp_metric=False,
    )

    trainer = pl.Trainer(
        logger=tensorboard_logger,
        callbacks=config.train_callbacks,
        num_sanity_val_steps=0,
        **kwargs,
    )

    trainer.fit(config.module, config.data_module, ckpt_path=ckpt_path)
