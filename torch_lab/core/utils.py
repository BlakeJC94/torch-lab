import shutil
import logging
import re
from pathlib import Path
from typing import List

from clearml import Task

from ..paths import ARTIFACTS_DIR


logger = logging.getLogger(__name__)


def verify_task_id(task_id: str):
    if re.match(r"^[a-f0-9]{32}$", task_id) is None or not Task.get_task(task_id):
        raise Exception("Invalid experiment_id")


def delete_tasks(prior_tasks: List[Task]):
    for task in prior_tasks:
        logger.info(f"Deleting experiment {task.name}, ID: {task.id}")
        task.delete(
            delete_artifacts_and_models=True,
            skip_models_used_by_other_tasks=True,
            raise_on_error=False,
        )


def check_existing_tasks(config, overwrite):
    prior_tasks = Task.get_tasks(
        task_name=rf"^{config.task_name}$"
    )  # Uses regex to search
    if len(prior_tasks) > 0:
        if not overwrite:
            raise FileExistsError()
        delete_tasks(prior_tasks)


def init_task(config, prior_task_id=None):
    continue_last_task = False
    if prior_task_id:
        continue_last_task = True
        # ClearML will let you resume training from a published experiment - prevent this
        if Task.get_task(prior_task_id).status == "published":
            raise ValueError(
                f"Experiment {prior_task_id} is already published, continuing training "
                f"would violate immutability."
            )

    task = Task.init(
        project_name=config.project_name,
        task_name=config.task_name,
        auto_connect_frameworks=dict(
            matplotlib=True,
            pytorch=False,
            tensorboard=True,
        ),
        auto_connect_arg_parser=False,
        reuse_last_task_id=prior_task_id or False,
        continue_last_task=continue_last_task,
    )
    return task


def connect_hparams_to_task(task, config, config_path, cli_params):
    if config_path is not None and Path(config_path).exists():
        task.connect_configuration(
            str(config_path),
            name="Config",
            description="Recipe for model",
        )

    task.connect(cli_params, "Args")
    task.connect(config.module.hparams, "Model")
    task.connect(config.data_module.hparams, "Data")

    task.add_tags(config.tags)
    task.set_user_properties(properties=config.user_properties)
    return task


def get_checkpoint_from_task(task_id, ckpt_name, ckpt_dir=None):
    task = Task.get_task(task_id=task_id)
    output_models = task.models["output"]
    selected_model = next((m for m in output_models if m.name == ckpt_name), None)
    if selected_model is None:
        raise FileNotFoundError()
    tmp_ckpt_path = Path(selected_model.get_weights())
    shutil.move(tmp_ckpt_path, ckpt_dir)
    return ckpt_dir / tmp_ckpt_path.name


def get_config_from_task(task_id):
    task = Task.get_task(task_id=task_id)
    ckpt_dir = ARTIFACTS_DIR / task.get_project_name() / task.name
    config_path = Path(ckpt_dir) / "config.py"
    with open(config_path, "w", encoding="utf-8") as weights_file:
        weights_file.write(task.get_configuration_object("Config"))
    return import_model_config(config_path)
