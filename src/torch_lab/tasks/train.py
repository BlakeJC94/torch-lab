"""Choo choo

Usage:

    $ train <path/to/hparams> [--gpu-devices <int> <int> ...] [--dev-run <float or int>] [--offline]

"""

import argparse
import concurrent.futures as cf
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from clearml import Model, Task
from torch_lab.callbacks import ClearMLModelCheckpoint, ClearMLTaskMarker, EpochProgress, NanMonitor
from torch_lab.loggers import ClearMlLogger
from torch_lab.paths import ARTIFACTS_DIR, get_task_dir_name
from torch_lab.utils import compile_config, dict_as_str, get_hparams_and_config_path

logger = logging.getLogger(__name__)


def main() -> str:
    return train(**vars(parse()))


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("hparams_paths", nargs="*", type=str)
    parser.add_argument(
        "-d",
        "--dev-run",
        type=str,
        default="",
        help="Overfit batches (float as as fraction of batches, negative integer for one batch)",
    )
    parser.add_argument(
        "-g",
        "--gpu-devices",
        nargs="*",
        type=int,
    )
    parser.add_argument("--pdb", action="store_true", default=False)
    parser.add_argument("-o", "--offline", action="store_true", default=False)
    return parser.parse_args()


def train(
    hparams_paths: List[str],
    dev_run: str = "",
    gpu_devices: Optional[List[int]] = None,
    pdb: bool = False,
    offline: bool = False,
):
    """Execute one or more single-GPU training tasks across multiple GPU devices.

    Will print PID as an INFO log at start of processes to enable task stopping with `$ kill -2
    <PID>`.

    Args:
        hparams_paths: List of paths to hyperparameters scripts to execute training jobs for.
        dev_run: Float or int as a string to trigger overfitting batches. Only allowed when
            executing a single training job.
        gpu_devices: List of ints specifying which GPU devices to target. Defaults to [0].
        pdb: Whether to launch PDB when an exception is raised in training, only allowed for a
            single training job.
        offline: Whether to disable ClearML logging.
    """
    if gpu_devices is None:
        gpu_devices = [None]

    if pdb or dev_run or len(hparams_paths) == 1:
        if len(hparams_paths) != 1:
            raise ValueError("Debugging only supported for one experiment at a time")
        _train(
            hparams_paths[0],
            gpu_device=gpu_devices[0],
            dev_run=dev_run,
            pdb=pdb,
            offline=offline,
        )
    else:
        logger.info(f"Main Process ID: {os.getpid()}")
        with cf.ProcessPoolExecutor(max_workers=len(hparams_paths)) as pool:
            future_to_hparams_path = {}
            for i, hparams_path in enumerate(hparams_paths):
                gpu_device = gpu_devices[i % len(gpu_devices)]
                future = pool.submit(
                    _train, hparams_path, gpu_device=gpu_device, offline=offline
                )
                future_to_hparams_path[future] = hparams_path

            for future in cf.as_completed(future_to_hparams_path):
                hparams_path = future_to_hparams_path[future]
                try:
                    _ = future.result()
                except Exception as exc:
                    print(f"Error occurred with '{hparams_path}': {exc}")
        logger.info("Finished!")


def _train(
    hparams_path: str,
    gpu_device: Optional[int] = None,
    dev_run: str = "",
    pdb: bool = False,
    offline: bool = False,
):
    """Launch a training job.

    Will log PID on info level at the start of the job to enable stopping the task with
    `$ kill -2 <PID>`.

    Args:
        hparams_path: Path to hparams script.
        gpu_device: Which GPU device index to use for training, used for `devices` kwarg in
            `Trainer.__init__` (by default 0 if a gpu is found, otherwise defaults to None).
        dev_run: Value to use for `overfit_batches` (int or float given as a string)
        pdb: Whether to launch a PDB session when an exception is encountered in config compilation
            or training.
        offline: Whether to skip logging to ClearML.
    """
    pl.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("high")

    logger.info(f"Process ID: {os.getpid()}")

    if gpu_device is None and torch.cuda.is_available():
        gpu_device = 0
    logger.info(f"GPU device: {gpu_device}")

    if dev_run:
        logger.info("DEV RUN")
    if pdb:
        logger.info("PDB")
    if offline:
        logger.info("OFFLINE")

    hparams, config_path = get_hparams_and_config_path(hparams_path, dev_run)

    # Initialise logger
    task_name = get_task_name(
        hparams_path,
        dev_run=dev_run,
        offline=offline,
        project_name=hparams.get("task", {}).get("project_name", "unnamed"),
    )
    save_dir = ARTIFACTS_DIR / task_name
    if not offline:
        exp_logger = ClearMlLogger(
            hparams=hparams,
            config_path=config_path,
            save_dir=save_dir,
        )
        save_dir = exp_logger.save_dir  # Get updated save_dir w/ task id
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
        exp_logger = pl.loggers.TensorBoardLogger(
            save_dir=save_dir,
            name="",
            default_hp_metric=False,
        )

    logger.info("hparams =")
    logger.info(dict_as_str(hparams))
    logger.info(f"Using config at '{config_path}'")

    # Compile config from hparams
    logger.info("Setting hparams on config")
    config = compile_config(config_path, hparams, field="train_config", pdb=pdb)
    hparams, config = load_weights(hparams, config)

    # Initialise callbacks
    callbacks = [
        EpochProgress(),
        NanMonitor(),
        ClearMLTaskMarker(),
        *config.get("callbacks", []),
    ]

    # Initialise trainer and kwargs
    trainer_fit_kwargs = hparams.get("trainer", {}).get("fit", {})
    trainer_init_kwargs = {
        "logger": exp_logger,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": (
            "auto"
            if gpu_device is None or not torch.cuda.is_available()
            else [gpu_device]
        ),
        "callbacks": callbacks,
        "num_sanity_val_steps": 0,
        "enable_progress_bar": False,
        "max_epochs": -1,
        **hparams.get("trainer", {}).get("init", {}),
        "default_root_dir": save_dir,
    }
    trainer = pl.Trainer(**trainer_init_kwargs)

    logger.info("trainer_init_kwargs =")
    logger.info(dict_as_str(trainer_init_kwargs))
    logger.info("trainer_fit_kwargs =")
    logger.info(dict_as_str(trainer_fit_kwargs))

    trainer_validate_kwargs = {}
    if "ckpt_path" in trainer_fit_kwargs:
        trainer_validate_kwargs["ckpt_path"] = trainer_fit_kwargs["ckpt_path"]

    # Validate, then fit model
    try:
        trainer.validate(
            config["module"],
            dataloaders=config["val_dataloaders"],
            **trainer_validate_kwargs,
        )
        trainer.fit(
            config["module"],
            train_dataloaders=config["train_dataloaders"],
            val_dataloaders=config["val_dataloaders"],
            **trainer_fit_kwargs,
        )
    except Exception as err:
        logger.error(f"Error when running task: {str(err)}")
        if pdb:
            model = config["model"]
            train_dls = config["train_dataloaders"]
            val_ds = config["val_dataloaders"]
            breakpoint()
        raise err


def get_task_name(hparams_path: Path, dev_run: bool, offline: bool, project_name: str):
    task_base_name = hparams_path.parts[-2]
    task_stem_name = hparams_path.parts[-1].removesuffix(".py")

    if dev_run:
        task_base_name = f"dev_{task_base_name}"

    for k, v in {
        "project name": project_name,
        "task base name": task_base_name,
        "task stem name": task_stem_name,
    }.items():
        if "-" in v:
            raise ValueError(f"The character '-' is forbidden in the {k} ('{v}')")

    if offline:
        task_version_name = "offline"
    else:
        prev_tasks = Task.get_tasks(
            project_name=project_name,
            task_name=f"^{task_base_name}-{task_stem_name}",
        )
        max_task_v = -1
        for t in prev_tasks:
            version_suffix = t.name.split("-", 2)[-1]
            version_search = re.search(r"\d+", version_suffix)
            if version_search is None:
                continue
            version = int(version_search.group(0))
            max_task_v = max(max_task_v, version)
        task_version_name = "v" + str(max_task_v + 1)

    task_stem_name = f"{task_stem_name}_{task_version_name}"
    return "-".join([task_base_name, task_stem_name])


def load_weights(
    hparams: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
]:
    ckpt_params = hparams.get("checkpoint", {})
    checkpoint_task_id = ckpt_params.get("checkpoint_task_id")
    if not checkpoint_task_id:
        return hparams, config

    checkpoint_name = ckpt_params.get("checkpoint_name", "last")
    weights_only = bool(ckpt_params.get("weights_only", False))

    prev_task = Task.get_task(task_id=checkpoint_task_id)
    prev_ckpt_id = prev_task.output_models_id[checkpoint_name]
    model = Model(model_id=prev_ckpt_id)
    try:
        temp_path = model.get_local_copy(raise_on_error=True)
    except ValueError as error:
        raise FileNotFoundError(
            f"Failed to download weights '{checkpoint_name}' from {checkpoint_task_id}."
        ) from error

    weights_path = (
        ARTIFACTS_DIR
        / f"{get_task_dir_name(prev_task)}/train/model_weights/{checkpoint_name}.ckpt"
    )
    weights_path.parent.mkdir(exist_ok=True, parents=True)
    shutil.move(temp_path, weights_path)

    # If not weights only, add to fit kwargs
    if weights_only:
        logger.info(f"Loading weights '{checkpoint_name}' from {checkpoint_task_id}")
        ckpt = torch.load(weights_path, map_location="cpu")
        config["module"].model.load_state_dict(ckpt["state_dict"])
    else:
        logger.info(f"Loading checkpoint '{checkpoint_name}' from {checkpoint_task_id}")
        hparams["trainer"]["fit"] = {
            **hparams["trainer"].get("fit", {}),
            "ckpt_path": str(weights_path),
        }

    return hparams, config


if __name__ == "__main__":
    main()
