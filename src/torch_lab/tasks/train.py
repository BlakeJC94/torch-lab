import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from clearml import Task
from torch_lab.callbacks import EpochProgress, NanMonitor
from torch_lab.loggers import ClearMlLogger
from torch_lab.paths import ARTIFACTS_DIR, get_task_dir_name
from torch_lab.utils import compile_config, get_hparams_and_config_path, dict_as_str

logger = logging.getLogger(__name__)


def main() -> str:
    return train(**vars(parse()))


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("hparams_path")
    parser.add_argument(
        "-d",
        "--dev-run",
        type=str,
        default="",
        help="Overfit batches (float as as fraction of batches, negative integer for one batch)",
    )
    parser.add_argument(
        "-g",
        "--gpu-device",
        type=int,
        default=None,
    )
    parser.add_argument("--pdb", action="store_true", default=False)
    parser.add_argument("-o", "--offline", action="store_true", default=False)
    return parser.parse_args()


def train(
    hparams_path: str,
    gpu_device: Optional[int] = None,
    dev_run: bool = False,
    pdb: bool = False,
    offline: bool = False,
):
    """Choo choo"""
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
    if not offline:
        exp_logger = ClearMlLogger(
            hparams=hparams,
            config_path=config_path,
            task_name=get_task_name(hparams_path, dev_run),
            root_dir=ARTIFACTS_DIR,
        )
        task_dir_name = get_task_dir_name(exp_logger.task)
    else:
        task_dir_name = f"{get_task_name(hparams_path, dev_run)}-offline"
        exp_logger = pl.loggers.TensorBoardLogger(
            save_dir=str(ARTIFACTS_DIR / f"{task_dir_name}/logs"),
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
        pl.callbacks.LearningRateMonitor(),
        *config.get("callbacks", []),
    ]
    if not dev_run:
        save_dir = ARTIFACTS_DIR / f"{task_dir_name}"
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                save_dir / "model_weights",
                monitor=hparams["config"]["monitor"],
                save_last=True,
            ),
            *callbacks,
        ]

    # Initialise trainer and kwargs
    trainer_fit_kwargs = hparams["trainer"].get("fit", {})
    trainer_init_kwargs = {
        "logger": exp_logger,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": "auto" if gpu_device is None else [gpu_device],
        "callbacks": callbacks,
        "num_sanity_val_steps": 0,
        "enable_progress_bar": False,
        "max_epochs": -1,
        **hparams["trainer"].get("init", {}),
    }
    trainer = pl.Trainer(**trainer_init_kwargs)

    logger.info("trainer_init_kwargs =")
    logger.info(dict_as_str(trainer_init_kwargs))
    logger.info("trainer_fit_kwargs =")
    logger.info(dict_as_str(trainer_fit_kwargs))

    # Validate, then fit model
    try:
        trainer.validate(config["module"], dataloaders=config["val_dataloaders"])
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


def get_task_name(hparams_path: Path, dev_run: bool):
    task_name = "-".join(Path(hparams_path).parts[-2:]).removesuffix(".py")
    if dev_run:
        task_name = f"dev_{task_name}"
    return task_name


def load_weights(
    hparams: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
]:
    ckpt_params = hparams["checkpoint"]
    checkpoint_task_id = ckpt_params.get("checkpoint_task_id")
    if not checkpoint_task_id:
        return hparams, config

    checkpoint_name = ckpt_params.get("checkpoint_name", "last")
    weights_only = bool(ckpt_params.get("weights_only", False))

    # TODO update to use remote checkpoint storage
    prev_task = Task.get_task(task_id=checkpoint_task_id)
    temp_path = (
        ARTIFACTS_DIR
        / f"{get_task_dir_name(prev_task)}/train/model_weights/{checkpoint_name}.ckpt"
    )

    # If not weights only, add to fit kwargs
    if weights_only:
        logger.info(f"Loading weights '{checkpoint_name}' from {checkpoint_task_id}")
        ckpt = torch.load(temp_path, map_location="cpu")
        config["model"].load_state_dict(ckpt["state_dict"])
    else:
        logger.info(f"Loading checkpoint '{checkpoint_name}' from {checkpoint_task_id}")
        hparams["trainer"]["fit"] = {
            **hparams["trainer"].get("init", {}),
            "ckpt_path": str(temp_path),
        }

    return hparams, config


if __name__ == "__main__":
    main()
