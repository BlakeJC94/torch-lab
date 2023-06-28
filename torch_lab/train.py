import logging
from pathlib import Path
from os import PathLike
from importlib.util import module_from_spec, spec_from_file_location
from typing import Union

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from . import Config


logger = logging.getLogger(__name__)


TORCHLAB_DIR = Path(__file__).absolute().parent
ROOT_DIR = TORCHLAB_DIR.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def import_model_config(config: Union[PathLike, Config]) -> Config:
    if isinstance(config, Config):
        return config
    logger.debug(f"Importing config '{config}'...")

    config_spec = spec_from_file_location("config", config)
    config_module = module_from_spec(config_spec)
    config_spec.loader.exec_module(config_module)
    try:
        model_config = config_module.main()
        assert isinstance(model_config, Config)
    except (ModuleNotFoundError, AssertionError, AttributeError):
        # pylint: disable=raise-missing-from
        raise AttributeError(
            "Config file must define function `main()` that returns a `Config`."
        )
    logger.debug("Done.")
    return model_config


def train(config: PathLike, **kwargs):
    pl.seed_everything(0, workers=True)

    config = import_model_config(config)

    # TODO Manage checkpoints
    # IDEA: Could submit stats to ClearML using on_checkpoint hook?

    save_dir = ARTIFACTS_DIR / config.project / config.experiment_name
    save_dir.mkdir(exist_ok=True, parents=True)
    logs_path = save_dir / "logs"

    tensorboard_logger = TensorBoardLogger(
        save_dir=str(logs_path),
        name="",
        default_hp_metric=False,
    )

    trainer = pl.Trainer(
        logger=tensorboard_logger,
        callbacks=config.callbacks,
        **kwargs,
    )
    trainer.fit(config.module, config.data_module)
