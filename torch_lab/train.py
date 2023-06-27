import logging
from os import PathLike
from importlib.util import module_from_spec, spec_from_file_location
from typing import Union

import pytorch_lightning as pl

from . import Config


logger = logging.getLogger(__name__)


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
        raise AttributeError("Config file must define function `main()` that returns a `Config`.")
    logger.debug("Done.")
    return model_config

def train(config: PathLike, **kwargs):
    pl.seed_everything(0, workers=True)

    # TODO Get config from path
    config = ...

    # TODO Add standard callbacks
    # TODO Manage checkpoints

    trainer = pl.Trainer(**kwargs)
    trainer.fit(config.module, config.data_module)
