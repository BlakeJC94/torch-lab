import logging
from pathlib import Path
from os import PathLike
from importlib.util import module_from_spec, spec_from_file_location
from typing import Union

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
