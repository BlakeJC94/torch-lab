"""Location for common utilities used in the task scripts."""
import json
import logging
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Dict, Tuple

from torch_lab.typing import JsonDict


def import_script_as_module(script_path: str | Path) -> Any:
    """Execute a python script and return object where symbols are defined as attributes.

    Args:
        config_path: Path to a python script.
    """
    config_path = Path(script_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Couldn't find '{str(config_path)}'.")
    if not config_path.is_file():
        raise FileNotFoundError(f"Path '{str(config_path)}' is not a file.")
    name = Path(config_path).stem
    spec = spec_from_file_location(name, config_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dict_as_str(d: Dict[str, Any], indent: int = 2) -> str:
    """Convert a Python dict to a Json-style string for printing and debugging.

    Non-Json data such as arbitrary Python objects are replaced by their `__str__` representation.

    Args:
        d: Python dictionary.
        indent: Number of spaces to indent values of dict by.
    """
    return json.dumps(d, indent=indent, default=lambda o: str(o))


def get_hparams_and_config_path(
    hparams_path: Path,
    dev_run: str = "",
) -> Tuple[JsonDict, str]:
    """Import the hparams from script at `path` as a Python dict, and infer path of config script.

    Expects that the hparams script defines a dict with the symbol `hparams`. Config script is
    inferred to be in the same directory as hparams script with the name `__main__.py`.

    Args:
        hparams_path: Path to a script defining a dict with the symbol `hparams`.
        dev_run: An int or a float as a string value.

    Returns:
        Hparams dict defied as the symbol `hparams` in the file `hparams_path`, and the path to the
        config script `__main__.py` in the same directory.
    """
    hparams = import_script_as_module(hparams_path).hparams
    if dev_run:
        hparams = set_hparams_debug_overrides(hparams, dev_run)

    config_path = Path(hparams_path).parent / "__main__.py"
    return hparams, config_path


def set_hparams_debug_overrides(hparams: JsonDict, dev_run: str) -> JsonDict:
    """Override values of hparams when dev mode is active.

    Args:
        hparams: A hparams dict with keys "task", "config", and "trainer".
        dev_run: An int or a float as a string value. Will be parsed and set as `overfit_batches`
            for trainer.
    """
    # Task overrides
    hparams["task"]["project_name"] = "test"
    # Config overrides
    hparams["config"]["num_workers"] = 0
    # Trainer overrides
    hparams["trainer"]["init"]["log_every_n_steps"] = 1
    hparams["trainer"]["init"]["overfit_batches"] = (
        float(dev_run) if "." in dev_run else int(dev_run)
    )
    return hparams


def compile_config(
    config_path: Path,
    hparams: Dict[str, Any],
    *config_args,
    field: str = "train_config",
    pdb: bool = False,
) -> Dict[str, Any]:
    """Execute a function defined in a script.

    Expects that the config script defines a function which accepts a hparams dict as the first
    argument. Additional arguments can be defined and passed through.

    Args:
        config_path: Location of config script.
        hparams: Dict containing the "config" key, which has the dict of values required by the
            config function to be executed.
        config_args: Additional args to be passed to the config function in addition to hparams.
            Used for inference configs.
        pdb: Wether to launch pdb when an exception is encountered.
        field: Name of the config function symbol defined at the config_path.
    """
    config_fn = getattr(import_script_as_module(config_path), field)

    try:
        config = config_fn(hparams, *config_args)
    except Exception as err:
        logging.getLogger(__name__).error(f"Couldn't import config: {str(err)}")
        if pdb:
            breakpoint()
        raise err

    return config
