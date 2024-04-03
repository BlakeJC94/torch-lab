import json
import logging
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Tuple, Dict, Optional

from attrdict import AttrDict


def import_script_as_module(config_path) -> Any:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Couldn't find '{str(config_path)}'.")
    if not config_path.is_file():
        raise FileNotFoundError(f"Path '{str(config_path)}' is not a file.")
    name = Path(config_path).stem
    spec = spec_from_file_location(name, config_path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def print_dict(d: dict) -> str:
    return json.dumps(d, indent=2, default=lambda o: str(o))


def get_hparams_and_config_path(
    hparams_path: Path,
    dev_run: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    hparams = import_script_as_module(hparams_path).hparams
    if dev_run:
        hparams = set_hparams_debug_overrides(hparams, dev_run)

    config_path = Path(hparams_path).parent / "__main__.py"
    return hparams, config_path


def set_hparams_debug_overrides(hparams, dev_run):
    # Task overrides
    hparams["task"]["init"]["project_name"] = "test"
    # Config overrides
    hparams["config"]["num_workers"] = 0
    # Trainer overrides
    hparams["trainer"]["init"]["log_every_n_steps"] = 1
    hparams["trainer"]["init"]["overfit_batches"] = (
        float(dev_run) if "." in dev_run else int(dev_run)
    )
    return hparams


def compile_config(
    hparams: Dict[str, Any],
    config_path: Path,
    *config_args,
    pdb: bool = False,
    field: str = "train_config",
) -> Dict[str, Any]:
    config_fn = getattr(import_script_as_module(config_path), field)

    try:
        config = config_fn(AttrDict(hparams), *config_args)
    except Exception as err:
        logging.getLogger(__name__).error(f"Couldn't import config: {str(err)}")
        if pdb:
            breakpoint()
        raise err

    return config


