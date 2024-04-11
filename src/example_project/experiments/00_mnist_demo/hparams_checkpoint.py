from pathlib import Path

from torch_lab.utils import import_script_as_module

hparams = import_script_as_module(Path(__file__).parent / "hparams.py").hparams
hparams["checkpoint"]["checkpoint_task_id"] = "deadbeef"
