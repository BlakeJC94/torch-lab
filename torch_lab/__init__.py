import os
import sys
import subprocess
from pathlib import Path
from importlib import metadata

from .core import Config, LabModule, LabDataModule
from .log import setup_logging
from .build import build
from .train import train
from .test import test
from .predict import predict


setup_logging()

if "--help" not in sys.argv:
    input_argv_0 = sys.argv[0]
    for file_path in [input_argv_0, os.getcwd(), __file__]:
        file_path = Path(file_path)
        if not file_path.exists():
            continue
        # Ensure file_path is always a file and not a directory
        if file_path.is_dir():
            file_path = next(file_path.glob("*"))
        is_repository = (
            "true"
            in subprocess.run(
                "git rev-parse --is-inside-work-tree",
                shell=True,
                check=False,
                encoding="utf-8",
                cwd=str(file_path.parent),
                capture_output=True,
            ).stdout
        )  # pylint: disable=subprocess-run-check
        if is_repository:
            sys.argv[0] = str(file_path)
            break

__all__ = [
    "__version__",
    "build",
    "train",
    "test",
    "predict",
]


__version__ = metadata.version("torch_lab")
