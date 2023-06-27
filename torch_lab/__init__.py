from importlib import metadata

from .core import Config, LabModule, LabDataModule
from .log import setup_logging
from .train import train
from .build import build


setup_logging()

__all__ = [
    "__version__",
    "train",
    "build",
]

__version__ = metadata.version("torch_lab")
