from importlib import metadata

from .core import Config, LabModule, LabDataModule
from .log import setup_logging
from .build import build
from .train import train
from .test import test
from .predict import predict


setup_logging()

__all__ = [
    "__version__",
    "build",
    "train",
    "test",
    "predict",
]

__version__ = metadata.version("torch_lab")
