import logging
from os import PathLike

import pytorch_lightning as pl


logger = logging.getLogger(__name__)

def train(config_path: PathLike):
    print(f"yo from train {config_path}")
    ...
