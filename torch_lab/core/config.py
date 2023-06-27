from dataclasses import dataclass, field
from typing import List

import pytorch_lightning as pl



@dataclass
class Config:
    module: pl.LightningModule = field(repr=False)
    data_module: pl.LightningDataModule = field(repr=False)
    project: str
    experiment_name: str
    tags: List[str] = field(default_factory=list)
