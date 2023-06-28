from dataclasses import dataclass, field
from typing import List

import pytorch_lightning as pl



@dataclass
class Config:
    project: str
    experiment_name: str
    module: pl.LightningModule = field(repr=False)
    data_module: pl.LightningDataModule = field(repr=False)
    callbacks: List[pl.Callback] = field(default_factory=list, repr=False)
    tags: List[str] = field(default_factory=list)
