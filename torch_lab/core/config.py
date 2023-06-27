from dataclasses import dataclass, field

import pytorch_lightning as pl

from


@dataclass
class Config:
    module: pl.LightningModule = field(repr=False)
    data_module: pl.LightningDataModule = field(repr=False)
    project: str
    experiment_name: str
    tags: List[str]] = field(default_factory=list)
