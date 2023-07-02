from dataclasses import dataclass, field
from typing import List

import pytorch_lightning as pl



@dataclass
class Config:
    project_name: str
    task_name: str
    module: pl.LightningModule = field(repr=False)
    data_module: pl.LightningDataModule = field(repr=False)
    train_callbacks: List[pl.Callback] = field(default_factory=list, repr=False)
    test_callbacks: List[pl.Callback] = field(default_factory=list, repr=False)
    tags: List[str] = field(default_factory=list)
