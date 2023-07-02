from dataclasses import dataclass, field
from typing import List, Dict, Any

import pytorch_lightning as pl



@dataclass
class Config:
    project_name: str
    task_name: str
    module: pl.LightningModule = field(repr=False)
    data_module: pl.LightningDataModule = field(repr=False)
    callbacks: Dict[str, List[pl.Callback]] = field(default_factory=dict, repr=False)
    tags: List[str] = field(default_factory=list)
    user_properties: Dict[str, Any] = field(default_factory=dict)
