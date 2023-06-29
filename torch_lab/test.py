import logging
from os import PathLike
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from .paths import import_model_config, ARTIFACTS_DIR


logger = logging.getLogger(__name__)


def test(config: PathLike, weights_path: Optional[PathLike] = None, **kwargs):
    pl.seed_everything(0, workers=True)

    lab_config = import_model_config(config)

    save_dir = ARTIFACTS_DIR / lab_config.project / lab_config.experiment_name
    save_dir.mkdir(exist_ok=True, parents=True)
    logs_path = save_dir / "logs"

    tensorboard_logger = TensorBoardLogger(
        save_dir=str(logs_path),
        name="",
        default_hp_metric=False,
    )

    trainer = pl.Trainer(
        logger=tensorboard_logger,
        callbacks=lab_config.callbacks,
        **kwargs,
    )

    if weights_path:
        lab_config.module.load_state_dict(torch.load(weights_path)['state_dict'])

    trainer.test(lab_config.module, lab_config.data_module)
