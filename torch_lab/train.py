import logging
from os import PathLike

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from .paths import import_model_config, ARTIFACTS_DIR


logger = logging.getLogger(__name__)


def train(config: PathLike, **kwargs):
    pl.seed_everything(0, workers=True)

    lab_config = import_model_config(config)

    # TODO Manage checkpoints
    # IDEA: Could submit stats to ClearML using on_checkpoint hook?

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
    trainer.fit(lab_config.module, lab_config.data_module)
