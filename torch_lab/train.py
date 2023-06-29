import logging
from os import PathLike

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from .paths import import_model_config, ARTIFACTS_DIR


logger = logging.getLogger(__name__)
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
logging.getLogger("torch._inductor").setLevel(logging.WARNING)


def train(config: PathLike, compile: bool = False, **kwargs):
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

    module = lab_config.module
    data_module = lab_config.data_module
    # TODO: Re-enable this when lightning gets upgraded to 2.1, logger just isn't ready yet!
    if compile:
        module.model = torch.compile(module.model)

    trainer.fit(module, data_module)
