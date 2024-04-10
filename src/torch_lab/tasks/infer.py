import argparse
import logging
from typing import List

import pytorch_lightning as pl
import torch
from torch_lab.utils import compile_config, get_hparams_and_config_path, dict_as_str

logger = logging.getLogger(__name__)


def main() -> str:
    return infer(**vars(parse()))


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("hparams_path")
    parser.add_argument("predict_args", nargs="*")
    return parser.parse_args()


def infer(hparams_path: str, predict_args: List[str]):
    """Compile an inference configuration from a hparams script and external parameters.

    Results are expected to be written by defining and passing a pl.Callback object.

    Args:
        hparams_path: Path to hyperparameters script which defines a function `infer_config` which
            accepts at least one arg and returns an infer task config.
        predict_args: List of args to apply to `infer_config`, can be obtained from the CLI input.
    """
    hparams, config_path = get_hparams_and_config_path(hparams_path)

    logger.info("hparams =")
    logger.info(dict_as_str(hparams))

    logger.info(f"Using config at '{config_path}'")
    logger.info(f"Using predict args: {predict_args}")

    config = compile_config(config_path, hparams, *predict_args, field="infer_config")

    trainer = pl.Trainer(
        callbacks=config.get("callbacks", []),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[0] if torch.cuda.is_available() else "auto",
    )
    trainer.predict(
        config["module"],
        dataloaders=config["predict_dataloaders"],
        return_predictions=False,
    )

    logger.info("Finished predictions")


if __name__ == "__main__":
    main()
