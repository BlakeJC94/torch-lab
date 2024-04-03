import logging
from pathlib import Path

import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class MnistWriter(pl.callbacks.BasePredictionWriter):
    """Example project callback to write predictions to file during inference."""

    def __init__(self, output_path: Path):
        super().__init__(write_interval="batch")
        self.output_path = Path(output_path)
        if self.output_path.exists():
            logger.warning(f"Removing existing '{str(self.output_path)}'.")
            self.output_path.unlink()

        with open(self.output_path, "w") as f:
            f.write("index,prediction\n")

    def write_on_batch_end(
        self,
        _trainer,
        _pl_module,
        prediction,
        _batch_indices,
        _batch,
        _batch_idx,
        _dataloader_idx=0,
    ):
        out, mds = prediction["out"], prediction["md"]
        idxs = mds["idx"]

        with open(self.output_path, "a") as f:
            for prediction, idx in zip(out, idxs):
                f.write(f"{idx},{prediction}\n")
