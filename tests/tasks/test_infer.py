from unittest.mock import ANY, call, patch

import pytorch_lightning as pl
import torch

import example_project.datasets
from torch_lab.tasks.infer import infer


@patch.object(torch.cuda, "is_available", autospec=True, return_value=False)
@patch.object(pl, "Trainer", autospec=True)
@patch.object(torch.nn.Module, "load_state_dict", autospec=True)
@patch.object(example_project.datasets, "PredictDataset", autospec=True)
class TestInfer:
    def test_infer(
        self,
        mock_predict_dataset,
        mock_torch_module_load_state_dict,
        mock_trainer,
        mock_cuda,
        hparams_path,
    ):
        """Test that trainer is called as expected using infer task."""
        infer(
            hparams_path,
            [
                "mock/path/to/weights.ckpt",
                "mock/path/to/test.gz",
            ],
        )

        assert mock_trainer.mock_calls == [
            call(
                callbacks=ANY,
                accelerator="cpu",
                devices="auto",
            ),
            call().predict(
                ANY,
                dataloaders=ANY,
                return_predictions=False,
            ),
        ]
