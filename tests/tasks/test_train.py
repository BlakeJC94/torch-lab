import shutil
from unittest.mock import ANY, call, patch

import pytest
import pytorch_lightning as pl
import torch.cuda
from clearml import Model, Task

from torch_lab.tasks.train import train
from torch_lab.utils import import_script_as_module


@patch.object(torch.cuda, "is_available", autospec=True, return_value=False)
@patch.object(Task, "init", autospec=True)
@patch.object(Task, "get_tasks", autospec=True, return_value=[])
@patch.object(pl, "Trainer", autospec=True)
class TestTrain:
    """Tests for the train.py task."""

    @pytest.mark.parametrize(
        "test_case",
        [
            dict(
                cuda_available=True,
                train_kwargs={},
                expected_devices={"accelerator": "gpu", "devices": [0]},
            ),
            dict(
                cuda_available=True,
                train_kwargs={"gpu_devices": [1]},
                expected_devices={"accelerator": "gpu", "devices": [1]},
            ),
            dict(
                cuda_available=False,
                train_kwargs={},
                expected_devices={"accelerator": "cpu", "devices": "auto"},
            ),
            dict(
                cuda_available=False,
                train_kwargs={"gpu_devices": [1]},
                expected_devices={"accelerator": "cpu", "devices": "auto"},
            ),
        ],
    )
    def test_train(
        self,
        mock_trainer,
        mock_task_get_tasks,
        mock_task_init,
        mock_cuda,
        hparams_path,
        test_case,
    ):
        """Test that Experiment logging and Trainer are called correctly."""
        mock_cuda.return_value = test_case["cuda_available"]
        train([hparams_path], **test_case["train_kwargs"])

        assert mock_task_init.mock_calls[:3] == [
            call(
                continue_last_task=False,
                reuse_last_task_id=False,
                auto_connect_frameworks=ANY,
                project_name="test",
                task_name="00_mnist_demo-hparams-v0",
            ),
            call().connect_configuration(hparams_path.parent / "__main__.py", "config"),
            call().connect(import_script_as_module(hparams_path).hparams, "hparams"),
        ]

        assert mock_trainer.mock_calls == [
            call(
                logger=ANY,
                **test_case["expected_devices"],
                callbacks=ANY,
                num_sanity_val_steps=0,
                enable_progress_bar=True,
                max_epochs=-1,
            ),
            call().validate(ANY, dataloaders=ANY),
            call().fit(
                ANY,
                train_dataloaders=ANY,
                val_dataloaders=ANY,
            ),
        ]

    def test_train_offline(
        self,
        mock_trainer,
        mock_task_get_tasks,
        mock_task_init,
        mock_cuda,
        hparams_path,
    ):
        """Test that offline triggers the expected changes to hparams."""
        train([hparams_path], offline=True)
        mock_task_init.assert_not_called()
        mock_task_get_tasks.assert_not_called()
        assert mock_trainer.mock_calls == [
            call(
                logger=ANY,
                accelerator="cpu",
                devices="auto",
                callbacks=ANY,
                num_sanity_val_steps=0,
                enable_progress_bar=True,
                max_epochs=-1,
            ),
            call().validate(ANY, dataloaders=ANY),
            call().fit(
                ANY,
                train_dataloaders=ANY,
                val_dataloaders=ANY,
            ),
        ]

    @pytest.mark.parametrize("dev_run", ["0.7", "3"])
    def test_train_dev_mode(
        self,
        mock_trainer,
        mock_task_get_tasks,
        mock_task_init,
        mock_cuda,
        hparams_path,
        dev_run,
    ):
        """Test that dev_mode triggers the expected changes to hparams."""
        mock_cuda.return_value = False
        train([hparams_path], dev_run=dev_run)
        assert mock_task_init.mock_calls[0] == call(
            continue_last_task=False,
            reuse_last_task_id=False,
            auto_connect_frameworks=ANY,
            project_name="test",
            task_name="dev_00_mnist_demo-hparams-v0",
        )
        assert mock_trainer.mock_calls[0] == call(
            logger=ANY,
            accelerator="cpu",
            devices="auto",
            callbacks=ANY,
            num_sanity_val_steps=0,
            enable_progress_bar=True,
            max_epochs=-1,
            log_every_n_steps=1,
            overfit_batches=float(dev_run) if "." in dev_run else int(dev_run),
        )

    def test_raise_train_dev_mode_multi(
        self,
        mock_trainer,
        mock_task_get_tasks,
        mock_task_init,
        mock_cuda,
        hparams_path,
    ):
        """Test that attempting to use dev_run for multiple experiments raises an error."""
        with pytest.raises(ValueError):
            train([hparams_path, hparams_path], dev_run="7")

    @patch.object(Task, "get_task", autospec=True)
    @patch.object(Model, "get_local_copy", autospec=True)
    @patch.object(Model, "__init__", autospec=True, return_value=None)
    @patch.object(shutil, "move", autospec=True)
    def test_train_from_checkpoint(
        self,
        mock_shutil_move,
        mock_model_init,
        mock_model_get_local_copy,
        mock_task_get_task,
        mock_trainer,
        mock_task_get_tasks,
        mock_task_init,
        mock_cuda,
        hparams_path_checkpoint,
    ):
        """Test that loading a checkpoint changes the call to the trainer."""
        train([hparams_path_checkpoint])

        assert mock_trainer.mock_calls == [
            call(
                logger=ANY,
                accelerator="cpu",
                devices="auto",
                callbacks=ANY,
                num_sanity_val_steps=0,
                enable_progress_bar=True,
                max_epochs=-1,
            ),
            call().validate(ANY, dataloaders=ANY, ckpt_path=ANY),
            call().fit(
                ANY,
                train_dataloaders=ANY,
                val_dataloaders=ANY,
                ckpt_path=ANY,
            ),
        ]

    @patch.object(Task, "get_task", autospec=True)
    @patch.object(Model, "get_local_copy", autospec=True)
    @patch.object(Model, "__init__", autospec=True, return_value=None)
    @patch.object(shutil, "move", autospec=True)
    @patch.object(torch, "load", autospec=True)
    @patch.object(torch.nn.Module, "load_state_dict", autospec=True)
    def test_train_weights_only(
        self,
        mock_torch_load_state_dict,
        mock_torch_load,
        mock_shutil_move,
        mock_model_init,
        mock_model_get_local_copy,
        mock_task_get_task,
        mock_trainer,
        mock_task_get_tasks,
        mock_task_init,
        mock_cuda,
        hparams_path_weights_only,
    ):
        """Test that loading a weights_only doesn't change the call to the trainer."""
        train([hparams_path_weights_only])
        assert mock_trainer.mock_calls == [
            call(
                logger=ANY,
                accelerator="cpu",
                devices="auto",
                callbacks=ANY,
                num_sanity_val_steps=0,
                enable_progress_bar=True,
                max_epochs=-1,
            ),
            call().validate(ANY, dataloaders=ANY),
            call().fit(
                ANY,
                train_dataloaders=ANY,
                val_dataloaders=ANY,
            ),
        ]

    # def test_train_offline(): ...
