import pytest
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch_lab.modules import LabModule, TrainLabModule


class MockModel(nn.Sequential):
    """Example model"""

    def __init__(self, n_channels, n_features, n_classes):
        self.n_channels = n_channels
        self.n_features = n_features
        self.n_classes = n_classes

        super().__init__(
            nn.Sequential(
                nn.Conv2d(n_channels, n_features, 3),
                nn.BatchNorm2d(n_features),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.AvgPool2d(2),
                nn.Conv2d(n_features, n_classes, 3),
                nn.BatchNorm2d(n_classes),
                nn.ReLU(),
            ),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        x = super().forward(x)
        x = x.squeeze(-1).squeeze(-1)
        return x


@pytest.fixture
def mock_model():
    return MockModel(1, 16, 10)


@pytest.fixture
def train_module(mock_model):
    return TrainLabModule(
        mock_model,
        loss_function=nn.BCEWithLogitsLoss(),
        optimizer_factory=lambda params: optim.Adam(params, lr=3e-4),
    )


@pytest.fixture
def trainer():
    return pl.Trainer(enable_progress_bar=False)


@pytest.fixture
def ckpt_path(train_module, tmp_path, mock_dataloader, trainer):
    trainer.validate(train_module, mock_dataloader)
    ckpt_path = tmp_path / "mock.ckpt"
    trainer.save_checkpoint(ckpt_path)
    return ckpt_path


@pytest.fixture
def mock_dataloader(mock_dataset):
    return DataLoader(mock_dataset, num_workers=0)


@pytest.fixture
def train_module_checkpoint(ckpt_path):
    return torch.load(ckpt_path)


class TestTrainModuleCheckpoint:
    """Test that TrainModule saves a checkpoint that can be loaded."""

    def test_load_checkpoint_into_raw_model(self, mock_model, train_module_checkpoint):
        """Test that TrainModule checkpoint can be loaded directly into model instance."""
        mock_model.load_state_dict(train_module_checkpoint["state_dict"])

    def test_load_checkpoint_into_lab_module(self, mock_model, train_module_checkpoint):
        """Test that state_dict can be loaded into a LabModule"""
        LabModule(mock_model).model.load_state_dict(
            train_module_checkpoint["state_dict"]
        )

    def test_load_checkpoint_into_train_lab_module(
        self, train_module, train_module_checkpoint
    ):
        """Test that state_dict can be loaded into a TrainLabModule (i.e. load checkpoint weights
        only).
        """
        train_module.model.load_state_dict(train_module_checkpoint["state_dict"])

    def test_load_checkpoint(self, train_module, mock_dataloader, trainer, ckpt_path):
        trainer.validate(
            train_module,
            mock_dataloader,
            ckpt_path=ckpt_path,
        )
