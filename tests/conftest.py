from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torch_lab.datasets import BaseDataset
from torch_lab.modules import TrainLabModule
from torch_lab.transforms import BaseDataTransform


@pytest.fixture
def hparams():
    return {
        "task": {
            "init": {
                "project_name": "test",
            },
        },
        "checkpoint": {
            "checkpoint_task_id": None,
            # "checkpoint_name": "last",
            # "weights_only": False,
        },
        "trainer": {
            "init": {
                "enable_progress_bar": True,
            },
            "fit": {},
            "predict": {},
        },
        "config": {
            "data": "./train-images-idx3-ubyte.gz",
            "annotations": "./train-labels-idx1-ubyte.gz",
            "n_features": 32,
            "learning_rate": 1.5 * 1e-3,
            "weight_decay": 0.01,
            "num_workers": 10,
            "batch_size": 2048,
            "patience": 5,
            "milestones": [5, 8],
            "gamma": 0.2,
            "monitor": "loss/validate",
        },
    }


@pytest.fixture
def hparams_path():
    return Path("./src/example_project/experiments/00_mnist_demo/hparams.py")


@pytest.fixture
def hparams_path_checkpoint():
    return Path("./src/example_project/experiments/00_mnist_demo/hparams_checkpoint.py")


@pytest.fixture
def hparams_path_weights_only():
    return Path(
        "./src/example_project/experiments/00_mnist_demo/hparams_weights_only.py"
    )


## Transforms
class MockTransform(BaseDataTransform):
    def compute(self, x):
        return 2 * x


## Datasets
class MockDataset(BaseDataset):
    def __init__(self, n_samples, n_classes, transform):
        super().__init__(transform)
        self.n_samples = n_samples
        self.n_classes = n_classes

        self.data = [i * torch.ones((1, 10, 10)) for i in range(n_samples)]
        self.labels = [i * torch.ones((n_classes,)) % 2 for i in range(n_samples)]

    def __len__(self):
        return self.n_samples

    def get_raw_data(self, md):
        i = md["i"]
        return self.data[i]

    def get_raw_label(self, md):
        i = md["i"]
        return self.labels[i]

    def get_additional_metadata(self, i):
        return {"foo": "bar"}


class MockPredictDataset(MockDataset):
    def get_raw_label(self, *_):
        return None


@pytest.fixture
def n_samples():
    return 6


@pytest.fixture
def n_classes():
    return 10


@pytest.fixture
def mock_dataset(n_samples, n_classes):
    return MockDataset(n_samples, n_classes, MockTransform())


@pytest.fixture
def mock_predict_dataset(n_samples, n_classes):
    return MockPredictDataset(n_samples, n_classes, MockTransform())


@pytest.fixture
def mock_dataloader(mock_dataset):
    return DataLoader(mock_dataset, num_workers=0)


## Modules
class MockModel(nn.Sequential):
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
def mock_model(n_classes):
    return MockModel(1, 16, n_classes)


@pytest.fixture
def train_module(mock_model):
    return TrainLabModule(
        mock_model,
        loss_function=nn.BCEWithLogitsLoss(),
        optimizer_factory=lambda params: optim.Adam(params, lr=3e-4),
    )


@pytest.fixture
def trainer():
    return pl.Trainer(logger=None, enable_progress_bar=False)


@pytest.fixture
def ckpt_path(train_module, tmp_path, mock_dataloader, trainer):
    trainer.validate(train_module, mock_dataloader)
    ckpt_path = tmp_path / "mock.ckpt"
    trainer.save_checkpoint(ckpt_path)
    return ckpt_path


@pytest.fixture
def train_module_checkpoint(ckpt_path):
    return torch.load(ckpt_path)
