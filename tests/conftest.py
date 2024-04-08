import pytest
import torch
from torch_lab.datasets import BaseDataset
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

## Datasets
class MockTransform(BaseDataTransform):
    def compute(self, x):
        return 2 * x


class MockAugmentation(BaseDataTransform):
    def compute(self, x):
        return x + 1


class MockDataset(BaseDataset):
    def __init__(self, n_samples, transform, augmentation=None):
        super().__init__(transform, augmentation)
        self.n_samples = n_samples

        self.data = [i * torch.ones((1, 10, 10)) for i in range(n_samples)]
        self.labels = [i * torch.ones((10)) % 2 for i in range(n_samples)]

    def __len__(self):
        return self.n_samples

    def get_raw_data(self, i):
        return self.data[i]

    def get_raw_label(self, i):
        return self.labels[i]

    def get_additional_metadata(self, i):
        return {"foo": "bar"}


class MockPredictDataset(MockDataset):
    def get_raw_label(self, i):
        return None


@pytest.fixture
def n_samples():
    return 6


@pytest.fixture
def mock_dataset(n_samples):
    return MockDataset(n_samples, MockTransform())


@pytest.fixture
def mock_predict_dataset(n_samples):
    return MockPredictDataset(n_samples, MockTransform())


@pytest.fixture
def mock_dataset_augmentation(n_samples):
    return MockDataset(
        n_samples,
        MockTransform(),
        MockAugmentation(),
    )
