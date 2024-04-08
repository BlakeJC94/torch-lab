import pytest
import torch
from torch_lab.datasets import BaseDataset
from torch_lab.transforms import BaseDataTransform


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

        self.data = [i * torch.ones((2, 2)) for i in range(n_samples)]
        self.labels = [i % 2 for i in range(n_samples)]

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


class TestBaseDataset:
    def test_getitem(self, mock_dataset, n_samples):
        """Test that transforms are applied and all metadata is returned."""
        for i, (x, md) in enumerate(mock_dataset):
            assert (x == 2 * i * torch.ones_like(x)).all()
            assert md["idx"] == i
            assert md["y"] == i % 2
            assert md["foo"] == "bar"

    def test_getitem_augmentation(self, mock_dataset_augmentation, n_samples):
        """Test that augmentations are applied before the transform."""
        for i, (x, md) in enumerate(mock_dataset_augmentation):
            assert (x == 2 * (1 + i * torch.ones_like(x))).all()
            assert md["idx"] == i
            assert md["y"] == i % 2
            assert md["foo"] == "bar"

    def test_getitem_predict(self, mock_predict_dataset, n_samples):
        """Test that metadata excludes key "y" from metadata if get_raw_label returns None."""
        for i, (x, md) in enumerate(mock_predict_dataset):
            assert (x == 2 * i * torch.ones_like(x)).all()
            assert md["idx"] == i
            assert md["foo"] == "bar"
            assert "y" not in md
