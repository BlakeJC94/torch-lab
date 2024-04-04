import gzip
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
from torch_lab.datasets import BaseDataset


class TrainDataset(BaseDataset):
    """Example dataset used for training"""

    def __init__(
        self,
        data_paths: Tuple[str | Path, str | Path],
        data_slice: slice,
        augmentation: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        self.data_path, self.annotations_path = [Path(fp) for fp in data_paths]
        self.data_slice = data_slice

        self.augmentation = augmentation
        self.transform = transform

        image_size = 28
        num_images = 60000

        with gzip.open(self.data_path, "r") as f:
            f.read(16)
            buf = f.read(image_size * image_size * num_images)
        images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        images = images.reshape(num_images, image_size, image_size)
        images = images[self.data_slice]
        self.images = images

        with gzip.open(self.annotations_path, "r") as f:
            f.read(8)
            buf = f.read(num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
        labels = labels[self.data_slice]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def get_raw_data(self, i: int) -> Any:
        x = self.images[i]
        return x.reshape(1, *self.images.shape[1:])

    def get_raw_label(self, i: int) -> Any:
        class_idx = self.labels[i]
        label = np.zeros(10)
        label[class_idx] = 1
        return label


class PredictDataset(BaseDataset):
    def __init__(self, data_path: str | Path, transform):
        self.data_path = Path(data_path)
        self.transform = transform

        self.images = None
        with gzip.open(self.data_path, "r") as f:
            image_size = 28
            num_images = 10000

            f.read(16)
            buf = f.read(image_size * image_size * num_images)

        images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        images = images.reshape(num_images, image_size, image_size)
        self.images = images

    def __len__(self):
        return len(self.images)

    def get_raw_data(self, i: int) -> Any:
        return self.images[i].reshape(1, *self.images.shape[1:])
