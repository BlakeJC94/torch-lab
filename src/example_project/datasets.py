import gzip
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from torch_lab.datasets import BaseDataset


class TrainDataset(BaseDataset):
    """Example dataset used for training"""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        augmentation: Optional[Callable] = None,
        transform: Optional[Callable] = None,
    ):
        self.images = images
        self.labels = labels
        self.augmentation = augmentation
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def get_raw_data(self, i: int) -> Any:
        x = self.images[i]
        if x.shape != (28, 28):
            breakpoint()
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
            num_images = 1000

            f.read(16)
            buf = f.read(image_size * image_size * num_images)

        images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        images = images.reshape(num_images, image_size, image_size)
        self.images = images

    def __len__(self):
        return len(self.images)

    def get_raw_data(self, i: int) -> Any:
        return self.images[i].reshape(1, *self.images.shape[1:])
