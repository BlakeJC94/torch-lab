"""Base class for usage with torch_lab modules.

Datasets are implemented as iterable objects which return (data, metadata) tuples on each iteration,
where metadata is a dictionary with collate-friendly values (int, float, str, np.ndarray, or
torch.Tensor).

Labels are located in `metadata["y"]`, which are automatically looked up in the train/validation
methods of the LabModule classes.

Custom transforms can be implemented from the BaseTransform classes, wrapped into a TransformCompose
object and passed in as an attribute
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

from torch import Tensor
from torch.utils.data import Dataset
from torch_lab.transforms import BaseTransform
from torch_lab.typing import CollateData


class BaseDataset(Dataset, ABC):
    """Base class for implementing datasets in torch_lab."""

    def __init__(
        self,
        transform: Optional[BaseTransform | Callable] = None,
    ):
        """Initialise a BaseDataset.

        Args:
            transform: Transform to apply to (data, metadata) tuple output of __getitem__.
        """
        self.transform = transform

    @abstractmethod
    def __len__(self):
        """Return length of dataset."""
        pass

    def get_additional_metadata(self, i: int) -> Dict[str, CollateData]:
        """Add additional information to metadata."""
        return {}

    @abstractmethod
    def get_raw_data(self, md: Dict[str, Any]) -> Any:
        """Return raw instance of data."""
        pass

    def get_raw_label(self, i: int) -> Any:
        """Return raw instance of labels used for training/validation. Used to create
        `metadata["y"]`.
        """
        return None

    def get_raw(self, i: int) -> Any:
        metadata = {
            "i": i,
            **self.get_additional_metadata(i),
        }

        if (label := self.get_raw_label(metadata)) is not None:
            metadata["y"] = label

        data = self.get_raw_data(metadata)
        return data, metadata

    def __getitem__(self, i: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        x, y = self.get_raw(i)

        if self.transform is not None:
            x, y = self.transform(x, y)

        return x, y
