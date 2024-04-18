"""Base classes and wrappers for dataset transforms and augmentations.

Base classes are intended to be subclassed to implement custom transforms to data and transforms,
and these implementations are intended to be used in a TransformCompose object, and passed into an
implementation of the BaseDataset class as the transform/augmentation attribute.
"""

import abc
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from torch import nn

from torch_lab.typing import CollateData
from torch_lab.utils import default_separate


class BaseTransform(nn.Module, abc.ABC):
    """Base class for general transforms that map (data, metadata) to (data_transformed,
    metadata_transformed).
    """

    def compute(
        x: Any,
        md: Dict[str, CollateData],
    ) -> Tuple[Any, Dict[str, CollateData]]:
        """Override this method in subclasses to implement transform actions."""
        return x, md

    def forward(
        self,
        x: Any,
        md: Dict[str, CollateData],
    ) -> Tuple[Any, Dict[str, CollateData]]:
        """Apply transform."""
        x, md = self.compute(x, md)
        return x, md

    def __add__(self, other):
        """Chain transforms together by adding them."""
        if isinstance(other, TransformCompose):
            return TransformCompose(self, *other.transforms)
        if isinstance(other, BaseTransform):
            return TransformCompose(self, other)


class BaseDataTransform(BaseTransform, abc.ABC):
    """Base class for transforms that only act on data."""

    def compute(x: Any) -> Any:
        """Override this method in subclasses to implement data transform actions."""
        return x

    def forward(
        self,
        x: Any,
        md: Dict[str, CollateData],
    ) -> Tuple[Any, Dict[str, CollateData]]:
        """Apply transform."""
        x = self.compute(x)
        return x, md


class BaseMetadataTransform(BaseTransform, abc.ABC):
    """Base class for transforms that only act on metadata."""

    def compute(md: Dict[str, CollateData]) -> Dict[str, CollateData]:
        """Override this method in subclasses to implement metadata transform actions."""
        return md

    def forward(
        self,
        x: Any,
        md: Dict[str, CollateData],
    ) -> Tuple[Any, Dict[str, CollateData]]:
        """Apply transform."""
        md = self.compute(md)
        return x, md


class TransformIterable(BaseTransform):
    """Wrapper class to apply a transform to a subset of keys in an iterable.

    Iterates over selected keys of data, and applies the specified transform.
    """

    def __init__(
        self, transform: BaseTransform | Callable, apply_to: Optional[List[Any]] = None
    ):
        """Initialise TransformIterable.

        Args:
            transform: Subclass of BaseTransform (or a callable that maps a (data, metadata) tuple
                to another (data, metadata) tuple) top apply to each selected key.
            apply_to: List of keys to apply a transform to.
        """
        super().__init__()
        self.transform = transform
        self.apply_to = apply_to

    def compute(
        self,
        x: Iterable,
        md: Dict[str, CollateData],
    ) -> Tuple[Iterable, Dict[str, CollateData]]:
        """Apply transform to selected keys, and raise any exception that occurs."""
        if self.apply_to is None:
            if isinstance(x, (dict, set)):
                apply_to = x
            elif isinstance(x, (list, tuple)):
                apply_to = list(range(len(x)))
            else:
                raise ValueError(
                    "TransformIterable can only accept sets, dicts, lists, or tuples."
                )
        else:
            apply_to = self.apply_to

        for i in apply_to:
            try:
                x[i], md = self.transform(x[i], md)
            except Exception as err:
                name = self.transform.__class__.__name__
                raise ValueError(
                    f"Error when applying transform '{name}' to key '{i}': {str(err)}"
                ) from err
        return x, md


class TransformCompose(BaseTransform):
    """Indexable container class that composes transforms of (data, metadata) tuples."""

    def __init__(self, *transforms: List[BaseTransform]):
        """Initialise TransformCompose.

        Args:
            transforms: Transforms to compose together. Each arg must be a BaseTransform subclass or
                a callable that tranforms a (data, metadata) tuple.
        """
        super().__init__()
        for t in transforms:
            if not callable(t):
                raise ValueError("TransformCompose must be initialised with callables.")
        self.transforms = transforms

    def compute(self, x, md):
        """Apply transforms in sequence, and raise any exception that occurs."""
        for transform in self.transforms:
            try:
                x, md = transform(x, md)
            except Exception as err:
                name = transform.__class__.__name__
                raise ValueError(
                    f"Error when applying transform '{name}': {str(err)}"
                ) from err
        return x, md

    def __len__(self):
        return len(self.transforms)

    def __getitem__(self, i):
        transforms = self.transforms[i]
        if isinstance(transforms, (tuple, list)):
            return TransformCompose(*transforms)
        return transforms

    def __eq__(self, other):
        if not isinstance(other, TransformCompose):
            return False
        return list(other.transforms) == list(self.transforms)

    def __add__(self, other):
        """Chain transforms together by adding them."""
        if isinstance(other, TransformCompose):
            return TransformCompose(*self.transforms, *other.transforms)
        if isinstance(other, BaseTransform):
            return TransformCompose(*self.transforms, other)


class TransformSeperate(BaseTransform):
    """Separate a batch into list of samples (inverse of collate_fn) before applying wrapped
    transform.
    """

    def __init__(
        self,
        transform: BaseTransform | Callable,
        separate_fn: Optional[Callable] = None,
    ):
        """Initialise TransformSeperate.

        Args:
            transform: Transform to apply to each sample in the batch.
            separate_fn: Optional callable used to invert the collate_fn operation when applying
                transform to each sample of the batch of predictions and metadata (uses
                torch_lab.utils.default_separate by default).
        """
        super().__init__()
        self.transform = transform
        self.separate_fn = separate_fn or default_separate

    def compute(self, x, md):
        x_out, md_out = [], []
        for x_sample, md_sample in zip(*self.separate_fn((x, md))):
            x_out_sample, md_out_sample = self.transform(x_sample, md_sample)
            x_out.append(x_out_sample)
            md_out.append(md_out_sample)
        return x_out, md_out
