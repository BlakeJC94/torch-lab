import abc
from typing import Any, Callable, Iterable, List

from torch import nn


class BaseTransform(nn.Module, abc.ABC):
    def compute(x, md):
        return x, md

    def forward(self, x, md=None):
        x, md = self.compute(x, md)
        return x, md


class BaseDataTransform(BaseTransform):
    def compute(x):
        return x

    def forward(self, x, md=None):
        x = self.compute(x)
        return x, md


class BaseMetadataTransform(BaseTransform):
    def compute(md):
        return md

    def forward(self, x, md=None):
        md = self.compute(md)
        return x, md


class TransformIterable(BaseTransform):
    def __init__(self, apply_to: List[Any], transform: Callable):
        super().__init__()
        self.transform = transform
        self.apply_to = apply_to

    def compute(self, x: Iterable, md=None):
        for i in self.apply_to:
            try:
                x[i], md = self.transform(x[i], md)
            except Exception as err:
                name = self.transform.__class__.__name__
                raise ValueError(
                    f"Error when applying transform '{name}' to key '{i}': {str(err)}"
                ) from err
        return x, md


class TransformCompose(BaseTransform):
    def __init__(self, *transforms):
        super().__init__()
        self.transforms = transforms

    def compute(self, x, md):
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
