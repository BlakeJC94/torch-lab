import contextlib
import copy
from collections.abc import Mapping, MutableMapping, MutableSequence, Sequence
from typing import Callable, Dict, Tuple, Type

import torch

default_separate_err_msg_format = (
    "default_collate: batch must be a tensor, numpy array, number, "
    "dict or list; found {}"
)


def separate(batch, *, separate_fn_map=None):
    """Opposite to torch.utils.data.collate"""
    batch_type = type(batch)
    if separate_fn_map is not None:
        if batch_type in separate_fn_map:
            return separate_fn_map[batch_type](batch, separate_fn_map=separate_fn_map)

    if isinstance(batch, Mapping):
        try:
            if isinstance(batch, MutableMapping):
                # The mapping type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new mapping.
                # Create a clone and update it if the mapping type is mutable.
                clone = copy.copy(batch)
                result = []
                for v in zip(
                    *[
                        separate(batch[key], separate_fn_map=separate_fn_map)
                        for key in batch
                    ]
                ):
                    clone = copy.copy(clone)
                    clone.update({ki: vi for ki, vi in zip(batch, v)})
                    result.append(clone)
                return result
            else:
                return [
                    batch_type({ki: vi for ki, vi in zip(batch, v)})
                    for v in zip(
                        *[
                            separate(batch[key], separate_fn_map=separate_fn_map)
                            for key in batch
                        ]
                    )
                ]
        except TypeError:
            # The mapping type may not support `copy()` / `update(mapping)`
            # or `__init__(iterable)`.
            return [
                {ki: vi for ki, vi in zip(batch, v)}
                for v in zip(
                    *[
                        separate(batch[key], separate_fn_map=separate_fn_map)
                        for key in batch
                    ]
                )
            ]
    elif isinstance(batch, tuple) and hasattr(batch, "_fields"):  # namedtuple
        return [
            batch_type(*sample)
            for sample in zip(
                *(separate(values, separate_fn_map=separate_fn_map) for values in batch)
            )
        ]
    elif isinstance(batch, Sequence):
        try:
            if isinstance(batch, MutableSequence):
                # The sequence type may have extra properties, so we can't just
                # use `type(data)(...)` to create the new sequence.
                # Create a clone and update it if the sequence type is mutable.
                clone = copy.copy(batch)  # type: ignore[arg-type]
                for i, samples in enumerate(batch):
                    clone[i] = separate(samples, separate_fn_map=separate_fn_map)
                return clone
            else:
                return batch_type(
                    [
                        separate(samples, separate_fn_map=separate_fn_map)
                        for samples in batch
                    ]
                )
        except TypeError:
            # The sequence type may not support `copy()` / `__setitem__(index, item)`
            # or `__init__(iterable)` (e.g., `range`).
            return [
                separate(samples, separate_fn_map=separate_fn_map) for samples in batch
            ]

    raise TypeError(default_separate_err_msg_format.format(batch_type))


def separate_array_fn(batch, *, separate_fn_map=None):
    return [b for b in batch]


def separate_str_fn(batch, *, separate_fn_map=None):
    return batch


default_separate_fn_map: Dict[Type | Tuple[Type, ...], Callable] = {
    torch.Tensor: separate_array_fn
}
with contextlib.suppress(ImportError):
    import numpy as np

    # For both ndarray and memmap (subclass of ndarray)
    default_separate_fn_map[np.ndarray] = separate_array_fn
    # See scalars hierarchy: https://numpy.org/doc/stable/reference/arrays.scalars.html
    # Skip string scalars
    default_separate_fn_map[(np.bool_, np.number, np.object_)] = separate_str_fn

default_separate_fn_map[float] = separate_str_fn
default_separate_fn_map[int] = separate_str_fn
default_separate_fn_map[str] = separate_str_fn
default_separate_fn_map[bytes] = separate_str_fn


def default_separate(batch):
    return separate(batch, separate_fn_map=default_separate_fn_map)
