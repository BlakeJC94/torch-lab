import random

import numpy as np
import pytest

from torch_lab.transforms import (
    BaseDataTransform,
    BaseMetadataTransform,
    BaseTransform,
    TransformCompose,
    TransformIterable,
)


@pytest.fixture
def x():
    return np.ones((3, 4))


@pytest.fixture
def md():
    return {
        "foo": 1,
        "bar": 2,
    }


class MockTransform(BaseTransform):
    def __init__(self, foo):
        super().__init__()
        self.foo = foo

    def compute(self, x, md):
        md["foo"] += self.foo
        return 2 * x, md


class MockDataTransform(BaseDataTransform):
    def compute(self, x):
        return 2 * x


class MockMetadataTransform(BaseMetadataTransform):
    def compute(self, md):
        md["bar"] += 1
        return md


def test_base_transform(x, md):
    transform = MockTransform(3)

    out, md_out = transform(x, md)

    assert (out == 2 * np.ones_like(x)).all()
    assert md_out["foo"] == 4


def test_base_data_transform(x, md):
    transform = MockDataTransform()

    out, md_out = transform(x, md)

    assert (out == 2 * np.ones_like(x)).all()
    assert md == md_out


def test_base_metadata_transform(x, md):
    transform = MockMetadataTransform()

    out, md_out = transform(x, md)

    assert (out == x).all()
    assert md_out["bar"] == 3


def test_transform_compose(x, md):
    transform_0, transform_1, transform_2 = (
        MockTransform(3),
        MockDataTransform(),
        MockMetadataTransform(),
    )
    transform = TransformCompose(
        transform_0,
        transform_1,
        transform_2,
    )

    out, md_out = transform(x, md)

    assert transform[0] == transform_0
    assert transform[1:] == TransformCompose(transform_1, transform_2)

    assert (out == 4 * np.ones_like(x)).all()
    assert md_out == {"foo": 4, "bar": 3}


def test_transform_iterable(x, md):
    data = {k: i * x for i, k in enumerate(list("abc"))}
    apply_to = random.choice([None, ["a", "b"]])

    transform = TransformIterable(MockDataTransform(), apply_to)

    out, md_out = transform(data, md)

    if apply_to is None:
        apply_to = list(data.keys())

    for i, k in enumerate(apply_to):
        assert (out[k] == i * x * 2).all()


def test_add_transforms():
    t1 = MockDataTransform()
    t2 = MockMetadataTransform()
    t = t1 + t2
    assert isinstance(t, TransformCompose)
    assert t == TransformCompose(t1, t2)


def test_add_transform_compose():
    t1 = MockDataTransform()
    t2 = MockMetadataTransform()
    t3 = TransformCompose(t1, t2)
    t = t3 + t1
    assert isinstance(t, TransformCompose)
    assert t == TransformCompose(t1, t2, t1)


def test_add_compose_transform():
    t1 = MockDataTransform()
    t2 = MockMetadataTransform()
    t3 = TransformCompose(t1, t2)
    t = t1 + t3
    assert isinstance(t, TransformCompose)
    assert t == TransformCompose(t1, t1, t2)


def test_add_compose():
    t1 = MockDataTransform()
    t2 = MockMetadataTransform()
    t3 = TransformCompose(t1, t2)
    t = t3 + t3
    assert isinstance(t, TransformCompose)
    assert t == TransformCompose(t1, t2, t1, t2)
