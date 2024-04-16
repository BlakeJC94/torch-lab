import random
import re

import pytest
import torch

from torch_lab.utils import (
    compile_config,
    default_separate,
    dict_as_str,
    get_hparams_and_config_path,
    import_script_as_module,
    set_hparams_debug_overrides,
)


class TestCompileConfig:
    """Tests and fixtures for `compile_config`."""

    @pytest.fixture
    def mock_config(self):
        return """
def train_config(config):
    return dict(foo=2*config["bar"])

def infer_config(config, baz):
    return dict(foo=2*config["bar"] + baz)
"""

    @pytest.fixture
    def mock_config_path(self, tmp_path, mock_config):
        config_path = tmp_path / "__main__.py"
        with open(config_path, "w") as f:
            f.write(mock_config)
        return config_path

    def test_train_config(self, mock_config_path):
        """Test that a train config can be compiled from hparams dict and a function with one arg."""
        hparams = {"config": dict(bar=2)}
        output = compile_config(mock_config_path, hparams)
        assert output["foo"] == 4

    def test_infer_config(self, mock_config_path):
        """Test that a infer config can be compiled from hparams dict and a function with multiple
        args.
        """
        hparams = {"config": dict(bar=2)}
        args = [3]
        output = compile_config(mock_config_path, hparams, *args, field="infer_config")
        assert output["foo"] == 7


def test_get_hparams_and_config_path():
    path = random.choice(
        [
            "./src/example_project/experiments/00_mnist_demo/hparams.py",
            "./src/example_project/experiments/00_mnist_demo/hparams_modified.py",
        ]
    )
    dev_run = random.choice(["", "1", "3", "0.7"])

    hparams, config_path = get_hparams_and_config_path(path, dev_run)

    expected = import_script_as_module(path).hparams
    if dev_run:
        expected = set_hparams_debug_overrides(expected, dev_run)

    assert expected == hparams
    assert config_path.name == "__main__.py"


def test_import_script_as_module():
    """Test that hparams can be imported using import_script_as_module."""
    path = random.choice(
        [
            "./src/example_project/experiments/00_mnist_demo/hparams.py",
            "./src/example_project/experiments/00_mnist_demo/hparams_modified.py",
        ]
    )
    output = import_script_as_module(path)
    assert hasattr(output, "hparams")
    assert isinstance(output.hparams, dict)
    assert all(k in output.hparams for k in ["config", "task", "trainer"])


def test_set_hparams_debug_overrides(hparams):
    """Test that set_hparams_debug_overrides sets the correct values"""
    dev_run = random.choice(["1", "3", "0.7"])
    input_hparams = hparams.copy()
    updated = set_hparams_debug_overrides(hparams, dev_run)
    assert updated["task"]["project_name"] == "test"
    assert updated["trainer"]["init"]["log_every_n_steps"] == 1
    assert (
        updated["trainer"]["init"]["overfit_batches"] == float(dev_run)
        if "." in dev_run
        else int(dev_run)
    )
    assert updated["config"]["num_workers"] == 0

    updated["config"].pop("num_workers")
    input_hparams["config"].pop("num_workers", None)
    assert input_hparams["config"] == updated["config"]


class TestDictAsStr:
    """Fixtures and tests for `dict_as_str`."""

    @pytest.fixture
    def mock_dict(self):
        return {
            "a": "foo",
            "b": {
                "c": 123,
                "d": 456.7,
                "e": [8, 9, 10],
            },
            "f": [
                11,
                12.0,
                "13",
            ],
            "g": None,
        }

    @pytest.fixture
    def mock_dict_str(self):
        return """{
  "a": "foo",
  "b": {
    "c": 123,
    "d": 456.7,
    "e": [
      8,
      9,
      10
    ]
  },
  "f": [
    11,
    12.0,
    "13"
  ],
  "g": null
}"""

    def test_basic(self, mock_dict, mock_dict_str):
        """Test that dicts are printed as Json-formatted strings."""
        assert dict_as_str(mock_dict) == mock_dict_str

    def test_object(self):
        """Test that Python objects are converted to their `__str__` representations."""
        output = dict_as_str({"foo": lambda x: 2 * x})
        assert re.match(
            r'{\n  "foo": "<function TestDictAsStr.test_object.<locals>.<lambda> at 0x[0-9a-f]+>"\n}',
            output,
        )


class TestSeparate:
    def test_tensor(self):
        batch = torch.rand(2, 3, 4)
        result = default_separate(batch)
        assert isinstance(result, list)
        assert all(isinstance(elem, torch.Tensor) for elem in result)
        assert all(elem.shape == batch.shape[1:] for elem in result)
        for i, elem in enumerate(result):
            assert (batch[i] == elem).all()

    def test_list_tensor(self):
        batch = [torch.rand(2, 3, 4) for _ in range(5)]
        result = default_separate(batch)
        assert isinstance(result, list)
        assert all(isinstance(elem, list) for elem in result)
        assert all(e.shape == batch[0].shape[1:] for elem in result for e in elem)
        for i, elem in enumerate(result):
            for j, e in enumerate(elem):
                assert (e == batch[i][j]).all()

    def test_dict_tensor(self):
        batch = {k: torch.rand(2, 3, 4) for k in list("qwe")}
        result = default_separate(batch)
        assert isinstance(result, list)
        assert all(batch.keys() == r.keys() for r in result)
        assert all(batch[k].shape[1:] == v.shape for r in result for k, v in r.items())
        for i, r in enumerate(result):
            assert all((r[k] == batch[k][i]).all() for k in batch)

    def test_list_str(self):
        batch = list("qwert")
        result = default_separate(batch)
        assert batch == result

    def test_list_float(self):
        batch = [float(i) for i in range(5)]
        result = default_separate(batch)
        assert batch == result

    def test_list_tensor_and_dict_tensor(self):
        batch_data = torch.rand(5, 6, 7)
        batch_md = {
            "foo": torch.rand(5, 3, 4),
            "bar": ["bar" for _ in batch_data],
            "baz": [float(5) for _ in batch_data],
        }
        batch = (batch_data, batch_md)

        result = default_separate(batch)
        data, mds = result

        assert isinstance(data, list)
        assert all(isinstance(elem, torch.Tensor) for elem in data)
        assert all(elem.shape == batch_data.shape[1:] for elem in data)

        assert isinstance(mds, list)
        for i, md in enumerate(mds):
            assert isinstance(md, dict)
            assert md.keys() == batch_md.keys()

            assert (md["foo"] == batch_md["foo"][i]).all()
            assert md["bar"] == batch_md["bar"][i]
            assert md["baz"] == batch_md["baz"][i]
