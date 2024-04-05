import pytest

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
