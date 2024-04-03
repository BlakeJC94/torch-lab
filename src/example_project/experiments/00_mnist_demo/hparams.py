hparams = {
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
        "init": {},
        "fit": {},
        "predict": {},
    },
    "config": {
        "n_features": 32,
        "learning_rate": 1.5 * 1e-3,
        "weight_decay": 0.01,
        "num_workers": 10,
        "batch_size": 128,
        "patience": 5,
        "milestones": [5, 8],
        "gamma": 0.2,
        "monitor": "loss/validate",
    },
}

