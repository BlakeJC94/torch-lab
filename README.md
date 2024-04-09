# torch-lab

Core components for experiments with Pytorch

## Usage

Add this package to a projects requirements
```
torch_lab @ git+ssh://git@github.com/BlakeJC94/torch-lab.git
```

A project should be laid out as
```
.
├── data         # Git-ignored dir for links and dirs for raw/processed data and artifacts
└── project_name
    ├── __init__.py
    ├── transforms.py
    ├── datasets.py
    ├── ...
    ├── preprocess
    │   ├── __init__.py
    │   └── dataset_name.py     # Script for builing and uploading datasets to ClearML
    └── experiments
        └── 00_experiment_name       # Numbered and titled experiment stem
            ├── __init__.py
            ├── __main__.py          # Configuration written as functions of hparams dictionaries
            ├── decrease_lr.py       # Hyperparmeter names appended to experiment name in clearml
            └── hyperparam_name.py   # Hyperparameters defined as a JSON-like python dictionary
```

### Experiment layout

Each experiment directory has at least 2 key files: a hyperparameters script and a config
script (`__main__.py`):
- The hyperparameters script defines the configuration of the experiment in a single python
  dictionary named `hparams`,
- The `hparams` dictionary is used in the `__main__.py` config script to configure the model and
  dataset objects.

Using `torch_lab`, a experiment is launched by selecting the specific hyperparameters script, which
will then apply the hyperparameters to the config in the experiment directory.

An annotated example is provided in this repo under `./src/example_project`.

#### Hyperparameters script

The `hparams` dictionary defined in the hyperparameters can be used to control aspects of the `clearml.Task` and `pytorch_lightning.Trainer`, and also loading of checkpoints from previous experiments with the following keys:
```python
hparams = {
    "task": {                         # Kwargs for `clearml.Task.init(..)`
        "project_name": "test",       # E.g. project name
    },
    "checkpoint": {                   # (Optional) Options For using checkpoint
        "checkpoint_task_id": None,   # ClearML ID for previous task, by default set to None
        # "checkpoint_name": "last",  # Name of Checkpoint to load, by default set to "last"
        # "weights_only": False,      # Whether to only load the state dict, by default False
    },
    "trainer": {         # (Optional) Inputs for pytorch lightning Trainer
        "init": {},      # Kwargs for pl.Trainer(..)
        "fit": {},       # Kwargs for pl.Trainer.fit(..)
        "predict": {},   # Kwargs for pl.Trainer.predict(..)
    },
    "config": {   # Keys and value used in config
        ...
    },
}
```

As demonstrated in `./src/example_project/experiments/00_mnist_demo/hparams_modified.py`, multiple
hyperparameters can be defined for a single experiment config. As demonstrated here, hyperparameters
can also be layered using `torch_lab.utils.import_script_as_module`.

#### Config script

For training, the `__main__.py` script must define a function called `train_config` which accepts
a single arg and returns a dictionary with the following keys:
```python
def train_config(hparams):
    ...
    return dict(
        module=...,             # torch_lab.modules.TrainLabModule
        train_dataloaders=...,  # torch.utils.data.DataLoader
        val_dataloaders=...,    # torch.utils.data.DataLoader
        callbacks=[...],        # List of pytorch_lightning.Callback objects (optional)
    )
```

The core of this function should use the values of `hparams` to define and intiliase datasets and
dataloaders, and the model to be trained using python code defined within `__main__.py` and/or
importing custom code from project modules. See the `src/example_project` for a worked example.

The `torch_lab.module.TrainLabModule` provides a scaffold for the `torch.nn.Module` object,
the loss function, `torchmetrics.Metrics`, the optimizer and the and scheduler. See the docs for
further details.

For inference, the `__main__.py` script must define a function called `infer_config` which
accepts at least one arg and returns a dictionary with the following keys:
```python
def infer_config(hparams, ...) -> Dict[str, Any]:
    ...
    return dict(
        module=...,               # torch_lab.modules.LabModule
        predict_dataloaders=...,  # torch.utils.data.DataLoader
        callbacks=[...],          # List of pytorch_lighting.Callback objects (optional)
    )
```

The core of this function should use the values of `hparams` and extra args to define and initialize
datasets and dataloaders, and the model to be trained using python code defined within `__main__.py`
and/or importing custom code from project modules. See the `src/example_project` for a worked
example.

The variable args can be accessed via the `infer` CLI, any args after the `hparams_path` will be
passed into `infer_config` as strings.


### Endpoint usage

Once added to your project dependencies, install your virtual environment and launch tasks using the
provided endpoints.

To launch a training job (after setting up ClearML):
```bash
$ train <path/to/hparams.py> [--dev-run <float or int>] [--offline] [--debug]
```

To launch an inference job:
```bash
$ infer <path/to/hparams.py> [<extra args specified in config>]
```

## Development

A simple git clone will suffice
```bash
$ git clone https://github.com/BlakeJC94/torch-lab
```

Pull requests and issues are welcome!
