# torch-lab

Core components for experiments with Pytorch

TODO
- [x] Merge artifacts into data
- [ ] Module checkpoint hooks
- [ ] MNIST config
- [x] `--debug` parse as int or float
- [x] `predict` --> `infer`
- [ ] Parse hparams as attrdicts
- [x] Tasks to core
- [x] Core -> torch_lab

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

TODO

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
