from torch_lab import Config, LabModule, LabDataModule

def main() -> Config:

    data_module = LabDataModule(
        ...
    )

    module = LabModule(
        ...
    )

    return Config(
        module=module,
        data_module=data_module,
    )
