import mnist
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from torch_lab import Config, LabModule, LabDataModule



class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: Tensor):
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def main() -> Config:
    train_images = torch.as_tensor(mnist.train_images()).float() / 255
    train_labels = torch.as_tensor(mnist.train_labels()).long()
    val_images = torch.as_tensor(mnist.test_images()).float() / 255
    val_labels = torch.as_tensor(mnist.test_labels()).long()

    data_module = LabDataModule(
        train_dataset=TensorDataset(train_images, train_labels),
        val_dataset=TensorDataset(val_images, val_labels),
        test_dataset=TensorDataset(val_images, val_labels),
        num_workers=8,
        batch_size_train=256,
        batch_size_test=256,
    )

    module = LabModule(
        model=Classifier(),
        loss_function=nn.CrossEntropyLoss(),
        metrics={"Accuracy": Accuracy(task="multiclass", num_classes=10)},
        optimizer_config=(torch.optim.Adam, dict(lr=0.0015, weight_decay=0.0001)),
    )

    callbacks = dict(
        train=[
            EarlyStopping(
                monitor="loss/validate",
                min_delta=0.001,
                patience=3,
                verbose=True,
                mode="min",
            ),
            ModelCheckpoint(
                monitor="loss/validate",
            ),
        ],
    )

    return Config(
        project_name="examples",
        task_name="MNIST",
        module=module,
        data_module=data_module,
        callbacks=callbacks,
    )


# IDEA: Write predict function in config!! torh--lab should just be for training and testing
