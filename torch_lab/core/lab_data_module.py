import os
import sys
from typing import Callable, Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Sampler

class LabDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        num_workers: int = os.cpu_count(),
        batch_size_train: int = 16,
        batch_size_test: int = 16,
        samplers: Optional[Dict[str, Sampler]] = None,
        collate_fn: Optional[Callable] = None,
    ):
        super().__init__()
        if all([ds is None for ds in [train_dataset, val_dataset, test_dataset, predict_dataset]]):
            raise ValueError("Must supply at least one dataset.")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.predict_dataset = predict_dataset
        self.num_workers = num_workers
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.samplers = samplers if samplers else {}
        self.collate_fn = collate_fn

        # Must fork instead of spawn on Mac or can't import any configs with lambda functions:
        # _pickle.PicklingError: Can't pickle <function <lambda> at {address}>:\
        # import of module 'config' failed
        self.multiprocess_context = (
            "fork" if self.num_workers > 0 and sys.platform == "darwin" else None
        )

        self.save_hyperparameters("num_workers", "batch_size_train", "batch_size_test")

    def setup(self, stage: Optional[str] = None) -> None:
        """Assign train/val/test datasets to attributes for use in dataloaders.

        NB this method is called in every GPU/machine.
        """
        if stage == "fit":
            assert self.train_dataset is not None, "Train dataset not provided."
        elif stage == "validate":
            assert self.val_dataset is not None, "Validation dataset not provided."
        elif stage == "test":
            assert self.test_dataset is not None, "Test dataset not provided."
        elif stage == "predict":
            assert self.predict_dataset is not None, "Prediction dataset not provided."

    def train_dataloader(self) -> DataLoader:
        """Return a DataLoader for training."""
        sampler = self.samplers.get("train", RandomSampler(self.train_dataset))
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            sampler=sampler,
            collate_fn=self.collate_fn,
            multiprocessing_context=self.multiprocess_context,
        )

    def val_dataloader(self) -> DataLoader:
        """Return a DataLoader for validation."""
        sampler = self.samplers.get("val", SequentialSampler(self.val_dataset))
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_test,
            num_workers=self.num_workers,
            sampler=sampler,
            collate_fn=self.collate_fn,
            multiprocessing_context=self.multiprocess_context,
        )

    def test_dataloader(self) -> DataLoader:
        """Return a DataLoader for testing."""
        sampler = self.samplers.get("test", SequentialSampler(self.test_dataset))
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size_test,
            num_workers=self.num_workers,
            sampler=sampler,
            collate_fn=self.collate_fn,
            multiprocessing_context=self.multiprocess_context,
        )

    def predict_dataloader(self) -> DataLoader:
        """Return a DataLoader for prediction."""
        sampler = self.samplers.get("predict", SequentialSampler(self.predict_dataset))
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size_test,
            num_workers=self.num_workers,
            sampler=sampler,
            collate_fn=self.collate_fn,
            multiprocessing_context=self.multiprocess_context,
        )
