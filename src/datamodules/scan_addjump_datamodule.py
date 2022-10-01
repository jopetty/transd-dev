from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from src.datamodules.datasets.scan_dataset import ScanDataset


class ScanAddJumpDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[float, float, float] = (0.9, 0.1, 0.0),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train = None
        self.data_val = None
        self.data_test = None

    def setup(self, stage: Optional[str] = None) -> None:

        train_val_dataset = ScanDataset(
            self.hparams.data_dir, split="tasks_train_addprim_jump"
        )

        train_val_split = (
            self.hparams.train_val_test_split[0],
            self.hparams.train_val_test_split[1],
        )

        # Ensures that the sum is always correct, in case
        # two splits end up with XX.5 values before rounding
        split_lengths = list(
            round(p * len(train_val_dataset)) for p in train_val_split[:-1]
        )
        split_lengths.append(len(train_val_dataset) - sum(split_lengths))

        self.data_train, self.data_val = random_split(
            dataset=train_val_dataset,
            lengths=split_lengths,
            generator=torch.manual_seed(42),
        )

        self.data_test = ScanDataset(
            self.hparams.data_dir, split="tasks_test_addprim_jump"
        )

    def pad_collate(self, data) -> Tuple[Tensor, Tensor, Tensor]:

        source_arr = [torch.Tensor(r["source"]).long() for r in data]
        transf_arr = [torch.Tensor(r["transformation"]).long() for r in data]
        target_arr = [torch.Tensor(r["target"]).long() for r in data]

        source_stack = pad_sequence(source_arr, batch_first=True)
        transf_stack = pad_sequence(transf_arr, batch_first=True)
        target_stack = pad_sequence(target_arr, batch_first=True)

        return (source_stack, transf_stack, target_stack)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.pad_collate,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.pad_collate,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.pad_collate,
            shuffle=False,
        )
