from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, random_split

from src.datamodules.datasets.anaphora_dataset import AnaphoraDataset


class AnaphoraDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_gen: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if (
            not self.data_train
            and not self.data_val
            and not self.data_test
            and not self.data_gen
        ):
            in_domain_dataset = AnaphoraDataset(self.hparams.data_dir, split="train")

            # Ensures that the sum is always correct, in case
            # two splits end up with XX.5 values before rounding
            split_lengths = list(
                round(p * len(in_domain_dataset))
                for p in self.hparams.train_val_test_split[:-1]
            )
            split_lengths.append(len(in_domain_dataset) - sum(split_lengths))

            self.data_train, self.data_val, self.data_test = random_split(
                dataset=in_domain_dataset,
                lengths=split_lengths,
                generator=torch.manual_seed(42),
            )

            self.data_gen = AnaphoraDataset(self.hparams.data_dir, split="gen")

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

    def gen_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_gen,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.pad_collate,
            shuffle=False,
        )
