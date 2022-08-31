import logging
import os
from typing import List

import pandas as pd
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

log = logging.getLogger(__name__)


DATASET_NAME = "SCAN"


class ScanDataset(Dataset):
    @property
    def special_tokens(self) -> List[str]:
        return ["<PAD>", "<unk>", "<BOS>", "<EOS>"]

    def __init__(self, root: str, split: str) -> None:

        super().__init__()

        data_file = os.path.join(root, DATASET_NAME, split + ".txt")
        self.data = pd.read_csv(
            data_file,
            header=None,
            names=["source", "target"],
            sep="OUT:|IN:",
            engine="python",
        )
        self.data["transformation"] = "SCAN"

        # print(self.data.iloc[0])
        # raise SystemExit

        # Construct tokenizer
        self.tokenizer = get_tokenizer("basic_english")

        # Construct vocabularies
        for col in self.data.columns:
            vocab = build_vocab_from_iterator(
                self._yield_tokens(self.data[col]), specials=self.special_tokens
            )
            setattr(self, f"{col}_vocab", vocab)

            assert vocab.get_stoi()["<PAD>"] == 0, "<PAD> must have index 0!"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        item = {}
        for col in self.data.columns:
            vocab = getattr(self, f"{col}_vocab")
            item[col] = (
                [vocab["<BOS>"]]
                + [vocab[token] for token in self.tokenizer(row[col])]
                + [vocab["<EOS>"]]
            )

        return item

    def _yield_tokens(self, data: pd.Series):
        for row in data:
            yield self.tokenizer(row)

    def convert_tokens_to_string(self, tokens, col: str):
        mapping = getattr(self, f"{col}_vocab").get_itos()
        return [mapping[t] for t in tokens]
