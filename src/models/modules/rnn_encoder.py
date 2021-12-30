from typing import Dict

import torch
from torch import Tensor, nn


class RNNEncoder(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        self.model = nn.Sequential(
            nn.Embedding(hparams["enc_vocab_size"], hparams["enc_embedding_size"]),
            nn.RNN(
                hparams["enc_embedding_size"],
                hparams["enc_hidden_size"],
                num_layers=hparams["enc_num_layers"],
                batch_first=True,
            ),
        )

    def forward(self, enc_input: Dict[str, Tensor]):

        outputs, last_state = self.model(enc_input["source"])

        enc_output = {
            "encoder_output": outputs,
            "encoder_last_state": torch.transpose(last_state, 0, 1),
        }

        return enc_output
