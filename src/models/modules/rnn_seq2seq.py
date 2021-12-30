from typing import Dict

from torch import Tensor, nn

from src.models.modules.rnn_decoder import RNNDecoder
from src.models.modules.rnn_encoder import RNNEncoder


class RNNSeq2Seq(nn.Module):
    def __init__(self, hparams: dict) -> None:
        super().__init__()

        self.encoder = RNNEncoder(hparams)
        self.decoder = RNNDecoder(hparams)

    def forward(self, enc_input: Dict[str, Tensor]):

        dec_input = self.encoder(enc_input) | enc_input
        output = self.decoder(dec_input, tf_ratio=1.0)

        return output
