from typing import Dict

from torch import Tensor, nn

from src.models.modules.gru_decoder import GRUDecoder
from src.models.modules.gru_encoder import GRUEncoder


class GRUSeq2Seq(nn.Module):
    def __init__(self, hparams: dict) -> None:
        super().__init__()

        self.encoder = GRUEncoder(hparams)
        self.decoder = GRUDecoder(hparams)

    def forward(self, enc_input: Dict[str, Tensor], tf_ratio: float):

        dec_input = self.encoder(enc_input) | enc_input
        output = self.decoder(dec_input, tf_ratio=tf_ratio)

        return output
