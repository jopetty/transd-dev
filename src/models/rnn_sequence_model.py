from typing import Any, Dict

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.modules.rnn_seq2seq import RNNSeq2Seq


class RNNSequenceModel(LightningModule):
    def __init__(
        self,
        enc_vocab_size: int = 47,
        enc_embedding_size: int = 256,
        enc_hidden_size: int = 256,
        enc_num_layers: int = 1,
        dec_vocab_size: int = 49,
        dec_input_size: int = 100,
        dec_embedding_size: int = 256,
        dec_hidden_size: int = 256,
        dec_num_layers: int = 1,
        dec_EOS_idx: int = 2,
        dec_max_gen_length: int = 20,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.model = RNNSeq2Seq(hparams=self.hparams)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.val_acc_best = MaxMetric()

    def forward(self, enc_input: Dict[str, Tensor]):

        return self.model(enc_input)

    def normalize_lengths(self, logits, target):

        diff = int(target.shape[1] - logits.shape[1])
        if diff == 0:
            pass
        elif diff > 0:
            padding = torch.zeros(logits.shape[0], diff, logits.shape[-1])
            padding[:, :, 0] = 1.0
            logits = torch.concat((logits, padding), dim=1)
        else:
            target = F.pad(input=target, pad=(0, -diff), value=0)

        logits = torch.transpose(logits, 1, 2)
        return logits, target

    def step(self, batch: Any):
        source, transform, target = batch
        enc_input = {"source": source, "transform": transform, "target": target}
        output = self.forward(enc_input)
        logits = output["logits"]
        preds = output["predictions"]

        logits, target = self.normalize_lengths(logits, target)
        loss = self.criterion(logits, target)

        return loss, preds, target

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # print("target: ", targets[0])
        # print("prediction: ", preds[0])

        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # print(loss)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)
        self.log(
            "val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True
        )

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_epoch_end(self) -> None:
        self.train_acc.reset()
        self.val_acc.reset()
        self.test_acc.reset()

    def configure_optimizers(self):
        return torch.optim.SGD(
            params=self.parameters(),
            lr=self.hparams.lr,
            # weight_decay=self.hparams.weight_decay,
        )
