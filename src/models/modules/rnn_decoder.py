import random
from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RNNDecoder(nn.Module):
    @property
    def max_gen_length(self) -> int:
        return self.hparams["dec_max_gen_length"]

    @property
    def EOS_idx(self) -> int:
        return self.hparams["dec_EOS_idx"]

    def __init__(self, hparams: dict) -> None:
        super().__init__()

        self.hparams = hparams

        self.embedding = nn.Embedding(
            hparams["dec_vocab_size"], hparams["dec_embedding_size"]
        )
        self.unit = nn.RNN(
            hparams["dec_embedding_size"],
            hparams["dec_hidden_size"],
            num_layers=hparams["dec_num_layers"],
            batch_first=True,
        )
        self.output = nn.Linear(hparams["dec_hidden_size"], hparams["dec_vocab_size"])

    def forward_step(self, step_input: Dict[str, Tensor]) -> Dict[str, Tensor]:

        # Unsqueeze if only one batch is present
        no_squeeze = lambda a: a.unsqueeze(0) if a.shape == 2 else a

        # print("Step Input")
        # for key in step_input:
        #     print(f"{key}: {step_input[key].shape}")

        h = no_squeeze(step_input["h"])
        unit_input = no_squeeze(F.relu(self.embedding(step_input["x"])))
        _, state = self.unit(unit_input, h)
        y = self.output(no_squeeze(state[-1, :, :]))

        # print(f"h: {h.shape}")
        # print(f"unit_input: {unit_input.shape}")
        # print(f"unk: {unk.shape}")
        # print(f"state: {state.shape}")
        # print(f"state[-1]: {state[-1].shape}")
        # print(f"y: {y.shape}")

        return {"y": y, "h": state}

    def get_step_input(self, dec_input: Dict[str, Tensor]) -> Dict[str, Tensor]:

        if "h" in dec_input:
            h = dec_input["h"]
        elif "encoder_last_state" in dec_input:
            h = torch.transpose(dec_input["encoder_last_state"], 0, 1)
        else:
            raise ValueError(
                f"You must provide a hidden input in dec_input '{dec_input}'"
            )

        if "x" in dec_input:
            x = dec_input["x"]
        elif "transform" in dec_input:
            # print("No x found")
            # print(dec_input["transform"][:, 1:-1].shape)
            x = dec_input["transform"][:, 1:-1]
        else:
            raise ValueError(
                f"You must provide a step input in dec_input '{dec_input}'"
            )

        step_input = {"x": x, "h": h}

        if "encoder_output" in dec_input:
            step_input["encoder_output"] = dec_input["encoder_output"]

        return step_input

    def forward(self, dec_input: Dict[str, Tensor], tf_ratio) -> Dict[str, Tensor]:

        is_teacher_forcing = random.random() < tf_ratio

        batch_size: int = dec_input["encoder_output"].shape[0]
        hidden_size: int = self.output.in_features
        vocab_size: int = self.output.out_features
        gen_length = (
            dec_input["target"][0].shape[0]
            if is_teacher_forcing
            else self.max_gen_length
        )

        dec_step_input = self.get_step_input(dec_input)

        has_finished = torch.zeros(batch_size, dtype=torch.bool)
        dec_output = torch.zeros(gen_length, batch_size, vocab_size)
        dec_hidden = torch.zeros(gen_length, batch_size, hidden_size)

        for i in range(gen_length):

            # print(f"STEP {i} (tf={is_teacher_forcing})")

            step_result = self.forward_step(dec_step_input)
            step_prediction = step_result["y"].argmax(dim=-1)

            # for key in step_result:
            #     print(f"step_result[{key}]: {step_result[key].shape}")
            # print("dec_hidden: ", dec_hidden.shape)

            dec_output[i] = step_result["y"]
            dec_hidden[i] = step_result["h"]

            has_finished[step_prediction == self.EOS_idx] = True
            if all(has_finished):
                break
            else:
                x = dec_input["target"][:, i] if is_teacher_forcing else step_prediction
                step_result["x"] = x.unsqueeze(-1)
                step_result["encoder_output"] = dec_input["encoder_output"]

                dec_step_input = self.get_step_input(step_result)

        output = {
            "logits": torch.transpose(dec_output, 0, 1),
            "predictions": torch.transpose(dec_output, 0, 1).argmax(dim=-1),
            "decoder_hiddens": dec_hidden,
        }

        return output
