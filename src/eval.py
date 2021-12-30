import os
from typing import Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule

from src.utils import utils

log = utils.get_logger(__name__)


def eval(config: DictConfig) -> Optional[float]:

    if config.get("seed"):
        utils.set_all_seeds(config.seed, workers=True)

    log.info(f"Creating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    log.info(f"Creating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    ckpt_path = os.path.join(
        config.work_dir, config.checkpoint_dir, "checkpoints", "last.ckpt"
    )
    log.info(f"Loading model weights from <{ckpt_path}>")
    model = model.__class__.load_from_checkpoint(ckpt_path)

    model.eval()
    model.freeze()
    datamodule.setup()
    ds = datamodule.data_train.dataset
    for batch in datamodule.train_dataloader():

        source, transform, target = batch
        enc_input = {"source": source, "transform": transform, "target": target}
        y_hat = model(enc_input)["predictions"]
        for i in range(target.shape[0]):
            print("target: ", ds.convert_tokens_to_string(target[i], "target"))
            print("pred: ", ds.convert_tokens_to_string(y_hat[i], "target"))
            print("")
        # print(target.shape)
        # print(y_hat.shape)
        raise SystemExit

    # raise NotImplementedError

    return None
