from typing import List, Optional

import hydra
from omegaconf import DictConfig, open_dict
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils import utils

log = utils.get_logger(__name__)


def train(config: DictConfig) -> Optional[float]:

    if config.get("seed"):
        utils.set_all_seeds(config.seed, workers=True)

    log.info(f"Creating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Set model config based on dataset values
    datamodule.setup()
    dec_vocab_size: int = len(datamodule.data_train.dataset.target_vocab)
    enc_vocab_size: int = len(datamodule.data_train.dataset.source_vocab)
    dec_EOS_idx: int = datamodule.data_train.dataset.target_vocab.get_stoi()["<EOS>"]
    with open_dict(config):
        config.model.dec_vocab_size = dec_vocab_size
        config.model.enc_vocab_size = enc_vocab_size
        config.model.dec_EOS_idx = dec_EOS_idx

    log.info(f"Creating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Creating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Creating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    log.info(f"Instiantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
    )

    log.info("Logging hyperparameters")
    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    log.info("Training model")
    trainer.fit(model=model, datamodule=datamodule)

    score = trainer.callback_metrics.get(config.get("fast_dev_run"))

    if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Testing model")
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    log.info("Finalizing training run")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    if not config.trainer.get("fast_dev_run"):
        log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    return score
