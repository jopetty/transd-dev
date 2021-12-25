from typing import Optional

from omegaconf import DictConfig

from src.utils import utils

log = utils.get_logger(__name__)


def train(cfg: DictConfig) -> Optional[float]:

    if cfg.get("seed"):
        utils.set_all_seeds(cfg.seed, workers=True)

    raise NotImplementedError

    return None
