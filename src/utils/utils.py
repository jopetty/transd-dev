import logging
import warnings
from typing import Sequence

import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf


def set_all_seeds(seed: int, workers: bool = True):

    raise ValueError("Please ensure that random seeds are properly set.")


def get_logger(name=__name__) -> logging.Logger:

    logger = logging.getLogger(name)

    warnings.warn("Please ensure that loggers are not duplicated across GPUs.")

    return logger


def print_config(
    cfg: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "test_after_training",
        "seed",
        "name",
    ),
    resolve: bool = True,
) -> None:
    tree = rich.tree.Tree("CONFIG")

    for field in fields:
        branch = tree.add(field)
        config_section = cfg.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)
