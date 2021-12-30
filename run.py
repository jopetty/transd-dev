import dotenv
import hydra
from omegaconf import DictConfig

dotenv.load_dotenv(override=True)


@hydra.main(config_path="conf/", config_name="config.yaml")
def main(config: DictConfig):

    from src.eval import eval
    from src.train import train
    from src.utils import utils

    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    if config.mode.name == "train":
        return train(config)
    elif config.mode.name == "eval":
        return eval(config)
    else:
        raise ValueError(f"Unknown mode '{config.mode}'")


if __name__ == "__main__":
    main()
