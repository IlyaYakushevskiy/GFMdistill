import hashlib
import os as os
import pathlib
import pprint
import time

import hydra
import torch
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_loaders.base import GeoFMDataset, GeoFMSubset, RawGeoFMDataset
from decoders.base import Decoder
from encoders.base import Encoder
from engine.evaluator import Evaluator
from engine.trainer import Trainer
from utils.collate_fn import get_collate_fn
from utils.logger import init_logger
from utils.subset_sampler import get_subset_indices
from utils.utils import (
    fix_seed,
    get_best_model_ckpt_path,
    get_final_model_ckpt_path,
    get_generator,
    seed_worker,
)

def get_exp_info(hydra_config: HydraConf) -> dict[str, str]:
    """Create a unique experiment name based on the choices made in the config.

    Args:
        hydra_config (HydraConf): hydra config.

    Returns:
        str: experiment information.
    """
    choices = OmegaConf.to_container(hydra_config.runtime.choices)
    cfg_hash = hashlib.sha1(
        OmegaConf.to_yaml(hydra_config).encode(), usedforsecurity=False
    ).hexdigest()[:6]
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    fm = choices["encoder"]
    decoder = choices["decoder"]
    ds = choices["dataset"]
    task = choices["task"]
    exp_info = {
        "timestamp": timestamp,
        "fm": fm,
        "decoder": decoder,
        "ds": ds,
        "task": task,
        "exp_name": f"{timestamp}_{cfg_hash}_{fm}_{decoder}_{ds}",
    }
    return exp_info


###TRAIN 
@hydra.main(version_base=None, config_path="./config", config_name="train")
def main(cfg: DictConfig) -> None:
    fix_seed(cfg.seed)
    rank = cfg.rank
    print(f"Rank: {rank}")


if __name__ == "__main__":
    main()