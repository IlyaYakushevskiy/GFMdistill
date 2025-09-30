import os as os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import logging

from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
from torch.nn import Module
from collections import OrderedDict

_log = logging.getLogger(__name__)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# to make flops calculator work
def prepare_input(input_res):
    image = {}
    x1 = torch.FloatTensor(*input_res)
    # input_res[-2] = 2
    input_res = list(input_res)
    input_res[-3] = 2
    x2 = torch.FloatTensor(*tuple(input_res))
    image["optical"] = x1
    image["sar"] = x2
    return dict(img=image)


def _find_ckpt(exp_dir: str | Path, suffix: str) -> Optional[str]:
    """Return the *first* file that ends with `suffix`; None if nothing found."""
    exp_dir = Path(exp_dir)
    for fname in exp_dir.iterdir():
        if fname.name.endswith(suffix):
            return str(fname)
    # Nothing found – warn once.
    _log.warning(
        "No checkpoint matching '*%s' found in %s. "
        "If this was a k-NN probe (no training), you can ignore this warning. Otherwise, check your experiment directory.",
        suffix, exp_dir,
    )
    return None


def get_best_model_ckpt_path(exp_dir: str | Path) -> Optional[str]:
    """Return '<exp_dir>/…_best.pth' or None when it does not exist."""
    return _find_ckpt(exp_dir, "_best.pth")


def get_final_model_ckpt_path(exp_dir: str | Path) -> Optional[str]:
    """Return '<exp_dir>/…_final.pth' or None when it does not exist."""
    return _find_ckpt(exp_dir, "_final.pth")



def load_teacher_model(encoder_cfg: DictConfig, decoder_cfg: DictConfig, ckpt_dir: str, logger) -> Module:
    """
    Instantiates a full teacher model and loads its weights from a checkpoint directory.

    Args:
        encoder_cfg: The OmegaConf config for the teacher's encoder.
        decoder_cfg: The OmegaConf config for the teacher's decoder.
        ckpt_dir: The directory containing the teacher's checkpoint file.
        logger: The logger instance.

    Returns:
        A torch.nn.Module with loaded weights, on the CPU.
    """
    logger.info(f"--- Loading teacher model from directory: {ckpt_dir} ---")

    # 1. Instantiate the model architecture from the config
    encoder = instantiate(encoder_cfg)
    model = instantiate(decoder_cfg, encoder=encoder)

    # 2. Find the best checkpoint file within the provided directory
    model_ckpt_path = get_best_model_ckpt_path(ckpt_dir)
    if model_ckpt_path is None:
        raise FileNotFoundError(f"Could not find 'model_best.pth' in directory {ckpt_dir}")

    logger.info(f"Found teacher checkpoint. Loading weights from: {model_ckpt_path}")

    # 3. Load the checkpoint file
    checkpoint = torch.load(model_ckpt_path, map_location='cpu', weights_only=False)

    # 4. Create a new state dict to handle the 'module.' prefix from DDP training
    state_dict = checkpoint['model']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.`
        new_state_dict[name] = v

    # 5. Load the cleaned state dict into the model
    model.load_state_dict(new_state_dict)
    logger.info("Successfully loaded teacher model weights.")
    return model
