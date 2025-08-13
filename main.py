import hashlib
import os as os
import pathlib
import pprint
import time

from fvcore.nn import FlopCountAnalysis
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
        "exp_name": f"{timestamp}_{fm}_{decoder}_{ds}_{cfg_hash}",
    }
    return exp_info


@hydra.main(version_base=None, config_path="./configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Geofm-bench main function.

    Args:
        cfg (DictConfig): main_config
    """
    # fix all random seeds
    fix_seed(cfg.seed)
    # distributed training variables
    # rank = cfg.rank
    # local_rank = int(cfg.local_rank)


    # device = torch.device("cuda", local_rank)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)


    # torch.cuda.set_device(device)
    # torch.distributed.init_process_group(backend="nccl")

    # true if training else false
    train_run = cfg.train
    if train_run:
        exp_info = get_exp_info(HydraConfig.get())
        exp_name = exp_info["exp_name"]
        task_name = exp_info["task"]
        exp_dir = pathlib.Path(cfg.work_dir) / "logs"/ exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        logger_path = exp_dir / "train.log"
        config_log_dir = exp_dir / "configs"
        config_log_dir.mkdir(exist_ok=True)
        # init wandb
        if cfg.task.trainer.use_wandb and rank == 0:
            import wandb

            wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(
                project="geofm-bench",
                name=exp_name,
                config=wandb_cfg,
            )
            cfg["wandb_run_id"] = wandb.run.id
        OmegaConf.save(cfg, config_log_dir / "config.yaml")

    else:
        exp_dir = pathlib.Path(cfg.ckpt_dir)
        exp_name = exp_dir.name
        logger_path = exp_dir / "test.log"
        # load training config
        #cfg_path = exp_dir / "configs" / "config.yaml" ####always evaluate with new config 
        #cfg = OmegaConf.load(cfg_path)
        if cfg.task.trainer.use_wandb and rank == 0:
            import wandb

            wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(
                project="geofm-bench",
                name=exp_name,
                config=wandb_cfg,
                id=cfg.get("wandb_run_id"),
                resume="allow",
            )

    logger = init_logger(logger_path, rank=rank)
    logger.info("============ Initialized logger ============")
    logger.info(pprint.pformat(OmegaConf.to_container(cfg), compact=True).strip("{}"))
    logger.info("The experiment is stored in %s\n" % exp_dir)
    logger.info(f"Device used: {device}")

    encoder: Encoder = instantiate(cfg.encoder)
    encoder.load_encoder_weights(logger)
    logger.info("Built {}.".format(encoder.model_name))

    # prepare the decoder (segmentation/regression)
    decoder: Decoder = instantiate(
        cfg.decoder,
        encoder=encoder,
    )
    decoder.to(device)
    decoder = torch.nn.parallel.DistributedDataParallel(
        decoder,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=cfg.finetune,
    )
    logger.info(
        "Built {} for with {} encoder.".format(
            decoder.module.model_name, type(encoder).__name__
        )
    )

    if rank == 0:
        # Calculate complexity on the underlying model, not the DDP wrapper
        model_to_analyze = decoder.module

        # --- Calculate Trainable Parameters ---
        params_in_M = sum(p.numel() for p in model_to_analyze.parameters() if p.requires_grad) / 1e6
        logger.info(f"Model Trainable Parameters: {params_in_M:.3f}M")

        # --- Calculate GMACs ---
        # Create a dummy input tensor that matches the model's forward pass signature.
        # The decoder expects a 5D tensor (B, C, T, H, W) in a dictionary.
        try:
            # Assuming a single modality for simplicity in this example.
            # You might need to adjust this if your model takes multiple modalities.
            modality_key = list(encoder.input_bands.keys())[0]
            num_input_channels = len(encoder.input_bands[modality_key])
            
            dummy_tensor = torch.randn(
                1,  # Batch size
                num_input_channels,
                1,  # Time dimension
                cfg.dataset.img_size,
                cfg.dataset.img_size,
            ).to(device)
            
            dummy_input = {modality_key: dummy_tensor}
            
            flop_analyzer = FlopCountAnalysis(model_to_analyze, dummy_input)
            gmacs = flop_analyzer.total() / 1e9
            logger.info(f"Model GMACs: {gmacs:.3f} (for input size {dummy_tensor.shape})")

        except Exception as e:
            gmacs = -1 # Assign a sentinel value if calculation fails
            logger.warning(f"Could not calculate GMACs due to an error: {e}")

        # --- Log to W&B ---
        if cfg.task.trainer.use_wandb:
            import wandb
            wandb.summary["parameters_M"] = params_in_M
            wandb.summary["gmacs"] = gmacs

    modalities = list(encoder.input_bands.keys())
    collate_fn = get_collate_fn(modalities)

    # training

    torch.cuda.empty_cache() ### clear cache before training

    if train_run or cfg.task.trainer.model == "knn_probe":
        # get preprocessor
        train_preprocessor = instantiate(
            cfg.preprocessing.train,
            dataset_cfg=cfg.dataset,
            encoder_cfg=cfg.encoder,
            _recursive_=False,
        )
        val_preprocessor = instantiate(
            cfg.preprocessing.val,
            dataset_cfg=cfg.dataset,
            encoder_cfg=cfg.encoder,
            _recursive_=False,
        )

        # get datasets
        raw_train_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="train")
        raw_val_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="val")

        if 0 < cfg.limited_label_train < 1:
            indices = get_subset_indices(
                raw_train_dataset,
                task=task_name,
                strategy=cfg.limited_label_strategy,
                label_fraction=cfg.limited_label_train,
                num_bins=cfg.stratification_bins,
                logger=logger,
            )
            raw_train_dataset = GeoFMSubset(raw_train_dataset, indices)

        if 0 < cfg.limited_label_val < 1:
            indices = get_subset_indices(
                raw_val_dataset,
                task=task_name,
                strategy=cfg.limited_label_strategy,
                label_fraction=cfg.limited_label_val,
                num_bins=cfg.stratification_bins,
                logger=logger,
            )
            raw_val_dataset = GeoFMSubset(raw_val_dataset, indices)

        train_dataset = GeoFMDataset(
            raw_train_dataset, train_preprocessor, cfg.data_replicate
        )
        val_dataset = GeoFMDataset(
            raw_val_dataset, val_preprocessor, cfg.data_replicate
        )

        logger.info("Built {} dataset.".format(cfg.dataset.dataset_name))

        logger.info(
            f"Total number of train patches: {len(train_dataset)}\n"
            f"Total number of validation patches: {len(val_dataset)}\n"
        )

        # get train val data loaders
        train_loader = DataLoader(
            train_dataset,
            sampler=DistributedSampler(train_dataset),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True,
            # persistent_workers=True causes memory leak
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=get_generator(cfg.seed),
            drop_last=True,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_dataset,
            sampler=DistributedSampler(val_dataset),
            batch_size=cfg.test_batch_size,
            num_workers=cfg.test_num_workers,
            pin_memory=True,
            persistent_workers=False,
            worker_init_fn=seed_worker,
            # generator=g,
            drop_last=False,
            collate_fn=collate_fn,
        )

        criterion = instantiate(cfg.criterion)
        optimizer = instantiate(cfg.optimizer, params=decoder.parameters())
        lr_scheduler = instantiate(
            cfg.lr_scheduler,
            optimizer=optimizer,
            total_iters=len(train_loader) * cfg.task.trainer.n_epochs,
        )

        val_evaluator: Evaluator = instantiate(
            cfg.task.evaluator, val_loader=val_loader, exp_dir=exp_dir, device=device
        )
        trainer: Trainer = instantiate(
            cfg.task.trainer,
            model=decoder,
            train_loader=train_loader,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            criterion=criterion,
            evaluator=val_evaluator,
            exp_dir=exp_dir,
            device=device,
        )
        # resume training if model_checkpoint is provided
        if cfg.ckpt_dir is not None:
            trainer.load_model(cfg.ckpt_dir)

        trainer.train()

    
    # Evaluation
    test_preprocessor = instantiate(
        cfg.preprocessing.test,
        dataset_cfg=cfg.dataset,
        encoder_cfg=cfg.encoder,
        _recursive_=False,
    )

    # get datasets
    raw_test_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="test")
    test_dataset = GeoFMDataset(raw_test_dataset, test_preprocessor)

    test_loader = DataLoader(
        test_dataset,
        sampler=DistributedSampler(test_dataset),
        batch_size=cfg.test_batch_size,
        num_workers=cfg.test_num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    test_evaluator: Evaluator = instantiate(
        cfg.task.evaluator, val_loader=test_loader, exp_dir=exp_dir, device=device
    )

    if cfg.use_final_ckpt:
        model_ckpt_path = get_final_model_ckpt_path(exp_dir)
    else:
        model_ckpt_path = get_best_model_ckpt_path(exp_dir)
        
    if model_ckpt_path is None and not cfg.task.trainer.model_name == "knn_probe":
        raise ValueError(f"No model checkpoint found in {exp_dir}")
    
    test_evaluator.evaluate(decoder, "test_model", model_ckpt_path)

    if cfg.use_wandb and rank == 0:
        wandb.finish()


if __name__ == "__main__":
    main()