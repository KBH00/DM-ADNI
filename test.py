from data.data import get_dataloaders
import argparse
import yaml
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models.DDPM_2D_patched import DDPM_2D

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)
    print(cfg)
    return cfg

# get_dataloaders(train_base_dir="/home/kbh/Downloads/nii", csv_path="./data/anomal.csv", modality="FLAIR",
#                 batch_size=4 ,validation_split=0.1, test_split=0.1, seed=42, config=config, inf=False)

def main(cfg):
    cfg = cfg['cfg']
    train_loader, val_loader, test_loader = get_dataloaders(
        train_base_dir=cfg.get('train_base_dir', "/home/kbh/Downloads/nii"),
        csv_path=cfg.get("csv_path"),
        modality="FLAIR",
        batch_size=cfg.get("batch_size", 64),
        validation_split=0.1,
        test_split=0.1,
        config=cfg,
        inf=False,
    )
    print(train_loader)


main(load_config('config.yaml'))