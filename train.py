import argparse
import torch
import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from data.data import ADNIDataset, get_dataloaders  # Ensure correct path
from models.DDPM_2D_patched import DDPM_2D  # Ensure correct path


def main(cfg):
    # Set random seed for reproducibility
    seed_everything(cfg.get("seed", 42))
    cfg=cfg['cfg']
    # Prepare data
    train_loader, val_loader, test_loader = get_dataloaders(
        train_base_dir=cfg['train_base_dir'],
        csv_path=cfg['csv_path'],
        modality=cfg['modality'],
        batch_size=cfg['batch_size'],
        transform=None, 
        validation_split=0.1,
        test_split=0.1,
        seed=cfg['seed'],
        config=cfg
    )

    # Initialize model
    model = DDPM_2D(cfg)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/Loss_comb",
        dirpath=cfg['checkpoint_dir'],
        filename="best-checkpoint",
        save_top_k=1,
        mode="min"
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="val/Loss_comb",
        patience=cfg.get('early_stopping_patience', 10),
        verbose=True,
        mode="min"
    )
    
    # Logger
    logger = TensorBoardLogger("logs", name="DDPM_2D")

    # Trainer
    trainer = Trainer(
        max_epochs=cfg['max_epochs'],
        gpus=cfg['gpus'],
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Test
    trainer.test(model, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file.")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
