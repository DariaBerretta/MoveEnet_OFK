#!/usr/bin/env python3

# Fix CUDA multiprocessing issue - MUST be at the very top
import multiprocessing as mp
mp.set_start_method('spawn', force=True)


from logging import config
import os
import json
import argparse
import numpy as np
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Cartesian

# Import fixed metrics first
from graph_enet.test_scripts.fixed_metrics import pck_error, mpjpe_error

# Monkey patch the metrics module before importing the model
import graph_enet.hpe_gnn.utils.metrics
graph_enet.hpe_gnn.utils.metrics.pck_error = pck_error
graph_enet.hpe_gnn.utils.metrics.mpjpe_error = mpjpe_error

from graph_enet.data.scarfDataset_splineConv import scarfDataset_splineConv
from graph_enet.hpe_gnn.model.hpegnn import hpeGnn_splineConv, hpeGnn_splineConv_single_weight
from graph_enet.hpe_gnn.utils.dataset_utils import new_dataset_split, dataset_split
from graph_enet.hpe_gnn.utils.library_utils import MyProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from graph_enet.hpe_gnn.scripts.config import cfg_SCARF as cfg

import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)

def setup_training(cfg):
    """Setup dataset, model, and trainer."""
    
    print("="*60)
    print("GRAPHENET-V2 TRAINING")
    print("="*60)
    print(f"Configuration:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Check data path exists
    if not os.path.exists(cfg['data_path']):
        raise FileNotFoundError(f"Data path not found: {cfg['data_path']}")
    
    # Load dataset
    dataset = scarfDataset_splineConv(
        cfg['data_path'],
        transform=None,
        pre_transform=None, 
        pre_filter=None,
        rf_size=14, 
        alpha=1.0, 
        C=0.3,
        res=(640, 480)
    )
    
    dataset = dataset.shuffle()
    print(f"Total dataset size: {len(dataset)}")
    
    # Split dataset
    train_dataset, val_dataset = new_dataset_split(
        dataset,
        style=cfg['dataset_split'], 
        fraction=cfg['data_fraction'], 
        dataset_label=cfg['dataset']
    )

    # train_dataset, val_dataset = dataset_split(
    #     dataset,
    #     style=cfg['dataset_split'], 
    #     fraction=cfg['data_fraction'], 
    #     dataset_label='scarfDataset_splineConv'
    # )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['batch_size'], 
        shuffle=True, 
        num_workers=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg['batch_size'],  
        num_workers=2,
        persistent_workers=True
    )
    
    print("DataLoaders created")
    
    # Create model
    num_joints = 13
    
    if cfg['arch'] == 'single_weight':
        model = hpeGnn_splineConv_single_weight(
            dataset.num_features, 
            cfg['hidden'], 
            num_joints, 
            learning_rate=cfg['learning_rate'],
            batch_size=cfg['batch_size'], 
            data_fraction=cfg['data_fraction'], 
            label=cfg['label'],
            task=cfg['task'], 
            transforms=None, 
            node_loss_weight=[cfg['target_loss_weight'], cfg['node_loss_weight']],
            pck_multiplier=0.6
        )
    else:  # two_weights
        model = hpeGnn_splineConv(
            dataset.num_features, 
            cfg['hidden'], 
            num_joints, 
            learning_rate=cfg['learning_rate'],
            batch_size=cfg['batch_size'], 
            data_fraction=cfg['data_fraction'], 
            label=cfg['label'],
            task='all', 
            transforms=None, 
            node_loss_weight=[cfg['target_loss_weight'], cfg['node_loss_weight']],
            pck_multiplier=0.6
        )
    
    print(f"Model created: {cfg['arch']} architecture")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Setup trainer
    if cfg['dev']:
        trainer = pl.Trainer(fast_dev_run=5, enable_progress_bar=True)
    else:
        # Create logger
        logger = pl.loggers.TensorBoardLogger("lightning_logs", name=cfg['label'])
        
        # Callbacks
        early_stop_callback = EarlyStopping(
            monitor="loss/val_epoch", 
            min_delta=0.001,  # Smaller delta for more sensitivity
            patience=cfg['patience'], 
            verbose=True, 
            mode="min",
            check_finite=True
        )
        
        # Trainer with improved settings
        trainer = pl.Trainer(
            max_epochs=cfg['epochs'],
            check_val_every_n_epoch=5,
            callbacks=[MyProgressBar(), early_stop_callback], 
            logger=logger, 
            min_epochs=int(cfg['epochs']/2),  # Minimum 50% of epochs
            # gradient_clip_val=1.0,  # Gradient clipping to prevent exploding gradients
            # accumulate_grad_batches=1,  # No gradient accumulation
            # log_every_n_steps=10,  # More frequent logging
            enable_checkpointing=True,
            #val_check_interval=0.5  # Validate twice per epoch
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1 if torch.cuda.is_available() else None
        )
        
        # Create log directory and save config
        os.makedirs(logger.log_dir, exist_ok=True)
        with open(os.path.join(logger.log_dir, 'cfg.json'), 'w') as fp:
            json.dump(cfg, fp, indent=2)
        
        print(f"Logging to: {logger.log_dir}")
    
    return model, trainer, train_loader, val_loader

def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        default=cfg['data_path'],
                        help='Path to dataset')
    parser.add_argument('--arch', type=str, 
                        default=cfg['arch'],
                        choices=['single_weight', 'two_weights'],
                        help='Model architecture')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=cfg['epochs'],
                        help='Number of epochs')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=cfg['batch_size'],
                        help='Batch size')
    parser.add_argument('--learning_rate', 
                        type=float, default=cfg['learning_rate'],
                        help='Learning rate')
    parser.add_argument('--data_fraction', 
                        type=float, 
                        default=cfg['data_fraction'],
                        help='Fraction of data to use')
    parser.add_argument('--node_loss_weight', 
                        type=float, 
                        default=cfg['node_loss_weight'],
                        help='Node loss weight')
    parser.add_argument('--target_loss_weight',
                        type=float, 
                        default=cfg['target_loss_weight'],
                        help='Target loss weight')
    parser.add_argument('--label', 
                        type=str, 
                        default=cfg['label'],
                        help='Experiment label')
    parser.add_argument('--hidden', 
                        type=str, 
                        default=cfg['hidden'],
                        help='Comma-separated hidden layer sizes')
    parser.add_argument('--patience', 
                        type=int, 
                        default=10,
                        help='Early stopping patience')
    parser.add_argument('--dev', 
                        action='store_true', 
                        default=False,
                        help='Development mode (fast run)')
    parser.add_argument('--resume', 
                        type=str, 
                        default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()


    # Create config from args
    config= {
        'data_path': args.data_path,
        'arch': args.arch,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'data_fraction': args.data_fraction,
        'dataset_split': 'dev',
        'hidden': [int(x) for x in args.hidden.split(',')],
        'node_loss_weight': args.node_loss_weight,
        'target_loss_weight': args.target_loss_weight,
        'label': args.label,
        'patience': args.patience,
        'dev': args.dev,
        'resume': args.resume
    }
    cfg.update(config)
    
    try:
        # Setup training
        model, trainer, train_loader, val_loader = setup_training(cfg)
        
        # Start training
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        trainer.fit(
            model=model, 
            train_dataloaders=train_loader, 
            val_dataloaders=val_loader, 
            ckpt_path=cfg['resume']
        )
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        
        # Get best checkpoint path
        if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
            best_ckpt = trainer.checkpoint_callback.best_model_path
            print(f"Best checkpoint: {best_ckpt}")
        
        if not cfg['dev']:
            print(f"Logs saved to: {trainer.logger.log_dir}")
            print("Use TensorBoard to monitor training:")
            print(f"  tensorboard --logdir {trainer.logger.log_dir}")
        
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    if success:
        print("\n🎉 Training completed successfully!")
        print("\n📋 Next steps:")
        print("1. Check TensorBoard for training progress")
        print("2. Test the new model with extract_predictions_visualize.py")
        print("3. Compare results with the previous poor model")
    else:
        print("\n❌ Training failed. Check the error messages above.")
