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
from graph_enet.hpe_gnn.utils.dataset_utils import dataset_split
from graph_enet.hpe_gnn.utils.library_utils import MyProgressBar
from graph_enet.hpe_gnn.scripts.config import cfg_SCARF
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import os.path as osp

import torch.nn as nn
import torch.optim as optim

import warnings
# silence the specific FutureWarning about torch.load weights_only default change
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)

# TO run in --dev mode in terminal:
# python graph_enet/test_scripts/test_scarf_train_splineConv.py --dev --label dev --data_fraction 0.01 --dataset_split dev --epochs 5 --batch_size 16 --learning_rate 0.001 --node_loss_weight 1.0 --target_loss_weight 1.0 --arch single_weight

# Argument parser creation and arguments definition
parser = argparse.ArgumentParser(description="Test SCARF SplineConv Training")

parser.add_argument('--data_fraction', 
                    type=float, 
                    default=cfg_SCARF['data_fraction'], 
                    help='Fraction of data to use')
parser.add_argument('--dataset_split', 
                    type=str, 
                    default=cfg_SCARF['dataset_split'], 
                    help="[subject, dev]. Dev will use 'fraction' with value 0.01.", 
                    choices=['subject','dev'])
parser.add_argument('--learning_rate', 
                    type=float, 
                    default=cfg_SCARF['learning_rate'], 
                    help='Constant Learning rate [0.0001,0.001,0.01]')
parser.add_argument('--epochs', 
                    type=int, 
                    default=cfg_SCARF['epochs'], 
                     help='Total epochs [1,10,20,50, 100]')
parser.add_argument('--batch_size', 
                    type=int, 
                    default=cfg_SCARF['batch_size'], 
                    help='Batch size [1,8,32,256,1024]')
parser.add_argument('--num_joints', 
                    type=int, 
                    default=13, 
                    help='Number of joints [1,13]')
parser.add_argument('--label', 
                    type=str, 
                    help='String to label the training log',
                    dest='label')
parser.add_argument('--node_feature', 
                    type=str, 
                    default=cfg_SCARF['node_feature'],
                    help='In scarfDataset_splineConv, node features are always 10')
parser.add_argument('--hidden', 
                    type=str, 
                    default=cfg_SCARF['hidden'], 
                    help='Number of channels in each layer, for at least 1 and at most 10 hidden layers')
parser.add_argument('--connectivity', 
                    type=str, 
                    default=cfg_SCARF['connectivity'], 
                    help='Connectivity set to 0 because graph is fully pre-connected')
parser.add_argument('--node_loss_weight', 
                    type=float, 
                    default=cfg_SCARF['node_loss_weight'], 
                    help='Node loss weight')
parser.add_argument('--target_loss_weight', 
                    type=float, 
                    default=cfg_SCARF['target_loss_weight'], 
                    help='Target loss weight')
parser.add_argument('--task', 
                    type=str, 
                    default= cfg_SCARF['task'],
                    help='HPE is the only task supported')
parser.add_argument('--dev', 
                    help='Use dev mode. dataset_split, epochs, batch_size and label will be overwritten.',
                    action='store_true', 
                    default=cfg_SCARF['dev'],
                    required=False)
parser.add_argument('--resume', 
                    type=str, 
                    help='Resume training from checkpoint PATH provided.'
                    'All relevant command line inputs are overwritten',  
                    default=cfg_SCARF['resume'], 
                    required=False)
parser.add_argument('--ckpt', 
                    help='Start training from checkpoint PATH provided.', 
                    default=cfg_SCARF['ckpt'], 
                    required=False,
                    type=str)
parser.add_argument('--data_path', 
                    help='Path to the dataset with the "raw" folder.', 
                    default=cfg_SCARF['data_path'],
                    required=False, 
                    type=str)
parser.add_argument('--data_path_dev', 
                    help='Path to the dataset with the "raw" folder for dev mode.',
                    default=cfg_SCARF['data_path_dev'], 
                    required=False, 
                    type=str)
parser.add_argument('--arch', 
                    type=str, 
                    default=cfg_SCARF['arch'],
                    choices=['single_weight', 'two_weights'], 
                    help='Model architecture type')
parser.add_argument('--dataset', 
                    type=str, 
                    default=cfg_SCARF['dataset'], 
                    help='Dataset class name',
                    required=False)

# Parse command line arguments
args = parser.parse_args()

# Update runtime configuration with parsed arguments
cfg = cfg_SCARF
cfg['data_fraction'] = args.data_fraction
cfg['dataset_split'] = args.dataset_split
cfg['learning_rate'] = args.learning_rate
cfg['epochs'] = args.epochs
cfg['batch_size'] = args.batch_size
cfg['label'] = args.label
cfg['node_feature'] = args.node_feature
cfg['hidden'] = args.hidden
cfg['connectivity'] = args.connectivity
cfg['node_loss_weight'] = args.node_loss_weight
cfg['target_loss_weight'] = args.target_loss_weight
cfg['task'] = args.task
cfg['dev'] = args.dev
cfg['resume'] = args.resume
cfg['data_path'] = args.data_path
cfg['data_path_dev'] = args.data_path_dev
cfg['arch'] = args.arch
cfg['dataset'] = args.dataset

# Determine dataset path and validate it
if cfg['dev']:
    data_path = cfg['data_path_dev']
else:
    data_path = cfg['data_path']

if not osp.exists(data_path):
    raise FileNotFoundError(f"Data path {data_path} does not exist. Please check the path.")
    exit()

# Model architecture configuration
if args.arch == None:
    args.arch = 'single_weight'

# # Check and validate ckpt and resume checkpoint 
# # paths (if provided)

# if args.ckpt is not None and not osp.exists(args.ckpt):
#     raise FileNotFoundError(f"Checkpoint path {args.ckpt} does not exist. Please check the path.")
# cfg['ckpt'] = args.ckpt

# if args.resume is not None and not osp.exists(args.resume):
#     raise FileNotFoundError(f"Resume path {args.resume} does not exist. Please check the path.")
# cfg['resume'] = args.resume


# Development mode setup
if cfg['dev']:
    cfg['dataset_split'] = 'dev'
    cfg['label'] = 'dev'
    cfg['batch_size'] = 16
    cfg['epochs'] = 5

# Experimental setup parsing
cfg['hidden'] = [int(x) for x in cfg['hidden'].split(',')]
num_joints = args.num_joints
exp_setup = {
    'connectivity': cfg['connectivity'],
    'node_feature': cfg['node_feature'],
    'arch': cfg['arch'],
    'dataset': cfg['dataset'],
    'task': cfg['task'],
    'num_joints': num_joints,
    'hidden': cfg['hidden'],
    'data_fraction': cfg['data_fraction'],
    'batch_size': cfg['batch_size'],
    'learning_rate': cfg['learning_rate'],
    'dataset_split': cfg['dataset_split']
    }
print(f"Experimental setup: {exp_setup}")

# Build tranfrorm pipeline
# No data trasformation is applied
# because the dataset is already pre-processed
# in the sense that the node features are already
# in the form of Cartesian coordinates.

# Dataset and dataloader creation
dataset = scarfDataset_splineConv(data_path, 
                                  transform=None,
                                  pre_transform=None, 
                                  pre_filter=None,
                                  rf_size=14, 
                                  alpha=1.0, 
                                  C=0.3,
                                  res=(640, 480))

dataset = dataset.shuffle()

train_dataset, val_dataset = dataset_split(dataset,
    style=cfg['dataset_split'], 
    fraction=cfg['data_fraction'], 
    dataset_label=cfg['dataset'])

train_loader = DataLoader(train_dataset, 
                          batch_size=cfg['batch_size'], 
                          shuffle=True, 
                          num_workers=4)

val_loader = DataLoader(val_dataset, 
                        batch_size=cfg['batch_size'],  
                        num_workers=4)

print(f'[INFO] DataLoaders created')

# Model initialization from scratch
# TODO: Add support for loading from checkpoint

if cfg['arch'] == 'single_weight':
    model = hpeGnn_splineConv_single_weight(dataset.num_features, 
                                            cfg['hidden'], 
                                            num_joints, 
                                            learning_rate=cfg['learning_rate'],
                                            batch_size=cfg['batch_size'], 
                                            data_fraction=cfg['data_fraction'], 
                                            label=cfg['label'],
                                            task=cfg['task'], 
                                            transforms= None, 
                                            node_loss_weight=[cfg['target_loss_weight'], cfg['node_loss_weight']],
                                            exp_setup=exp_setup,
                                            pck_multiplier=0.6)
elif cfg['arch'] == 'two_weights':
    model = hpeGnn_splineConv(dataset.num_features, 
                                         cfg['hidden'], 
                                         num_joints, 
                                         learning_rate=cfg['learning_rate'],
                                         batch_size=cfg['batch_size'], 
                                         data_fraction=cfg['data_fraction'], 
                                         label=cfg['label'],
                                         task=cfg['task'], 
                                         transforms=None, 
                                         node_loss_weight=[cfg['target_loss_weight'], cfg['node_loss_weight']],
                                         exp_setup=exp_setup, 
                                         pck_multiplier=0.6)

# print(f'[INFO] Model initialized: {model}')
print(f'[INFO] Model initialized')

# Set up training configuration

# dev mode
if cfg['dev']:
    trainer = pl.Trainer(fast_dev_run=5, enable_progress_bar=True)
else:
    logger = pl.loggers.TensorBoardLogger("lightning_logs", name=cfg['label'])
    early_stop_callback = EarlyStopping(monitor="loss/val_epoch", min_delta=0.00, patience=5, verbose=False, mode="min",
                                        check_finite=True)
    trainer = pl.Trainer(max_epochs=cfg['epochs'], check_val_every_n_epoch=3, callbacks=[MyProgressBar(),
                         early_stop_callback], logger=logger, min_epochs=int(cfg['epochs']/2))
    # # #Create directories if they do not exist and store cfg hyperparams
    # # os.makedirs(logger.log_dir, exist_ok=True)
    # # with open(os.path.join(logger.log_dir,'cfg.json'), 'w') as fp:
    # #     json.dump(cfg, fp)

print(f'[INFO] Trainer set up complete')
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg['resume'])
print(f'[INFO] Training complete')


    