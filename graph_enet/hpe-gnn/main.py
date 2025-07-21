import os.path
import json
import numpy as np
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import Tensor

from data import customDatasets
import data.transforms as my_transforms
from model import hpegnn
from utils.dataset_utils import hpe_filter, dataset_split, schema_spline
from utils.library_utils import MyProgressBar
import utils.training_utils as maps
from utils.training_utils import test_ckpt_path
from utils.model_utils import GraphVisualization
from argparse import ArgumentParser
from scripts.config import cfg

# Trainer arguments
parser = ArgumentParser()
parser.add_argument('--data_fraction', type=float, default=cfg['data_fraction'], help='Fraction of dataset to be used [0.01,0.1,1]')
parser.add_argument('--dataset_split', type=str, default=cfg['dataset_split'],
                    help="[subject, dev]. Dev will use 'fraction' with value 0.01.", choices=['subject','dev'])
parser.add_argument('--learning_rate', type=float, default=cfg['learning_rate'], help='Constant Learning rate [0.0001,0.001,01]')
parser.add_argument('--epochs', type=int, default=cfg['epochs'], help='Total epochs [1,10,20,50, 100]')
parser.add_argument('--batch_size', type=int, default=cfg['batch_size'], help='Batch size [1,8,32,256,1024]')
# parser.add_argument('--num_joints', type=int, default=13, help='Number of joints [1,13]')
parser.add_argument('--label', type=str, default=cfg['label'], help='String to label the training log',dest='label')
parser.add_argument('--node_feature', type=str, default=cfg['node_feature'],
                    help='Experimental setup [pos, pos_fit, pos_fit_lsg]', choices=['pos','pos_fit'])
parser.add_argument('--hidden', type=str, default=cfg['hidden'],
                    help='Number of channels in each layer, for at least 1 and at most 10 hidden layers')
parser.add_argument('--connectivity', type=int, default=cfg['connectivity'], choices=[10, 12, 15, 20, 30],
                    help='Added connection for pixel values [10, 12, 15, 20, 30]. Default = 15')
parser.add_argument('--node_loss_weight', type=float, default=cfg['node_loss_weight'],
                    help='Node_loss_weight')
parser.add_argument('--target_loss_weight', type=float, default=cfg['target_loss_weight'],
                    help='Target_loss_weight')
parser.add_argument('--task', type=str, default=cfg['task'], choices=["center", "head", "all", "right_hand", "hands"],
                    help='Task setup; [center, head, all, right_hand, hands]')
parser.add_argument('--dev', help='Use dev mode. dataset_split, epochs, batch_size and label will be overwritten.',
                    action='store_true', default=cfg['dev'], required=False)
parser.add_argument('--resume', type=str, help='Resume training from checkpoint PATH provided. '
                    'All relevant command line inputs are overwritten',  default=cfg['resume'], required=False)
parser.add_argument('--ckpt', help='Start training from checkpoint PATH provided.', default=cfg['ckpt'], required=False,
                    type=str)
parser.add_argument('--data_path', help='Path to the dataset with the "raw" folder.', default=cfg['data_path'],
                    required=False, type=str)
parser.add_argument('--data_path_dev', help='Path to the dataset with the "raw" folder for dev mode.',
                    default=cfg['data_path_dev'], required=False, type=str)
parser.add_argument('--arch', help='Network to use. [single_weight(default), two_weights, gat]', default=cfg['arch'],
                    required=False, type=str, choices=["single_weight(default)", "two_weights", "gat"])
parser.add_argument('--dataset', help='Dataset to use. [h36m(default), dhp19]', default=cfg['dataset'],
                    choices=['h36m', 'dhp19'], required=False, type=str)

args = parser.parse_args()

# Update the config file dynamically
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

# Dataset path on disk
if cfg['dev']:
    # data_path = '/home/ggoyal/data/h36m_cropped/ledge_toy/'
    # data_path = cfg['data_path_dev']

    # My data path
    data_path = '/home/dberretta-iit.local/data/toy_gamer/'
    # data_path = '/home/dberretta-iit.local/data/tast_gamer_GNN/'
else:
    # data_path = '/home/ggoyal/data/h36m_cropped/ledge/'
    # data_path = cfg['data_path']

    # My data path
    data_path = '/home/dberretta-iit.local/data/toy_gamer/'
    # data_path = '/home/dberretta-iit.local/data/tast_gamer_GNN/'

if not os.path.exists(data_path):
    print(data_path)
    print('Data path does not exist. Exiting.')
    exit()

elif args.arch == 'gat':
    print('This model is still under development, please use a different model')
    exit()
elif args.arch == None:
    args.arch = 'single_weight'


if args.ckpt != None:
    args.ckpt = test_ckpt_path(args.ckpt)
    if args.ckpt == None:
        print('Please check ckpt path. Exiting.')
        exit()
cfg['ckpt'] = args.ckpt

if args.resume != None:
    args.resume = test_ckpt_path(args.resume)
    if args.resume == None:
        print('Please check resume path. Exiting.')
        exit()
cfg['resume'] = args.resume


# Development mode setup
if cfg['dev']:
    cfg['dataset_split'] = 'dev'
    cfg['label'] = 'dev'
    cfg['batch_size'] = 16
    cfg['epochs'] = 5


# Experimental setup based on command line arguments and inputs
cfg['hidden'] = list(map(int, cfg['hidden'].split(',')))
task_to_num_joints = {'head': 1, 'center': 1, 'right_hand': 1, 'all': 13, 'hands': 2}
num_joints = task_to_num_joints[cfg['task']]
exp_setup = [cfg['connectivity'], cfg['node_feature'], args.arch]
print(exp_setup)

transforms_current = [my_transforms.check_x_size]
if cfg['node_feature'] != None:
    transforms_current.append(maps.node_feature(cfg['node_feature']))
if cfg['connectivity'] > 0:
    transforms_current.append(maps.connectivity_map(cfg['connectivity']))
if cfg['task'] != 'all':
    transforms_current.append(maps.task_map(cfg['task']))

transforms_namelist = [func.__name__ for func in transforms_current]
# print(transforms_namelist)
transforms_current = my_transforms.chain_transforms(transforms_current)


# Dataset and dataloader setup
# dataset = customDatasets.eh36m_spline_ledge(data_path, transform=transforms_current, pre_filter=hpe_filter, schema=schema_spline)

dataset = customDatasets.eh36m_spline_gamer(data_path,transform=transforms_current, pre_filter=hpe_filter, schema=schema_spline)

dataset = dataset.shuffle()
print(dataset.get(10).x.shape)
show_vals = (dataset.multi_get([i for i in range(100)]))
train_dataset, val_dataset = dataset_split(dataset, style=cfg['dataset_split'], fraction=cfg['data_fraction'], dataset_label = args.dataset)
train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], num_workers=2)
print('Dataloaders created')

if cfg['ckpt'] == None:
    if args.arch == None or args.arch == 'single_weight':
        model = hpegnn.hpeGnn_splineConv_single_weight(dataset.num_features, cfg['hidden'], num_joints, learning_rate=cfg['learning_rate'],
                                    batch_size=cfg['batch_size'], data_fraction=cfg['data_fraction'], label=cfg['label'],
                                    task=cfg['task'], transforms=transforms_namelist, node_loss_weight=[cfg['target_loss_weight'], cfg['node_loss_weight']],
                                    exp_setup=exp_setup, pck_multiplier=0.6)
    elif args.arch == 'two_weights':
        model = hpegnn.hpeGnn_splineConv(dataset.num_features, cfg['hidden'], num_joints, learning_rate=cfg['learning_rate'],
                                    batch_size=cfg['batch_size'], data_fraction=cfg['data_fraction'], label=cfg['label'],
                                    task=cfg['task'], transforms=transforms_namelist, node_loss_weight=[cfg['target_loss_weight'], cfg['node_loss_weight']],
                                    exp_setup=exp_setup, pck_multiplier=0.6)
else:
    if args.arch == None or args.arch == 'single_weight':
        model = hpegnn.hpeGnn_splineConv_single_weight.load_from_checkpoint(cfg['ckpt'], learning_rate=cfg['learning_rate'],
                                    batch_size=cfg['batch_size'], data_fraction=cfg['data_fraction'], label=cfg['label'],
                                    task=cfg['task'], transforms=transforms_namelist, node_loss_weight=[cfg['target_loss_weight'], cfg['node_loss_weight']], 
                                    exp_setup=exp_setup, pck_multiplier=0.6)
    elif args.arch == 'two_weights':
        model = hpegnn.hpeGnn_splineConv.load_from_checkpoint(cfg['ckpt'], learning_rate=cfg['learning_rate'],
                                    batch_size=cfg['batch_size'], data_fraction=cfg['data_fraction'], label=cfg['label'],
                                    task=cfg['task'], transforms=transforms_namelist, node_loss_weight=[cfg['target_loss_weight'], cfg['node_loss_weight']], 
                                    exp_setup=exp_setup, pck_multiplier=0.6)

    print('ckpt')
print('Model set up complete')


# Training code
if cfg['dev']:
    trainer = pl.Trainer(fast_dev_run=5, enable_progress_bar=True)
else:
    logger = pl.loggers.TensorBoardLogger("lightning_logs", name=cfg['label'])
    early_stop_callback = EarlyStopping(monitor="loss/val_epoch", min_delta=0.00, patience=5, verbose=False, mode="min",
                                        check_finite=True)
    trainer = pl.Trainer(max_epochs=cfg['epochs'], check_val_every_n_epoch=3, callbacks=[MyProgressBar(),
                         early_stop_callback], logger=logger, min_epochs=int(cfg['epochs']/2))
    #Create directories if they do not exist and store cfg hyperparams
    os.makedirs(logger.log_dir, exist_ok=True)
    with open(os.path.join(logger.log_dir,'cfg.json'), 'w') as fp:
        json.dump(cfg, fp)
print('Trainer set up complete')
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg['resume'])
print('Training complete')
