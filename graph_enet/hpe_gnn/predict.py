
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data.database import TensorInfo

from data import customDatasets
# from data import  transforms
from data.transforms import chain_transforms, center_detection, head_detection
from data.transforms import replace_node_feature_with_ones, check_x_size, use_pos_as_input, merge_pos_to_input
from model import hpegnn
import data.transforms as my_transforms
from utils.dataset_utils import hpe_filter, dataset_split, schema_spline
from utils.training_utils import test_ckpt_path
from utils.model_utils import GraphVisualization
from argparse import ArgumentParser

# Predict arguments
parser = ArgumentParser()
parser.add_argument('--data_path', 
                    help='PATH to base folder to save prediction', 
                    default=None, 
                    required=False,
                    type=str)
parser.add_argument('--model_csv_name', 
                    help='name of your model to save in csv', 
                    default='GraphEnet.csv',
                    required=False, 
                    type=str)
parser.add_argument('--ckpt_path', 
                    help='Start prediction from checkpoint PATH .', 
                    default= None, 
                    required=False,
                    type=str)
parser.add_argument('--visualise', 
                    help='set type of visualisation ["pose", None]', 
                    default= 'pose', 
                    required=False,
                    type=str, 
                    choices=['pose', None, 'vectors-head', 'vectors-handR'])
parser.add_argument('--video_path', 
                    help='Set to a path to save a video', 
                    default=None,
                    required=False, 
                    type=str)
parser.add_argument('--dataset', 
                    help='Input Dataset ["dhp19", "h36m"(default)]', 
                    default= 'h36m', 
                    required=False,
                    type=str, 
                    choices=['dhp19', 'h36m'])
parser.add_argument('--arch', 
                    help='Network to use. [single_weight(default), two_weights, gat]', 
                    default='single_weight',
                    required=False, 
                    type=str, 
                    choices=["single_weight(default)", "two_weights", "gat"])

args = parser.parse_args()

data_path = args.data_path
model_csv_name = args.model_csv_name

if data_path == None:
    print('Data path does not exist. Exiting.')
    exit()

# Training Parameters
data_fraction = 1  # fraction of data to use in total
num_joints = 13
visualise = args.visualise
video_path = args.video_path
# video_path = '/home/usr/data/h36m_gamer/videos/head_connectivity_15.mp4'

ckpt_path = args.ckpt_path
if ckpt_path != None:
    ckpt_path = test_ckpt_path(ckpt_path)

if ckpt_path == None:
    print('checkpoint path does not exist. Exiting.')
    exit()

lightning_checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage, weights_only=False)
hyperparams = lightning_checkpoint["hyper_parameters"]

for key, value in hyperparams.items():
    print(key, value)

if 'task' not in hyperparams.keys():
    hyperparams['task'] = 'head'  # head or center or all
hyperparams['transforms'] = None
if 'transforms' not in hyperparams.keys() or hyperparams['transforms'] == None:
    transforms = [my_transforms.check_x_size, my_transforms.use_pos_as_input, my_transforms.add_edges_15]

    if hyperparams['out_channels'] == 1:
        if hyperparams['task'] == 'center':
            transforms.append(center_detection)
        elif hyperparams['task'] == 'head':
            transforms.append(head_detection)
    hyperparams['transforms'] = chain_transforms(transforms)

if args.dataset == 'h36m':
    dataset = customDatasets.eh36m_spline_ledge(data_path, 
                                                transform=hyperparams['transforms'], 
                                                pre_filter=hpe_filter,
                                                schema=schema_spline)
elif args.dataset == 'dhp19':
    dataset = customDatasets.dhp19_spline_ledge(data_path, 
                                                transform=hyperparams['transforms'], 
                                                pre_filter=hpe_filter,
                                                schema=schema_spline)

test_loader = DataLoader(dataset, batch_size=1, num_workers=2)

print('Dataloaders created')

if args.arch == 'two_weights':
    model = hpegnn.hpeGnn_splineConv.load_from_checkpoint(ckpt_path, in_channels=hyperparams['in_channels'],
                              hidden_channels=hyperparams["hidden_channels"], 
                              out_channels=hyperparams['out_channels'],
                              learning_rate=hyperparams['learning_rate'], 
                              batch_size=1, 
                              visualise=visualise,
                              write_csv=data_path, 
                              file_name_eval = model_csv_name, 
                              image_size=hyperparams['image_size'],
                              node_loss_weight=hyperparams['node_loss_weight'], 
                              save_video=video_path)
elif args.arch == 'single_weight':
    model = hpegnn.hpeGnn_splineConv_single_weight.load_from_checkpoint(ckpt_path, in_channels=hyperparams['in_channels'],
                              hidden_channels=hyperparams["hidden_channels"], 
                              out_channels=hyperparams['out_channels'],
                              learning_rate=hyperparams['learning_rate'], 
                              batch_size=1, 
                              visualise=visualise,
                              write_csv=data_path, 
                              file_name_eval = model_csv_name, 
                              image_size=hyperparams['image_size'],
                              node_loss_weight=hyperparams['node_loss_weight'], 
                              save_video=video_path)
print('Model set up complete')
print('Setting up trainer')

trainer = pl.Trainer(max_epochs=1, enable_progress_bar=True)
trainer.predict(model, dataloaders=test_loader)
# print(out)

