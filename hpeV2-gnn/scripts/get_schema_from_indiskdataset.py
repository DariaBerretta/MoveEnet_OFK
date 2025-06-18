
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data.database import TensorInfo
from torch import Tensor

from data import customDatasets
from model import hpegnn
from utils.dataset_utils import hpe_filter, dataset_split
from utils.library_utils import MyProgressBar
from utils.model_utils import GraphVisualization
from typing import Dict, Any

def change_data(dataset):
    for data in dataset:
        data.y = torch.squeeze(data.y[0])




# data_path = '/home/ggoyal/data/DHP19/Gamer'
data_path = '/home/ggoyal/data/h36m_cropped/gamer/'
# Training Parameters

data_fraction = 0.02 #fraction of data to use in total
learning_rate = 0.0001
epochs = 10
hidden_channels = [2,8]
batch_size = 1
num_joints = 1
visualise = True

dataset = customDatasets.eh36m_gcn(data_path, pre_filter=hpe_filter)   # where the dataset is set up with InMemoryDataset class
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
count = 0

ref_data = dataset[0]

schema: Dict[str, Any] = {}
for key, value in ref_data.to_dict().items():
    if isinstance(value, (int, float, str)):
        schema[key] = value.__class__
    elif isinstance(value, Tensor) and value.dim() == 0:
        schema[key] = dict(dtype=value.dtype, size=(-1,))
    elif isinstance(value, Tensor):
        size = list(value.size())
        size[ref_data.__cat_dim__(key, value)] = -1
        schema[key] = dict(dtype=value.dtype, size=tuple(size))
    else:
        schema[key] = object
print(schema)