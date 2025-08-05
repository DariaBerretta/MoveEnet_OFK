
import torch
import sys
from torch_geometric.data import Data
from torch_geometric.nn import pool
sys.path.append('..')
from data.customDatasets import batchData
from torch_geometric.transforms import Cartesian

def check_and_join_tensors(tensor_first, tensor_additional):
    # Iterate over each column of B
    for col in tensor_additional.t():
        # Check if any column of A is equal to the current column of B
        match = torch.any(torch.all(tensor_first == col[:, None], dim=0))

        # If no match found, concatenate the column of B to A
        if not match:
            tensor_first = torch.cat((tensor_first, col[:, None]), dim=1)

    return tensor_first

def transform_test(data):
    x = data.x[0]
    data_2 = Data(x=x,y=data.y,pos=data.pos,edge_index=data.edge_index)
    print("bla")
    return data_2


def center_detection(data: batchData):
    y = torch.reshape(data.y,[-1,2])
    data.y = torch.mean(y,dim=0)
    return data

def head_detection(data: batchData):
    y = torch.reshape(data.y, [-1, 2])
    data.y = y[0,:]
    return data

def right_hand_detection(data: batchData):
    y = torch.reshape(data.y, [-1, 2])
    data.y = y[7,:]
    return data

def hands_detection(data: batchData):
    y = torch.reshape(data.y, [-1, 2])
    data.y = y[7:9,:]
    return data

def replace_node_feature_with_ones(data: batchData):
    data.x = torch.ones_like(data.x)
    return data


def merge_pos_to_input(data: batchData):
    # Concatenate along the second dimension
    # temp = data.x.unsqueeze(1)
    temp = data.x
    temp = torch.cat((data.pos, temp), dim=1)
    data.x = temp
    return data
def merge_pos_to_fit(data: batchData):
    # Concatenate along the second dimension
    temp = data.x[:,0].unsqueeze(1)
    temp = torch.cat((data.pos, temp), dim=1)
    data.x = temp

    return data

def use_pos_as_input(data: batchData):
    data.x = data.pos[:, :]
    return data

def check_x_size(data: batchData):
    if len(data.x.shape) == 1:
        if data.x.shape[:] != len(data.pos):
            # data.x = data.x.reshape(-1,2)
            data.x = data.x
        else:
            data.x = data.x[:, None]
    return data

def chain_transforms(transforms):
    if not isinstance(transforms, list):
        transforms = [transforms]

    def chained_transform(value):
        result = value
        for transform in transforms:
            result = transform(result)
        return result

    return chained_transform

def add_edges_10(data):
    return add_edges(data, r=10)

def add_edges_12(data):
    return add_edges(data, r=12)

def add_edges_15(data):
    return add_edges(data, r=15)

def add_edges_20(data):
    return add_edges(data, r=20)

def add_edges_30(data):
    return add_edges(data, r=30)

def add_edges(data, r=10):
    new_edges = pool.radius_graph(data.pos, r, num_workers=4)
    # print('\n Initial edge_index size:', data.edge_index.size())
    old_edges = data.edge_index
    # data.edge_index = new_edges
    data.edge_index = check_and_join_tensors(data.edge_index,new_edges)
    # print('Final edge_index size:', data.edge_index.size())
    # print('Initial edge_attr size:', data.edge_attr.size())
    # data.edge_attr = torch.cat((data.edge_attr, torch.tensor([0.5]*(data.edge_index.size()[1]-data.edge_attr.size()[0]), dtype=torch.float)))
    # print('Final edge_attr size:', data.edge_attr.size())
    edge_attr_cartesian = Cartesian(norm=True, cat=False)
    data = edge_attr_cartesian(data)
    return data
