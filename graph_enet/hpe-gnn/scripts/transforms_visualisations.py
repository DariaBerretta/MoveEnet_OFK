
import torch
from torch_geometric.data.database import TensorInfo
import sys

relative_module_directory = "/home/ggoyal/code/wp5-gamer/hpe-gnn"
sys.path.append(relative_module_directory)

from data import customDatasets, transforms
from model import hpegnn
from utils.dataset_utils import hpe_filter, dataset_split, schema_spline
from utils.library_utils import MyProgressBar
from utils.model_utils import GraphVisualization
from tqdm import tqdm

import cv2


def highlight_difference(original, augmented):
    # Read the grayscale images
    image1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the two images
    diff = cv2.absdiff(image1, image2)

    # Apply a colormap to visualize the difference in color
    diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

    # Overlay the difference onto one of the images
    result = cv2.addWeighted(cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR), 0.5, diff_color, 0.5, 0)

    return result, diff_color


data_path = '/home/ggoyal/data/h36m_cropped/ledge_toy'
# Training Parameters

visualise = True
save_video = True

transform_list = []
transform_list.append(transforms.check_x_size)
transform_list = transforms.chain_transforms(transform_list)

transform_list_changed = []
transform_list_changed.append(transforms.check_x_size)
transform_list_changed.append(transforms.add_edges_15)
transform_list_changed = transforms.chain_transforms(transform_list_changed)

dataset = customDatasets.eh36m_spline_ledge(data_path, transform=transform_list_changed, pre_filter=hpe_filter, schema=schema_spline)
dataset_original = customDatasets.eh36m_spline_ledge(data_path, transform=transform_list, pre_filter=hpe_filter, schema=schema_spline)

if save_video:
    file_path_orig = '/home/ggoyal/data/h36m_cropped/videos/graph_base.mp4'
    file_path_aug = '/home/ggoyal/data/h36m_cropped/videos/graph_base_r15.mp4'
    frame_width = 640
    frame_height = 480
    fps = 50
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    input = cv2.VideoWriter(file_path_orig, fourcc, fps, (frame_width, frame_height))
    output = cv2.VideoWriter(file_path_aug, fourcc, fps, (frame_width, frame_height))
    print('saving video')


for i in tqdm(range(1500)):
    data = dataset.multi_get([i])
    data_ori = dataset_original.multi_get([i])

    G = GraphVisualization(data[0])
    G_ori = GraphVisualization(data_ori[0])
    img = G.create_image(show_gt=False)
    img_ori = G_ori.create_image(show_gt=False)
    mixed , diff = highlight_difference(img_ori,img)
    if visualise:
        cv2.imshow('augmented', img)
        cv2.waitKey(1)
        cv2.imshow('original', img_ori)
        cv2.waitKey(1)
        cv2.imshow('diff', mixed)
        cv2.waitKey(1)

    if save_video:
        input.write(img_ori)
        output.write(mixed)
        # data = temporary_head_detecttion(data)

if save_video:
    output.release()
