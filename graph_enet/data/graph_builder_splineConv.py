from graph_enet.pyScarf.scarf.scarf_class import SCARF
from sklearn.decomposition import PCA
import numpy as np
import torch
from torch_geometric.nn import radius_graph
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.transforms import Cartesian


def build_scarf_graph_splineConv(
    scarf,
    current_skeleton,
    k_neighbour: int = 4,
    active_ratio: float = 0.15,
    radius: float = 25.0,
    device: str = 'auto',
):
    """
    Build a PyG Data graph for SplineConv models from a SCARF state + a 2D skeleton.

    - Node features order:
        x[:, 0:2] = (x_mean, y_mean)
        x[:, 2]   = RF_idx
        x[:, 3:5] = v1 (PCA first component)
        x[:, 5:7] = v2 (PCA second component)
        x[:, 7:9] = (lambda_1, lambda_2)
        x[:, 9]   = eccentricity

    - Edges: radius_graph over positions (x_mean, y_mean) with limit (k_neighbour) on neighbors.
    - edge_attr: (dx, dy) via Cartesian() for SplineConv (dim=2).
    - y: shape [1, 2J] --> where J is the number of joints in the skeleton.
    - th_pck: torso dimension (distance between joints 2 and 6)
    
    Args:
        device: 'auto', 'cuda', 'cpu', or specific device like 'cuda:0'
    """
    # Auto-detect device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    active_rfs = scarf.get_active_RF(active_ratio)
    
    # Pre-allocate arrays
    num_rfs = len(active_rfs)
    node_features = np.zeros((num_rfs, 10), dtype=np.float32)
    
    # Pre-create PCA object
    pca = PCA(n_components=2)
    
    # Determine the node features
    for i, rf in enumerate(active_rfs):
        RF_idx, _, events = rf

        # Compute means (vectorized) - use events directly
        x_mean, y_mean = np.mean(events[:, :2], axis=0)

        # PCA computation
        pca.fit(events[:, :2])

        v1, v2 = pca.components_
        lambda_1, lambda_2 = pca.explained_variance_

        # Avoid unnecessary float() conversions - direct assignment to float32 array
        # control on lambda_1 = 0 that means no variance (all events at same position)
        eccentricity = np.sqrt(1 - lambda_2/lambda_1) if lambda_1 > 0 else 0.0

        # Feature assignment
        node_features[i, :] = [
            x_mean, y_mean, 
            RF_idx, 
            v1[0], v1[1], 
            v2[0], v2[1], 
            lambda_1, lambda_2, 
            eccentricity
        ]

    
    if len(node_features) == 0:
        print("[ERROR] Not enogh avtive RFs to build the graph.")
        return None
    
    # Creation of the nodes tensor - direct conversion and move to GPU
    nodes = torch.from_numpy(node_features).float().to(device)

    # Creation of the edges 
    positions = nodes[:, 0:2].contiguous()    # Already on GPU

    edge_index = radius_graph(x=positions, r=radius, max_num_neighbors=k_neighbour, loop=False)

    # Ensure y is correct shape and type - move to GPU
    y = torch.tensor(current_skeleton, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Compute th_pck (distance between joint 2 and 6) 
    kp = y.reshape(-1, 2)
    th_pck = torch.norm(kp[2] - kp[6]).unsqueeze(0)

    # Graph creation
    graph = Data(x = nodes, edge_index = edge_index, pos=positions)
    graph = Cartesian(cat=False)(graph)    # edge_attr = (Δx, Δy) in pixel units normalised [0, 1]  Dose it need norm=False?
    graph.y = y
    graph.th_pck = th_pck

    return graph

