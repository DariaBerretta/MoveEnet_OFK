from graph_enet.pyScarf.scarf.scarf_class import SCARF
"""
Build a geometric graph representation from SCARF receptive fields.
This function constructs a PyTorch Geometric Data object by extracting active
receptive fields from a SCARF object and computing node features based on
spatial and statistical properties. Edges are established using radius-based
graph construction.
Args:
    scarf (SCARF): A SCARF object containing receptive field information.
    k_neighbour (int, optional): Maximum number of neighbors per node in the
        radius graph. Defaults to 4.
    active_ratio (float, optional): Ratio threshold for determining active
        receptive fields. Defaults to 0.15.
    radius (float, optional): Radius threshold for edge creation in the
        radius-based graph construction. Defaults to 25.
Returns:
    torch.geometric.data.Data or None: A PyTorch Geometric Data object with:
        - x (torch.Tensor): Node feature matrix of shape [num_nodes, 10]
          containing RF index, spatial center (x, y), PCA components,
          eigenvalues, and eccentricity.
        - edge_index (torch.Tensor): Edge index tensor defining graph connectivity.
        - pos (torch.Tensor): Node positions (x, y coordinates) of shape [num_nodes, 2].
        Returns None if no active receptive fields are found.
"""
from sklearn.decomposition import PCA
import numpy as np
import torch
from torch_geometric.nn import knn_graph, approx_knn_graph, radius_graph
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data




#def build_scarf_graph(scarf, current_skeleton, k_neighbour=4, active_ratio=0.15, radius=25):
def build_scarf_graph(
          scarf, 
          k_neighbour=4, 
          active_ratio=0.15, 
          radius=25):   
    
    active_rfs = scarf.get_active_RF(active_ratio)
        
    node_features= []                # List of node features

    # print(f"[INFO] Number of total RFs: {len(scarf.rfs)}, \n [INFO] Number of active RFs: {n_active_rfs}.")
    
    # Determine the node features
    for rf in active_rfs:
        RF_index, _, events = rf
        RF_idx = RF_index
        x_mean = np.mean(events[:,0])
        y_mean = np.mean(events[:,1])

        pca = PCA(n_components=2)
        pca.fit(events[:,:2])

        v1,v2 = pca.components_
        lambda_1,lambda_2 = pca.explained_variance_

        eccentricity = np.sqrt(1 - lambda_2/lambda_1) 

        feature = [RF_idx, 
                   x_mean, y_mean, 
                   v1[0], v1[1], 
                   v2[0],v2[1], 
                   lambda_1, lambda_2, 
                   eccentricity]
        
        node_features.append(feature)

        # print(f"RF_index: {int(RF_idx)}")
        # print(f"x_mean: {x_mean}")
        # print(f"y_mean: {y_mean}")
        # print(f"PCA_1: {v1}")
        # print(f"PCA_2: {v2}")
        # print(f"Lambda_1: {lambda_1}")
        # print(f"Lambda_2: {lambda_2}")
        # print(f" Eccentricity: {eccentricity}")
       

    # Creation of the nodes tensor
    if len(node_features) == 0:
        print("[ERROR] Not enogh avtive RFs to build the graph.")
        return None
    else:
          nodes = torch.tensor(np.array(node_features), dtype=torch.float32)

    # Creation of the edges 
    positions = nodes[:, 1:3]                                       # The kNN is based on the distance between the "center of mass" (x_mean, y_mean) of each RFs

    # edge_index = knn_graph(x=positions, k=k_neighbour, loop=False)
    # edge_index = approx_knn_graph(x=positions, k = k_neighbour, loop=False)
    # edge_index = radius_graph(x=positions, r=radius, loop=False)
    edge_index = radius_graph(x=positions, r=radius, max_num_neighbors=k_neighbour, loop=False)


    # Graph creation
    graph = Data(x = nodes, edge_index = edge_index, pos=positions)

    return graph

