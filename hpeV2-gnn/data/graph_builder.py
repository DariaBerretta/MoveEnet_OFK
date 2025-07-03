
from pyScarf.scarf.scarf_class import SCARF
from sklearn.decomposition import PCA
import numpy as np

def build_scarf_graph(scarf):
    
    active_rfs = scarf.get_active_RF()

    n_nodes = len(active_rfs)        # Number os nodes 
    n_features = 6                   # RF_index, x_mean, y_mean, PCA_1, PCA_2, Eccentricity


    # Determine the node features
    for rf in active_rfs:
        RF_index, carf, events = rf
        RF_idx = RF_index
        x_mean = np.mean(events[:,0])
        y_mean = np.mean(events[:,1])

        pca = PCA(n_components=2)
        pca.fit(events[:,:2])