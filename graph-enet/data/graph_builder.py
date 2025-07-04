from ..pyScarf.scarf.scarf_class import SCARF
from pyScarf.utils.event_loader import  load_batch_from_log
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

        v1,v2 = pca.compontents_
        lambda_1,lambda_2 = pca.explained_variance_

        print(f"RF_index: {RF_idx}, \n x_mean: {x_mean}, \n y_mean: {y_mean}, \n PCA_1: {v1}, \n PCA_2: {v2}, \n Lambda_1: {lambda_1}, \n Lambda_2: {lambda_2}")


# if the file is run as main
if __name__ == "__main__":

    # === SCARF Parameters ===
    rf_size = 14
    alpha = 1.0
    C = 0.3
    res = (640, 480)
    dt = 0.01

    # === Load batch of events from log file ===
    log_path = "/data/cam2_S1_Directions/ch0dvs"
    
    end_time = 20.0
    start_time = timer - dt 

    events = load_batch_from_log(log_path,start_time, end_time)

    print(f"[INFO] Loaded {len(events)} events")

    # === Init SCARF ===
    scarf = SCARF(res,rf_size,alpha,C)

    for ev in events:
        scarf.update(ev['x'], ev['y'], ev['pol'])

    # === Build the graph from SCARF ===
    build_scarf_graph(scarf)