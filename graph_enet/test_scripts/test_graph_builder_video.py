import cv2
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx
from graph_enet.pyScarf.utils.slt_ppr_filter import SpatialFilter
from graph_enet.utils.log_loader import load_events_from_log
from graph_enet.pyScarf.scarf.scarf_class import SCARF
from graph_enet.data.graph_builder import build_scarf_graph

# === SCARF Parameters ===
rf_size = 14
alpha = 1.0
C = 0.3
res = (640, 480)
dt = 0.01

# === Load events ===
events = load_events_from_log("/home/dberretta-iit.local/data/cam2_S1_Directions/ch0dvs/")
N = len(events)
print(f"[INFO] Loaded {N} events — Duration: {events['ts'][-1]:.2f}s")

# === Init SCARF and OpenCV display ===
scarf = SCARF(res, rf_size, alpha, C)

# === Init Salt&Pepper filter ===
filter = SpatialFilter()
filter.initialise(res[1], res[0], period=0.1, spatial_range=1)

# === Initialize Video Writer ===
output_path = "/home/dberretta-iit.local/data/graph_construction/graph_slt_ppr.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(1.0 / dt)
video_writer = cv2.VideoWriter(output_path, fourcc, fps, res)


# === Main loop ===
timer = 0.0
idx = 0

while timer < events['ts'][-1]:
    
    start_idx = idx
    while idx < N and events['ts'][idx] <= timer:
        idx += 1
    batch = events[start_idx:idx]

    for ev in batch:
        # Salt and Pepper noise removal
         if filter.check(ev['x'], ev['y'], ev['pol'], ev['ts']):
            scarf.update(ev['x'], ev['y'], ev['pol'])

    # === Create blank image for graph visualization only ===
    colored = np.full((res[1], res[0], 3), 255, dtype=np.uint8)

    # === Overlay graph if possible ===
    graph = build_scarf_graph(scarf)
    if graph is not None:
        G = to_networkx(graph, to_undirected=True, remove_self_loops=True)
        pos_dict = {
            i: (int(graph.pos[i][0].item()), int(graph.pos[i][1].item()))
            for i in range(graph.num_nodes)
        }

        # Draw edges
        for u, v in G.edges():
            p1 = pos_dict[u]
            p2 = pos_dict[v]
            cv2.line(colored, p1, p2, (100, 100, 255), 1)  # Red-ish edges

        # Draw nodes
        for node, (x, y) in pos_dict.items():
            cv2.circle(colored, (x, y), 3, (255, 255, 0), -1)  # Cyan nodes

    # === write on video ===
    video_writer.write(colored)  # Save frame to video

    timer += dt

video_writer.release()