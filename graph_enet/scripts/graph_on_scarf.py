import cv2
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx

from graph_enet.pyScarf.utils.event_loader import load_events_from_log
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
cv2.namedWindow("SCARF with Graph", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SCARF with Graph", res)

# === Main loop ===
timer = 0.0
idx = 0

while timer < events['ts'][-1]:
    
    start_idx = idx
    while idx < N and events['ts'][idx] <= timer:
        idx += 1
    batch = events[start_idx:idx]

    for ev in batch:
        scarf.update(ev['x'], ev['y'], ev['pol'])

    # === Get SCARF grayscale image ===
    img32 = scarf.get_surface()
    img8U = (img32 * 255).clip(0, 255).astype('uint8')
    inverted = cv2.bitwise_not(img8U)
    colored = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)

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

    # === Show final overlay ===
    cv2.imshow("SCARF with Graph", colored)
    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit early
        break

    timer += dt

cv2.destroyAllWindows()
