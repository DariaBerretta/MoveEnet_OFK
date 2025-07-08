from pyScarf.scarf.scarf_class import SCARF
from pyScarf.utils.event_loader import load_events_from_log
from data.graph_builder import build_scarf_graph
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

# === SCARF Parameters ===
rf_size = 14
alpha = 1.0
C = 0.3
res = (640, 480)
dt = 0.01

# === Load events from log file ===
log_path = "/home/dberretta-iit.local/data/cam2_S1_Directions/ch0dvs"
events = load_events_from_log(log_path)

# === Init SCARF ===
scarf= SCARF(res, rf_size, alpha, C)
N = len(events)

# === Main loop ===
timer = 0.0
idx = 0

# === Set up plot ===
fig, ax = plt.subplots(figsize=(8, 6))
plt.ion()  # Turn on interactive mode

while timer < events['ts'][-1]:
    
    start_idx = idx
    
    # Collect a batch of events
    while idx < N and events['ts'][idx] <= timer:
        idx += 1
    batch = events[start_idx:idx]

    # Update SCARF with the new batch
    for ev in batch:
        scarf.update(ev['x'], ev['y'], ev['pol'])

    # Build the new Graph  
    graph = build_scarf_graph(scarf)

    if graph is None:
        print(f"[INFO] Skipping frame at time {timer:.2f}s: Not enough active RFs.")
        timer += dt
        continue  # Skip this iteration

    # Update the graph plot
    # Conversion into NetworkX for graph visualization
    G = to_networkx(graph, to_undirected=True, remove_self_loops=True)

    # Extract node positions for plotting
    pos_dict = {
        i: (graph.pos[i][0].item(), graph.pos[i][1].item()) 
        for i in range(graph.num_nodes)
    }

    # === Clear and redraw graph ===
    ax.clear()
    nx.draw(G, pos=pos_dict,
            ax=ax,
            node_size=80,
            node_color='skyblue',
            edge_color='gray',
            with_labels=False)
    ax.set_title("SCARF Graph (active RFs)")
    # === Fix axes limits ===
    ax.set_xlim(0, res[0])
    ax.set_ylim(0, res[1])
    ax.set_aspect('equal')
    ax.axis("on")
    plt.tight_layout()
    plt.pause(0.01)  # You can adjust this pause time if needed
    
    # Increase timer
    timer += dt
