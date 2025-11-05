from graph_enet.pyScarf.scarf.scarf_class import SCARF
from graph_enet.utils.log_loader import load_events_from_log
from graph_enet.data.graph_builder import build_scarf_graph
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from graph_enet.pyScarf.utils.slt_ppr_filter import SpatialFilter

# === SCARF Parameters ===
rf_size = 14
alpha = 1.0
C = 0.3
res = (640, 480)
dt = 0.01

# === Load events from log file ===
log_path = "/home/dberretta-iit.local/data/new_scarfGNN_full/raw/cam2_S1_Directions/ch0dvs"
events = load_events_from_log(log_path)

# === Init SCARF ===
scarf= SCARF(res, rf_size, alpha, C)
N = len(events)

# === Init Salt&Pepper filter ===
filter = SpatialFilter()
filter.initialise(res[1], res[0], period=0.1, spatial_range=1)

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
        # Salt and Pepper noise removal
         if filter.check(ev['x'], ev['y'], ev['pol'], ev['ts']):
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
            node_size=30,
            node_color='skyblue',
            edge_color='red',
            with_labels=False)
    ax.set_title("SCARF Graph (active RFs)")
    # === Fix axes limits ===
    ax.set_xlim(0, res[0])
    ax.set_ylim(0, res[1])
    ax.set_aspect('equal')
    ax.invert_yaxis()    # Invert y-axis to match the event coordinates
    ax.axis("on")
    plt.tight_layout()
    plt.pause(0.01) 
    
    # Increase timer
    timer += dt
