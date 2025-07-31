from pyScarf.scarf.scarf_class import SCARF
from graph_enet.utils.log_loader import  load_batch_from_log
from data.graph_builder import build_scarf_graph



# === SCARF Parameters ===
rf_size = 14
alpha = 1.0
C = 0.3
res = (640, 480)
dt = 0.01

# === Load batch of events from log file ===
log_path = "/home/dberretta-iit.local/data/cam2_S1_Directions/ch0dvs"

end_time = 31.0
start_time = end_time - dt 

events = load_batch_from_log(log_path,start_time, end_time)


print(f"[INFO] Loaded {len(events)} events")

# # === Init SCARF ===
scarf = SCARF(res,rf_size,alpha,C)

for ev in events:
     scarf.update(ev['x'], ev['y'], ev['pol'])

# === Build the graph from SCARF ===
build_scarf_graph(scarf)

