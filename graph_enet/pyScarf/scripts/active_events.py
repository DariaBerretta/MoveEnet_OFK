import cv2
import sys
import os
import time 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph_enet.utils.log_loader import load_batch_from_log
from graph_enet.pyScarf.scarf.scarf_class import SCARF

# === SCARF Parameters ===
rf_size = 14
alpha = 1.0
C = 0.3
res = (640, 480)
dt = 0.01

# === Load a single batch of events ===
timer = 20.0
start_time = timer - dt 
idx = 0

events = load_batch_from_log("/home/dberretta-iit.local/data/cam2_S1_Directions/ch0dvs", start_time,timer)
N = len(events)
print(f"[INFO] Loaded {N} events — Duration: {events['ts'][-1]:.2f}s")

# === Init SCARF and OpenCV display ===
scarf = SCARF(res, rf_size, alpha, C)

active_events = []

for ev in events:
    scarf.update(ev['x'], ev['y'], ev['pol'])


for RFidx, _, rf_events in scarf.get_active_RF():
    print(f"\n[RF {idx}] — {len(rf_events)} active events:")
    for ev in rf_events:
        u, v, p, a = ev
        print(f"  • (u={u}, v={v}, p={p}, a={a})")
        time.sleep(0.1)