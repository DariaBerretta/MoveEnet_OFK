import cv2
import sys
import os
import time 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.event_loader import load_events_from_log
from scarf.scarf import init_scarf, update_scarf, get_all_active_events

# === SCARF Parameters ===
rf_size = 14
alpha = 1.0
C = 0.3
res = (640, 480)
dt = 0.01

# === Load events ===
events = load_events_from_log("/data/cam2_S1_Directions/ch0dvs")
N = len(events)
print(f"[INFO] Loaded {N} events — Duration: {events['ts'][-1]:.2f}s")

# === Init SCARF and OpenCV display ===
scarf = init_scarf(res, rf_size, alpha, C)

# === Main loop ===
timer = 20.0
idx = 0

start_time = timer - dt 

active_events = []

 # while timer < events['ts'][-1]:
    
start_idx = idx

while start_idx < N and events['ts'][start_idx] < start_time:
    start_idx += 1


idx = start_idx

while idx < N and events['ts'][idx] <= timer:
    idx += 1
batch = events[start_idx:idx]

for ev in batch:
    update_scarf(ev['x'], ev['y'], ev['pol'], scarf)


events_per_rf = get_all_active_events(scarf)
# active_events.append(events_per_rf)

# print(f"List of active events for RFs: {events_per_rf}")

for idx, rf_events in enumerate(events_per_rf):
    if rf_events:  # Only print RFs with active events
        print(f"\n[RF {idx}] — {len(rf_events)} active events:")
        for ev in rf_events:
            u, v, p, a = ev
            print(f"  • (u={u}, v={v}, p={p}, a={a})")
            time.sleep(0.1)

# timer += dt

# === check functioning of active events extraction === 