import cv2
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph_enet.utils.log_loader import load_events_from_log
from scarf.scarf_class import SCARF 

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
scarf= SCARF(res, rf_size, alpha, C)
cv2.namedWindow("SCARF", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SCARF", res)

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

    img32 = scarf.get_surface()
    img8U = (img32 * 255).clip(0, 255).astype('uint8')
    inverted = cv2.bitwise_not(img8U)

    cv2.imshow("SCARF", inverted)
    cv2.waitKey(1)
    timer += dt