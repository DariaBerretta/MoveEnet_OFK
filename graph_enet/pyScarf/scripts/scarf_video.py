import os
import cv2
import imageio
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.event_loader import load_events_from_log
from utils.video_writer import create_video_writer, write_frame
from scarf.scarf_class import SCARF
# === SCARF Parameters ===
rf_size = 14
alpha = 1.0
C = 0.3
res = (640, 480)
dt = 0.01

input_path = "/data/cam4_S9_Sitting/ch0dvs"
output_dir = "/data/scarf_snapshots_py"
# Name the output video according to the input 
video_path = os.path.join(output_dir, "try with class.mp4")

# === Load events ===
events = load_events_from_log(input_path)
N = len(events)
print(f"[INFO] Loaded {N} events — Duration: {events['ts'][-1]:.4f}s")

# === Initialize SCARF and output ===
scarf = SCARF(res, rf_size, alpha, C)
video_writer = create_video_writer(video_path, res, fps=int(1/dt))

# === Processing loop ===
timer = 0.0
idx = 0
img_count = 0

while timer < events['ts'][-1]:
    
    # Slice the current batch of events
    start_idx = idx
    while idx < N and events['ts'][idx] <= timer:
        idx += 1
    batch = events[start_idx:idx]

    # Update SCARF
    for ev in batch:
        scarf.update(ev['x'], ev['y'], ev['pol'])

    # Convert and save image
    img32 = scarf.get_surface()
    img8U = (img32 * 255).clip(0, 255).astype('uint8')
    inverted = cv2.bitwise_not(img8U)

    # Save EXR
    exr_name = f"scarf_py_t{img_count:04d}.exr"
    imageio.imwrite(os.path.join(output_dir, exr_name), img32.astype('float32'))

    # Write frame to video
    write_frame(video_writer, inverted)

    # Show progress
    #      print(f"[INFO] Frame {img_count:04d} | Time: {timer:.2f}s | Events: {len(batch)}")
    timer += dt
    img_count += 1

video_writer.release()
print("[INFO] Video saved:", video_path)
