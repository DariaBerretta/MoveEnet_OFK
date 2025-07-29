import sys
import os
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph_enet.pyScarf.utils.event_loader import load_skeleton_from_log
from graph_enet.pyScarf.scarf.scarf_class import SCARF
from graph_enet.pyScarf.utils.event_loader import load_events_from_log

# Example path to skeleton data file
sklt_fpath = "/home/dberretta-iit.local/data/cam2_S1_Discussion/ch0GT50Hzskeleton/"
ev_fpath = "/home/dberretta-iit.local/data/cam2_S1_Discussion/ch0dvs/"

# Load the skeleton data
sklt_data = load_skeleton_from_log(sklt_fpath)

# Load events data
events = load_events_from_log(ev_fpath)

# Initialize SCARF object
scarf = SCARF(res=(640, 480), rf_size=14, alpha=1.0, C=0.3)

# Synchronize skeleton timestamps with events
sklt_idx = 0
for event in events:
    # Check if we have processed all skeletons
    if sklt_idx >= len(sklt_data['ts']):
        print("All skeletons have been processed.")
        break

    # Get the timestamp of the current skeleton frame
    current_sklt_ts = sklt_data['ts'][sklt_idx]

    # If the event occurred after the current skeleton timestamp,
    # it means we have collected all events for that skeleton's time window.
    if event['ts'] > current_sklt_ts:
        # Process the accumulated events by getting the SCARF representation
        scarf_surface = scarf.get_surface()
        
        # Get the corresponding skeleton data
        current_skeleton = sklt_data['keypoints'][sklt_idx]

        # Plot SCARF surface and skeleton keypoints
        img32 = scarf.get_surface()
        img8U = (img32 * 255).clip(0, 255).astype('uint8')
        inverted = cv2.bitwise_not(img8U)

        # Convert to color image for drawing
        img_color = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)

        # Draw skeleton keypoints (Each skeleton: 13 joints, each joint has (x, y) -> 26 elements flattened)
        for i in range(0, len(current_skeleton), 2):
            x, y = int(current_skeleton[i]), int(current_skeleton[i + 1])
            cv2.circle(img_color, (x, y), 3, (0, 255, 0), -1)

        cv2.imshow("SCARF + Skeleton", img_color)
        cv2.waitKey(1)

        # Move to the next skeleton frame
        sklt_idx += 1

    # Update the SCARF representation with the current event
    scarf.update(event['x'], event['y'], event['pol'])
