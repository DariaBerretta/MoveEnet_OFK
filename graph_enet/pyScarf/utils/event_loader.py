import numpy as np
import os
import sys
import re

# Add bimvee path (adjust if needed)
sys.path.append("/home/dberretta-iit.local/Documents/Repos/SCARF_analysis/SCARF_analysis/submodule/bimvee")
from bimvee.importers.ImporterDataLog import ImporterDataLog


def load_events_from_log(folder_path, file_name="data.log"):
    """
    Load events using bimvee's ImporterDataLog.

    Returns:
        np.ndarray: structured array with dtype [('x', 'i4'), ('y', 'i4'), ('ts', 'f8'), ('pol', 'i1')]
    """
    importer = ImporterDataLog(folder_path, file_name)
    event_dict = importer.get_full_data_as_dict()

    N = len(event_dict['x'])
    dtype = [('x', 'i4'), ('y', 'i4'), ('ts', 'f8'), ('pol', 'i1')]
    events = np.zeros(N, dtype=dtype)

    events['x'] = event_dict['x']
    events['y'] = event_dict['y']
    events['ts'] = event_dict['ts']
    events['pol'] = event_dict['pol']

    return events

def load_batch_from_log(log_path, start_time, end_time):
    """
    Load a single batch of events from withing a specific time interval defined by a start and a end time
    """
    
    end_idx = 0

    events = load_events_from_log(log_path)

    start_idx = end_idx

    while start_idx < len(events) and events['ts'][start_idx] < start_time:
        start_idx += 1


    end_idx = start_idx

    while end_idx < len(events) and events['ts'][end_idx] <= end_time:
        end_idx += 1
    
    return events[start_idx:end_idx]

def load_skeleton_from_log(folder_path, file_name="data.log"):
    
    filename_gt_sklt = os.path.join(folder_path, file_name)
    if not os.path.exists(filename_gt_sklt):
        raise FileNotFoundError(f"Annotation file not found: {filename_gt_sklt}")

    with open(filename_gt_sklt, "r") as f:
        content = f.readlines()
    if len(content) == 0:
        raise Exception("No file, or no file content")

    # line format: 'id timestamp SKLT (<flattened 2D keypoints coordinates>) head_size torso_size'
    pattern = re.compile(r'(\d+) ([\d\.e\-]+) SKLT \((.*?)\) ([\d\.\-]+) ([\d\.\-]+)')

    gt_sklt_ts = []
    gt_sklt_jnts= []
    for line in content:
        match = pattern.match(line.strip())
        if match:
            _, ts, points_str, _, _ = match.groups()
            gt_sklt_ts.append(float(ts))
            points = np.array([float(p) for p in points_str.split() if p])
            gt_sklt_jnts.append(points)
        else:
            continue

    gt_sklt_ts = np.array(gt_sklt_ts)
    gt_sklt_jnts = np.array(gt_sklt_jnts)

    # Each skeleton: 13 joints, each joint has (x, y) -> 26 elements flattened
    num_joints = 13
    dtype = [('ts', 'f8'), ('keypoints', 'f4', (num_joints * 2,))]

    structured = np.zeros(len(gt_sklt_ts), dtype=dtype)
    structured['ts'] = gt_sklt_ts
    structured['keypoints'] = gt_sklt_jnts.reshape(-1, num_joints * 2)

    # Each skeleton: 13 joints, each joint has (x, y)
    #num_joints = 13
    #dtype = [('ts', 'f8'), ('keypoints', 'f4', (num_joints, 2))]

    #structured = np.zeros(len(gt_sklt_ts), dtype=dtype)
    #structured['ts'] = gt_sklt_ts
    #structured['keypoints'] = gt_sklt_jnts.reshape(-1, num_joints, 2)

    gt_sklt = structured
    return gt_sklt