import numpy as np
import os
import sys
import logging

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