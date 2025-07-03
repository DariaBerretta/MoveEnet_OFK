import numpy as np
import os
import sys
import logging

# Add bimvee path (adjust if needed)
# sys.path.append("/home/dberretta-iit.local/Documents/Repos/SCARF_analysis/SCARF_analysis/submodule/bimvee")
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
