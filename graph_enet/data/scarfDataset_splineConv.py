import os
import os.path as osp
from os.path import join, relpath
import torch
from torch_geometric.data import Dataset
from tqdm import tqdm
from graph_enet.pyScarf.scarf.scarf_class import SCARF
from graph_enet.utils.log_loader import load_events_from_log, load_skeleton_from_log
from graph_enet.data.graph_builder_splineConv import build_scarf_graph_splineConv
from graph_enet.pyScarf.utils.slt_ppr_filter import SpatialFilter
import numpy as np
import re

class scarfDataset_splineConv(Dataset):
    def __init__(
            
            self, 
            root, 
            transform=None, pre_transform=None, pre_filter=None,
            rf_size = 14, 
            alpha = 1.0, 
            C = 0.3,
            res = (640, 480)):
        
        """
        This initializer sets up the dataset configuration parameters before calling
        the parent class initialization.
        Args:
            root (str): Root directory where the dataset files are stored.
            transform (callable, optional): A function/transform that takes in
                a Data object and returns a transformed version. Default is None.
            pre_transform (callable, optional): A function/transform that pre-processes
                the raw data and saves it. Default is None.
            pre_filter (callable, optional): A function/transform that filters out
                data samples before processing. Default is None.
            rf_size (int, optional): Receptive field size for the graph construction.
                Default is 14.
            alpha (float, optional): Alpha parameter for spline convolution scaling.
                Default is 1.0.
            C (float, optional): Compression or clipping parameter. Default is 0.3.
            res (tuple, optional): Resolution (width, height) of the input data.
                Default is (640, 480).
        """
        
       
        self.rf_size = rf_size
        self.alpha = alpha
        self.C = C
        self.res = res

        super().__init__(root, transform, pre_transform, pre_filter)   # Init only after set the attributes


    @property
    def raw_file_names(self):
        raw_dir = join(self.root, 'raw')
        file_names = []

        for dirpath, _, filenames in os.walk(raw_dir):
            if 'data.log' in filenames:
                rel_path = relpath(join(dirpath, 'data.log'), raw_dir)
                file_names.append(rel_path)

        return file_names
    
    # def read_raw_paths(self, raw_paths):
    #     """
    #     Splits the raw_paths list into two lists:
    #     - event_paths: all paths containing '/ch0dvs/data.log'
    #     - skltn_paths: all paths containing '/ch0GT50Hzskeleton/data.log'
    #     """
    #     event_paths = []
    #     skltn_paths = []

    #     for path in raw_paths:
    #         if '/ch0dvs/data.log' in path:
    #             event_paths.append(path)
    #         elif '/ch0GT50Hzskeleton/data.log' in path:
    #             skltn_paths.append(path)

    #     return event_paths, skltn_paths


    def read_raw_paths(self, raw_paths, subject_ids=None):
        """
        Return (event_paths, skltn_paths) from raw_paths.
        Only accepts paths like:
        .../cam{cam}_S{subject}_{action}[_{take}]/(ch0dvs|ch0GT50Hzskeleton)/data.log

        subject_ids: None (no filter), an int, or an iterable of ints.
        """
        # Normalize allowed subjects
        if subject_ids is None:
            allowed = None
        elif isinstance(subject_ids, int):
            allowed = {subject_ids}
        else:
            allowed = {int(s) for s in subject_ids}

        event_paths, skltn_paths = [], []

        for path in raw_paths:
            s = path.replace('\\', '/')

            # Keep only the two target files
            if not s.endswith('/data.log'):
                continue
            is_event = '/ch0dvs/' in s
            is_skel  = '/ch0GT50Hzskeleton/' in s
            if not (is_event or is_skel):
                continue

            # Clip folder is two levels up: .../<clip>/<stream>/data.log
            parts = s.strip('/').split('/')
            if len(parts) < 3:
                continue
            clip = parts[-3]  # e.g. cam2_S11_Eating or cam2_S11_Phoning_2

            # Extract subject id from clip name
            m = re.search(r'_S(\d+)_', clip)
            if not m:
                continue
            sid = int(m.group(1))

            # Subject filter
            if allowed is not None and sid not in allowed:
                continue

            # Route to the right list
            if is_event:
                event_paths.append(path)
            else:
                skltn_paths.append(path)

        return event_paths, skltn_paths


    @property
    def processed_file_names(self):
        # ensure processed_dir exists, if not create it
        os.makedirs(self.processed_dir, exist_ok=True)
        # Dynamically list all .pt files in the processed directory
        return [f for f in os.listdir(self.processed_dir) if f.endswith('.pt')]

    def download(self):
        pass                    # No download needed, file are already be in `raw_dir`

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)      # Create processed dir if not there

        # === load events data from log file ===
        event_path, sklt_path = self.read_raw_paths(self.raw_paths)

        # Ensure we have both event and skeleton paths
        if not event_path or not sklt_path:
            raise ValueError("No valid event or skeleton paths found in the raw data.")


        graph_idx = 0

        with tqdm(total=len(event_path), desc="Processing files", unit="file") as file_pbar:  
            for i in range(len(event_path)):
                file_pbar.set_description(f"[INFO] Processing file {i+1}/{len(event_path)}")
                
                # === load events and skeleton data from log file ===
                efolder_path = os.path.dirname(event_path[i])
                sfolder_path = os.path.dirname(sklt_path[i])
                file_name = os.path.basename(event_path[i])

                events = load_events_from_log(efolder_path, file_name)
                sklt_data = load_skeleton_from_log(sfolder_path, file_name)
            
                # === Init SCARF object (for each new file)===
                scarf = SCARF(self.res, self.rf_size, self.alpha, self.C)
                N = len(events)

                # === Init Slt&Ppr filter ===
                filter = SpatialFilter()
                filter.initialise(self.res[1], self.res[0], period=0.1, spatial_range=1)

                # === Main loop over batches of events ===
                sklt_idx = 0

                with tqdm(total=len(events), desc="Processing events", 
                         unit="event", leave=False) as event_pbar:
                    
                    for ev in events:
                    # Check if we have processed all skeletons
                        if sklt_idx >= len(sklt_data['ts']):
                            print("All skeletons have been processed.")
                            break

                        # Get the timestamp of the current skeleton frame
                        current_sklt_ts = sklt_data['ts'][sklt_idx]

                        # If the event occurred after the current skeleton timestamp,
                        # it means we have collected all events for that skeleton's time window.
                        if ev['ts'] > current_sklt_ts:
                            # Create a graph for the current skeleton frame

                            # Get the corresponding skeleton data
                            current_skeleton = sklt_data['keypoints'][sklt_idx]

                            graph = build_scarf_graph_splineConv(scarf, current_skeleton)

                            if graph is None:
                                print(f"[INFO] Skipping skeleton at time {current_sklt_ts}s: Not enough active RFs.")
                                sklt_idx += 1
                                continue  # Skip this iteration

                            # === saving the graph in the processed folder ===
                            # Extract from event_path the folder name between 'raw/' and 'data.log'
                            folder_name = os.path.basename(os.path.dirname(os.path.dirname(event_path[i])))
                            graph_path = osp.join(self.processed_dir, f"{folder_name}_data_{graph_idx}.pt")
                            torch.save(graph, graph_path)
                            graph_idx += 1

                            # Move to the next skeleton frame
                            sklt_idx += 1

                        # Update the SCARF representation with the current event
                        if filter.check(ev['x'], ev['y'], ev['pol'], ev['ts']):
                                scarf.update(ev['x'], ev['y'], ev['pol'])

                        # Update progress bar every 1000 events to avoid overhead
                        if event_pbar.n % 1000 == 0:
                            event_pbar.update(1000)
                    
                    # Update remaining events
                    event_pbar.n = len(events)
                    event_pbar.refresh()
                
                file_pbar.update(1)
                file_pbar.set_postfix({"Graphs created": graph_idx})

    def len(self):
        # Cache result to avoid repeted directory scanning
        if not hasattr(self, '_cached_len'):
            self._cached_len = len([f for f in os.listdir(self.processed_dir) 
                                    if f.startswith('cam') and f.endswith('.pt')])
        return self._cached_len

    def get(self, idx):
        # Find the file matching 'cam*_{idx}.pt' in the processed directory
        files = [f for f in os.listdir(self.processed_dir) 
                 if f.startswith('cam') and f.endswith(f'_{idx}.pt')]
        if not files:
            raise IndexError(f"No processed file found for index {idx}")
        path = osp.join(self.processed_dir, files[0])
        return torch.load(path, weights_only=False)