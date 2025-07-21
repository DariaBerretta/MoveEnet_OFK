import os
import os.path as osp
import torch
from torch_geometric.data import Dataset
from graph_enet.pyScarf.scarf.scarf_class import SCARF
from graph_enet.pyScarf.utils.event_loader import load_events_from_log
from graph_enet.data.graph_builder import build_scarf_graph
from graph_enet.pyScarf.utils.slt_ppr_filter import SpatialFilter


class scarfDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None,
                 rf_size = 14, 
                 alpha = 1.0, 
                 C = 0.3,
                 res = (640, 480),
                 dt = 0.01):
       
        self.rf_size = rf_size
        self.alpha = alpha
        self.C = C
        self.res = res
        self.dt = dt

        super().__init__(root, transform, pre_transform, pre_filter)   # Init only after set the attributes


    @property
    def raw_file_names(self):
        return ['data.log']
    
    @property
    def processed_file_names(self):
        # This is needed by torch_geometric to decide if process() needs to run
        return ['data_0.pt']  # Just a placeholder, doesn't need to exist yet


    def download(self):
        pass                    # No download needed, file are already be in `raw_dir`

    def process(self):
        os.makedirs(self.processed_dir, exist_ok=True)      # Create processed dir if not there

        # === load data from log file ===
        event_path = self.raw_paths[0]          # Only one file
        folder_path = os.path.dirname(event_path)
        file_name = os.path.basename(event_path)
        
        events = load_events_from_log(folder_path, file_name)
        

        # === Init SCARF object ===

        scarf = SCARF(self.res, self.rf_size, self.alpha, self.C)
        N = len(events)

        # === Init Slt&Ppr filter ===
        filter = SpatialFilter()
        filter.initialise(self.res[1], self.res[0], period=0.1, spatial_range=1)

        # === Main loop over batches of events ===
        timer = 0.0
        idx = 0
        graph_idx = 0

        while timer < events['ts'][-1]:

            start_idx = idx

            # === load a batch ===
            while idx < N and events['ts'][idx] <= timer:
                idx += 1
            
            batch = events[start_idx:idx]

            # === Update scarf ===
            for ev in batch:
                # === Salt and Pepper noise removal ===
                if filter.check(ev['x'], ev['y'], ev['pol'], ev['ts']):
                    scarf.update(ev['x'], ev['y'], ev['pol'])

            # === create graph ===
            graph = build_scarf_graph(scarf)

            if graph is None:
                print(f"[INFO] Skipping frame at time {timer:.2f}s: Not enough active RFs.")
                timer += self.dt
                continue  # Skip this iteration
            
            # === saving the graph in the processed folder ===
            torch.save(graph, osp.join(self.processed_dir, f'data_{graph_idx}.pt'))
            graph_idx += 1

            # === Update for the next batch ===
            timer += self.dt
        

    def len(self):
        return len([f for f in os.listdir(self.processed_dir) if f.endswith('.pt')])

    def get(self, idx):
        path = osp.join(self.processed_dir, f'data_{idx}.pt')
        return torch.load(path)