
from torch import abs, sum, float32, int64
from torch_geometric.data.database import TensorInfo
from graph_enet.hpe_gnn.data import h36m_utils as h36m
from tqdm import tqdm

def hpe_filter(sample):
    return check_sample_length(sample) & check_y_values_sample(sample)

def dataset_split(dataset, style= None, fraction = None, dataset_label = 'h36m'):
    train_dataset = []
    val_dataset = []
    if style is None:
        print('Dataset split unclear. Please define. Exiting.')
        exit()
    if fraction is None:
        fraction = 1

    if style == 'dev':
        fraction = 0.01
        cutoff = round(dataset.len() * .8 * fraction)
        for i in range(int(len(dataset)*fraction)):
            if i < cutoff:
                train_dataset.append(dataset[i])
            else:
                val_dataset.append(dataset[i])

    elif style == 'subject':
        if dataset_label == 'h36m':
            train_set = h36m.SPLITS['train']
        elif dataset_label == 'dhp19':
            train_set = dhp19.SPLITS['train']
        else:
            print("dataset not correct:", dataset_label)
            exit()

        # val_set = h36m.SPLITS['val']
        for i, data in enumerate(dataset):
            if data.sample[1] in train_set:
                train_dataset.append(data)
            else:
                val_dataset.append(data)
        if fraction < 1:
            train_dataset = train_dataset[:int(len(train_dataset)*fraction)]
            val_dataset = val_dataset[:int(len(val_dataset)*fraction)]
    return train_dataset, val_dataset

def new_dataset_split(dataset, style=None, fraction=None, dataset_label='dev'):
    if style is None:
        print('Dataset split unclear. Please define. Exiting.')
        exit()
    if fraction is None:
        fraction = 1

    if style == 'dev':
        fraction = fraction
        dataset_size = len(dataset)
        total_items = int(dataset_size * fraction)      # the actual number of items to consider 
        
        # 80/10/10 split
        train_end = int(total_items * 0.8)
        val_end = int(total_items)
        
        print(f"Splitting dataset into {train_end}/{val_end-train_end} samples...")
        
        # Fast sequential slicing
        # Train split
        train_dataset = [dataset[i] for i in range(train_end)]
        
        # Val split
        val_dataset = [dataset[i] for i in range(train_end, val_end)]

        print(f"Dataset split complete: {len(train_dataset)}/{len(val_dataset)} (train/val)")

    return train_dataset, val_dataset

def check_y_values(dataset):
    count = 0
    values_to_remove = 0
    indices = []
    for data in dataset:
        if check_y_values_sample(data):
            indices.append(count)
            count += 1
        else:
            values_to_remove += 1
    return dataset.index_select(indices)

def check_y_values_sample(sample):
    return sum(abs(sample.y)) < 700000

def check_sample_length(sample):
    return len(sample.x) > 30

schema = {'x': TensorInfo(float32, (-1,)),
          'edge_index': TensorInfo(int64, (2, -1)),
          'edge_attr': TensorInfo(float32, (-1,)),
          'y': TensorInfo(float32, (-1,)),
          'th_pck': TensorInfo(float32, (-1,)),
          'pos': TensorInfo(float32, (-1, 2))}

schema_spline = {'x': TensorInfo(float32, (-1,)),
                 'edge_index': TensorInfo(int64, (2, -1)),
                 'edge_attr': TensorInfo(float32, (-1, 2)),
                 'y': TensorInfo(float32, (-1,)),
                 'th_pck': TensorInfo(float32, (-1,)),
                 'pos': TensorInfo(float32, (-1, 2)),
                 'ts': TensorInfo(float32, (-1,)),
                 'sample': TensorInfo(int64, (3, -1))
                 }

# schema_scarf_spline = {
#     'x': TensorInfo(float32, (-1, 10)),           # Node features: [N_nodes, 10]
#     'edge_index': TensorInfo(int64, (2, -1)),     # Edge indices: [2, N_edges]
#     'edge_attr': TensorInfo(float32, (-1, 2)),    # Edge attributes: [N_edges, 2] (dx, dy for SplineConv)
#     'y': TensorInfo(float32, (1, 26)),            # Target: [1, 26] (13 joints × 2 coordinates)
#     'th_pck': TensorInfo(float32, (1,)),          # PCK threshold: [1]
#     'pos': TensorInfo(float32, (-1, 2))           # Node positions: [N_nodes, 2]
#     }
