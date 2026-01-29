
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

def new_dataset_split(dataset, style=None, fraction=None, dataset_label='dev', train_subjects=None, val_subjects=None):

    if style is None:
        print('Dataset split unclear. Please define. Exiting.')
        exit()

    if style in ['total', 'dev']:
        if style == 'total':
            fraction = 1
        # fraction is already passed for 'dev'
        dataset_size = len(dataset)
        total_items = int(dataset_size * fraction)

        # 80/20 split
        train_end = int(total_items * 0.8)
        val_end = int(total_items)

        print(f"Splitting dataset into {train_end}/{val_end-train_end} samples...")

        # Train split
        # train_dataset = [dataset[i] for i in range(train_end)]
        train_dataset = dataset[:train_end]

        # Val split
        # val_dataset = [dataset[i] for i in range(train_end, val_end)]
        val_dataset = dataset[train_end:val_end]

        print(f"Dataset split complete: {len(train_dataset)}/{len(val_dataset)} (train/val)")

    elif style == 'subject':
        
        if train_subjects is None or val_subjects is None:
            print('Subject split requires train_subjects and val_subjects. Exiting.')
            exit()

        train_subjects = [int(sub) for sub in train_subjects]
        val_subjects = [int(sub) for sub in val_subjects]

        overlap = set(train_subjects).intersection(val_subjects)
        if overlap:
            print(f'Subject overlap between splits: {sorted(overlap)}. Exiting.')
            exit()

        if fraction is None:
            fraction = 1

        train_dataset = []
        val_dataset = []

        for data in dataset:
            if not hasattr(data, 'sample'):
                print('Dataset samples missing subject metadata. Exiting.')
                exit()

            subject_tensor = data.sample[1]
            subject_id = int(subject_tensor.item()) if hasattr(subject_tensor, 'item') else int(subject_tensor)

            if subject_id in train_subjects:
                train_dataset.append(data)
            elif subject_id in val_subjects:
                val_dataset.append(data)

        if fraction < 1:
            train_cutoff = int(len(train_dataset) * fraction)
            val_cutoff = int(len(val_dataset) * fraction)

            if fraction > 0:
                train_dataset = train_dataset[:max(train_cutoff, 1) if train_dataset else 0]
                val_dataset = val_dataset[:max(val_cutoff, 1) if val_dataset else 0]
            else:
                train_dataset = []
                val_dataset = []

        print(f"Dataset split complete: {len(train_dataset)}/{len(val_dataset)} (train/val)")


    # TO DO: verify this part
    # if style == 'subject':
    #     if dataset_label == 'h36m':
    #         train_set = h36m.SPLITS['train']
    #     #elif dataset_label == 'dhp19':
    #     #    train_set = dhp19.SPLITS['train']
    #     else:
    #         print("dataset not correct:", dataset_label)
    #         exit()

    #     # val_set = h36m.SPLITS['val']
    #     for i, data in enumerate(dataset):
    #         if data.sample[1] in train_set:
    #             train_dataset.append(data)
    #         else:
    #             val_dataset.append(data)
    #     if fraction < 1:
    #         train_dataset = train_dataset[:int(len(train_dataset)*fraction)]
    #         val_dataset = val_dataset[:int(len(val_dataset)*fraction)]

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
