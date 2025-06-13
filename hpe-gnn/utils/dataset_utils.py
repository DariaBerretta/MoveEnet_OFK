
from torch import abs, sum, float32, int64
from torch_geometric.data.database import TensorInfo
from data import h36m_utils as h36m

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

