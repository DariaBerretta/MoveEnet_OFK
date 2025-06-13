
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import Tensor

from data import customDatasets
from data.transforms import chain_transforms, center_detection, head_detection
from data.transforms import replace_node_feature_with_ones, add_pos_to_input
from model import hpegnn
from utils.dataset_utils import hpe_filter, dataset_split, schema_spline
from utils.library_utils import MyProgressBar
from utils.model_utils import GraphVisualization
from argparse import ArgumentParser



# Trainer arguments
parser = ArgumentParser()
parser.add_argument('--data_fraction', type=float, default=0.01, help='Fraction of dataset to be used [0.01,0.1,1]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Constant Learning rate [0.0001,0.001,01]')
parser.add_argument('--epochs', type=int, default=10, help='Total epochs [1,10,20,50]')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch size [1,8,32,256,1024]')
parser.add_argument('--num_joints', type=int, default=1, help='Number of joints [1,13]')
parser.add_argument('--label', type=str, default='default', help='String to label the output')
parser.add_argument('--dev', help='Use dev mode. data_fraction, epochs, batch_size and label will be overwritten.',
                    action='store_true', default=False, required=False)
parser.add_argument('--exp_setup', type=int, default=1,
                    help='Experimental setup [1, 2]; 1: node feature as energy, 2: node feature as 1')
parser.add_argument('--task', type=int, default=1,
                    help='Task setup [1, 2]; Only used if num_joint==1 1: center detection, 2: head detection')

args = parser.parse_args()


# max_edge = 15  # diagonal length of a GAMER block #TODO: Find this value

if args.dev:
    data_path = '/home/ggoyal/data/h36m_gamer/gamer_toy/'
    # data_path = '/mnt/disk1/data/h36m/gamer/'

else:
    data_path = '/home/ggoyal/data/h36m_gamer/gamer/'
    # data_path = '/mnt/disk1/data/h36m/gamer/'

# Training Parameters

learning_rate = args.learning_rate
hidden_channels = [8,32,8]
num_joints = args.num_joints
if args.dev:
    data_fraction = 1
    label = 'dev'
    batch_size = 16
    epochs = 5
else:
    data_fraction = args.data_fraction  # fraction of data to use in total
    label = args.label  # ['test', ]
    batch_size = args.batch_size
    epochs = args.epochs

image_size = [640, 480]
visualise = False


transforms = None
if args.num_joints == 1:
    transforms = []
    if args.exp_setup == 2:
        transforms.append(replace_node_feature_with_ones)
    if args.task == 1:
        transforms.append(center_detection)
    else:
        transforms.append(head_detection)
    transforms = chain_transforms(transforms)



if num_joints == 1:
    dataset = customDatasets.eh36m_spline(data_path, transform=transforms, pre_filter=hpe_filter,
                                          schema=schema_spline)
elif num_joints == 13:
    dataset = customDatasets.eh36m_spline(data_path, pre_filter=hpe_filter, schema=schema_spline)
else:
    print("joint value doesn't have corresponding GT available. Exiting.")
    exit()
dataset = dataset.shuffle()
# print(dataset.get(10))
# show_vals = (dataset.multi_get([i for i in range(100)]))


# if visualise:
#     for i in range(1500):
#         data = dataset.get(i)
#         G = GraphVisualization(data)
#         img = G.create_image(show_gt=False)
#         cv2.imshow('augmented', img)
#         cv2.waitKey(1)


train_dataset, val_dataset = dataset_split(dataset, data_fraction)
# change_data(val_dataset)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
print('Dataloaders created')


model = hpegnn.hpeGnn_splineConv(dataset.num_features, hidden_channels, num_joints, learning_rate=learning_rate,
                                 batch_size=batch_size, visualise=visualise, image_size=image_size,
                                 data_fraction=data_fraction, label=label)
print('Model set up complete')

print('Setting up trainer')
if args.dev:
    trainer = pl.Trainer(fast_dev_run=3, enable_progress_bar=True)
else:
    early_stop_callback = EarlyStopping(monitor="loss/val_epoch", min_delta=0.00, patience=5, verbose=False, mode="min",
                                        check_finite=True)
    trainer = pl.Trainer(max_epochs=epochs, check_val_every_n_epoch=3, callbacks=[MyProgressBar(), early_stop_callback])
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
print('Trainer set up complete')

# out = trainer.test(model, dataloaders=test_loader, verbose=True)
# print(out)

# exit()
# Program for a random test with the same parameters
hidden_channels = []
hidden_channels.append([8, 32, 64, 64, 32, 8])
hidden_channels.append([8, 16, 128, 128, 16, 8])
hidden_channels.append([8, 16, 64, 64, 16, 8])
hidden_channels.append([8, 16, 32, 32, 16, 8])
hidden_channels.append([8, 16, 64, 128, 128, 64, 16, 8])
hidden_channels.append([8, 16, 32, 128, 128, 32, 16, 8])
hidden_channels.append([8, 16, 32, 64, 64, 32, 16, 8])
hidden_channels.append([8, 16, 32, 64, 128, 128, 64, 32, 16, 8])
for n in hidden_channels:
    # dataset = dataset.shuffle()
    # train_dataset, val_dataset = dataset_split(dataset, data_fraction)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

    model = hpegnn.hpeGnn_splineConv(dataset.num_features, n, num_joints, learning_rate=learning_rate,
                                     batch_size=batch_size, visualise=visualise, image_size=image_size,
                                     data_fraction=data_fraction, label=label)
    print('Model set up complete')

    print('Setting up trainer')
    if args.dev:
        trainer = pl.Trainer(fast_dev_run=3, enable_progress_bar=True)
    else:
        early_stop_callback = EarlyStopping(monitor="loss/val_epoch", min_delta=0.00, patience=5, verbose=False,
                                            mode="min",
                                            check_finite=True)
        trainer = pl.Trainer(max_epochs=epochs, check_val_every_n_epoch=3,
                             callbacks=[MyProgressBar(), early_stop_callback])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print('Trainer set up complete for hidden_channels ', n)

