# GNN for HPE

Setup for a Graph Neural Network with Gamer representation.

To use this setup, 
1. copy the data into `<<dataDirectory>>/raw`. 
2. Open main.py and change the `data_path` variable to `<<dataDirectory>>`. 
3. Run main.py

## Installation
To install PyG, follow the instructionfrom [here](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

For the setup, please install the following libraries:

torch
pytorch-lightning
torch_geometric
matplotlib
opencv_python
hpe_core
albumentations
torchvision
tensorboard
torch-cluster

## Dataset
- Please download the eH3.6M and and DHP19 dataset [link](https://istitutoitalianotecnologia.sharepoint.com/:f:/r/sites/ISAAC_EXT/Shared%20Documents/WP5_Y8Q4?csf=1&web=1&e=lNR4Lf) and organise the downloaded files as follows:
```
/data
├── datasets
│   ├── eH36M
│   │   ├── EV2       # Contains Ground True data
│   │   ├── ledge 
│   │   │   ├── raw   # Contains validation data
│   ├── dhp19
│   │   ├── EV2       # Contains Ground True data
│   │   ├── ledge 
│   │   │   ├── raw   # Contains validation data

```

## Training

- To train the GraphEnet model on eh3.6m and dhp19, run the following command: 

```
python3 main.py --node_loss_weight 0.001 --epoch 50 --data_path /data/datasets/dhp19/ledge/ --exp two_weights --dataset dhp19
```
## Evaluation
- To evaluate a single checkpoint, run the following command with --ckpt to specify the checkpoint to be evaluated:
``` 
python3 predict.py --data_path /data/datasets/xxxx/ledge/ --ckpt_path path/to/checkpoints/model
```

Two models are available, one per dataset, in the ckpts folder of the repository. 
``` 
python3 predict.py --data_path /data/datasets/dhp19/ledge/ --ckpt_path ckpts/single_weight_dhp19.ckpts --arch single_weight --dataset dhp19 --visualise pose
```



