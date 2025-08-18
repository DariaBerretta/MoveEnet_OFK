"""
@Fire
https://github.com/fire717
"""
home = '/data'
cfg = {
    ##### Global Setting
    'connectivity': 0, #'Added connection for pixel values [12, 15, 20, 30]
    'node_loss_weight': 1,
    'target_loss_weight': 1,
    'dev': False,
    'task': 'all',  # Task setup; [center, head, all, right_hand, hands]
    'node_feature': 'pos', # Experimental setup [pos, pos_fit, pos_fit_lsg]
    'arch': 'single_weight',
    'dataset': 'h36m',

    ##### Train Setting
    'label': 'default',
    'hidden': '8, 16, 64, 128, 128, 64, 16, 8',

    ##### Train Hyperparameters
    'learning_rate': 0.001,  # 1.25e-4
    'batch_size': 1024,
    'epochs': 10,
    'optimizer': 'Adam',  # Adam  SGD
    # 'scheduler': 'MultiStepLR-70,100-0.1',  # default  SGDR-5-2  CVPR   step-4-0.8 MultiStepLR
    'dataset_split': 'subject', #[subject, dev]. Dev will use 'fraction' ""with value 0.01."
    'data_fraction': 0.1, #'Fraction of dataset to be used [0.01,0.1,1]'

    ##### File paths
    'data_path': '/mnt/disk1/data/h36m/ledge/',
    'data_path_dev':'/home/usr/data/h36m/ledge_toy',
    'ckpt': None,
    'resume': None #Resume training from checkpoint PATH provided.
}

cfg_SCARF = {
    ##### Global Setting
    'connectivity': 0, # Graph fully pre-connected
    'node_loss_weight': 1,
    'target_loss_weight': 1,
    'dev': False,
    'task': 'all',  # Only HPE task supported
    'node_feature': 'scarf_feature', # Scarf_splineConv always have 10 node features
    'arch': 'single_weight',
    'dataset': 'scarfDataset_splineConv',

    ##### Train Setting
    'label': 'default',
    'hidden': '8, 16, 64, 128, 128, 64, 16, 8',

    ##### Train Hyperparameters
    'learning_rate': 0.001,  # 1.25e-4
    'batch_size': 1024,
    'epochs': 10,
    'optimizer': 'Adam',  # Adam  SGD
    # 'scheduler': 'MultiStepLR-70,100-0.1',  # default  SGDR-5-2  CVPR   step-4-0.8 MultiStepLR
    'dataset_split': 'dev', #[subject, dev]. Dev will use 'fraction' ""with value 0.01."
    'data_fraction': 0.1, #'Fraction of dataset to be used [0.01,0.1,1]'

    ##### File paths
    'data_path': '/home/dberretta-iit.local/data/new_scarfGNN',
    'data_path_dev':'/home/dberretta-iit.local/data/new_scarfGNN',
    'ckpt': None,
    'resume': None #Resume training from checkpoint PATH provided.
}