"""
@Fire
https://github.com/fire717
"""
home = '/data'
cfg = {
    ##### Global Setting
    'connectivity': 15, #'Added connection for pixel values [12, 15, 20, 30]
    'node_loss_weight': 0.001,
    'target_loss_weight': 1,
    'dev': False,
    'task': 'all',  # Task setup; [center, head, all, right_hand, hands]
    'node_feature': 'pos', # Experimental setup [pos, pos_fit, pos_fit_lsg]
    'arch': 'single_weight', # Architecture setup [single_weight, two_weights]
    'dataset': 'h36m',

    ##### Train Setting
    'label': 'ledge_dataset', # Label for the experiment (ckpt name)
    # 'hidden': '8, 16, 64, 128, 128, 64, 16, 8',
    'hidden': '8, 16, 64,128, 128, 64, 16, 8',

    ##### Train Hyperparameters
    'learning_rate': 0.01,  # 1.25e-4
    'batch_size': 128, # 64
    'epochs': 20,
    'optimizer': 'Adam',  # Adam  SGD
    # 'scheduler': 'MultiStepLR-70,100-0.1',  # default  SGDR-5-2  CVPR   step-4-0.8 MultiStepLR
    'dataset_split': 'dev', #[subject, dev]. Dev will use 'fraction' ""with value 0.1."
    'data_fraction': 1, #'Fraction of dataset to be used [0.01,0.1,1], when you have small dataset use higher fraction

    ##### File paths
    'data_path': '/home/dberretta-iit.local/data/LEDGE_eh36m_train',
    'data_path_dev':'/home/dberretta-iit.local/data/LEDGE_eh36m_train',
    'ckpt': None,
    'resume': None #Resume training from checkpoint PATH provided.
}

cfg_SCARF = {
    ##### Global Setting
    'connectivity': 0, # Graph fully pre-connected
    'node_loss_weight': 0.001,
    'target_loss_weight': 1,
    'dev': False,
    'task': 'all',  # Only HPE task supported
    'node_feature': 'scarf_feature', # Scarf_splineConv always have 10 node features
    'arch': 'single_weight', # Architecture setup [single_weight, two_weights]
    'dataset': 'scarfDataset_splineConv',

    ##### Train Setting
    'label': 'scarf_dataset',     # Label for the experiment (ckpt name)
    'hidden': '8, 16, 64, 128, 128, 64, 16, 8',

    ##### Train Hyperparameters
    'learning_rate': 0.01,  # 1.25e-4
    'batch_size': 128,
    'epochs': 20,
    'optimizer': 'Adam',  # Adam  SGD
    # 'scheduler': 'MultiStepLR-70,100-0.1',  # default  SGDR-5-2  CVPR   step-4-0.8 MultiStepLR
    'dataset_split': 'dev', #[subject, dev]. Dev will use 'fraction' ""with value 0.01."
    'data_fraction': 1, #'Fraction of dataset to be used [0.01,0.1,1], when you have small dataset use higher fraction'

    ##### File paths
    'data_path': '/home/dberretta-iit.local/data/new_scarfGNN_train',
    'data_path_dev':'/home/dberretta-iit.local/data/new_scarfGNN_train',
    'ckpt': None,
    'resume': None #Resume training from checkpoint PATH provided.
}