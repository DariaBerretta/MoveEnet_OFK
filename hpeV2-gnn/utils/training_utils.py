# This file has mappings from command line arguments to their respective functions


import data.transforms as my_transforms
import os
import numpy as np


def connectivity_map(val: int):
    conn_map = {10: my_transforms.add_edges_10,
                12: my_transforms.add_edges_12,
                15: my_transforms.add_edges_15,
                20: my_transforms.add_edges_20,
                30: my_transforms.add_edges_30}
    return conn_map[val]


def task_map(val: str):
    task_mapping = {'center': my_transforms.center_detection,
                    'head': my_transforms.head_detection,
                    'right_hand': my_transforms.right_hand_detection,
                    'hands': my_transforms.hands_detection}
    return task_mapping[val]


def node_feature(val: str):
    feature_mapping = {'pos': my_transforms.use_pos_as_input,
                       'pos_fit': my_transforms.merge_pos_to_fit,
                       'pos_fit_lsg': my_transforms.merge_pos_to_input}
    return feature_mapping[val]


def test_ckpt_path(path):
    print(path)
    if not (os.path.isfile(path)):
        print("Checkpoint path is not a file.")

        if os.path.isdir(path):
            print("Checkpoint path is a folder! Testing if it's a label. Checking inside. ")
            files_0 = os.listdir(path)
            if len(files_0) > 0:
                if 'checkpoints' in files_0:
                    print('Might be a label folder.')
                    path_other = os.path.join(path,'checkpoints')
                    files_1 = os.listdir(path_other)
                    path = os.path.join(path_other, files_1[0])
                    print('Found a ckpt. Using that one.')
                    print(path)
                elif files_0[0].split('_')[0] == 'version':
                    versions = np.array([file.split('_')[1] for file in files_0])
                    latest = max(versions)
                    use_version = str('version_' + str(latest))
                    path = os.path.join(path, use_version, 'checkpoints')
                    if not (os.path.exists(path)):
                        print('No checkpoints folder inside. Exiting')
                        return None
                    else:
                        files_1 = os.listdir(path)
                        path = os.path.join(path, files_1[0])
                        print('Found a ckpt. Using that one.')
                        print(path)
                else:
                    print("Not a label folder. Checking if it's a checkpoints folder. ")

                    if path.split('/')[-1] == 'checkpoints':
                        files_1 = os.listdir(path)
                        print(path)
                        print(files_1)
                        path = os.path.join(path, files_1[0])
                        print('Found a ckpt. Using that one.')
                        print(path)
                    else:
                        print("Not a checkpoint folder either. Test your path. Exiting.")
                        return None
            else:
                print('Nothing inside folder. Test your path. Exiting.')
                return None
        else:
            print('Or a folder.')
            return None
    return path
