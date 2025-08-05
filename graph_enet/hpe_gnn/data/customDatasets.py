import os
import time
import os.path as osp
from os.path import join
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data import OnDiskDataset, Dataset
from torch_geometric.transforms import Cartesian
from torchmetrics.functional import pairwise_euclidean_distance as dist
import csv
import re
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Union, Iterable, Optional, List, Sequence
from torch_geometric.data.data import BaseData
import math
import copy

import data.h36m_utils as h36m
import utils.eval_utils as eval

IndexType = Union[slice, Tensor, np.ndarray, Sequence]

class batchData(Data):
    # This should contain the names of all keys in the graph that do not need to be re-indexed for batching
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'y':
            return None
        if key == 'th_pck':
            return None
        if key == 'ts':
            return None
        if key == 'sample':
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)
class eh36m_gcn(OnDiskDataset):
    def __init__(self, root, transform=None, pre_filter=None, backend='sqlite', schema=object, log=True):
        super().__init__(root, transform, pre_filter, backend, schema, log)
        self.db.schema = schema
        self.transform = transform
        self.log = log

    @property
    def raw_file_names(self):
        root = join(self.root,'raw')
        files = [join(direct, os.listdir(join(root, direct))[0]) for direct in os.listdir(root)]
        # files = ['cam2_S1_Directions/gamer.csv','cam2_S1_Directions_1/gamer.csv','cam2_S1_Discussion/gamer.csv']
        return files

    @property
    def processed_file_names(self):
        # name_list = [('data'+str(i)+'.pt')for i in range(10)]
        return f'{self.backend}.db'

    def get_delimiter(self):
        return ' '

    def get_gt_pattern(self):
        return re.compile('\d* (\d*.\d*) SKLT \((.*)\) (-?\d*.\d*) (\d*.\d*)')

    def index_select(self, idx: IndexType):
        r"""Creates a subset of the dataset from specified indices :obj:`idx`.
        Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
        list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
        long or bool.
        """
        indices = self.indices()
        if isinstance(idx, slice):
            start, stop, step = idx.start, idx.stop, idx.step
            # Allow floating-point slicing, e.g., dataset[:0.9]
            if isinstance(start, float):
                start = round(start * len(self))
            if isinstance(stop, float):
                stop = round(stop * len(self))
            idx = slice(start, stop, step)

            indices = indices[idx]

        elif isinstance(idx, Tensor) and idx.dtype == torch.long:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Tensor) and idx.dtype == torch.bool:
            idx = idx.flatten().nonzero(as_tuple=False)
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == np.int64:
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, np.ndarray) and idx.dtype == bool:
            idx = idx.flatten().nonzero()[0]
            return self.index_select(idx.flatten().tolist())

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            indices = [indices[i] for i in idx]

        else:
            raise IndexError(
                f"Only slices (':'), list, tuples, torch.tensor and "
                f"np.ndarray of dtype long or bool are valid indices (got "
                f"'{type(idx).__name__}')")

        dataset = copy.copy(self)
        dataset._indices = indices
        return dataset

    def get_torso_diameter(self, keypoints):
        kps = np.reshape(keypoints, [-1, 2])

        if len(keypoints) // 2 == 13:
            left_hip = kps[5, :]
            right_hip = kps[6, :]
            left_shoulder = kps[1, :]
            right_shoulder = kps[2, :]
            diagonal_1 = math.dist(left_hip, right_shoulder)
            diagonal_2 = math.dist(left_shoulder, right_hip)
            torso_diameter = np.mean([diagonal_1, diagonal_2])
            return torso_diameter
        else:
            return 0

    def download(self):
        pass

    def get(self, idx: int) -> BaseData:
        r"""Gets the data object at index :obj:`idx`."""
        data = self.deserialize(self.db.get(idx))
        # if self.transform is not None:
        #     data = self.transform(data)
        return data

    def multi_get(
        self,
        indices: Union[Iterable[int], Tensor, slice, range],
        batch_size: Optional[int] = None,
    ) -> List[batchData]:
        r"""Gets a list of data objects from the specified indices."""
        if len(indices) == 1:
            data_list = [self.db.get(indices[0])]
        else:
            data_list = self.db.multi_get(indices, batch_size)

        data_list = [self.deserialize(data) for data in data_list]
        if self.transform is not None:
            data_list = [self.transform(data) for data in data_list]
        return data_list

    def serialize(self, data: batchData) -> Dict[str, Any]:
        return data.to_dict()

    def deserialize(self, data: Dict[str, Any]) -> batchData:
        return batchData.from_dict(data)

    def process(self):
        # Read data into huge `Data` list. Data_list should be a list of graph Data objects
        # pre_transform is where we define how to read the data from these files, and pair the input with the GT
        log = self.log
        data_list: List[batchData] = []
        if log:  # pragma: no cover
            raw_paths = tqdm(self.raw_paths, desc='Converting to OnDiskDataset')

        for idx, file in enumerate(raw_paths):
            samples = self.read_files(file)
            if samples is None:
                print(f"Sample {file} being skipped due to incompatibility of the GT file.")
                continue

            for sample in samples:
                nodes = []
                label = []
                node_index = 0
                segments = sample['segments']
                if len(segments)<=15:
                    continue
                segments = np.asarray(segments).reshape(-1,5)
                edge_index0 = []
                edge_index1 = []
                edge_attr = []
                node_attr = []
                for i in range(segments.shape[0]):
                    point0 = [int(segments[i,0]),int(segments[i,1])]
                    point1 = [int(segments[i,2]),int(segments[i,3])]
                    energy = float(segments[i,4])
                    point_index = [0,0]
                    j = 0
                    for point in [point0,point1]:
                        if point in nodes:
                            point_index[j] = nodes.index(point)
                            node_attr[point_index[j]] += energy
                        else:
                            nodes.append(point)
                            node_attr.append(energy)
                            point_index[j] = node_index
                            node_index += 1
                        j+=1
                    edge_index0.extend([point_index[0],point_index[1]])
                    edge_index1.extend([point_index[1],point_index[0]])
                    # segment_length = self.get_distance(np.asarray(point0),np.asarray(point1))
                    edge_attr.extend([energy, energy])

                # Temporary fix for DHP19, not needed if the gamer is updated from the dataset again, where this bug has been fixed.
                y = np.reshape(sample['anno'], [-1, 2])
                y = y[:, 1::-1]
                sample['anno'] = np.reshape(y, [-1])

                th_pck = [self.get_torso_diameter(sample['anno'])]
                # label = (sample['anno'][0:2]).tolist()
                label = (sample['anno']).tolist()

                pos = torch.tensor(nodes, dtype=torch.float)
                y = torch.tensor(label, dtype=torch.float)
                th_pck = torch.tensor(th_pck, dtype=torch.float)
                node_attr = torch.tensor(node_attr, dtype=torch.float)
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)

                edge_index = torch.tensor([edge_index0,
                                           edge_index1], dtype=torch.long)

                data = batchData(x=node_attr, y=y, edge_index=edge_index, edge_attr=edge_attr, pos=pos, th_pck=th_pck)
                data.validate(raise_on_error=True)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                data_list.append(data)
            if len(data_list) !=0:

                self.extend(data_list)
                data_list = []


    def read_files(self, filename):
        filename_anno = self.get_anno_name(filename)
        anno_ts,anno_sk = self.read_anno(filename_anno)
        if anno_ts is None:
            return None

        anno_count = 0
        delimiter = self.get_delimiter()
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file,delimiter=delimiter)
            samples = []

            for line in csv_reader:

                while abs(anno_ts[anno_count] - float(line[0]))>0.008:
                    anno_count +=1
                    if anno_count >= len(anno_ts):
                        break
                if len(line)<3:
                    continue
                if anno_count >= len(anno_ts):
                    break
                sample = {}
                sample['ts'] = float(line[0])
                sample['segments'] =[float(i) for i in line[1:-1]]
                sample['anno'] = anno_sk[anno_count]
                sample['sample_name'] = self.get_sample_name(filename)
                samples.append(sample)

        return samples

    def get_sample_name(self, filename):
        return filename.split('/')[-2]

    def get_anno_name(self, filename):
        sample = self.get_sample_name(filename)
        filename_anno = '/' + join(*filename.split('/')[:-4], 'EV2', sample, 'ch0GT50Hzskeleton/data.log')
        return filename_anno


    def get_distance(self, p, q):
        """
        Return euclidean distance between points p and q
        assuming both to have the same number of dimensions
        """
        # sum of squared difference between coordinates
        s_sq_difference = 0
        for p_i, q_i in zip(p, q):
            s_sq_difference += (p_i - q_i) ** 2

        # take sq root of sum of squared difference
        distance = s_sq_difference ** 0.5
        return distance

    def read_anno(self, filename_anno):
        with open(filename_anno) as f:
            content = f.readlines()
        if (len(content) == 0):
            raise Exception("No file, or no file content")
        # line format is 'id timestamp SKLT (<flattened 2D keypoints coordinates>) head_size torso_size'
        pattern = self.get_gt_pattern()
        pattern2 = re.compile('\d* \d*e-\d* SKLT \((.*)\) (-?\d*.\d*) (\d*.\d*)')

        try:
            tss, points, _, _ = pattern.findall(content[0])[0]
        except:
            print("Dataset", filename_anno, "does not match pattern")
            print("required: [# TS SKLT (int x26) head torso]")
            print("got     :", content[0])
            return None, None

        anno_ts = np.zeros((len(content)), dtype=float)
        anno_sk = np.zeros((len(content), 26), dtype=float)
        for li, line in enumerate(content):
            try:
                anno_ts[li], points, _, _ = pattern.findall(line)[0]
            except IndexError:
                continue
            points = np.array(list(filter(None, points.split(' ')))).astype(int)
            anno_sk[li, :] = [int(str(i).replace("\n", "")) for i in points]
        return anno_ts, anno_sk


class eh36m_spline_gamer(eh36m_gcn):
    def __init__(self, root, transform=None, pre_filter=None, backend='sqlite', schema=object, log=True, max_value=1):
        self.max_value = max_value
        super().__init__(root, transform, pre_filter, backend, schema, log)
        self.db.schema = schema
        self.transform = transform
        self.log = log

    def set_features(self):
            self.node_features = 5

    def process(self):

        # Read data into huge `Data` list. Data_list should be a list of graph Data objects
        # pre_transform is where we define how to read the data from these files, and pair the input with the GT
        log = self.log
        data_list: List[batchData] = []
        edge_attr_cartesian = Cartesian(norm=True, cat=False)
        if log:  # pragma: no cover
            raw_paths = tqdm(self.raw_paths, desc='Converting to OnDiskDataset')
        self.set_features()
        for idx, file in enumerate(raw_paths):
            samples = self.read_files(file)
            if samples is None:
                print(f"Sample {file} being skipped due to incompatibility of the GT file.")
                continue
            start_sample = time.time()    
            for sample in samples:
                nodes = []
                label = []
                node_index = 0
                segments = sample['segments']
                if len(segments)<=15:
                    continue
                segments = np.asarray(segments).reshape(-1, self.node_features)
                edge_index0 = []
                edge_index1 = []
                node_attr = []
                for i in range(segments.shape[0]):
                    point0 = [int(segments[i,0]),int(segments[i,1])]
                    point1 = [int(segments[i,2]),int(segments[i,3])]
                    node_info = np.array(segments[i,4:]).astype(float)

                    point_index = [0,0]
                    j = 0
                    for point in [point0,point1]:
                        if point in nodes:
                            point_index[j] = nodes.index(point)
                            # node_attr[point_index[j]] += node_info
                        else:
                            nodes.append(point)
                            node_attr.extend(node_info)
                            point_index[j] = node_index
                            node_index += 1
                        j+=1
                    edge_index0.extend([point_index[0],point_index[1]])
                    edge_index1.extend([point_index[1],point_index[0]])
                    # segment_length = self.get_distance(np.asarray(point0),np.asarray(point1))

                # Temporary fix for DHP19, not needed if the gamer is updated from the dataset again, where this bug has been fixed.
                # y = np.reshape(sample['anno'], [-1, 2])
                # y = y[:, 1::-1]
                # sample['anno'] = np.reshape(y, [-1])

                th_pck = [self.get_torso_diameter(sample['anno'])]
                # label = (sample['anno'][0:2]).tolist()
                label = (sample['anno']).tolist()

                pos = torch.tensor(nodes, dtype=torch.float)
                y = torch.tensor(label, dtype=torch.float)
                th_pck = torch.tensor(th_pck, dtype=torch.float)
                node_attr = torch.tensor(np.array(node_attr), dtype=torch.float)

                edge_index = torch.tensor([edge_index0,
                                           edge_index1], dtype=torch.long)

                ts = torch.tensor(sample['ts'], dtype=torch.float)
                sample_embedding = torch.tensor(h36m.name_to_embedding(sample['sample_name']), dtype=torch.long)
                data = batchData(x=node_attr, y=y, edge_index=edge_index, pos=pos, th_pck=th_pck, ts=ts, sample=sample_embedding)
                data = edge_attr_cartesian(data)
                data.validate(raise_on_error=True)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                data_list.append(data)

            delay=time.time() - start_sample
            # Create csv output
            row = eval.create_row_time_graph_making(sample_name= sample.get('sample_name'), No_of_graph=len(data_list), delay = delay)
            # # Create a eval folder containing output data
            file_name ='total_time_making_graph.csv' #TODO:  self.__name__ +
            write_path = os.path.join('run_time', file_name)
            eval.ensure_loc(os.path.dirname(write_path))
            eval.write_results(write_path, row)
            if len(data_list) !=0:

                self.extend(data_list)
                data_list = []

class eh36m_spline_ledge(eh36m_spline_gamer):
    def __init__(self, root, transform=None, pre_filter=None, backend='sqlite', schema=object, log=True):
        super().__init__(root, transform, pre_filter, backend, schema, log)
        self.db.schema = schema
        self.transform = transform
        self.log = log
        self.node_features = None

    def set_features(self):
        self.node_features = 6
