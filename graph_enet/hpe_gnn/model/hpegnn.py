#
import torch, sys, time
import pytorch_lightning as pl
from torch.nn import Linear
from torch_geometric.nn import GCNConv, max_pool, voxel_grid
from torch_geometric.utils import softmax
from torch_geometric.nn.pool.pool import pool_batch, pool_edge, pool_pos
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.transforms import Cartesian
import torch.nn.functional as F
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.data import Data
from torch_geometric.nn.pool import max_pool_x
from typing import List, Optional, Tuple, Union
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import os

sys.path.append('..')
import cv2

from torch.nn.functional import elu
from torch_geometric.nn.conv import SplineConv
from graph_enet.hpe_gnn.utils.model_utils import GraphVisualization
from graph_enet.hpe_gnn.utils.metrics import pck_error, mpjpe_error
import graph_enet.hpe_gnn.utils.eval_utils as eval
import graph_enet.hpe_gnn.data.h36m_utils as h36m


class hpegnn(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, learning_rate=0.01, batch_size=1,
                 visualise=None, image_size=None, save_video=None, write_csv=None,file_name_eval = None, pck_multiplier=0.6):
        super(hpegnn, self).__init__()

        # super().__init__(*args, **kwargs)
        if image_size is None:
            image_size = [640, 480]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.joints = out_channels
        self.image_size = image_size
        self.save_hyperparameters()
        self.visualise = visualise
        self.pck_multiplier = pck_multiplier
        #self.count = 0
        if save_video is not None:
            self.save_video = True
            file_path = save_video
            # file_path = '/home/ggoyal/data/h36m_cropped/videos/test.mp4'
            frame_width = image_size[0]
            frame_height = image_size[1]
            fps = 30
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.output = cv2.VideoWriter(file_path, fourcc, fps, (frame_width, frame_height))
            print('saving video')
        else:
            self.save_video = False
        if write_csv is not None:
            self.write_path = write_csv
            self.write_csv = True
        else:
            self.write_csv=False

        if file_name_eval is not None:
            self.file_name = file_name_eval
            self.write_csv = True 
        else:
            self.write_csv= False
            
    def forward(self, x_in, edge_index, edge_attr=None, pos=None, batch=None):
        return NotImplementedError

    def training_step(self, data, batch_idx):
        loss, pck, mpjpe = self.basic_step(data)
        metrics = {"loss/train": loss, "pck/train": pck, "mpjpe/train": mpjpe}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=data.batch_size)
        # self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size=self.batch_size)
        # self.log_dict({'loss': loss}, batch_size=data.num_graphs)

        return {'loss': loss}

    def validation_step(self, data, batch_idx):
        loss, pck, mpjpe = self.basic_step(data)
        metrics = {"loss/val": loss, "pck/val": pck, "mpjpe/val": mpjpe}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=data.batch_size)

        return {'loss': loss}

    def predict_step(self, data, batch_idx):
        # ALl the metrics calculated in this function are stored and averaged for the output of trainer.test
        start_sample = time.time()
        out, data.contrib = self.forward(data.x, data.edge_index, data.edge_attr, data.pos, data.batch)
        if self.write_csv:
            kps_gamer = torch.reshape(out, [-1, 2])
            kps_gamer = torch.flip(kps_gamer,dims=(1,))
            kps_gamer = torch.reshape(kps_gamer, [-1])            
            # print(kps_gamer)
            # Create csv output
            row = eval.create_row(kps_gamer.cpu().numpy(), ts=data.ts.cpu().numpy(), delay=time.time() - start_sample)
            # print(row)
            # Create a eval folder containing output data
            base_dir = os.path.join(self.write_path, 'eval')
            # print(data.sample.squeeze())
            # print(data.sample.squeeze().shape)
            sample = h36m.embedding_to_name(data.sample.squeeze().cpu().numpy())
            # file_name ='hpeGnn_test.csv' #TODO:  self.__name__ +
            write_path = os.path.join(base_dir, sample, self.file_name)
            eval.ensure_loc(os.path.dirname(write_path))
            eval.write_results(write_path, row)
            # G.create_image(show_image=True, show_gt=True, show_pred=True, joints=self.joints, show_avg=True)
        G = GraphVisualization(data, pred=out, res=self.image_size, visualise=self.visualise)
        if self.visualise == 'pose':
            image = G.create_image(show_image=self.visualise, show_gt=True, show_pred=True, joints=self.joints)
            # image = G.visualise_vectors(show_image=self.visualise, show_gt=True, show_pred=True, joints=self.joints)
            # edge_mask = G.visualize()
            # print(edge_mask)
            if self.save_video:
                self.output.write(image)
                # cv2.imwrite('/home/cpham-iit.local/data/h36m_full/videos/cam2_S9_Directions/image_{}.jpg'.format(self.count), image)
                # self.count +=1
        elif self.visualise == 'vectors-head':
            # G = GraphVisualization(data, pred = out, visualise=self.visualise)
            image = G.visualise_vectors(show_image=self.visualise, show_gt=True, show_pred=True, joints=self.joints, histogram = False)
            if self.save_video:
                self.output.write(image)
        elif self.visualise == 'vectors-handR':
            # G = GraphVisualization(data, pred = out, visualise=self.visualise)
            image = G.visualise_vectors(show_image=self.visualise, show_gt=True, show_pred=True, joints=self.joints, histogram = False)
            if self.save_video:
                self.output.write(image)
        elif self.visualise == None:
            # G = GraphVisualization(data, pred=out)
            image = G.create_image(show_image=False, show_gt=True, show_pred=True, joints=self.joints)
            if self.save_video:
                self.output.write(image)

        return None

    def on_predict_end(self) -> None:
        if self.save_video:
            self.output.release()

    def target_loss(self, x_out, y):
        return F.mse_loss(x_out, y)
    
    def centre_of_body_loss(self, x_out, y):
        return F.mse_loss(torch.mean(x_out), torch.mean(y))

    def basic_step(self, data):
        out, _ = self.forward(data.x, data.edge_index, data.edge_attr, data.pos, data.batch)
        y = torch.reshape(data.y, (data.num_graphs, -1))
        loss = self.target_loss(out, y)
        pck = pck_error(y, out, data.th_pck, self.pck_multiplier)
        mpjpe = mpjpe_error(y, out)
        return loss, pck, mpjpe

    def custom_dot(self, x_in, pos_in, batch):
        # joints = x_in.shape[1]/2
        # print(joints)
        out_batch = torch.zeros(len(torch.unique(batch)), self.joints * 2, dtype=torch.float, device=x_in.device)
        for j in torch.unique(batch):
            x = x_in[batch == j]
            pos = pos_in[batch == j]
            out_xs = torch.zeros(self.joints * 2).type_as(x)
            for i in range(self.joints):
                out_xs[2 * i] = torch.dot(x[:, 2 * i], pos[:, 0])
                out_xs[2 * i + 1] = torch.dot(x[:, 2 * i + 1], pos[:, 1])
            out_batch[j, :] = out_xs
        return out_batch

    def check_bounds(self, pred):
        out = 0 <= pred[0] <= self.image_size[1]
        out = 0 <= pred[1] <= self.image_size[0] and out
        return out

    def configure_optimizers(self):
        for i, child in enumerate(self.children()):
            if i == 0:
                params = [dict(params=child.parameters(), weight_decay=5e-4)]
            else:
                params.append(dict(params=child.parameters(), weight_decay=0))
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)  # Only perform weight-decay on first convolution.
        scheduler = ReduceLROnPlateau(optimizer, patience=3, cooldown=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/train"
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def test_step(self, data, batch_idx):
        return self.predict_step(data, batch_idx)

    def on_test_end(self) -> None:
        self.on_predict_end()

class hpegnn_gcnconv(hpegnn):

    def __init__(self, in_channels, hidden_channels, out_channels, learning_rate=0.01, batch_size=1,
                 visualise=False, image_size=None):
        super(hpegnn_gcnconv, self).__init__()

        # super().__init__(*args, **kwargs)
        if image_size is None:
            image_size = [640, 480]
        self.conv1 = GCNConv(in_channels, hidden_channels[0], cached=True, normalize=False)
        self.conv_layers = list()
        for i in range(len(hidden_channels) - 1):
            self.conv_layers.append(GCNConv(hidden_channels[i], hidden_channels[i + 1], cached=True, normalize=False))
        # self.conv3 = GCNConv(hidden_channels[1], out_channels, cached=True, normalize=False)
        self.conv3 = GCNConv(hidden_channels[-1], 2 * out_channels, cached=True, normalize=False)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.joints = out_channels
        self.image_size = image_size
        self.save_hyperparameters()
        self.visualise = visualise

    def forward(self, x_in, edge_index, edge_attr=None, pos=None, batch=None):
        x = x_in[:, None]
        # x = F.dropout(x, p=0.2, training=self.training)
        x = elu(self.conv1(x, edge_index, edge_attr))
        device = next(self.conv1.parameters()).device
        x = F.dropout(x, p=0.1, training=self.training)
        for conv in self.conv_layers:
            conv.to(device)
            x = elu(conv(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        # x = self.custom_dot(x, pos, batch)
        # cluster = torch.zeros(len(x))
        # data = Data(x=x, pos=pos)
        # x = self.custom_pool(data)

        # x = gap(x, batch=None)
        x_out = self.custom_dot(x, pos, batch)
        return x_out, x

    def configure_optimizers(self):
        params = [dict(params=self.conv1.parameters(), weight_decay=5e-4)]
        params.extend([dict(params=conv.parameters(), weight_decay=0) for conv in self.conv_layers])
        params.append(dict(params=self.conv3.parameters(), weight_decay=0))

        optimizer = torch.optim.Adam(params, lr=self.learning_rate)  # Only perform weight-decay on first convolution.
        return optimizer

class dummy_model(hpegnn):
    # Usage: model = hpegnn.dummy_model(save_video=video_path)
    def __init__(self, in_channels=None, hidden_channels=None, out_channels=13, save_video=None):
        super(dummy_model, self).__init__(in_channels, hidden_channels, out_channels)
        self.joints = out_channels
        if save_video is not None:
            self.save_video = True
            file_path = save_video
            # file_path = '/home/ggoyal/data/h36m_cropped/videos/test.mp4'
            frame_width = 640
            frame_height = 480
            fps = 30
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.output = cv2.VideoWriter(file_path, fourcc, fps, (frame_width, frame_height))
            print('saving video')
        else:
            self.save_video = False


    def predict_step(self, data, batch_idx):
        G = GraphVisualization(data, pred=None)
        image = G.create_image(show_image=True, show_gt=True, show_pred=False, joints=self.joints)

        if self.save_video:
            self.output.write(image)
        cv2.imshow('', image)
        cv2.waitKey(1)
        return None



class spline_gnn(pl.LightningModule):
    def __init__(self, in_channels, hidden, out_channels, dim, kernel_size, bias=False, root_weight=False, dropout=None):
        super(spline_gnn, self).__init__()
        self.conv1 = SplineConv(in_channels, hidden, dim=dim, kernel_size=kernel_size, bias=bias,
                               root_weight=root_weight)
        self.norm1 = BatchNorm(in_channels=hidden)
        self.conv2 = SplineConv(hidden, hidden, dim=dim, kernel_size=kernel_size, bias=bias,
                                root_weight=root_weight)
        self.norm2 = BatchNorm(in_channels=hidden)
        self.conv3 = SplineConv(hidden, out_channels, dim=dim, kernel_size=kernel_size, bias=bias,
                                root_weight=root_weight)
        self.norm3 = BatchNorm(in_channels=out_channels)
        self.dropout = dropout
        if self.dropout is not None:
            if self.dropout > 1 or self.dropout <= 0:
                # print("Dropout value out of bounds (0,1). Skipping dropout")
                self.dropout = None

    def forward(self, x, edge_index, edge_attr=None):

        x = elu(self.conv1(x, edge_index, edge_attr))
        x = self.norm1(x)
        x = elu(self.conv2(x, edge_index, edge_attr))
        x = self.norm2(x)
        x = elu(self.conv3(x, edge_index, edge_attr))
        x = self.norm3(x)
        if self.dropout is not None:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class spline_module(pl.LightningModule):
    def __init__(self, in_channels, out_channels, dim, kernel_size, bias=False, root_weight=False, dropout=None):
        super(spline_module, self).__init__()
        self.conv = SplineConv(in_channels, out_channels, dim=dim, kernel_size=kernel_size, bias=bias,
                               root_weight=root_weight)
        self.norm = BatchNorm(in_channels=out_channels)
        self.dropout = dropout
        if self.dropout is not None:
            if self.dropout > 1 or self.dropout <= 0:
                # print("Dropout value out of bounds (0,1). Skipping dropout")
                self.dropout = None

    def forward(self, x, edge_index, edge_attr=None):

        x = elu(self.conv(x, edge_index, edge_attr))
        x = self.norm(x)
        if self.dropout is not None:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class hpeGnn_splineConv(hpegnn):
    def __init__(self, in_channels, hidden_channels, out_channels, learning_rate=0.01, batch_size=1,
                 visualise=False, image_size=None, label=None, data_fraction=1, task='head', transforms=None,
                 node_loss_weight=None, save_video=None, exp_setup=None, write_csv=None, file_name_eval = None, pck_multiplier=0.6):
        super().__init__(in_channels, hidden_channels, out_channels, learning_rate, batch_size, visualise, image_size,
                         save_video, write_csv, file_name_eval, pck_multiplier)
        if image_size is None:
            image_size = [640, 480]
        dim = 2
        kernel_size = 2
        # bias = False
        # root_weight = False
        n = len(hidden_channels)
        self.batch_size = batch_size
        self.image_size = torch.Tensor(image_size)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.joints = out_channels
        self.label = label
        self.data_fraction = data_fraction
        self.task = task
        self.transforms = transforms
        self.exp_setup = exp_setup
        self.node_loss_weight = node_loss_weight

        last_hidden = hidden_channels[0]
        self.spline1 = spline_module(in_channels, hidden_channels[0], dim=dim, kernel_size=kernel_size, dropout=0.4)
        if n > 1:
            last_hidden = hidden_channels[1]
            self.spline2 = spline_module(hidden_channels[0], last_hidden, dim=dim, kernel_size=kernel_size, dropout=0)
        if n > 2:
            last_hidden = hidden_channels[2]
            self.spline3 = spline_module(hidden_channels[1], last_hidden, dim=dim, kernel_size=kernel_size, dropout=0)
        if n > 3:
            last_hidden = hidden_channels[3]
            self.spline4 = spline_module(hidden_channels[2], last_hidden, dim=dim, kernel_size=kernel_size, dropout=0)
        if n > 4:
            last_hidden = hidden_channels[4]
            self.spline5 = spline_module(hidden_channels[3], last_hidden, dim=dim, kernel_size=kernel_size, dropout=0)
        if n > 5:
            last_hidden = hidden_channels[5]
            self.spline6 = spline_module(hidden_channels[4], last_hidden, dim=dim, kernel_size=kernel_size, dropout=0)

        if n > 6:
            last_hidden = hidden_channels[6]
            self.spline7 = spline_module(hidden_channels[5], last_hidden, dim=dim, kernel_size=kernel_size, dropout=0)

        if n > 7:
            last_hidden = hidden_channels[7]
            self.spline8 = spline_module(hidden_channels[6], last_hidden, dim=dim, kernel_size=kernel_size, dropout=0)
        if n > 8:
            last_hidden = hidden_channels[8]
            self.spline9 = spline_module(hidden_channels[7], last_hidden, dim=dim, kernel_size=kernel_size, dropout=0)

        if n > 9:
            last_hidden = hidden_channels[9]
            self.spline10 = spline_module(hidden_channels[8], last_hidden, dim=dim, kernel_size=kernel_size, dropout=0)

        self.spline_last = spline_module(last_hidden, out_channels * 4, dim=dim, kernel_size=kernel_size, dropout=0)
        self.pck_multiplier = pck_multiplier
        self.save_hyperparameters()
        self.visualise = visualise
        self.save_video = save_video

    def custom_vect_dot(self, x_in, pos_in, batch):
        idx = torch.stack(
            (torch.arange(0, 4 * self.joints, 4), torch.arange(1, 4 * self.joints, 4))).t().flatten().tolist()
        # print(idx)
        x_node_vect = x_in[:, idx]
        idx = torch.stack(
            (torch.arange(2, 4 * self.joints, 4), torch.arange(3, 4 * self.joints, 4))).t().flatten().tolist()
        # print(idx)
        x_node_weight = x_in[:, idx]
        out_batch = torch.zeros(len(torch.unique(batch)), self.joints * 2, dtype=torch.float, device=x_in.device)
        for j in torch.unique(batch):
            x_vect = x_node_vect[batch == j].reshape(-1,self.joints,2) * self.image_size
            x_weight = x_node_weight[batch == j]#.reshape(-1,self.joints,2)
            pos = pos_in[batch == j]
            x_vect = x_vect.movedim(0,1)
            new_pos = x_vect + pos
            new_pos = new_pos.movedim(0,1)
            # out_xs = torch.zeros(self.joints * 2).type_as(x_in)
            new_pos = new_pos.reshape(-1, self.joints * 2)
            out_xs = torch.mul(x_weight,new_pos).sum(0)
            out_batch[j, :] = out_xs
        return out_batch

    def custom_vect_pool(self, x_in, pos_in, batch):
        out_batch = torch.zeros(len(torch.unique(batch)), self.joints * 2, dtype=torch.float, device=x_in.device)
        for j in torch.unique(batch):
            x = x_in[batch == j]
            pos = pos_in[batch == j]
            out_xs = torch.zeros(self.joints * 2).type_as(x)
            for i in range(self.joints):
                out_xs[2 * i] = torch.mean(pos[:, 0] + x[:, 4 * i]*self.image_size[0])
                out_xs[2 * i + 1] = torch.mean(pos[:, 1] + x[:, 4 * i + 1]*self.image_size[1])
            out_batch[j, :] = out_xs
        return out_batch

    def node_vect_loss(self, y, x_node=None, x_in=None, batch=None):
        if x_node is None or x_in is None:
            print('Invalid setup. Check for correct parameters being bpassed to the loss function. Exiting.')
            exit()
        loss = 0
        idx = torch.stack(
            (torch.arange(0, 4 * self.joints, 4), torch.arange(1, 4 * self.joints + 1, 4))).t().flatten().tolist()
        x_node_selected = x_node[:, idx]
        for j in torch.unique(batch):
            x = x_node_selected[batch == j]
            pos = x_in[:, 0:2][batch == j]
            pos = pos.unsqueeze(1).expand(-1,self.joints,-1)
            nodes = len(x)
            x = x.reshape(-1,self.joints,2)*self.image_size
            out_xs = x + pos
            out_xs = out_xs.reshape(-1, self.joints*2)
            loss += torch.mean(F.mse_loss(out_xs, torch.broadcast_to(y[j, :], (nodes, self.joints * 2))))
        loss = loss / len(torch.unique(batch))
        return loss

    def custom_softmax(self, x, index):
        idx = torch.stack(
            (torch.arange(2, 4 * self.joints + 2, 4), torch.arange(3, 4 * self.joints + 3, 4))).t().flatten().tolist()
        x[:, idx] = softmax(x[:, idx], index=index)
        return x

    def custom_sigmoid(self, x, index):
        idx = torch.stack(
            (torch.arange(0, 4 * self.joints, 4), torch.arange(1, 4 * self.joints + 1, 4))).t().flatten().tolist()
        x[:, idx] = (torch.sigmoid(x[:, idx])*2)-1
        return x
    
    def basic_step(self, data):
        out, x_node = self.forward(data.x, data.edge_index, data.edge_attr, batch=data.batch)
        y = torch.reshape(data.y, (data.num_graphs, -1))
        target_loss = self.target_loss(out, y)
        node_loss = 0
        if self.node_loss_weight != None:
            node_loss = self.node_vect_loss(y, x_node=x_node, x_in=data.x, batch=data.batch)
        if isinstance(self.node_loss_weight, list):
            loss = (self.node_loss_weight[0]*target_loss) + (self.node_loss_weight[1]*node_loss)
        else:
            loss = node_loss + target_loss
        pck = pck_error(y, out, data.th_pck, self.pck_multiplier)
        mpjpe = mpjpe_error(y, out)
        return loss, pck, mpjpe

    def forward(self, x_in, edge_index, edge_attr=None, pos=None, batch=None):
        """Forward.

        Args:
            x_in: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        self.image_size = self.image_size.to(self.device)
        # x = x[:, None]
        x = torch.clone(x_in)
        for layer in self.children():
            x = layer(x, edge_index, edge_attr)
        x = self.custom_softmax(x, index=batch)
        x = self.custom_sigmoid(x, index=batch)
        x_out = self.custom_vect_dot(x, x_in[:, 0:2], batch)
        return x_out, x

class hpeGnn_splineConv_single_weight(hpegnn):
    def __init__(self, in_channels, hidden_channels, out_channels, learning_rate=0.01, batch_size=1,
                 visualise=False, image_size=None, label=None, data_fraction=1, task='head', transforms=None,
                 node_loss_weight=None, save_video=None, exp_setup=None, write_csv=None, file_name_eval = None, pck_multiplier=0.6):
        super().__init__(in_channels, hidden_channels, out_channels, learning_rate, batch_size, visualise, image_size,
                         save_video, write_csv, file_name_eval, pck_multiplier)
        if image_size is None:
            image_size = [640, 480]
        dim = 2
        kernel_size = 2
        # bias = False
        # root_weight = False
        n = len(hidden_channels)
        self.batch_size = batch_size
        self.image_size = torch.Tensor(image_size)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.joints = out_channels
        self.label = label
        self.data_fraction = data_fraction
        self.task = task
        self.transforms = transforms
        self.exp_setup = exp_setup
        self.node_loss_weight = node_loss_weight

        last_hidden = hidden_channels[0]
        self.spline1 = spline_module(in_channels, hidden_channels[0], dim=dim, kernel_size=kernel_size, dropout=0.4)
        if n > 1:
            last_hidden = hidden_channels[1]
            self.spline2 = spline_module(hidden_channels[0], last_hidden, dim=dim, kernel_size=kernel_size, dropout=0)
        if n > 2:
            last_hidden = hidden_channels[2]
            self.spline3 = spline_module(hidden_channels[1], last_hidden, dim=dim, kernel_size=kernel_size, dropout=0)
        if n > 3:
            last_hidden = hidden_channels[3]
            self.spline4 = spline_module(hidden_channels[2], last_hidden, dim=dim, kernel_size=kernel_size, dropout=0)
        if n > 4:
            last_hidden = hidden_channels[4]
            self.spline5 = spline_module(hidden_channels[3], last_hidden, dim=dim, kernel_size=kernel_size, dropout=0)
        if n > 5:
            last_hidden = hidden_channels[5]
            self.spline6 = spline_module(hidden_channels[4], last_hidden, dim=dim, kernel_size=kernel_size, dropout=0)

        if n > 6:
            last_hidden = hidden_channels[6]
            self.spline7 = spline_module(hidden_channels[5], last_hidden, dim=dim, kernel_size=kernel_size, dropout=0)

        if n > 7:
            last_hidden = hidden_channels[7]
            self.spline8 = spline_module(hidden_channels[6], last_hidden, dim=dim, kernel_size=kernel_size, dropout=0)
        if n > 8:
            last_hidden = hidden_channels[8]
            self.spline9 = spline_module(hidden_channels[7], last_hidden, dim=dim, kernel_size=kernel_size, dropout=0)

        if n > 9:
            last_hidden = hidden_channels[9]
            self.spline10 = spline_module(hidden_channels[8], last_hidden, dim=dim, kernel_size=kernel_size, dropout=0)

        self.spline_last = spline_module(last_hidden, out_channels * 4, dim=dim, kernel_size=kernel_size, dropout=0)
        self.pck_multiplier = pck_multiplier
        self.save_hyperparameters()
        self.visualise = visualise
        self.save_video = save_video

    def custom_vect_dot_single_weight_vector_old(self, x_in, pos_in, batch):
        # joints = x_in.shape[1]/2
        # print(joints)
        out_batch = torch.zeros(len(torch.unique(batch)), self.joints * 2, dtype=torch.float, device=x_in.device)
        for j in torch.unique(batch):
            x = x_in[batch == j]
            pos = pos_in[batch == j]
            # min_val = x.min(0).values
            # max_val = x.max(0).values
            # x = (x - min_val[None, :]) / (max_val[None, :] - min_val[None, :])
            out_xs = torch.zeros(self.joints * 2).type_as(x)
            # out_ys = torch.zeros(self.joints).type_as(x)
            for i in range(self.joints):
                out_xs[2 * i] = torch.dot(x[:, 4 * i + 2], pos[:, 0] + x[:, 4 * i]*self.image_size[0])
                out_xs[2 * i + 1] = torch.dot(x[:, 4 * i + 2], pos[:, 1] + x[:, 4 * i + 1]*self.image_size[1])
            out_batch[j, :] = out_xs
        return out_batch

    def custom_vect_dot_single_weight_vector(self, x_in, pos_in, batch):
        idx = torch.stack(
            (torch.arange(0, 4 * self.joints, 4), torch.arange(1, 4 * self.joints, 4))).t().flatten().tolist()
        # print(idx)
        x_node_vect = x_in[:, idx]
        # print(x_node_vect)
        idx = torch.stack(
            (torch.arange(2, 4 * self.joints, 4), torch.arange(2, 4 * self.joints, 4))).t().flatten().tolist()
        # print(idx)
        x_node_weight = x_in[:, idx]
        # print(x_node_weight)
        out_batch = torch.zeros(len(torch.unique(batch)), self.joints * 2, dtype=torch.float, device=x_in.device)
        for j in torch.unique(batch):
            x_vect = x_node_vect[batch == j].reshape(-1,self.joints,2) * self.image_size
            x_weight = x_node_weight[batch == j]#.reshape(-1,self.joints,2)
            pos = pos_in[batch == j]
            x_vect = x_vect.movedim(0,1)
            new_pos = x_vect + pos
            new_pos = new_pos.movedim(0,1)
            # out_xs = torch.zeros(self.joints * 2).type_as(x_in)
            new_pos = new_pos.reshape(-1, self.joints * 2)
            out_xs = torch.mul(x_weight,new_pos).sum(0)
            out_batch[j, :] = out_xs
        return out_batch
    
    def custom_vect_pool(self, x_in, pos_in, batch):
        out_batch = torch.zeros(len(torch.unique(batch)), self.joints * 2, dtype=torch.float, device=x_in.device)
        for j in torch.unique(batch):
            x = x_in[batch == j]
            pos = pos_in[batch == j]
            out_xs = torch.zeros(self.joints * 2).type_as(x)
            for i in range(self.joints):
                out_xs[2 * i] = torch.mean(pos[:, 0] + x[:, 4 * i]*self.image_size[0])
                out_xs[2 * i + 1] = torch.mean(pos[:, 1] + x[:, 4 * i + 1]*self.image_size[1])
            out_batch[j, :] = out_xs
        return out_batch

    def node_vect_loss(self, y, x_node=None,x_in=None, batch=None):
        if x_node is None or x_in is None:
            print('Invalid setup. Check for correct parameters being bpassed to the loss function. Exiting.')
            exit()
        loss = 0
        out_batch = torch.zeros(len(torch.unique(batch)), self.joints * 2, dtype=torch.float, device=x_in.device)
        for j in torch.unique(batch):
            x = x_node[batch == j]
            pos = x_in[:, 0:2][batch == j]
            nodes = len(x)
            out_xs = torch.zeros((pos.shape[0],self.joints*2)).type_as(x)
            for i in range(self.joints):
                # print(pos.shape)
                # print(x.shape)
                out_xs[:, 2 * i] = pos[:, 0] + x[:, 4 * i]*self.image_size[0]
                out_xs[:, 2 * i + 1] = pos[:, 1] + x[:, 4 * i + 1]*self.image_size[1]
            loss += torch.mean(F.mse_loss(out_xs, torch.broadcast_to(y[j,:], (nodes, self.joints*2))))
        loss = loss/len(torch.unique(batch))
        # final_loss = loss*0.01 + self.target_loss(x_out,y[j,:])
        return loss

    def custom_softmax(self, x, index):
        idx = torch.stack(
            (torch.arange(2, 4 * self.joints + 2, 4), torch.arange(3, 4 * self.joints + 3, 4))).t().flatten().tolist()
        x[:, idx] = softmax(x[:, idx], index=index)
        return x

    def custom_softmax_last(self, x, index):
            idx = torch.stack(
                (torch.arange(2, 4 * self.joints + 2, 4), torch.arange(3, 4 * self.joints + 3, 4))).t().flatten().tolist()
            # non_zero_indices = torch.where(x[:, idx] != 0)
            # non_zero_indices = torch.nonzero(x[:, idx], as_tuple = False)
            mask = (x[:, idx]!=0).float()
            softmax_result = softmax(x[:, idx] + (1 - mask) * -1e9, index = index)
            x[:, idx] = softmax_result * mask
            # output = output_flat.view(x.shape) 
            # x[:, non_zero_indices] = torch.exp(x[:, non_zero_indices]) / torch.sum(torch.exp(x[:, non_zero_indices]), axis = -1)
            # x[:, idx] = softmax(x[:, idx], index = index)
            return x
    
    def custom_sigmoid(self, x, index):
        idx = torch.stack(
            (torch.arange(0, 4 * self.joints, 4), torch.arange(1, 4 * self.joints + 1, 4))).t().flatten().tolist()
        x[:, idx] = (torch.sigmoid(x[:, idx])*2)-1
        return x

    def selective_pooling(self, x, index, a = 5, b = 1, step_func = True, custom_sigmoid = False, thres = 0.001):
        idx = torch.stack(
            (torch.arange(2, 4 * self.joints + 2, 4), torch.arange(3, 4 * self.joints + 3, 4))).t().flatten().tolist()
        #hard code set threshold to clustering weights from outside or inside person
        # top_contrib_nodes = int(0.5* len(x))
        if step_func:
            mask = x[:, idx] <= torch.tensor(thres, device='cuda:0') 
            x[:, idx] = torch.where(mask, torch.tensor(0, device='cuda:0'), x[:, idx])
        print('before slective pooling:', x[:, idx])
        if custom_sigmoid:
            out_xs = torch.zeros(self.joints * 2).type_as(x)
            # out_ys = torch.zeros(self.joints).type_as(x)
            for i in range(self.joints):
                # print('selective values: ', 1 / (1 + torch.exp(-20000 * x[:, 4 * i + 2] + 1e1)))
                # print('x min values: ', x[:, 4 * i + 2].max(0).values)
                x[:, 4 * i + 2] = 1 / (1 + torch.exp(-a * x[:, 4 * i + 2]) + b) * x[:, 4 * i + 2]
                x[:, 4 * i + 3] = 1 / (1 + torch.exp(-a * x[:, 4 * i + 3]) + b) * x[:, 4 * i + 3]
        # x[:, idx][mask] = 0  # Apply the mask to set values <= 0.02 to 0
        # print('After selective values: ', x[:, idx])
        return x
    
    def basic_step(self, data):
        out, x_node = self.forward(data.x, data.edge_index, data.edge_attr, batch=data.batch)
        y = torch.reshape(data.y, (data.num_graphs, -1))
        target_loss = self.target_loss(out, y)
        node_loss = 0
        if self.node_loss_weight != None:
            node_loss = self.node_vect_loss(y, x_node=x_node, x_in=data.x, batch=data.batch)
        if isinstance(self.node_loss_weight, list):
            loss = (self.node_loss_weight[0]*target_loss) + (self.node_loss_weight[1]*node_loss)
        else:
            loss = node_loss + target_loss
        # pck = pck_error(y, out, data.th_pck, self.pck_multiplier)                     # Inverted arguments y and out
        # mpjpe = mpjpe_error(y, out)                                                   # Inverted arguments y and out
        pck = pck_error(out, y, data.th_pck, self.pck_multiplier)
        mpjpe = mpjpe_error(out, y)
        return loss, pck, mpjpe

    def forward(self, x_in, edge_index, edge_attr=None, pos=None, batch=None):
        """Forward.

        Args:
            x_in: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        self.image_size = self.image_size.to(self.device)
        # x = x[:, None]
        x = torch.clone(x_in)
        for layer in self.children():                           # iterate through all SplineConv layers, here all the parameters of the nodes are used and updated
            x = layer(x, edge_index, edge_attr)
        x = self.custom_softmax(x, index=batch)
        x = self.custom_sigmoid(x, index=batch)
        # x = self.selective_pooling(x, index = batch)
        # x = self.custom_softmax_last(x, index=batch)
        x_out = self.custom_vect_dot_single_weight_vector(x, x_in[:, 0:2], batch)           # to predict the final 2D co-ordinates only the coordinate vectors params are used
        return x_out, x