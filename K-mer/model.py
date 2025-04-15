import argparse
import os.path as osp
import pickle
from math import ceil

import torch_geometric
from gtrick.pyg import VirtualNode
from matplotlib import pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data

from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool, SAGPooling, GraphNorm, GPSConv, GINEConv, \
    TransformerConv, dense_diff_pool, DenseSAGEConv
from torch_geometric.nn import GINConv, JumpingKnowledge, GCNConv, Sequential, SAGEConv, GATConv, PNAConv, SimpleConv, \
    GraphConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set, TopKPooling
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
import warnings

from torch_geometric.utils import to_dense_adj, to_dense_batch

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class GNN(torch.nn.Module):

    def __init__(self, in_fea, hidden_channels, num_layers, dropout, conv_type, out_channels=20, edge_dim=None):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.edge_dim = edge_dim

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.vns = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                if conv_type == 'gcn':
                    conv = GraphConv(in_fea, hidden_channels)
                elif conv_type == 'gin':
                    conv = GINConv(nn.Linear(in_fea, hidden_channels))
                elif conv_type == 'sage':
                    conv = SAGEConv(in_fea, hidden_channels, normalize=True)
                elif conv_type == 'gat':
                    conv = GATConv(in_fea, hidden_channels, edge_dim=edge_dim, dropout=dropout)
                elif conv_type == 'transformer':
                    conv = TransformerConv(in_fea, hidden_channels, edge_dim=edge_dim)
                bn = torch.nn.BatchNorm1d(hidden_channels)
                vn = VirtualNode(in_fea, hidden_channels, dropout=dropout)
            else:
                if conv_type == 'gcn':
                    conv = GraphConv(in_channels=hidden_channels, out_channels=hidden_channels)
                elif conv_type == 'gin':
                    conv = GINConv(nn.Linear(hidden_channels, hidden_channels))
                elif conv_type == 'sage':
                    conv = SAGEConv(hidden_channels, hidden_channels, normalize=True)
                elif conv_type == 'gat':
                    conv = GATConv(hidden_channels, hidden_channels, edge_dim=edge_dim, dropout=dropout)
                elif conv_type == 'transformer':
                    conv = TransformerConv(hidden_channels, hidden_channels, edge_dim=edge_dim)
                bn = torch.nn.BatchNorm1d(hidden_channels)
                vn = VirtualNode(hidden_channels, hidden_channels, dropout=dropout)
            self.vns.append(vn)
            self.convs.append(conv)
            self.batch_norms.append(bn)
        self.pool = SAGPooling(hidden_channels, 1e-4)
        self.mlp = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()
            self.batch_norms[i].reset_parameters()
            self.vns[i].reset_parameters()
        self.pool.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for i in range(self.num_layers):
            x, vx = self.vns[i].update_node_emb(x, edge_index, batch)
            if self.conv_type == 'gat' or self.conv_type == 'transformer':
                if self.edge_dim is not None:
                    x = self.convs[i](x, edge_index, edge_attr)
                else:
                    x = self.convs[i](x, edge_index)
            else:
                x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.dropout(F.relu(x), p=self.dropout)
        x, edge_index, edge_attr, batch, perm, select_output_weight = self.pool(x, edge_index, batch=batch)
        x = self.mlp(x)
        return x


class GNN_diff(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, self.conv1(x0, adj, mask).relu())
        x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        x3 = self.bn(3, self.conv3(x2, adj, mask).relu())

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = self.lin(x).relu()

        return x

class TopkNet(torch.nn.Module):
    def __init__(self, in_fea, hidden_channels, out_channels=2):
        super().__init__()

        self.conv1 = GraphConv(in_fea, hidden_channels)
        self.pool1 = SAGPooling(hidden_channels, 1e-4)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.pool2 = SAGPooling(hidden_channels, 1e-4)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.pool3 = SAGPooling(hidden_channels, 1e-4)

        self.mlp = nn.Linear(hidden_channels*2, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        return self.mlp(x)

class DiffNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        max_nodes = 500
        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN_diff(in_channels, hidden_channels, num_nodes)
        self.gnn1_embed = GNN_diff(in_channels, hidden_channels, hidden_channels, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN_diff(3 * hidden_channels, hidden_channels, num_nodes)
        self.gnn2_embed = GNN_diff(3 * hidden_channels, hidden_channels, hidden_channels, lin=False)

        self.gnn3_embed = GNN_diff(3 * hidden_channels, hidden_channels, hidden_channels, lin=False)

        self.lin1 = torch.nn.Linear(3 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data, mask=None):
        if isinstance(data, Data):
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
            if data.x is not None:
                x, mask = to_dense_batch(x, batch)
                x = x.to(torch.float32)
            adj = to_dense_adj(edge_index, batch).to(torch.float32)
        else:
            x = data.x
            adj = data.adj
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x


class PNANet(torch.nn.Module):
    def __init__(self, deg, in_fea, hidden_channels, out_dim, edge_dim, layers=2):
        super().__init__()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(layers):
            if _ == 0:
                conv = PNAConv(in_channels=in_fea, out_channels=hidden_channels,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=edge_dim, towers=1, pre_layers=1, post_layers=1,
                           divide_input=False)
                bn = BatchNorm(hidden_channels)
            else:
                conv = PNAConv(in_channels=hidden_channels, out_channels=hidden_channels,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=edge_dim, towers=4, pre_layers=1, post_layers=1,
                           divide_input=False)
                bn = BatchNorm(hidden_channels)
            self.convs.append(conv)
            self.batch_norms.append(bn)
        self.pool = SAGPooling(hidden_channels, 1e-4)
        self.mlp = nn.Sequential(Linear(hidden_channels, hidden_channels//2), ReLU(), Linear(hidden_channels//2, hidden_channels//4), ReLU(),
                              Linear(hidden_channels//4, out_dim))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
        x, edge_index, edge_attr, batch, perm, select_output_weight = self.pool(x, edge_index, batch=batch)
        return self.mlp(x)