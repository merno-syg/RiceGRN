import argparse
import os.path as osp
import pickle
import random
from math import ceil
from sklearn.metrics import roc_auc_score
import numpy as np
# from gtrick.pyg import VirtualNode
from matplotlib import pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ModuleList, ReLU, Sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.nn import BatchNorm, PNAConv, global_add_pool, SAGPooling, GraphNorm, GPSConv, GINEConv, \
    TransformerConv, dense_diff_pool, DenseSAGEConv
from torch_geometric.nn import GINConv, JumpingKnowledge, GCNConv, Sequential, SAGEConv, GATConv, PNAConv, SimpleConv, \
    GraphConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set, TopKPooling
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
import warnings
from model import *

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


def draw_fig(ylist, name, epoch=200, savename='PNA', namelist=None, show=False, savedir='./plots'):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('AGG')
    x1 = range(1, epoch + 1)
    plt.figure(figsize=(20, 10))
    plt.xlabel('epoch', fontsize=30)
    plt.ylabel(f'{name}', fontsize=30)
    plt.title(f'{name}', fontsize=30)
    if len(ylist) == epoch and not namelist:
        plt.plot(x1, ylist, label=f'{name}', linewidth=3.0)
    elif len(ylist[0]) == epoch and namelist:
        for i in range(len(ylist)):
            plt.plot(x1, ylist[i], label=f'{namelist[i]}', linewidth=3.0)
    else:
        print('Error! Pleas check your input ylist.')
    plt.legend(prop={'size': 20})
    os.makedirs(savedir, exist_ok=True)
    savepath = osp.join(savedir, f"{savename}.png")
    plt.savefig(savepath, dpi=600)
    if show:
        plt.show()


def train_one_epoch(model, optimizer, criterion, loader, device):
    model.to(device)
    criterion.to(device)
    model.train()
    total_loss = 0
    correct = 0
    all_probs = []
    all_labels = []
    
    for i, data in enumerate(loader):
        data = data.to(device, non_blocking=True)
        y = data.y.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        
        # 计算概率
        probs = F.softmax(output, dim=1)[:, 1]  # 取正类的概率
        all_probs.append(probs.detach().cpu())
        all_labels.append(y.cpu())
        
        loss = criterion(output, y.long().squeeze())
        loss.backward()
        total_loss += loss.item() * loader.batch_size
        optimizer.step()
        correct += ((output.argmax(dim=1) == y.squeeze()).sum())
    
    # 计算AUC
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5  # 如果只有一个类别出现时设为0.5
    
    return (total_loss / len(loader.dataset), 
            correct.item() / len(loader.dataset),
            auc)

# 修改评估函数
@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.to(device)
    total_loss = 0
    correct = 0
    all_probs = []
    all_labels = []
    model.eval()
    
    for i, data in enumerate(loader):
        data = data.to(device, non_blocking=True)
        y = data.y.to(device, non_blocking=True)
        output = model(data)
        
        # 计算概率
        probs = F.softmax(output, dim=1)[:, 1]  # 取正类的概率
        all_probs.append(probs.detach().cpu())
        all_labels.append(y.cpu())
        
        loss = criterion(output, y.long().squeeze())
        correct += ((output.argmax(dim=1) == y.squeeze()).sum())
        total_loss += loss.item() * data.num_graphs
    
    # 计算AUC
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    
    return (total_loss / len(loader.dataset), 
            correct.item() / len(loader.dataset),
            auc)


# @torch.no_grad()
# def eval_one_epoch(model, loader, criterion, device):
#     model.to(device)
#     total_loss = 0
#     correct = 0
#     model.eval()
#     for i, data in enumerate(loader):
#         data = data.to(device, non_blocking=True)
#         y = data.y.to(device, non_blocking=True)
#         output = model(data)
#         loss = criterion(output, y.long().squeeze())
#         correct += ((output.argmax(dim=1) == y.squeeze()).sum())
#         total_loss += loss.item() * data.num_graphs
#     return total_loss / len(loader.dataset), correct.item() / len(loader.dataset)


def compute_deg(train_loader):
    from torch_geometric.utils import degree
    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_loader:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_loader:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg

def pkl_load(dataset_path):
    start = time.perf_counter()
    with open(dataset_path, 'rb') as f:
        dat = pickle.load(f)
    end = time.perf_counter()
    print(f"Data loading {(end - start):.4f}s")

    return dat


def correct_data(datalist, in_fea):
    if len(datalist[0]) > 1:
        datalist = datalist[0]
    ok_list = []
    for d in datalist:
        print(d)
        if d.x.shape[1] == in_fea:
            ok_list.append(d)
    return ok_list

def compute_deg(train_loader):
    from torch_geometric.utils import degree
    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_loader:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_loader:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess the raw protein structures in PDB format.")
    parser.add_argument("--project_dir", default="/mnt/data2024/hrren/Settle_code/kmer-data", type=str,
                        help="Directory for storing output datasets. Default: `./`.")
    parser.add_argument("--name", default="grn", type=str,
                        help="Default: grn.")
    parser.add_argument("--cpu", default=16, type=int, help="CPU processors. Default: `16`.")
    parser.add_argument("--epoch", default=300, type=int, help="Number of training epochs. Default: `300`.")
    parser.add_argument("--batch", default=64, type=int, help="Batch size. Default: `64`.")
    parser.add_argument("--conv", default='gcn', type=str,
                        help="GNN conv operators used in model. Default: `gcn`.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU number. Default:0.")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate. Default:1e-3.")
    parser.add_argument("--layer", default=5, type=int, help="GNN layer. Default:5.")
    parser.add_argument("--hidden", default=512, type=int, help="GNN hidden size. Default:512.")
    parser.add_argument("--dropout", default=0.5, type=float, help="Default:0.5.")
    parser.add_argument("--seed", default=42, type=int, help="random seed value. Default:42.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    seed_value = args.seed  # 设定随机数种子

    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

    torch.manual_seed(seed_value)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)  # 为所有GPU设置随机种子（多块GPU）

    project_dir = args.project_dir
    name = args.name
    dataset = ['train', 'val', 'test']
    train_path = osp.join(project_dir, f"{dataset[0]}.pkl")
    val_path = osp.join(project_dir, f"{dataset[1]}.pkl")
    test_path = osp.join(project_dir, f"{dataset[2]}.pkl")
    conv_type = args.conv
    batch_size = args.batch
    hidden_channels = args.hidden
    num_layers = args.layer
    cpus = args.cpu
    gpu = args.gpu
    num_epochs = args.epoch
    lr = args.lr
    dropout = args.dropout
    prefetch_factor = 2
    num_classes = 2
    train_data_list = pkl_load(train_path)
    train_loader = DataLoader(train_data_list, batch_size=batch_size, pin_memory=True, num_workers=cpus,
                              prefetch_factor=prefetch_factor, persistent_workers=True, shuffle=True)
    print(f"train dataset length: {len(train_data_list)} link subgraphs.")
    val_data_list = pkl_load(val_path)
    val_loader = DataLoader(val_data_list, batch_size=batch_size, pin_memory=True, num_workers=cpus,
                            prefetch_factor=prefetch_factor, persistent_workers=True, shuffle=True)
    print(f"val loader length: {len(val_data_list)} link subgraphs.")
    test_data_list = pkl_load(test_path)
    test_loader = DataLoader(test_data_list, batch_size=batch_size, pin_memory=True, num_workers=cpus,
                             prefetch_factor=prefetch_factor, persistent_workers=True, shuffle=True)
    print(f"test loader length: {len(test_data_list)} link subgraphs.")

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    if conv_type == 'diff':
        model = DiffNet(in_channels=test_data_list[0].x.shape[1], hidden_channels=hidden_channels,
                        out_channels=num_classes)
    elif conv_type == 'topk':
        model = TopkNet(in_fea=113, hidden_channels=hidden_channels,
                        out_channels=num_classes)
    elif conv_type == 'pna':
        model = PNANet(deg=compute_deg(train_loader), in_fea=test_data_list[0].x.shape[1], hidden_channels=hidden_channels, out_dim=num_classes, edge_dim=test_data_list[0].edge_attr.shape[1])
    else:
        model = GNN(in_fea=test_data_list[0].x.shape[1], hidden_channels=hidden_channels, num_layers=num_layers,
                    dropout=dropout, conv_type=conv_type, out_channels=num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                  min_lr=0.00001)

    train_loss_list = []
    train_acc_list = []
    train_auc_list = []
    val_acc_list = []
    val_auc_list = []
    best_acc = 0
    
    for epoch in range(1, num_epochs + 1):
        # 训练阶段
        start = time.perf_counter()
        train_loss, train_acc, train_auc = train_one_epoch(model, optimizer, criterion, train_loader, device)
        end = time.perf_counter()
        
        # 验证阶段
        start_val = time.perf_counter()
        val_loss, val_acc, val_auc = eval_one_epoch(model, val_loader, criterion, device)
        end_val = time.perf_counter()
        
        # 记录指标
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        train_auc_list.append(train_auc)
        val_acc_list.append(val_acc)
        val_auc_list.append(val_auc)
        
        # 打印结果
        print(f"Train | {(end - start):.4f}s | Epoch {epoch} | "
              f"Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | AUC: {train_auc:.4f}")
        print(f"Valid | {(end_val - start_val):.4f}s | Epoch {epoch} | "
              f"Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")
        
        # 测试阶段（每50个epoch）
        if epoch % 50 == 0:
            test_loss, test_acc, test_auc = eval_one_epoch(model, test_loader, criterion, device)
            print(f"Test | Epoch {epoch} | "
                  f"Acc: {test_acc:.4f} | AUC: {test_auc:.4f}")
        
        # 保存最佳模型
        if epoch > 50 and val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'./{name}_model.pth')
    
    # 绘制包含AUC的图表
    draw_fig([train_loss_list, train_acc_list, train_auc_list, val_acc_list, val_auc_list], 
             name='Metrics', 
             epoch=num_epochs,
             savename=f'{name}_with_auc',
             namelist=['Train Loss', 'Train Acc', 'Train AUC', 'Val Acc', 'Val AUC'])