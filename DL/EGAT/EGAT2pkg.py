import dgl
from dgl.nn.pytorch import Set2Set, EGATConv
from dgl.data.utils import load_graphs

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse

import os
import time
from utils import *

parser = argparse.ArgumentParser(description='EGAT for PCE prediction')

parser.add_argument('--max_epochs', type =int, default=2000, help='Maximum number of epochs for training')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size of epochs for training')
parser.add_argument('--in_feats_don_node', type=int, default=45, help='Number of input features for donor nodes')
parser.add_argument('--in_feats_don_edge', type=int, default=10, help='Number of input features for donor edges')
parser.add_argument('--in_feats_acc_node', type=int, default=45, help='Number of input features for acceptor nodes')
parser.add_argument('--in_feats_acc_edge', type=int, default=10, help='Number of input features for acceptor edges')
parser.add_argument('--out_feats', type=int, default=45, help='Number of output features')
parser.add_argument('--n_hidden', type=int, default=128, help='Number of hidden units')
parser.add_argument('--n_classes', type=int, default=1, help='Number of output classes')
parser.add_argument('--n_layers', type=int, default=3, help='Number of GNN layers')
parser.add_argument('--n_step_s2s', type=int, default=2, help='Number of steps for Set2Set readout function')
parser.add_argument('--n_layer_s2s', type=int, default=1, help='Number of layers for Set2Set readout function')
parser.add_argument('--seed', type=int, default=21, help='Random seed')
parser.add_argument('--K', type=int, default=5, help='Number of folds for K-fold cross validation')
parser.add_argument('--cuda', type=int, default=0, help='Specify the ID of the CUDA device to use')

folder_name = 'n_loss_curve'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'navy', 'turquoise', 'cornflowerblue',
          'teal', 'salmon', 'purple', 'sienna', 'limegreen', 'olive', 'indigo', 'darkslategrey', 'peru']

def collate(samples):
    don_graphs, acc_graphs, PCEs = map(list, zip(*samples))
    don_graphs = dgl.batch(don_graphs)
    acc_graphs = dgl.batch(acc_graphs)
    return don_graphs, acc_graphs, PCEs

class GraphDataset(Dataset):
    def __init__(self, don_graphs, acc_graphs, PCEs):
        self.don_graphs = don_graphs
        self.acc_graphs = acc_graphs
        self.PCEs = PCEs
    
    def __len__(self):
        return len(self.PCEs)
    
    def __getitem__(self, item):
        don_graphs = self.don_graphs[item]
        acc_graphs = self.acc_graphs[item]
        PCEs = self.PCEs[item]
        return [don_graphs, acc_graphs, PCEs]
    
class EGAT(nn.Module):

    def __init__(self, in_node_feats, in_edge_feats, n_hidden,
                 out_node_feats, out_edge_feats, n_layers, n_step_s2s, n_layer_s2s):
        super(EGAT, self).__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.layers.append(EGATConv(in_node_feats=in_node_feats, in_edge_feats=in_edge_feats,
                                    out_node_feats=n_hidden, out_edge_feats=n_hidden, num_heads=4))
        self.bns.append(nn.BatchNorm1d(n_hidden))

        for l in range(n_layers):
            self.layers.append(EGATConv(in_node_feats = n_hidden//2**l,
                                        in_edge_feats = n_hidden//2**l,
                                        out_node_feats = n_hidden//2**(l+1),
                                        out_edge_feats = n_hidden//2**(l+1),
                                        num_heads = 4, bias = True))
            self.bns.append(nn.BatchNorm1d(n_hidden//2**(l+1)))

        self.layers.append(EGATConv(in_node_feats = n_hidden // 2**(n_layers-1),
                                    in_edge_feats = n_hidden // 2**(n_layers-1),
                                    out_node_feats = out_node_feats,
                                    out_edge_feats = out_edge_feats,
                                    num_heads = 4))
        
        self.readout = Set2Set(out_node_feats, n_iters = n_step_s2s, n_layers = n_layer_s2s)
        self.dropout = nn.Dropout(p = 0.5)
        self.n_layers = n_layers

    def forward(self, graph, node_feat, edge_feat):
        init = node_feat.clone()
        for i in range(self.n_layers):
            node_feat, edge_feat = self.layers[i](graph, node_feat, edge_feat)
            node_feat = torch.mean(node_feat, dim = 1, keepdim = False)
            edge_feat = torch.mean(edge_feat, dim = 1, keepdim = False)
            node_feat = self.dropout(node_feat)
            node_feat = self.bns[i](node_feat)

        node_feat,_ = self.layers[-1](graph, node_feat, edge_feat)
        node_feat = torch.mean(node_feat, dim = 1, keepdim = False)
        node_feat = node_feat + init
        graph_feat = self.readout(graph, node_feat)

        return graph_feat
    

class EGATModel(nn.Module):

    def __init__(self, in_feats_don_node, in_feats_don_edge,
                 in_feats_acc_node, in_feats_acc_edge, n_hidden,
                 out_feats, n_layers, n_step_s2s, n_layer_s2s, n_classes):
        super(EGATModel, self).__init__()

        self.in_feats_don_node = in_feats_don_node
        self.in_feats_don_edge = in_feats_don_edge
        self.in_feats_acc_node = in_feats_acc_node
        self.in_feats_acc_edge = in_feats_acc_edge
        self.n_hidden = n_hidden
        self.out_feats = out_feats
        self.n_layers = n_layers
        self.n_step_s2s = n_step_s2s
        self.n_layer_s2s = n_layer_s2s

        self.don_EGAT = EGAT(in_node_feats = in_feats_don_node, 
                             in_edge_feats = in_feats_don_edge,
                             n_hidden = n_hidden,
                             out_node_feats = in_feats_don_node,
                             out_edge_feats = in_feats_don_edge,
                             n_layers = n_layers,
                             n_step_s2s = n_step_s2s,
                             n_layer_s2s = n_layer_s2s)
        self.acc_EGAT = EGAT(in_node_feats = in_feats_acc_node,
                             in_edge_feats = in_feats_acc_edge,
                             n_hidden = n_hidden,
                             out_node_feats = in_feats_acc_node,
                             out_edge_feats = in_feats_acc_edge,
                             n_layers = n_layers,
                             n_step_s2s = n_step_s2s,
                             n_layer_s2s = n_layer_s2s)
                            
        self.fc1 = nn.Linear(4*self.out_feats, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_hidden//4)
        self.fc3 = nn.Linear(self.n_hidden//4, n_classes)

    def forward(self, don_graph, acc_graph):

        don_node_feats = don_graph.ndata['x'].float()
        don_edge_feats = don_graph.edata['w'].float()
        acc_node_feats = acc_graph.ndata['x'].float()
        acc_edge_feats = acc_graph.edata['w'].float()

        don_graph_feats = self.don_EGAT(graph=don_graph, node_feat=don_node_feats, edge_feat=don_edge_feats)
        acc_graph_feats = self.acc_EGAT(graph=acc_graph, node_feat=acc_node_feats, edge_feat=acc_edge_feats)
        toal_features = torch.cat((don_graph_feats, acc_graph_feats), dim=1)

        predictions = torch.relu(self.fc1(toal_features))
        predictions = torch.relu(self.fc2(predictions))
        predictions = self.fc3(predictions)

        return predictions


def train(model, optimizer, data_loader, loss_fn, device):
    model.train()
    epoch_loss = 0
    for i, (dg, ag, label) in enumerate(data_loader):
        dg = dg.to(device)
        ag = ag.to(device)
        prediction = model(dg, ag)
        loss = loss_fn(prediction.view(-1), torch.tensor(label, dtype=torch.float32, device=device).clone().detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (i+1)
    return epoch_loss

def evaluate(model, data_loader, loss_fn,device):
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        for i, (dg, ag, label) in enumerate(data_loader):
            dg = dg.to(device)
            ag = ag.to(device)
            prediction = model(dg, ag)
            loss = loss_fn(prediction.view(-1), torch.tensor(label, dtype=torch.float32, device=device).clone().detach())
            epoch_loss += loss.detach().item()
        epoch_loss /= (i+1)
    return epoch_loss

def main():

    start_time = time.time()

    general_don_graphs = load_graphs('train_don.dgl')[0]
    general_acc_graphs = load_graphs('train_acc.dgl')[0]

    data = pd.read_excel('train_dataset.xlsx')

    general_PCEs = data.iloc[:, 2].values.tolist()
    np.save('general_PCEs.npy',np.array(general_PCEs))
    general_PCEs = np.load('general_PCEs.npy').tolist()
    general_PCEs = label_normalization(general_PCEs)

    general_don_graphs = [dgl.add_self_loop(g) for g in general_don_graphs]
    general_acc_graphs = [dgl.add_self_loop(g) for g in general_acc_graphs]
    general_PCEs = np.array(general_PCEs)

    total_dataset = GraphDataset(don_graphs=general_don_graphs, acc_graphs=general_acc_graphs, PCEs= general_PCEs)

    print(len(general_don_graphs), len(general_acc_graphs), len(general_PCEs), len(total_dataset))

    args = parser.parse_args()
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    in_feats_don_node = args.in_feats_don_node
    in_feats_don_edge = args.in_feats_don_edge
    in_feats_acc_node = args.in_feats_acc_node
    in_feats_acc_edge = args.in_feats_acc_edge
    out_feats = args.out_feats
    n_hidden = args.n_hidden
    n_classes = args.n_classes
    n_layers = args.n_layers
    n_step_s2s = args.n_step_s2s
    n_layer_s2s = args.n_layer_s2s
    seed = args.seed
    K = args.K
    cuda_device = args.cuda
    torch.manual_seed(seed)

    device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')
    model = EGATModel(in_feats_don_node=in_feats_don_node, in_feats_don_edge=in_feats_don_edge,
                     in_feats_acc_node=in_feats_acc_node, in_feats_acc_edge=in_feats_acc_edge,
                     n_hidden=n_hidden, out_feats=out_feats, n_layers=n_layers,
                     n_step_s2s=n_step_s2s, n_layer_s2s=n_layer_s2s, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')
    avg_val_loss = 0

    train_losses_list = []
    val_losses_list =[]
    max_train_len = 0
    max_val_len = 0

    for fold in range(K):
        train_index = [i for i in range(len(total_dataset)) if i % K != fold]
        val_index = [i for i in range(len(total_dataset)) if i % K == fold]
        train_dataset = [total_dataset[i] for i in train_index]
        val_dataset = [total_dataset[i] for i in val_index]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate, shuffle=True)

        train_losses = []
        val_losses = []
        patience = 0

        for epoch in range(max_epochs):
            train_loss = train(model, optimizer, train_loader, loss_fn, device)
            train_losses.append(train_loss)
            val_loss = evaluate(model, val_loader, loss_fn, device)
            val_losses.append(val_loss)
            scheduler.step()

            if val_loss < best_val_loss:
                torch.save(model.state_dict(),'best_model.pth')
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
            
            print(f'Fold {fold + 1:02d}, Epoch {epoch + 1:03d}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}')

            if patience == 70:
                print('Early stopping!')
                break

        plt.plot(train_losses, label=f'fold {fold+1} train loss', color=colors[fold % len(colors)])
        plt.plot(val_losses, label=f'fold {fold+1} val loss', color=colors[fold+1 % len(colors)])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        file_path = os.path.join(folder_name, f'fold_{fold+1}_loss_curve.png')
        plt.savefig(file_path)

        train_losses_list.append(train_losses)
        val_losses_list.append(val_losses)

        max_train_len = max(max_train_len, len(train_losses))
        max_val_len = max(max_val_len, len(val_losses))

        avg_val_loss += val_loss / K

    ##
    for i in range(len(train_losses_list)):
        train_losses_list[i] += [np.nan] * (max_train_len - len(train_losses_list[i]))
    
    for i in range(len(val_losses_list)):
        val_losses_list[i] +=[np.nan] * (max_val_len - len(val_losses_list[i]))
    
    train_losses_array = np.array(train_losses_list)
    np.save('train_losses.npy', train_losses_array)
    val_losses_array = np.array(val_losses_list)
    np.save('val_losses.npy', val_losses_array)

    plt.figure()
    for i in range(K):
        plt.plot(train_losses_list[i], label=f'fold{i+1} train loss', color=colors[i % len(colors)])
        plt.plot(val_losses_list[i], label=f'fold{i+1} val loss', color=colors[i+1 % len(colors)])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    file_path = os.path.join(folder_name, 'all_folds_loss.png')
    plt.savefig(file_path)

    print(f'Average validation loss: {avg_val_loss:.4f}')

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'Processing time: {int(hours)}h {int(minutes)}min {seconds:.2f}s')


if __name__ == '__main__':
    main()
        
