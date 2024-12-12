import dgl
import dgl.function as fn
from dgl.nn.pytorch import Set2Set, NNConv
from dgl.data.utils import load_graphs

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse

import os
import time
from utils import *

parser = argparse.ArgumentParser(description='GraphTransformer for PCE prediction')

parser.add_argument('--max_epochs', type =int, default=2000, help='Maximum number of epochs for training')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size of epochs for training')
parser.add_argument('--in_feats_don_node', type=int, default=45, help='Number of input features for donor nodes')
parser.add_argument('--in_feats_don_edge', type=int, default=10, help='Number of input features for donor edges')
parser.add_argument('--in_feats_acc_node', type=int, default=45, help='Number of input features for acceptor nodes')
parser.add_argument('--in_feats_acc_edge', type=int, default=10, help='Number of input features for acceptor edges')
parser.add_argument('--out_feats', type=int, default=128, help='Number of output features')
parser.add_argument('--n_hidden', type=int, default=128, help='Number of hidden units')
parser.add_argument('--n_classes', type=int, default=1, help='Number of output classes')
parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
parser.add_argument('--n_step_s2s', type=int, default=2, help='Number of steps for Set2Set readout function')
parser.add_argument('--n_layer_s2s', type=int, default=1, help='Number of layers for Set2Set readout function')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
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

# The model architecture of GraphTransformer is modified from https://github.com/BioinfoMachineLearning/DeepInteract

def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2)+tensor.size(-1))*tensor.var())
        tensor.data *= scale.sqrt()

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, num_input_feats, num_output_feats,
                 num_heads, using_bias=False, update_edge_feats=True):
        super(MultiHeadAttentionLayer, self).__init__()

        self.num_output_feats = num_output_feats
        self.num_heads = num_heads
        self.using_bias = using_bias
        self.update_edge_feats = update_edge_feats

        self.Q = nn.Linear(num_input_feats, self.num_output_feats*self.num_heads, bias=using_bias)
        self.K = nn.Linear(num_input_feats, self.num_output_feats*self.num_heads, bias=using_bias)
        self.V = nn.Linear(num_input_feats, self.num_output_feats*self.num_heads, bias=using_bias)
        self.edge_feats_projection = nn.Linear(num_input_feats, self.num_output_feats*self.num_heads, bias=using_bias)

        self.reset_parameters()

    def reset_parameters(self):
        scale = 2.0
        if self.using_bias:
            glorot_orthogonal(self.Q.weight, scale=scale)
            self.Q.bias.data.fill_(0)
            glorot_orthogonal(self.K.weight, scale=scale)
            self.K.bias.data.fill_(0)
            glorot_orthogonal(self.V.weight, scale=scale)
            self.V.bias.data.fill_(0)
            glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)
            self.edge_feats_projection.bias.data.fill_(0)
        else:
            glorot_orthogonal(self.Q.weight, scale=scale)
            glorot_orthogonal(self.K.weight, scale=scale)
            glorot_orthogonal(self.V.weight, scale=scale)
            glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)

    def propagate_attention(self, g):
        g.apply_edges(lambda edges: {'score':edges.src['K_h']*edges.dst['Q_h']})
        g.apply_edges(lambda edges: {'score':(edges.data['score']/np.sqrt(self.num_output_feats)).clamp(-5.0, 5.0)})
        g.apply_edges(lambda edges: {'score':edges.data['score']*edges.data['proj_e']})
        if self.update_edge_feats:
            g.apply_edges(lambda edges: {'e_out': edges.data['score']})

        g.apply_edges(lambda edges: {'score': torch.exp((edges.data['score'].sum(-1,keepdim=True)).clamp(-5.0, 5.0))})
        g.update_all(fn.u_mul_e('V_h','score','V_h'), fn.sum('V_h','wV'))
        g.update_all(fn.copy_e('score','score'), fn.sum('score','z'))

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            e_out = None
            node_feats_q = self.Q(node_feats)
            node_feats_k = self.K(node_feats)
            node_feats_v = self.V(node_feats)
            edge_feats_projection = self.edge_feats_projection(edge_feats)

            g.ndata['Q_h'] = node_feats_q.view(-1, self.num_heads, self.num_output_feats)
            g.ndata['K_h'] = node_feats_k.view(-1, self.num_heads, self.num_output_feats)
            g.ndata['V_h'] = node_feats_v.view(-1, self.num_heads, self.num_output_feats)
            g.edata['proj_e'] = edge_feats_projection.view(-1, self.num_heads, self.num_output_feats)

            self.propagate_attention(g)

            h_out = g.ndata['wV']/(g.ndata['z']+torch.full_like(g.ndata['z'], 1e-6))

            if self.update_edge_feats:
                e_out = g.edata['e_out']

        return h_out, e_out


class GraphTransformerModule(nn.Module):

    def __init__(self, num_hidden_channels, num_layers, dropout_rate, 
                 activ_fn=nn.SiLU(), residual=True, num_attention_heads=4, norm_to_apply='batch'):
        super(GraphTransformerModule, self).__init__()

        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()

        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
        if self.apply_layer_norm:
            self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:
            self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.mha_module = MultiHeadAttentionLayer(self.num_hidden_channels, 
                                                  self.num_output_feats // self.num_attention_heads,
                                                  self.num_attention_heads,
                                                  self.num_hidden_channels != self.num_output_feats,
                                                  update_edge_feats = True)
        self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)
        self.O_edge_feats = nn.Linear(self.num_output_feats, self.num_output_feats)

        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([nn.Linear(self.num_output_feats, self.num_output_feats*2, bias=False),
                                             self.activ_fn, dropout, 
                                             nn.Linear(self.num_output_feats*2, self.num_output_feats, bias=False)])
        
        if self.apply_layer_norm:
            self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm2_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:
            self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm2_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.edge_feats_MLP = nn.ModuleList([nn.Linear(self.num_output_feats, self.num_output_feats*2, bias=False),
                                             self.activ_fn, dropout,
                                             nn.Linear(self.num_output_feats*2, self.num_output_feats, bias=False)])
        self.reset_parameters()

    def reset_parameters(self):
        scale = 2.0
        glorot_orthogonal(self.O_node_feats.weight, scale=scale)
        self.O_node_feats.bias.data.fill_(0)
        glorot_orthogonal(self.O_edge_feats.weight, scale=scale)
        self.O_edge_feats.bias.data.fill_(0)

        for layer in self.node_feats_MLP:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=scale)
        for layer in self.edge_feats_MLP:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=scale)

    def run_gt_layer(self, g, node_feats, edge_feats):

        node_feats_in1 = node_feats
        edge_feats_in1 = edge_feats

        if self.apply_layer_norm:
            node_feats = self.layer_norm1_node_feats(node_feats)
            edge_feats = self.layer_norm1_edge_feats(edge_feats)
        else:
            node_feats = self.batch_norm1_node_feats(node_feats)
            edge_feats = self.batch_norm1_edge_feats(edge_feats)

        node_attn_out, edge_attn_out = self.mha_module(g, node_feats, edge_feats)

        node_feats = node_attn_out.view(-1, self.num_output_feats)
        edge_feats = edge_attn_out.view(-1, self.num_output_feats)

        node_feats = F.dropout(node_feats, self.dropout_rate, training = self.training)
        edge_feats = F.dropout(edge_feats, self.dropout_rate, training = self.training)

        node_feats = self.O_node_feats(node_feats)
        edge_feats = self.O_edge_feats(edge_feats)

        if self.residual:
            node_feats = node_feats_in1 + node_feats
            edge_feats = edge_feats_in1 + edge_feats

        node_feats_in2 = node_feats
        edge_feats_in2 = edge_feats

        if self.apply_layer_norm:
            node_feats = self.layer_norm2_node_feats(node_feats)
            edge_feats = self.layer_norm2_edge_feats(edge_feats)
        else:
            node_feats = self.batch_norm2_node_feats(node_feats)
            edge_feats = self.batch_norm2_edge_feats(edge_feats)

        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)
        for layer in self.edge_feats_MLP:
            edge_feats = layer(edge_feats)

        if self.residual:
            node_feats = node_feats_in2 + node_feats
            edge_feats = edge_feats_in2 + edge_feats

        return node_feats, edge_feats
    
    def forward(self, g, node_feats, edge_feats):
        node_feats, edge_feats = self.run_gt_layer(g, node_feats, edge_feats)
        return node_feats, edge_feats
    

class FinalGraphTransformerModule(nn.Module):

    def __init__(self, out_node_feats, num_hidden_channels,
                 num_step_set2_set, n_layer_s2s, activ_fn = nn.SiLU(), residual = True,
                 num_attention_heads = 4, norm_to_apply = 'batch',
                 dropout_rate = 0.1, num_layers = 4):
        super(FinalGraphTransformerModule, self).__init__()

        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()

        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels

        self.readout = Set2Set(self.num_output_feats, n_iters = num_step_set2_set, n_layers = n_layer_s2s)

        if self.apply_layer_norm:
            self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:
            self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.mha_module = MultiHeadAttentionLayer(self.num_hidden_channels, 
                                                  self.num_output_feats // self.num_attention_heads,
                                                  self.num_attention_heads,
                                                  self.num_hidden_channels != self.num_output_feats,
                                                  update_edge_feats=False)
        self.O_node_feats = nn.Linear(self.num_output_feats, self.num_output_feats)

        dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([nn.Linear(self.num_output_feats, self.num_output_feats*2, bias=False),
                                             self.activ_fn, dropout,
                                             nn.Linear(self.num_output_feats*2, self.num_output_feats, bias=False)])
        if self.apply_layer_norm:
            self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
        else:
            self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)
        
        self.reset_parameters()

    def reset_parameters(self):
        scale =2.0
        glorot_orthogonal(self.O_node_feats.weight, scale=scale)
        self.O_node_feats.bias.data.fill_(0)
        
        for layer in self.node_feats_MLP:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=scale)

    def run_gt_layer(self, g, node_feats, edge_feats):

        node_feats_in1 = node_feats

        if self.apply_layer_norm:
            node_feats = self.layer_norm1_node_feats(node_feats)
            edge_feats = self.layer_norm1_edge_feats(edge_feats)
        else:
            node_feats = self.batch_norm1_node_feats(node_feats)
            edge_feats = self.batch_norm1_edge_feats(edge_feats)
        
        node_attn_out, _ = self.mha_module(g, node_feats, edge_feats)
        node_feats = node_attn_out.view(-1, self.num_output_feats)
        node_feats = F.dropout(node_feats, self.dropout_rate, training = self.training)
        node_feats = self.O_node_feats(node_feats)

        if self.residual:
            node_feats = node_feats_in1 + node_feats

        node_feats_in2 = node_feats

        if self.apply_layer_norm:
            node_feats = self.layer_norm2_node_feats(node_feats)
        else:
            node_feats = self.batch_norm2_node_feats(node_feats)

        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)

        if self.residual:
            node_feats = node_feats_in2 + node_feats

        return node_feats
    
    def forward(self, g, node_feats, edge_feats):
        node_feats = self.run_gt_layer(g, node_feats, edge_feats)
        graph_feat = self.readout(g, node_feats)

        return graph_feat
    

class DGLGraphTransformer(nn.Module):

    def __init__(self, in_feats_don_node, in_feats_don_edge,
                 in_feats_acc_node, in_feats_acc_edge, 
                 n_hidden, n_layers, out_feats,
                 n_step_s2s, n_layer_s2s, n_classes,
                 activ_fn = nn.SiLU(), transformer_residual = True,
                 num_attention_heads = 4, norm_to_apply = 'batch',
                 dropout_rate = 0.1, **kwargs):
        super(DGLGraphTransformer, self).__init__()

        self.activ_fn = activ_fn
        self.transformer_residual = transformer_residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.n_layers = n_layers
        self.out_feats = out_feats
        self.n_hidden = n_hidden
        
        self.don_node_encoder = nn.Linear(in_feats_don_node, n_hidden)
        self.don_edge_encoder = nn.Linear(in_feats_don_edge, n_hidden)
        self.acc_node_encoder = nn.Linear(in_feats_acc_node, n_hidden)
        self.acc_edge_encoder = nn.Linear(in_feats_acc_edge, n_hidden)

        num_intermediate_layers = max(0, n_layers-1)
        gt_block_modules = [GraphTransformerModule(num_hidden_channels=n_hidden,
                                                   num_layers=n_layers,
                                                   dropout_rate=dropout_rate,
                                                   activ_fn=activ_fn,
                                                   residual=True,
                                                   num_attention_heads=4,
                                                   norm_to_apply='batch')
                                                   for _ in range(num_intermediate_layers)]
        if n_layers > 0:
            gt_block_modules.extend([FinalGraphTransformerModule(out_node_feats=in_feats_don_node,
                                                                 num_hidden_channels=n_hidden,
                                                                 num_step_set2_set=n_step_s2s,
                                                                 n_layer_s2s=n_layer_s2s,
                                                                 activ_fn=nn.SiLU(),
                                                                 residual=True,
                                                                 num_attention_heads = 4,
                                                                 norm_to_apply = 'batch',
                                                                 dropout_rate = 0.1,
                                                                 num_layers = 4)])
        self.gt_block = nn.ModuleList(gt_block_modules)

        self.fc1 = nn.Linear(4*self.out_feats, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_hidden//4)
        self.fc3 = nn.Linear(self.n_hidden//4, n_classes)

    def forward(self, don_graph, acc_graph):

        don_node_feats = don_graph.ndata['x'].float()
        don_edge_feats = don_graph.edata['w'].float()
        acc_node_feats = acc_graph.ndata['x'].float()
        acc_edge_feats = acc_graph.edata['w'].float()

        don_node_feats = self.don_node_encoder(don_node_feats)
        don_edge_feats = self.don_edge_encoder(don_edge_feats)
        acc_node_feats = self.acc_node_encoder(acc_node_feats)
        acc_edge_feats = self.acc_edge_encoder(acc_edge_feats)

        for gt_layer in self.gt_block[:-1]:
            don_node_feats, don_edge_feats = gt_layer(don_graph, don_node_feats, don_edge_feats)
            acc_node_feats, acc_edge_feats = gt_layer(acc_graph, acc_node_feats, acc_edge_feats)

        don_graph_feats = self.gt_block[-1](don_graph, don_node_feats, don_edge_feats)
        acc_graph_feats = self.gt_block[-1](acc_graph, acc_node_feats, acc_edge_feats)
        total_features = torch.cat((don_graph_feats, acc_graph_feats), dim = 1)

        predictions = torch.relu(self.fc1(total_features))
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
    model = DGLGraphTransformer(in_feats_don_node=in_feats_don_node,
                                in_feats_don_edge=in_feats_don_edge,
                                in_feats_acc_node=in_feats_acc_node,
                                in_feats_acc_edge=in_feats_acc_edge,
                                n_hidden=n_hidden, 
                                n_layers=n_layers,
                                out_feats=out_feats,
                                n_step_s2s=n_step_s2s,
                                n_layer_s2s=n_layer_s2s,
                                n_classes=n_classes,
                                activ_fn=nn.SiLU(),
                                transformer_residual=True,
                                num_attention_heads=4,
                                norm_to_apply='batch',
                                dropout_rate=0.5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
    scheduler = CosineAnnealingLR(optimizer, T_max = max_epochs)
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
        



                





