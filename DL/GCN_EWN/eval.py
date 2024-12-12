import dgl
from dgl.data.utils import load_graphs

import torch
from torch.utils.data import DataLoader

from GCN_EWNpkg import GCN_EWNModel, GraphDataset, collate

import numpy as np
import pandas as pd

from utils import *

print(f'Loading model...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN_EWNModel(in_feats_don_node=45, in_feats_don_edge=10, 
                     in_feats_acc_node=45, in_feats_acc_edge=10,
                     n_hidden=128, out_feats=45, n_layers=2, 
                     n_step_s2s=2, n_layer_s2s=1, n_classes=1)

state_dict = torch.load('best_model.pth', map_location = device)
model.load_state_dict(state_dict)
model.to(device)

print(f'Loading data...')
test_don_graphs = load_graphs('test_don.dgl')[0]
test_acc_graphs = load_graphs('test_acc.dgl')[0]

data = pd.read_excel('test_dataset.xlsx')

don_sms = data.iloc[:, 4].values.tolist()
acc_sms = data.iloc[:, 3].values.tolist()

test_PCEs = data.iloc[:, 2].values.tolist()
np.save('test_PCEs.npy', np.array(test_PCEs))
test_PCEs = np.load('test_PCEs.npy').tolist()
test_PCEs = label_normalization(test_PCEs)

test_don_graphs = [dgl.add_self_loop(g) for g in test_don_graphs]
test_acc_graphs = [dgl.add_self_loop(g) for g in test_acc_graphs]

print(len(test_acc_graphs), len(test_don_graphs), len(test_PCEs))

test_dataset =GraphDataset(don_graphs=test_don_graphs, acc_graphs=test_acc_graphs, PCEs=test_PCEs)
test_loader = DataLoader(test_dataset, collate_fn=collate, batch_size=1000)
predictions, labels = [], []

def eval(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for _, (dg, ag, label) in enumerate(data_loader):
            dg = dg.to(device)
            ag = ag.to(device)
            label = torch.tensor(label).to(device)
            prediction = model(dg, ag)
            label = label.cpu().detach().numpy()
            prediction = prediction.view(-1).cpu().detach().numpy()
            labels.extend(label)
            predictions.extend(prediction)
            mae_value = mae(prediction, label)
            rmse_value = rmse(prediction, label)
            sd_value = sd(prediction, label)
            pearsonr_value = pearson(prediction, label)
    return mae_value, rmse_value, sd_value, pearsonr_value

print(f'Evaluating...')
mae_value, rmse_value, sd_value, pearsonr_value = eval(model, test_loader, device)

print(f'MAE={mae_value:.4f}, RMSE={rmse_value:.4f}, SD={sd_value:.4f}, R={pearsonr_value:.4f}')

predictions_df = pd.DataFrame({'Label': labels, 'Prediction': predictions})
predictions_df.to_excel('GCN_EWN_P.xlsx', index=False)
