import optuna
import torch
import numpy as np
import pandas as pd
import random
import deepdish as dd
import torch.nn.functional as F
from os.path import join
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from sklearn.model_selection import GroupKFold

seed = 121314
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SILCSDataset(InMemoryDataset):
    def __init__(self, root, node_features, edge_idxs, edge_atts, activity_csv, transform=None, pre_transform=None):
        self.root = root
        self.node_features = node_features
        self.edge_idxs = edge_idxs
        self.edge_atts = edge_atts
        self.activity_csv = activity_csv
        super(SILCSDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        self.node_data = dd.io.load(join(self.root, self.node_features))
        self.edge_idx_data = dd.io.load(join(self.root, self.edge_idxs))
        self.edge_att_data = dd.io.load(join(self.root, self.edge_atts))
        self.activity = pd.read_csv(join(self.root, self.activity_csv))

        data_list = []
        for key in self.activity.CMPD:
            group = self.activity[self.activity.CMPD == key].KGRP.iloc[0]
            y_val = self.activity[self.activity.CMPD == key].DG.iloc[0]

            data = Data(
                x=torch.FloatTensor(self.node_data[key]),
                edge_index=torch.LongTensor(self.edge_idx_data[key]),
                edge_attr=torch.FloatTensor(self.edge_att_data[key]),
                y=torch.FloatTensor([y_val])
            )
            data.id = key
            data.group = group
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


dataset = SILCSDataset("sdf",
                          "training_features.h5",
                          "training_edgeidx.h5",
                          "training_edgeatt.h5",
                          "exp.csv")


class Net(torch.nn.Module):
    def __init__(self, input_dim, gcn_hidden_dims, mlp_hidden_dims, prevoutdim, dropout_rate):
        super(Net, self).__init__()

        self.gcn_layers = torch.nn.ModuleList()
        self.bn_layers = torch.nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in gcn_hidden_dims:
            self.gcn_layers.append(GCNConv(prev_dim, hidden_dim, cached=False))
            self.bn_layers.append(BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        self.mlp_layers = torch.nn.ModuleList()
        self.mlp_bn_layers = torch.nn.ModuleList()
        for hidden_dim in mlp_hidden_dims:
            self.mlp_layers.append(Linear(prev_dim, hidden_dim))
            self.mlp_bn_layers.append(BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim

        self.prevoutdim = prevoutdim
        self.preout = Linear(prev_dim, prevoutdim)
        self.out = Linear(prevoutdim, 1)
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for conv, bn in zip(self.gcn_layers, self.bn_layers):
            x = F.relu(conv(x, edge_index))
            x = bn(x)

        x = global_add_pool(x, data.batch)

        for fc, bn in zip(self.mlp_layers, self.mlp_bn_layers):
            x = F.relu(fc(x))
            x = bn(x)

        x = F.relu(self.preout(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.out(x)
        return F.relu(x).view(-1)


def k_fold_group_train_val(dataset, folds, group_names):
    gkf = GroupKFold(n_splits=folds)
    train_indices, val_indices = [], []
    for train_idx, val_idx in gkf.split(X=torch.zeros(len(dataset)), y=None, groups=group_names):
        train_indices.append(torch.from_numpy(train_idx).to(torch.long))
        val_indices.append(torch.from_numpy(val_idx).to(torch.long))
    return train_indices, val_indices


def train(model, train_loader, epoch, device, optimizer, scheduler):
    model.train()
    loss_all = 0
    error = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        error += (model(data) - data.y).abs().sum().item()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()
    return loss_all / len(train_loader.dataset), error / len(train_loader.dataset)


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    error = 0
    for data in loader:
        data = data.to(device)
        error += (model(data) - data.y).abs().sum().item()
    return error / len(loader.dataset)


@torch.no_grad()
def test_predictions(model, loader):
    model.eval()
    pred, true = [], []
    for data in loader:
        data = data.to(device)
        pred += model(data).detach().cpu().numpy().tolist()
        true += data.y.detach().cpu().numpy().tolist()
    return pred, true


def objective(trial):
    learning_rate = trial.suggest_categorical("learning_rate", [1e-4, 1e-3, 1e-2, 1e-1])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    gcn_layers = trial.suggest_int("gcn_layers", 2, 4)
    gcn_hidden_dims = [trial.suggest_categorical(f"gcn_dim_{i}", [32, 64, 128, 256, 512]) for i in range(gcn_layers)]
    mlp_layers = trial.suggest_int("mlp_layers", 1, 3)
    mlp_hidden_dims = [trial.suggest_categorical(f"mlp_dim_{i}", [16, 32, 64, 128, 256]) for i in range(mlp_layers)]
    dropout = trial.suggest_categorical("dropout_rate", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    outdim = trial.suggest_categorical("prevoutdim", [8, 16, 32, 64])

    group_names = [data.group for data in dataset]
    folds = 5
    train_idx_list, val_idx_list = k_fold_group_train_val(dataset, folds=folds, group_names=group_names)

    val_errors = []

    for fold in range(folds):
        train_subset = torch.utils.data.Subset(dataset, train_idx_list[fold])
        val_subset = torch.utils.data.Subset(dataset, val_idx_list[fold])

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = Net(input_dim=18, gcn_hidden_dims=gcn_hidden_dims, mlp_hidden_dims=mlp_hidden_dims, prevoutdim=outdim, dropout_rate=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10, min_lr=1e-5)

        best_val_error = None
        for epoch in range(1, 101):
            loss, train_error = train(model, train_loader, epoch, device, optimizer, scheduler)
            val_error = test(model, val_loader, device)
            scheduler.step(val_error)
            if best_val_error is None or val_error < best_val_error:
                best_val_error = val_error

        val_errors.append(best_val_error)
    return np.mean(val_errors)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
print("Best trial:")
print(study.best_trial)
