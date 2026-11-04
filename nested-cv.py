import sys
import copy
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
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr

seed = 121314
N_TRIALS = 100     
N_EPOCHS = 100    
INNER_EPOCHS = N_EPOCHS

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


def compute_pc(pred, exp):
    n = len(exp)

    if n <= 1:
        return 1.0

    correct = 0
    total = 0

    for i in range(n):
        for j in range(i + 1, n):

            exp_diff = exp[j] - exp[i]
            pred_diff = pred[j] - pred[i]

            if exp_diff == 0:
                continue

            if exp_diff * pred_diff > 0:
                correct += 1

            total += 1

    return correct / total if total > 0 else np.nan


def train(model, train_loader, device, optimizer):
    model.train()
    loss_all = 0
    error = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        error += (out - data.y).abs().sum().item()
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
    pred, true, ids = [], [], []
    for data in loader:
        data = data.to(device)
        pred += model(data).detach().cpu().numpy().tolist()
        true += data.y.detach().cpu().numpy().tolist()
        ids += list(data.id)
    return pred, true, ids


def suggest_params(trial):
    p = {}
    p["learning_rate"] = trial.suggest_categorical("learning_rate", [1e-4, 1e-3, 1e-2, 1e-1])
    p["batch_size"] = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    p["gcn_layers"] = trial.suggest_int("gcn_layers", 2, 4)
    for i in range(p["gcn_layers"]):
        p[f"gcn_dim_{i}"] = trial.suggest_categorical(f"gcn_dim_{i}", [32, 64, 128, 256, 512])
    p["mlp_layers"] = trial.suggest_int("mlp_layers", 1, 3)
    for i in range(p["mlp_layers"]):
        p[f"mlp_dim_{i}"] = trial.suggest_categorical(f"mlp_dim_{i}", [16, 32, 64, 128, 256])
    p["dropout_rate"] = trial.suggest_categorical("dropout_rate", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    p["prevoutdim"] = trial.suggest_categorical("prevoutdim", [8, 16, 32, 64])
    return p


def make_model(p):
    return Net(
        input_dim=18,
        gcn_hidden_dims=[p[f"gcn_dim_{i}"] for i in range(p["gcn_layers"])],
        mlp_hidden_dims=[p[f"mlp_dim_{i}"] for i in range(p["mlp_layers"])],
        prevoutdim=p["prevoutdim"],
        dropout_rate=p["dropout_rate"],
    ).to(device)


def subset(idx):
    return torch.utils.data.Subset(dataset, [int(i) for i in idx])

groups = np.array([data.group for data in dataset])
all_groups = sorted(np.unique(groups).tolist())
print(f"Groups (KGRP): {all_groups}")

if len(sys.argv) > 1:
    outer_groups = [type(all_groups[0])(g) for g in sys.argv[1:]]
else:
    outer_groups = all_groups

all_rows, summary = [], []
suffix = ("_kgrp" + "_".join(str(g) for g in outer_groups)) if len(sys.argv) > 1 else ""

for outer_g in outer_groups:
    print(f"\n================ OUTER FOLD: KGRP {outer_g} held out ================")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    outer_test_idx = np.where(groups == outer_g)[0]
    tune_idx = np.where(groups != outer_g)[0]
    tune_groups = groups[tune_idx]

    inner_folds = []
    for ig in sorted(np.unique(tune_groups).tolist()):
        inner_folds.append((tune_idx[tune_groups != ig], tune_idx[tune_groups == ig]))

    def objective(trial):
        p = suggest_params(trial)
        val_errors = []
        for inner_train_idx, inner_val_idx in inner_folds:
            train_loader = DataLoader(subset(inner_train_idx), batch_size=p["batch_size"], shuffle=True, drop_last=True)
            val_loader = DataLoader(subset(inner_val_idx), batch_size=p["batch_size"], shuffle=False)

            model = make_model(p)
            optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.95, patience=10, min_lr=1e-5)

            best_val_error = None
            for epoch in range(1, INNER_EPOCHS + 1):
                train(model, train_loader, device, optimizer)
                val_error = test(model, val_loader, device)
                scheduler.step(val_error)
                if best_val_error is None or val_error < best_val_error:
                    best_val_error = val_error
            val_errors.append(best_val_error)
        return float(np.mean(val_errors))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=N_TRIALS)
    best = study.best_params
    print(f"Inner best MAE: {study.best_value:.4f}")
    print(f"Best params: {best}")

    train_loader = DataLoader(subset(tune_idx), batch_size=best["batch_size"], shuffle=True, drop_last=True)
    model = make_model(best)
    optimizer = torch.optim.Adam(model.parameters(), lr=best["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.95, patience=10, min_lr=1e-5)
    best_train_error, best_model = None, None
    for epoch in range(1, N_EPOCHS + 1):
        loss, train_error = train(model, train_loader, device, optimizer)
        if best_train_error is None or train_error <= best_train_error:
            best_train_error = train_error
            best_model = copy.deepcopy(model)

    test_loader = DataLoader(subset(outer_test_idx), batch_size=1, shuffle=False)
    pred, true, ids = test_predictions(best_model, test_loader)
    mae = float(np.mean(np.abs(np.array(pred) - np.array(true))))
    rmse = root_mean_squared_error(true, pred)
    r_value, _ = pearsonr(true, pred)
    pc = compute_pc(pred, true)
    print(f"OUTER KGRP {outer_g}: MAE {mae:.4f}  RMSE {rmse:.4f}  Pearson r {r_value:.4f}  PC {pc:.4f}")

    for t, pr, i in zip(true, pred, ids):
        all_rows.append((t, pr, i, outer_g))
    summary.append(dict(kgrp=outer_g, mae=mae, rmse=rmse, r=r_value, pc=pc,
                        inner_best_mae=study.best_value, params=best))

if len(summary) > 1:
    for key, label in [("mae", "MAE"), ("rmse", "RMSE"), ("r", "Pearson r"), ("pc", "PC")]:
        vals = [s[key] for s in summary]
        print(f"Nested CV {label}: {np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.4f}")
