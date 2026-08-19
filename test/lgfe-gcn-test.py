import torch
import numpy as np
import pandas as pd
import random
import deepdish as dd
import torch.nn.functional as F
import csv
from os.path import join
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
from scipy.stats import pearsonr
from collections import defaultdict

seed = 326187
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
            y_val = self.activity[self.activity.CMPD == key].DG.iloc[0]

            data = Data(
                x=torch.FloatTensor(self.node_data[key]),
                edge_index=torch.LongTensor(self.edge_idx_data[key]),
                edge_attr=torch.FloatTensor(self.edge_att_data[key]),
                y=torch.FloatTensor([y_val])
            )
            data.id = key
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
                          "test.csv")


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(18, 64, cached=False)
        self.bn1 = BatchNorm1d(64)
        self.conv2 = GCNConv(64, 128, cached=False)
        self.bn2 = BatchNorm1d(128)
        self.conv3 = GCNConv(128, 512, cached=False)
        self.bn3 = BatchNorm1d(512)
        self.fc1 = Linear(512, 32)
        self.bn4 = BatchNorm1d(32)
        self.fc2 = Linear(32, 256)
        self.bn5 = BatchNorm1d(256)
        self.fc3 = Linear(256, 32)
        self.bn6 = BatchNorm1d(32)
        self.fc4 = Linear(32, 64)
        self.fc5 = Linear(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = global_add_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        x = self.bn4(x)
        x = F.relu(self.fc2(x))
        x = self.bn5(x)
        x = F.relu(self.fc3(x))
        x = self.bn6(x)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc5(x)
        x = F.relu(x).view(-1)
        return x


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
    pred, true, id = [], [], []
    for data in loader:
        data = data.to(device)
        pred += model(data).detach().cpu().numpy().tolist()
        true += data.y.detach().cpu().numpy().tolist()
        id += data.id
    return pred, true, id


test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = Net()
model.load_state_dict(torch.load('../model.pt'))
pred, true, id = test_predictions(model, test_loader)

rows = zip(true, pred, id)
with open('pred.csv', "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

groups = defaultdict(lambda: {"true": [], "pred": []})
for t, p, i in zip(true, pred, id):
    group_name = i.split('_')[0]
    groups[group_name]["true"].append(t)
    groups[group_name]["pred"].append(p)

group_rmses = {}
group_rs = {}

for group_name, vals in groups.items():
    g_true = vals["true"]
    g_pred = vals["pred"]

    g_rmse = root_mean_squared_error(g_true, g_pred)
    if len(g_true) > 1:
        g_r, _ = pearsonr(g_true, g_pred)
    else:
        g_r = float('nan')

    group_rmses[group_name] = g_rmse
    group_rs[group_name] = g_r

    print(f"{group_name}: RMSE = {g_rmse:.3f}, R = {g_r:.3f}  (n={len(g_true)})")

avg_rmse = np.nanmean(list(group_rmses.values()))
avg_r = np.nanmean(list(group_rs.values()))
print(f"Average RMSE across groups: {avg_rmse:.3f}")
print(f"Average Pearson R across groups: {avg_r:.3f}")

# overall (ungrouped) metrics
##rmse = root_mean_squared_error(true, pred)
##r_value, _ = pearsonr(true, pred)
##print(f"\nOverall Validation RMSE : {rmse:.3f}")
##print(f"Overall Validation Pearson R : {r_value:.3f}")

