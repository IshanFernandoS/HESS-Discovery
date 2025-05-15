import numpy as np
import pandas as pd
import torch
from pymatgen.core import Composition

# ------------------- Load Features -------------------
train_features_file = "./data/final_features_train.npy"
final_features_train = np.load(train_features_file)
print(f"Loaded HEC5 full data shape: {final_features_train.shape}")

# ------------------- Load Target Values -------------------
hec5_file = "./data/HEC5_output.csv"
hec5_df = pd.read_csv(hec5_file)
efa_values = hec5_df['EFA'].values
assert len(final_features_train) == len(efa_values), "Mismatch between features and EFA values!"

features = np.array(final_features_train, dtype=np.float32)
efa = np.array(efa_values, dtype=np.float32)

# ------------------- Load Test Features -------------------
test_features_file = "./data/final_features_test.npy"    # replace with your actual test .npy filename
final_features_test = np.load(test_features_file)
print(f"Loaded HEC8 test data shape: {final_features_test.shape}")

# ------------------- Load Test Target Values -------------------
hec8_file = "./data/HEC8_output.csv"                   # replace if your filename differs
hec8_df = pd.read_csv(hec8_file)
efa_test_values = hec8_df['EFA'].values
assert len(final_features_test) == len(efa_test_values), (
    f"Mismatch between test features ({len(final_features_test)}) "
    f"and test EFA values ({len(efa_test_values)})!"
)

# Cast to float32 for consumption by your model
features_test = np.array(final_features_test, dtype=np.float32)
efa_test     = np.array(efa_test_values,  dtype=np.float32)

# ------------------- Train-Test Split -------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    features,
    efa,
    test_size=0.2,
    random_state=42
)
print(f"Split into {len(X_train)} train and {len(X_test)} test samples.")

# ------------------- Create Edge Index & Edge Attr -------------------
from torch_geometric.utils import dense_to_sparse

def create_edge_index(n):
    adj = torch.ones((n, n)) - torch.eye(n)
    return dense_to_sparse(adj)[0]

def create_edge_attr(x, edge_index):
    src, dst = edge_index
    n = x.size(0)
    last = n - 1
    ea = torch.zeros(src.size(0), 1, dtype=x.dtype, device=x.device)
    mask_s = (src == last)
    ea[mask_s] = x[dst[mask_s], -1].unsqueeze(-1)
    mask_d = (dst == last) & ~mask_s
    ea[mask_d] = x[src[mask_d], -1].unsqueeze(-1)
    return ea

edge_index_train = create_edge_index(6)

# ------------------- Create PyG Data Objects -------------------
from torch_geometric.data import Data

train_set = []
for i in range(len(X_train)):
    x_i = torch.tensor(X_train[i], dtype=torch.float32)
    ei = edge_index_train
    ea = create_edge_attr(x_i, ei)
    data_i = Data(x=x_i, edge_index=ei, edge_attr=ea, y=torch.tensor(y_train[i], dtype=torch.float32))
    train_set.append(data_i)

test_set = []
for i in range(len(X_test)):
    x_i = torch.tensor(X_test[i], dtype=torch.float32)
    ei = edge_index_train
    ea = create_edge_attr(x_i, ei)
    data_i = Data(x=x_i, edge_index=ei, edge_attr=ea, y=torch.tensor(y_test[i], dtype=torch.float32))
    test_set.append(data_i)

from torch_geometric.data import Data

edge_index_val = create_edge_index(9)

val_set = []
for i in range(len(features_test)):
    x_i = torch.tensor(features_test[i], dtype=torch.float32)
    ei = edge_index_val  
    ea = create_edge_attr(x_i, ei)
    data_i = Data(
        x         = x_i,
        edge_index= ei,
        edge_attr = ea,
        y         = torch.tensor(efa_test[i], dtype=torch.float32)
    )
    val_set.append(data_i)

# ------------------- Create Data Loaders -------------------
from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
test_loader = DataLoader(test_set, batch_size=4, shuffle=True)

# ------------------- Define the GATv2 Model -------------------
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GlobalAttention
from torch_geometric.nn import GraphNorm

class GCNEncoder(nn.Module):
    def __init__(self, in_channels=35, hidden_dim=50, num_layers=2, heads=1, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Initial layer
        self.convs.append(
            GATv2Conv(
                in_channels=in_channels - 1,
                out_channels=hidden_dim,
                heads=heads,
                dropout=dropout,
                edge_dim=1,
                concat=False
            )
        )
        # self.norms.append(nn.LayerNorm(hidden_dim * heads))

        # Additional layers
        for _ in range(1, num_layers):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=False
                )
            )
            self.norms.append(GraphNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        # Attention pooling
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.att_pool = GlobalAttention(gate_nn=self.gate_nn)

        # Final readout MLP
        self.readout_mlp = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_features = x[:, :-1].float()
        num_nodes = x.size(0)
        src, tgt = edge_index

        edge_attr = torch.zeros(src.size(0), 1, device=x.device, dtype=x.dtype)
        mask_src = (src == (num_nodes - 1))
        if mask_src.sum() > 0:
            edge_attr[mask_src] = x[tgt[mask_src], -1].unsqueeze(-1)
        mask_tgt = (tgt == (num_nodes - 1)) & ~(src == (num_nodes - 1))
        if mask_tgt.sum() > 0:
            edge_attr[mask_tgt] = x[src[mask_tgt], -1].unsqueeze(-1)
        edge_attr = edge_attr.float()

        out = node_features
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = out
            if i == 0:
                out = conv(out, edge_index,edge_attr)
            else:
                out = conv(out, edge_index)  # No edge_attr in subsequent layers
            
            out = norm(out)
            out = self.activation(out)
            out = self.dropout(out)

            if out.size() == residual.size():
                out = out + residual

        graph_emb = self.att_pool(out, batch=batch)
        out = self.readout_mlp(graph_emb)
        return out
# ------------------- Set Up Training -------------------
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = GCNEncoder(
    in_channels=35, 
    hidden_dim=52,
    num_layers=8,
    heads=4,
    dropout=0.3
).to(device)

# model.load_state_dict(torch.load("_model_50999_after_8l.pth",weights_only=True))

epochs = 1000000  # reduce for faster runs
lr = 5e-5
optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=False)
train_loss_fn = nn.MSELoss()
val_loss_fn = nn.L1Loss()



from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score

# 0) build loaders once
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=4, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=4, shuffle=False)

train_losses = []
val_losses   = []
test_losses  = []
val_r2s      = []
test_r2s     = []

eval_every = 5

print("Starting training...")
for epoch in range(1, epochs + 1):
    # ---- train ----
    model.train()
    total_train_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out    = model(batch).view(-1)
        target = batch.y.view(-1)
        loss   = train_loss_fn(out, target)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * batch.num_graphs

    avg_train_loss = total_train_loss / len(train_set)
    train_losses.append(avg_train_loss)

    # ---- validate & test ----
    if epoch % eval_every == 0:
        model.eval()

        # validation
        total_val_loss = 0.0
        all_val_preds  = []
        all_val_targs  = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out    = model(batch).view(-1)
                target = batch.y.view(-1)
                loss   = val_loss_fn(out, target)
                total_val_loss += loss.item() * batch.num_graphs
                all_val_preds.append(out.cpu().numpy())
                all_val_targs.append(target.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_set)
        val_losses.append(avg_val_loss)
        all_val_preds = np.concatenate(all_val_preds)
        all_val_targs = np.concatenate(all_val_targs)
        val_r2 = r2_score(all_val_targs, all_val_preds)
        val_r2s.append(val_r2)

        # test
        total_test_loss = 0.0
        all_test_preds  = []
        all_test_targs  = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out    = model(batch).view(-1)
                target = batch.y.view(-1)
                loss   = val_loss_fn(out, target)
                total_test_loss += loss.item() * batch.num_graphs
                all_test_preds.append(out.cpu().numpy())
                all_test_targs.append(target.cpu().numpy())

        avg_test_loss = total_test_loss / len(test_set)
        test_losses.append(avg_test_loss)
        all_test_preds = np.concatenate(all_test_preds)
        all_test_targs = np.concatenate(all_test_targs)
        test_r2 = r2_score(all_test_targs, all_test_preds)
        test_r2s.append(test_r2)

        print(f"[Epoch {epoch}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} (R²={val_r2:.3f}) | "
              f"Test Loss: {avg_test_loss:.4f} (R²={test_r2:.3f})")
    else:
        print(f"[Epoch {epoch}/{epochs}] Train Loss: {avg_train_loss:.4f}")

    # optional checkpoint
    if epoch % 1000 == 0:
        torch.save(model.state_dict(), f"_model_{epoch}_after_xl.pth")


# ------------------- Plot Training Curve -------------------
import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
val_x = list(range(5, epochs+1, 5))
plt.plot(val_x, val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('GATv2 Training & Validation Loss (HEC5 Split)')
plt.legend()
plt.savefig("loss_curve.png")
plt.close()
print("Saved final loss curve to 'loss_curve.png'.")