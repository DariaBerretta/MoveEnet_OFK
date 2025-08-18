import torch
from torch_geometric.loader import DataLoader
from graph_enet.data.scarfDataset import scarfDataset
from graph_enet.model.test_models import DoubleGCNConv

import matplotlib.pyplot as plt

def plot_pred_vs_gt(pred, gt):
    pred = pred.view(-1, 2).cpu().numpy()
    gt = gt.view(-1, 2).cpu().numpy()
    plt.figure()
    plt.scatter(gt[:,0], gt[:,1], c='g', label='GT')
    plt.scatter(pred[:,0], pred[:,1], c='r', label='Pred')
    for i in range(len(gt)):
        plt.plot([gt[i,0], pred[i,0]], [gt[i,1], pred[i,1]], 'k--', alpha=0.3)
    plt.legend()
    plt.title('Predicted vs Ground Truth Skeletons')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.xlim(0, 640)
    plt.ylim(0, 480)
    plt.gca().invert_yaxis()
    plt.show()

# === Parameters ===
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Load Dataset ===
dataset = scarfDataset(root='/home/dberretta-iit.local/data/new_scarfGNN')
print(f"Dataset loaded with {len(dataset)} graphs.")
dataset = dataset.shuffle()
train_len = int(0.8 * len(dataset))
train_dataset = dataset[:train_len]
val_dataset = dataset[train_len:]

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Init model ===
in_channels = dataset[0].x.shape[1]      # e.g., 10
model = DoubleGCNConv(in_channels=in_channels).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.MSELoss()

# === Training loop ===
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data)       # [batch_size, 26]
        loss = loss_fn(out, data.y.view(-1, 26))  # [batch_size, 26]
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

# === Validation loop ===
@torch.no_grad()
def evaluate(epoch=None):
    model.eval()
    total_loss = 0
    show_plot = epoch is not None and epoch % 20 == 0  # Only plot every 5 epochs

    for i, data in enumerate(val_loader):
        data = data.to(DEVICE)
        out = model(data)  # [batch_size, 26]
        loss = loss_fn(out, data.y)
        total_loss += loss.item() * data.num_graphs

        # === Only plot one sample every 5 epochs ===
        if show_plot and i == 0:
            plot_pred_vs_gt(out[0], data.y[0])

    return total_loss / len(val_loader.dataset)

# === Run Training ===
for epoch in range(1, EPOCHS + 1):
    train_loss = train()
    val_loss = evaluate(epoch)
    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
