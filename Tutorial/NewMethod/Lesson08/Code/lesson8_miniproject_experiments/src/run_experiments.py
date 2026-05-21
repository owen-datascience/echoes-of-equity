"""
Lesson 8 Mini Project: Experiment Tracking & Hyperparameter Tuning

This script:
1. Loads a synthetic spectrogram dataset.
2. Defines a small adversarial CNN (feature extractor + main + group heads).
3. Runs multiple experiments over different (learning_rate, lambda_grl) configs.
4. Logs results (val main acc, val group acc) to data/experiment_results.csv.

Run with:
    python src/run_experiments.py
"""

from pathlib import Path
import csv
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ---------------- Data ----------------

class AdvCnnDataset(Dataset):
    def __init__(self, X, y_main, y_group):
        self.X = torch.from_numpy(X)
        self.y_main = torch.from_numpy(y_main)
        self.y_group = torch.from_numpy(y_group)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y_main[idx], self.y_group[idx]

def load_data():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "synthetic_experiment_data.npz"
    data = np.load(data_path)
    return data["X"], data["y_main"], data["y_group"]

# ---------------- Model ----------------

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GRL.apply(x, lambda_)

class CnnFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return x

class MainHead(nn.Module):
    def __init__(self, in_dim=16*8*8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, h):
        return self.fc(h)

class GroupHead(nn.Module):
    def __init__(self, in_dim=16*8*8, n_groups=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_groups),
        )

    def forward(self, h):
        return self.fc(h)

# ---------------- Training & Evaluation ----------------

def run_single_experiment(lr=1e-3, lambda_grl=0.0, num_epochs=8, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, y_main, y_group = load_data()
    X_train, X_val, y_main_train, y_main_val, y_group_train, y_group_val = train_test_split(
        X, y_main, y_group, test_size=0.25, random_state=42, stratify=y_group
    )

    train_ds = AdvCnnDataset(X_train, y_main_train, y_group_train)
    val_ds = AdvCnnDataset(X_val, y_main_val, y_group_val)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat = CnnFeatureExtractor().to(device)
    main_head = MainHead().to(device)
    group_head = GroupHead().to(device)

    main_criterion = nn.CrossEntropyLoss()
    group_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        list(feat.parameters()) + list(main_head.parameters()) + list(group_head.parameters()),
        lr=lr,
    )

    for epoch in range(1, num_epochs + 1):
        feat.train(); main_head.train(); group_head.train()
        for xb, yb_main, yb_group in train_loader:
            xb = xb.to(device)
            yb_main = yb_main.to(device)
            yb_group = yb_group.to(device)

            optimizer.zero_grad()
            h = feat(xb)

            logits_main = main_head(h)
            loss_main = main_criterion(logits_main, yb_main)

            h_rev = grad_reverse(h, lambda_grl)
            logits_group = group_head(h_rev)
            loss_group = group_criterion(logits_group, yb_group)

            loss = loss_main + loss_group
            loss.backward()
            optimizer.step()

    # Validation metrics
    feat.eval(); main_head.eval(); group_head.eval()
    correct_main_v = 0
    correct_group_v = 0
    total_v = 0
    with torch.no_grad():
        for xb, yb_main, yb_group in val_loader:
            xb = xb.to(device)
            yb_main = yb_main.to(device)
            yb_group = yb_group.to(device)

            h = feat(xb)
            logits_main = main_head(h)
            logits_group = group_head(h)

            preds_main = logits_main.argmax(dim=1)
            preds_group = logits_group.argmax(dim=1)

            correct_main_v += (preds_main == yb_main).sum().item()
            correct_group_v += (preds_group == yb_group).sum().item()
            total_v += xb.size(0)

    val_main_acc = correct_main_v / total_v
    val_group_acc = correct_group_v / total_v
    return val_main_acc, val_group_acc

def main():
    base_dir = Path(__file__).resolve().parents[1]
    out_csv = base_dir / "data" / "experiment_results.csv"

    configs = [
        {"lr": 1e-3, "lambda_grl": 0.0},
        {"lr": 1e-3, "lambda_grl": 0.5},
        {"lr": 1e-3, "lambda_grl": 1.0},
        {"lr": 5e-4, "lambda_grl": 0.5},
        {"lr": 5e-4, "lambda_grl": 1.0},
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lr", "lambda_grl", "val_main_acc", "val_group_acc"])

        for cfg in configs:
            lr = cfg["lr"]
            lambda_grl = cfg["lambda_grl"]
            print(f"Running experiment lr={lr}, lambda_grl={lambda_grl}")
            val_main_acc, val_group_acc = run_single_experiment(
                lr=lr, lambda_grl=lambda_grl, num_epochs=8, seed=0
            )
            print(f"  -> val_main_acc={val_main_acc:.3f}, val_group_acc={val_group_acc:.3f}")
            writer.writerow([lr, lambda_grl, f"{val_main_acc:.4f}", f"{val_group_acc:.4f}"])

    print(f"\nSaved results to: {out_csv}")

if __name__ == "__main__":
    main()
