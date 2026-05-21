"""
Lesson 6 Mini Project: Adversarial Debiasing with Gradient Reversal

This script trains a small neural network with two heads:
- Main head: predicts a binary label (e.g., trustworthy vs neutral).
- Adversarial head: predicts a 3-class group (e.g., ethnicity).

A Gradient Reversal Layer (GRL) is used so that the feature extractor
learns features that are good for the main task but *bad* for predicting
the sensitive group, reducing encoded bias.

Run with:
    python src/train_adv_debias.py
"""

from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse gradient and scale by lambda
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GRL.apply(x, lambda_)

class AdvDataset(Dataset):
    def __init__(self, X, y_main, y_group):
        self.X = torch.from_numpy(X)
        self.y_main = torch.from_numpy(y_main)
        self.y_group = torch.from_numpy(y_group)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y_main[idx], self.y_group[idx]

class FeatureExtractor(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class MainHead(nn.Module):
    def __init__(self, in_dim=32):
        super().__init__()
        self.fc = nn.Linear(in_dim, 2)  # binary classification

    def forward(self, h):
        return self.fc(h)

class GroupHead(nn.Module):
    def __init__(self, in_dim=32, n_groups=3):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_groups)

    def forward(self, h):
        return self.fc(h)

def load_data():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "synthetic_adv_data.npz"
    data = np.load(data_path)
    X = data["X"]
    y_main = data["y_main"]
    y_group = data["y_group"]
    return X, y_main, y_group

def train(num_epochs=15, lambda_grl=1.0):
    X, y_main, y_group = load_data()
    X_train, X_val, y_main_train, y_main_val, y_group_train, y_group_val = train_test_split(
        X, y_main, y_group, test_size=0.25, random_state=42, stratify=y_group
    )

    train_ds = AdvDataset(X_train, y_main_train, y_group_train)
    val_ds = AdvDataset(X_val, y_main_val, y_group_val)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat = FeatureExtractor().to(device)
    main_head = MainHead().to(device)
    group_head = GroupHead().to(device)

    main_criterion = nn.CrossEntropyLoss()
    group_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        list(feat.parameters()) + list(main_head.parameters()) + list(group_head.parameters()),
        lr=1e-3,
    )

    for epoch in range(1, num_epochs + 1):
        feat.train(); main_head.train(); group_head.train()
        total_main_loss = 0.0
        total_group_loss = 0.0
        correct_main = 0
        correct_group = 0
        total = 0

        for xb, yb_main, yb_group in train_loader:
            xb = xb.to(device)
            yb_main = yb_main.to(device)
            yb_group = yb_group.to(device)

            optimizer.zero_grad()

            # Shared feature extractor
            h = feat(xb)

            # Main task prediction
            logits_main = main_head(h)
            loss_main = main_criterion(logits_main, yb_main)

            # Adversarial group prediction with gradient reversal
            h_rev = grad_reverse(h, lambda_grl)
            logits_group = group_head(h_rev)
            loss_group = group_criterion(logits_group, yb_group)

            # Combined loss: main loss + group loss
            loss = loss_main + loss_group
            loss.backward()
            optimizer.step()

            total_main_loss += loss_main.item() * xb.size(0)
            total_group_loss += loss_group.item() * xb.size(0)

            preds_main = logits_main.argmax(dim=1)
            preds_group = logits_group.argmax(dim=1)
            correct_main += (preds_main == yb_main).sum().item()
            correct_group += (preds_group == yb_group).sum().item()
            total += xb.size(0)

        train_main_loss = total_main_loss / total
        train_group_loss = total_group_loss / total
        train_main_acc = correct_main / total
        train_group_acc = correct_group / total

        # Validation: check main accuracy and group accuracy on representation
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

        print(
            f"Epoch {epoch:02d} | "
            f"Train main acc: {train_main_acc:.3f}, Train group acc: {train_group_acc:.3f} | "
            f"Val main acc: {val_main_acc:.3f}, Val group acc: {val_group_acc:.3f}"
        )

if __name__ == "__main__":
    # Try lambda_grl = 1.0 (stronger debiasing effect).
    train(num_epochs=15, lambda_grl=1.0)
