"""
Lesson 4 Mini Project: Simple CNN on Synthetic Spectrograms

This script trains a tiny CNN on a synthetic dataset that mimics
Mel-spectrogram "images" for two classes: neutral vs trustworthy.

Run with:
    python src/train_cnn.py
"""

import numpy as np
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SpectrogramDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 16x16
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 8x8
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)

def load_data():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "synthetic_spectrograms.npz"
    data = np.load(data_path)
    X, y = data["X"], data["y"]
    return X, y

def train():
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    train_ds = SpectrogramDataset(X_train, y_train)
    val_ds = SpectrogramDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 11):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * Xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                preds = logits.argmax(dim=1)
                correct_val += (preds == yb).sum().item()
                total_val += yb.size(0)

        val_acc = correct_val / total_val
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f}, "
              f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

if __name__ == "__main__":
    train()
