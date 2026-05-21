"""
Lesson 9 Mini Project: Visualizing CNN Attention with Grad-CAM

This script will:
1. Load a small synthetic spectrogram-like dataset.
2. Train a simple CNN classifier for a few epochs.
3. Pick a few test examples and compute Grad-CAM heatmaps to see
   which parts of the "spectrogram" the model focuses on.
4. Save the original spectrograms and Grad-CAM overlays as PNG images.

Run with:
    python src/gradcam_demo.py
"""

from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------------- Dataset ----------------

class ExplainDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_data():
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "synthetic_explain_data.npz"
    data = np.load(data_path)
    return data["X"], data["y_main"]

# ---------------- CNN Model ----------------

class SimpleCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(16 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# ---------------- Grad-CAM Utilities ----------------

def compute_gradcam(model, x, target_class, device):
    """
    Compute Grad-CAM heatmap for a single input x and target_class.

    Args:
        model: trained SimpleCnn
        x: tensor of shape (1, 1, H, W)
        target_class: int (0 or 1)
        device: torch.device
    Returns:
        heatmap: numpy array of shape (H, W) normalized to [0, 1]
    """
    model.eval()

    # We will hook the last conv layer
    conv_layer = model.conv[-2]  # the second Conv2d
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    handle_f = conv_layer.register_forward_hook(forward_hook)
    handle_b = conv_layer.register_backward_hook(backward_hook)

    x = x.to(device)
    x.requires_grad_(True)

    logits = model(x)
    loss = logits[0, target_class]
    model.zero_grad()
    loss.backward()

    # Get activations and gradients
    act = activations[0]       # shape: (1, C, Hc, Wc)
    grad = gradients[0]        # shape: (1, C, Hc, Wc)

    # Global average pool over spatial dims
    weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

    # Weighted sum of activations
    cam = (weights * act).sum(dim=1, keepdim=False)  # (1, Hc, Wc)
    cam = cam[0]

    # ReLU
    cam = torch.relu(cam)

    # Normalize to [0,1]
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    # Upsample to original size (H, W)
    cam = torch.nn.functional.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=x.shape[2:],
        mode="bilinear",
        align_corners=False
    )[0, 0]

    heatmap = cam.detach().cpu().numpy()

    handle_f.remove()
    handle_b.remove()

    return heatmap

# ---------------- Training and Visualization ----------------

def train_model(num_epochs=8):
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    train_ds = ExplainDataset(X_train, y_train)
    val_ds = ExplainDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCnn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        train_loss = total_loss / total
        train_acc = correct / total

        # Simple validation accuracy
        model.eval()
        correct_v = 0
        total_v = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                correct_v += (preds == yb).sum().item()
                total_v += xb.size(0)
        val_acc = correct_v / total_v

        print(f"Epoch {epoch:02d} | Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f} | Val acc: {val_acc:.3f}")

    return model, (X_val, y_val), device

def save_gradcam_examples(model, val_data, device, num_examples=4):
    X_val, y_val = val_data
    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)

    # Pick the first few examples
    for i in range(min(num_examples, len(X_val))):
        x_np = X_val[i:i+1]  # shape (1, 1, H, W)
        y_true = int(y_val[i])
        x = torch.from_numpy(x_np).float()

        # Predict
        model.eval()
        with torch.no_grad():
            logits = model(x.to(device))
            pred_class = int(logits.argmax(dim=1).item())

        # Compute Grad-CAM for predicted class
        heatmap = compute_gradcam(model, x, pred_class, device)

        # Plot original spectrogram and Grad-CAM overlay
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(x_np[0, 0], aspect="auto", origin="lower")
        axes[0].set_title(f"Original (y={y_true}, pred={pred_class})")
        axes[0].axis("off")

        axes[1].imshow(x_np[0, 0], aspect="auto", origin="lower", cmap="gray")
        axes[1].imshow(heatmap, cmap="jet", alpha=0.5, origin="lower", aspect="auto")
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")

        fig.tight_layout()
        out_path = out_dir / f"example_{i}_gradcam.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved Grad-CAM visualization to {out_path}")

def main():
    model, val_data, device = train_model(num_epochs=8)
    save_gradcam_examples(model, val_data, device, num_examples=4)

if __name__ == "__main__":
    main()
