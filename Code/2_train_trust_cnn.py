import os
import random
import numpy as np
import pandas as pd
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, classification_report

current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current Working Directory: {current_dir}")

METADATA_CSV = os.path.join(current_dir, "data/metadata.csv")   
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


def make_speaker_splits(df: pd.DataFrame,
                        train_ratio=0.7,
                        val_ratio=0.15,
                        test_ratio=0.15) -> pd.DataFrame:
    """
    Split dataset by speaker_id so no speaker appears in more than one split.
    """
    assert abs(train_ratio + val_ratio - 1.0 + test_ratio) < 1e-6 or \
           abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1."

    speakers = df["speaker_id"].unique()
    np.random.shuffle(speakers)

    n_speakers = len(speakers)
    n_train = int(n_speakers * train_ratio)
    n_val = int(n_speakers * val_ratio)

    train_speakers = speakers[:n_train]
    val_speakers = speakers[n_train:n_train + n_val]
    test_speakers = speakers[n_train + n_val:]

    def assign_split(row):
        sid = row["speaker_id"]
        if sid in train_speakers:
            return "train"
        elif sid in val_speakers:
            return "val"
        else:
            return "test"

    df["split"] = df.apply(assign_split, axis=1)
    return df


def intent_to_label(intent: str) -> int:
    """
    Map text label to integer:
    neutral -> 0
    trustworthy -> 1
    """
    intent = intent.lower()
    if intent == "neutral":
        return 0
    elif intent == "trustworthy":
        return 1
    else:
        raise ValueError(f"Unknown intent: {intent}")

def collate_fn(batch):
    """
    Custom collate function to pad spectrograms to the same length in a batch.
    Each item in batch is (mel_tensor, label_tensor) where mel_tensor is [1, n_mels, T].
    """
    mels, labels = zip(*batch)

    labels = torch.stack(labels)
    
    max_time = max(mel.shape[2] for mel in mels)
    
    padded_mels = []
    for mel in mels:
        current_time = mel.shape[2]
        if current_time < max_time:
            padding = torch.zeros(1, mel.shape[1], max_time - current_time, dtype=mel.dtype)
            mel = torch.cat([mel, padding], dim=2)
        padded_mels.append(mel)

    mels_batch = torch.stack(padded_mels)
    
    return mels_batch, labels


class MelSpecDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        spec_path = row["spec_path"]
        label = intent_to_label(row["intent"])

        mel = np.load(spec_path) 

        mean = mel.mean()
        std = mel.std() + 1e-9
        mel_norm = (mel - mean) / std

        mel_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return mel_tensor, label_tensor

class CNNTrustNet(nn.Module):
    """
    Simple 2D CNN for spectrogram classification.
    Input: [B, 1, n_mels, T]
    """

    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), 
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),         
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def eval_model(model, loader, criterion) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            running_loss += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)


def main():
    df = pd.read_csv(METADATA_CSV)
    df = df[(df["intent"].isin(["neutral", "trustworthy"]))]
    df = make_speaker_splits(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    print(df["split"].value_counts())

    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]
    test_df  = df[df["split"] == "test"]

    train_dataset = MelSpecDataset(train_df)
    val_dataset   = MelSpecDataset(val_df)
    test_dataset  = MelSpecDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = CNNTrustNet(n_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = eval_model(model, val_loader, criterion)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    test_loss, test_acc, y_true, y_pred = eval_model(model, test_loader, criterion)
    print("\n=== TEST RESULTS ===")
    print(f"Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.3f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["neutral", "trustworthy"]))


if __name__ == "__main__":
    main()
