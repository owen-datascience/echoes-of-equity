import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

from transformers import AutoProcessor, AutoModel
from transformers import Wav2Vec2FeatureExtractor


current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current Working Directory: {current_dir}")

METADATA_CSV = os.path.join(current_dir, "data/metadata_acoustic_prosody.csv")

MODEL_NAME = "microsoft/wavlm-base-plus"
TARGET_SR = 16000

BATCH_SIZE = 8
NUM_EPOCHS = 10
LR = 1e-4
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACOUSTIC_FEATURE_COLS = [
    "duration_sec",
    "f0_mean_hz",
    "f0_std_hz",
    "hnr_mean_db",
    "jitter_local",
    "cpp_mean_db",
]



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_SEED)


def intent_to_label(intent: str) -> int:
    """
    neutral -> 0, trustworthy -> 1
    """
    intent = intent.lower()
    if intent == "neutral":
        return 0
    elif intent == "trustworthy":
        return 1
    else:
        raise ValueError(f"Unknown intent: {intent}")


def make_speaker_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> pd.DataFrame:
    """
    Speaker-independent splits: each speaker_id appears in only one split.
    """
    speakers = df["speaker_id"].unique()
    np.random.shuffle(speakers)

    n_speakers = len(speakers)
    n_train = int(n_speakers * train_ratio)
    n_val = int(n_speakers * val_ratio)

    train_speakers = speakers[:n_train]
    val_speakers = speakers[n_train:n_train + n_val]
    test_speakers = speakers[n_train + n_val:]

    def assign_split(sid):
        if sid in train_speakers:
            return "train"
        elif sid in val_speakers:
            return "val"
        else:
            return "test"

    df["split"] = df["speaker_id"].apply(assign_split)
    return df



class HybridTrustDataset(Dataset):
    """
    Dataset that returns:
      - raw audio (numpy array)
      - normalized acoustic features (tensor [F])
      - label (0/1)
    """

    def __init__(self, df: pd.DataFrame, feat_means: pd.Series, feat_stds: pd.Series):
        self.df = df.reset_index(drop=True)
        self.feat_means = feat_means
        self.feat_stds = feat_stds.replace(0, 1.0)  

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav_path = row["wav_path"]

        y, sr = sf.read(wav_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

        feats = row[ACOUSTIC_FEATURE_COLS].values.astype(np.float32)
        feats_norm = (feats - self.feat_means.values) / (self.feat_stds.values + 1e-9)

        label = intent_to_label(row["intent"])

        return y.astype(np.float32), feats_norm.astype(np.float32), label


def collate_fn(batch: List):
    """
    Custom collate:
    - batch: list of (audio_np, feats_np, label_int)
    Returns:
      audios: list[np.ndarray] (variable length)
      feats:  tensor [B, F]
      labels: tensor [B]
    """
    audios = [item[0] for item in batch]
    feats = np.stack([item[1] for item in batch], axis=0)
    labels = np.array([item[2] for item in batch], dtype=np.int64)

    feats_tensor = torch.tensor(feats, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return audios, feats_tensor, labels_tensor


class WavLMHybridClassifier(nn.Module):
    """
    Hybrid classifier: [WavLM embedding] + [acoustic features] -> dense layers -> logits
    """

    def __init__(self, wavlm_hidden_dim: int, acoustic_dim: int, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(wavlm_hidden_dim + acoustic_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, wavlm_embeds, acoustic_feats):
        x = torch.cat([wavlm_embeds, acoustic_feats], dim=1)
        logits = self.classifier(x)
        return logits


def train_one_epoch(model, wavlm_model, processor, loader, optimizer, criterion):
    model.train()
    wavlm_model.eval() 
    running_loss = 0.0
    all_preds, all_labels = [], []

    for audios, feats, labels in loader:
        inputs = processor(
            audios,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = wavlm_model(input_values=input_values, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  
            wavlm_embeds = hidden_states.mean(dim=1)   

        feats = feats.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(wavlm_embeds, feats)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def eval_model(model, wavlm_model, processor, loader, criterion) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    wavlm_model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for audios, feats, labels in loader:
            inputs = processor(
                audios,
                sampling_rate=TARGET_SR,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs["input_values"].to(DEVICE)
            attention_mask = inputs["attention_mask"].to(DEVICE)

            outputs = wavlm_model(input_values=input_values, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state 
            wavlm_embeds = hidden_states.mean(dim=1)   

            feats = feats.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(wavlm_embeds, feats)
            loss = criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)



def group_metrics(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, group_col: str):
    """
    Print accuracy per group (ethnicity, age_group, sex).
    """
    print(f"\n=== Group metrics by {group_col} ===")
    df_local = df.reset_index(drop=True).copy()
    df_local["y_true"] = y_true
    df_local["y_pred"] = y_pred

    for group_value in sorted(df_local[group_col].dropna().unique()):
        subset = df_local[df_local[group_col] == group_value]
        if len(subset) == 0:
            continue
        acc = accuracy_score(subset["y_true"], subset["y_pred"])
        print(f"{group_col} = {group_value:12s} | N = {len(subset):3d} | Accuracy = {acc:.3f}")


def main():
    df = pd.read_csv(METADATA_CSV)

    df = df[df["intent"].isin(["neutral", "trustworthy"])]

    df = df.dropna(subset=ACOUSTIC_FEATURE_COLS)

    df = make_speaker_splits(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    print("Split distribution:")
    print(df["split"].value_counts())

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    feat_means = train_df[ACOUSTIC_FEATURE_COLS].mean()
    feat_stds  = train_df[ACOUSTIC_FEATURE_COLS].std().replace(0, 1.0)

    train_dataset = HybridTrustDataset(train_df, feat_means, feat_stds)
    val_dataset   = HybridTrustDataset(val_df, feat_means, feat_stds)
    test_dataset  = HybridTrustDataset(test_df, feat_means, feat_stds)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print("Loading WavLM model...")
    try:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        print("Loaded Wav2Vec2FeatureExtractor successfully.")
    except Exception as e:
        print(f"Error loading Wav2Vec2FeatureExtractor: {e}")
        print("Falling back to AutoProcessor...")
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
    wavlm_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    for p in wavlm_model.parameters():
        p.requires_grad = False

    wavlm_hidden_dim = wavlm_model.config.hidden_size
    acoustic_dim = len(ACOUSTIC_FEATURE_COLS)

    classifier = WavLMHybridClassifier(
        wavlm_hidden_dim=wavlm_hidden_dim,
        acoustic_dim=acoustic_dim,
        num_classes=2,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            classifier, wavlm_model, processor, train_loader, optimizer, criterion
        )
        val_loss, val_acc, _, _ = eval_model(
            classifier, wavlm_model, processor, val_loader, criterion
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = classifier.state_dict()

    if best_state is not None:
        classifier.load_state_dict(best_state)

    test_loss, test_acc, y_true, y_pred = eval_model(
        classifier, wavlm_model, processor, test_loader, criterion
    )
    print("\n=== HYBRID MODEL – TEST RESULTS ===")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.3f}")
    print("\nClassification report (0=neutral, 1=trustworthy):")
    print(classification_report(y_true, y_pred, target_names=["neutral", "trustworthy"]))

    for group_col in ["ethnicity", "age_group", "sex"]:
        group_metrics(test_df, y_true, y_pred, group_col)


if __name__ == "__main__":
    main()
