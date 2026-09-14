import os
import random
from typing import Tuple, List

current_dir = os.path.dirname(os.path.abspath(__file__))

try:
    from pathlib import Path
    default_cache = Path.home() / ".cache" / "huggingface"
    hf_cache_dir = str(default_cache)
    os.makedirs(hf_cache_dir, exist_ok=True)
except Exception:
    hf_cache_dir = os.path.join(current_dir, ".cache", "transformers")
    os.makedirs(hf_cache_dir, exist_ok=True)

hf_cache_dir = os.path.abspath(hf_cache_dir)

if not os.path.isdir(hf_cache_dir):
    raise RuntimeError(f"Could not create cache directory: {hf_cache_dir}")

for key in ["HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE"]:
    if key in os.environ:
        value = os.environ[key]
        if value is None or value == "" or not isinstance(value, str):
            del os.environ[key]

os.environ["HF_HOME"] = hf_cache_dir
os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir

if "HF_TOKEN" not in os.environ:
    os.environ["HF_TOKEN"] = ""  
if "HUGGINGFACE_HUB_TOKEN" not in os.environ:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = ""

try:
    import huggingface_hub
    huggingface_hub.constants.HF_HUB_CACHE = hf_cache_dir
    if hasattr(huggingface_hub, 'file_download'):
        pass
except (ImportError, AttributeError):
    pass  

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

from transformers import AutoProcessor, AutoModel
try:
    from transformers import WavLMModel, WavLMProcessor
    WAVLM_AVAILABLE = True
except ImportError:
    WAVLM_AVAILABLE = False
    WavLMModel = None
    WavLMProcessor = None

def patch_transformers_paths():
    """Patch transformers to replace None paths with valid cache directory"""
    try:
        import transformers.utils.hub as hub_utils
        if hasattr(hub_utils, 'default_cache_path'):
            if hub_utils.default_cache_path is None:
                hub_utils.default_cache_path = hf_cache_dir
    except (AttributeError, ImportError):
        pass
    
    try:
        import transformers.file_utils as file_utils
        if hasattr(file_utils, 'default_cache_path'):
            if file_utils.default_cache_path is None:
                file_utils.default_cache_path = hf_cache_dir
    except (AttributeError, ImportError):
        pass

    try:
        from transformers.utils import cached_path
        original_cached_path = cached_path
        
        def patched_cached_path(path_or_repo_id, *args, **kwargs):
            if path_or_repo_id is None:
                raise ValueError("Path cannot be None")
            if 'cache_dir' in kwargs and kwargs['cache_dir'] is None:
                kwargs['cache_dir'] = hf_cache_dir
            elif 'cache_dir' not in kwargs:
                kwargs['cache_dir'] = hf_cache_dir
            return original_cached_path(path_or_repo_id, *args, **kwargs)
        
        import transformers.utils
        transformers.utils.cached_path = patched_cached_path
    except (AttributeError, ImportError):
        pass

patch_transformers_paths()

def wrap_from_pretrained(original_method):
    """Wrapper for from_pretrained that handles NoneType errors"""
    def wrapped(*args, **kwargs):
        try:
            return original_method(*args, **kwargs)
        except TypeError as e:
            if "NoneType" in str(e) or "expected str" in str(e):
                fixed_kwargs = {}
                for key, value in kwargs.items():
                    if value is None and key in ['cache_dir', 'token', 'local_files_only']:
                        if key == 'cache_dir':
                            fixed_kwargs[key] = hf_cache_dir
                        elif key == 'local_files_only':
                            fixed_kwargs[key] = False
                        else:
                            fixed_kwargs[key] = value
                    else:
                        fixed_kwargs[key] = value
                
                if 'cache_dir' not in fixed_kwargs:
                    fixed_kwargs['cache_dir'] = hf_cache_dir
                
                try:
                    return original_method(*args, **fixed_kwargs)
                except Exception:
                    kwargs['cache_dir'] = hf_cache_dir
                    return original_method(*args, **kwargs)
            else:
                raise
    return wrapped

try:
    AutoProcessor.from_pretrained = wrap_from_pretrained(AutoProcessor.from_pretrained)
    AutoModel.from_pretrained = wrap_from_pretrained(AutoModel.from_pretrained)
except:
    pass

try:
    import transformers
    if hasattr(transformers, 'TRANSFORMERS_CACHE'):
        transformers.TRANSFORMERS_CACHE = hf_cache_dir
except (AttributeError, ImportError):
    pass


import os.path as _original_os_path_module
import inspect
_original_join = _original_os_path_module.join

def _safe_join(*args):
    """os.path.join wrapper that replaces None with cache directory for transformers"""
    try:
        stack = inspect.stack()
        is_transformers_call = any(
            'transformers' in str(frame.filename) or 'huggingface' in str(frame.filename)
            for frame in stack[1:6]  # Check a few frames up
        )
    except:
        is_transformers_call = False
    
    filtered = []
    for arg in args:
        if arg is None:
            if is_transformers_call:
                filtered.append(hf_cache_dir)
            else:
                return _original_join(*args)
        else:
            filtered.append(arg)
    
    return _original_join(*filtered)

import os
os.path.join = _safe_join
_original_os_path_module.join = _safe_join

_original_fspath = os.fspath
def _safe_fspath(path):
    """os.fspath wrapper that replaces None with cache directory for transformers"""
    if path is None:

        try:
            import inspect
            stack = inspect.stack()
            is_transformers_call = any(
                'transformers' in str(frame.filename) or 'huggingface' in str(frame.filename)
                for frame in stack[1:6]
            )
            if is_transformers_call:
                return hf_cache_dir
        except:
            pass
        return _original_fspath(path)
    return _original_fspath(path)

os.fspath = _safe_fspath

print(f"Current Working Directory: {current_dir}")
print(f"HuggingFace cache directory: {hf_cache_dir}")

METADATA_CSV = os.path.join(current_dir, "data/metadata_acoustic.csv") 

MODEL_NAME = "microsoft/wavlm-base-plus"     
TARGET_SR = 16000                             
BATCH_SIZE = 8
NUM_EPOCHS = 10
LR = 1e-4
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)


set_seed(RANDOM_SEED)

def make_speaker_splits(df: pd.DataFrame,
                        train_ratio=0.7,
                        val_ratio=0.15,
                        test_ratio=0.15) -> pd.DataFrame:
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


def intent_to_label(intent: str) -> int:
    intent = intent.lower()
    if intent == "neutral":
        return 0
    elif intent == "trustworthy":
        return 1
    else:
        raise ValueError(f"Unknown intent: {intent}")


ACOUSTIC_FEATURE_COLS = ["duration_sec", "f0_mean_hz", "f0_std_hz", "hnr_mean_db"]


class HybridTrustDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

        missing_cols = [col for col in ACOUSTIC_FEATURE_COLS if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing acoustic feature columns: {missing_cols}. Please run step 3 (3_extract_acoustic_features.py) first.")

        self.df[ACOUSTIC_FEATURE_COLS] = self.df[ACOUSTIC_FEATURE_COLS].fillna(
            self.df[ACOUSTIC_FEATURE_COLS].mean()
        )

        self.means = self.df[ACOUSTIC_FEATURE_COLS].mean()
        self.stds = self.df[ACOUSTIC_FEATURE_COLS].std().replace(0, 1.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav_path = row["wav_path"]

        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio file not found: {wav_path}")

        try:
            y, sr = sf.read(wav_path)
        except Exception as e:
            raise RuntimeError(f"Error reading audio file {wav_path}: {e}")
        
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

        feats = row[ACOUSTIC_FEATURE_COLS].values.astype(np.float32)
        feats_norm = (feats - self.means.values) / (self.stds.values + 1e-9)

        label = intent_to_label(row["intent"])

        return y.astype(np.float32), feats_norm, label


def collate_fn(batch: List):
    """
    Custom collate function:
    - batch: list of (audio_np, feats_np, label_int)
    Returns:
      - list of 1D numpy arrays (audio signals)
      - tensor of acoustic features [B, F]
      - tensor of labels [B]
    (We keep audio as list because WavLM processor handles padding.)
    """
    audios = [item[0] for item in batch]
    feats = np.stack([item[1] for item in batch], axis=0)
    labels = np.array([item[2] for item in batch], dtype=np.int64)

    feats_tensor = torch.tensor(feats, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return audios, feats_tensor, labels_tensor

class WavLMHybridClassifier(nn.Module):
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

    def forward(self, wavlm_embeddings, acoustic_feats):
        """
        wavlm_embeddings: [B, H]
        acoustic_feats:   [B, F]
        """
        x = torch.cat([wavlm_embeddings, acoustic_feats], dim=1)
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
    df: test dataframe aligned with y_true/y_pred rows
    group_col: one of ["ethnicity", "age_group", "sex"]
    """
    if group_col not in df.columns:
        print(f"\n=== Group metrics by {group_col} ===")
        print(f"Warning: Column '{group_col}' not found in dataframe. Skipping group analysis.")
        return
    
    print(f"\n=== Group metrics by {group_col} ===")
    df_local = df.reset_index(drop=True)
    df_local["y_true"] = y_true
    df_local["y_pred"] = y_pred

    unique_values = df_local[group_col].dropna().unique()
    if len(unique_values) == 0:
        print(f"No valid values found in column '{group_col}'. Skipping group analysis.")
        return

    for group_value in sorted(unique_values):
        subset = df_local[df_local[group_col] == group_value]
        if len(subset) == 0:
            continue
        acc = accuracy_score(subset["y_true"], subset["y_pred"])
        print(f"{group_col} = {group_value:12s} | N = {len(subset):3d} | Accuracy = {acc:.3f}")


def main():
    if not os.path.exists(METADATA_CSV):
        raise FileNotFoundError(f"Metadata file not found: {METADATA_CSV}. Please run step 3 (3_extract_acoustic_features.py) first.")
    
    df = pd.read_csv(METADATA_CSV)

    required_cols = ["wav_path", "intent", "speaker_id"] + ACOUSTIC_FEATURE_COLS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata: {missing_cols}")

    df = df[df["intent"].isin(["neutral", "trustworthy"])]
    
    if len(df) == 0:
        raise ValueError("No valid samples found with 'neutral' or 'trustworthy' intent labels.")

    df = make_speaker_splits(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    print("Split distribution:")
    print(df["split"].value_counts())

    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]
    test_df  = df[df["split"] == "test"]

    train_dataset = HybridTrustDataset(train_df)
    val_dataset   = HybridTrustDataset(val_df)
    test_dataset  = HybridTrustDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print("Loading WavLM model…")
    print(f"Cache directory: {hf_cache_dir}")
    print(f"Cache dir exists: {os.path.exists(hf_cache_dir)}")
    print(f"Cache dir is directory: {os.path.isdir(hf_cache_dir)}")
    print(f"HF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
    print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', 'Not set')}")

    try:
        import transformers
        tf_version = transformers.__version__
        print(f"Transformers version: {tf_version}")

        if tf_version.startswith("4.57"):
            print("\n" + "="*60)
            print("WARNING: You are using transformers 4.57.x which has a known")
            print("bug causing NoneType errors when loading models.")
            print("="*60)
            print("If you encounter 'expected str, bytes or os.PathLike object, not NoneType'")
            print("errors, please try downgrading transformers:")
            print("  pip uninstall transformers -y")
            print("  pip install transformers==4.40.0")
            print("="*60 + "\n")
    except:
        pass

    if not os.path.isdir(hf_cache_dir):
        raise RuntimeError(f"Cache directory is not valid: {hf_cache_dir}")

    cache_dir_str = str(os.path.abspath(hf_cache_dir))
    assert os.path.isdir(cache_dir_str), f"Cache directory must exist: {cache_dir_str}"

    print("Checking for cached model files...")
    try:
        from huggingface_hub import snapshot_download
        try:
            cached_path = snapshot_download(
                repo_id=MODEL_NAME,
                cache_dir=cache_dir_str,
                local_files_only=True  
            )
            if cached_path and os.path.isdir(cached_path):
                print(f"Found cached model at: {cached_path}")
                print("Attempting to load from cache with local_files_only=True...")
                try:
                    processor = AutoProcessor.from_pretrained(cached_path, local_files_only=True)
                    wavlm_model = AutoModel.from_pretrained(cached_path, local_files_only=True).to(DEVICE)
                    print("Model loaded successfully from cache!")
                    model_loaded = True
                except Exception as cache_error:
                    print(f"Error loading from cache: {cache_error}")
                    model_loaded = False
            else:
                model_loaded = False
        except Exception:
            model_loaded = False
    except ImportError:
        model_loaded = False
    
    if not model_loaded:
        print("Attempting to load model directly (letting transformers handle cache)...")
        print("This may take a few minutes on first run as it downloads the model.")
        
    try:
        print("Loading processor (this will download the model if not cached)...")
        print("This may take a few minutes on first run.")
        
        try:
            from transformers import Wav2Vec2FeatureExtractor
            processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME, resume_download=True)
            print("Processor (Wav2Vec2FeatureExtractor) loaded successfully.")
        except (ImportError, Exception) as feat_err:
            print(f"Could not load Wav2Vec2FeatureExtractor: {feat_err}")
            print("Trying AutoProcessor...")
            processor = AutoProcessor.from_pretrained(
                MODEL_NAME, 
                resume_download=True,
                trust_remote_code=False
            )
            print("Processor loaded successfully.")
        
        print("Loading model...")
        wavlm_model = AutoModel.from_pretrained(MODEL_NAME, resume_download=True).to(DEVICE)
        print("Model loaded successfully.")
        
    except (TypeError, ValueError, OSError) as e:
        if "NoneType" in str(e) or "expected str" in str(e) or "Can't load tokenizer" in str(e) or "tokenizer" in str(e).lower():
            print(f"Error with default loading: {e}")
            print("Trying with explicit cache_dir...")
            try:
                processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=cache_dir_str, resume_download=True)
                wavlm_model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=cache_dir_str, resume_download=True).to(DEVICE)
                print("Model loaded successfully with explicit cache_dir.")
            except Exception as e2:
                if "NoneType" in str(e2) or "expected str" in str(e2) or "Can't load tokenizer" in str(e2) or "tokenizer" in str(e2).lower():
                    print(f"Error with explicit cache_dir: {e2}")
                    print("Trying Wav2Vec2FeatureExtractor (WavLM doesn't use tokenizer)...")
                    try:
                        from transformers import Wav2Vec2FeatureExtractor
                        processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME, cache_dir=cache_dir_str, resume_download=True)
                        wavlm_model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=cache_dir_str, resume_download=True).to(DEVICE)
                        print("Model loaded successfully using Wav2Vec2FeatureExtractor.")
                    except (ImportError, Exception) as feat_err:
                        print(f"Error with Wav2Vec2FeatureExtractor: {feat_err}")
                        try:
                            from transformers import Wav2Vec2Processor
                            processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME, cache_dir=cache_dir_str, resume_download=True)
                            wavlm_model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=cache_dir_str, resume_download=True).to(DEVICE)
                            print("Model loaded successfully using Wav2Vec2Processor.")
                        except (ImportError, Exception) as proc_err:
                            print(f"Error with Wav2Vec2Processor: {proc_err}")
                            pass

                    print("Trying alternative: download with huggingface_hub first...")
                    try:
                        from huggingface_hub import snapshot_download
                        print("Downloading model files...")
                        local_model_path = snapshot_download(
                            repo_id=MODEL_NAME,
                            cache_dir=cache_dir_str,
                            local_files_only=False
                        )
                        local_model_path = str(os.path.abspath(local_model_path))
                        print(f"Files downloaded to: {local_model_path}")

                        required_files = ["config.json", "preprocessor_config.json"]
                        missing = [f for f in required_files if not os.path.exists(os.path.join(local_model_path, f))]
                        if missing:
                            print(f"Warning: Some files may be missing: {missing}")

                        print("Loading from downloaded files...")
                        try:
                            from transformers import Wav2Vec2FeatureExtractor
                            processor = Wav2Vec2FeatureExtractor.from_pretrained(local_model_path, local_files_only=True)
                            wavlm_model = AutoModel.from_pretrained(local_model_path, local_files_only=True).to(DEVICE)
                            print("Model loaded successfully using Wav2Vec2FeatureExtractor (local_files_only).")
                        except (ImportError, Exception) as feat_err:
                            print(f"Error with Wav2Vec2FeatureExtractor: {feat_err}")
                            try:
                                from transformers import Wav2Vec2Processor
                                processor = Wav2Vec2Processor.from_pretrained(local_model_path, local_files_only=True)
                                wavlm_model = AutoModel.from_pretrained(local_model_path, local_files_only=True).to(DEVICE)
                                print("Model loaded successfully using Wav2Vec2Processor (local_files_only).")
                            except (ImportError, Exception) as proc_err:
                                print(f"Error with Wav2Vec2Processor: {proc_err}")
                                try:
                                    processor = AutoProcessor.from_pretrained(local_model_path, local_files_only=True)
                                    wavlm_model = AutoModel.from_pretrained(local_model_path, local_files_only=True).to(DEVICE)
                                    print("Model loaded successfully from downloaded path (local_files_only).")
                                except Exception as local_err:
                                    print(f"Error with local_files_only: {local_err}")
                                    print("Trying without local_files_only (may download additional files)...")
                                    processor = AutoProcessor.from_pretrained(local_model_path, resume_download=True)
                                    wavlm_model = AutoModel.from_pretrained(local_model_path, resume_download=True).to(DEVICE)
                                    print("Model loaded successfully from downloaded path.")
                    except Exception as e3:
                        error_msg = (
                            f"\n{'='*70}\n"
                            f"CRITICAL ERROR: Model Loading Failed\n"
                            f"{'='*70}\n"
                            f"\nAll model loading attempts have failed.\n"
                            f"\n{'='*70}\n"
                            f"Error Details:\n"
                            f"  Error 1: {e}\n"
                            f"  Error 2: {e2}\n"
                            f"  Error 3: {e3}\n"
                            f"{'='*70}\n"
                            f"\nSOLUTIONS:\n"
                            f"1. Clear cache and retry:\n"
                            f"   Delete: {hf_cache_dir}\n"
                            f"   Then run this script again\n"
                            f"2. Check your internet connection\n"
                            f"3. Try manually downloading the model\n"
                            f"{'='*70}\n"
                        )
                        raise RuntimeError(error_msg)
                else:
                    raise
        else:
            raise
    except ImportError as import_err:
            if "snapshot_download" in str(import_err) or "huggingface_hub" in str(import_err):
                print("huggingface_hub not available, trying direct loading...")
                try:
                    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=cache_dir_str, resume_download=True)
                    wavlm_model = AutoModel.from_pretrained(MODEL_NAME, cache_dir=cache_dir_str, resume_download=True).to(DEVICE)
                    print("Model loaded successfully with explicit cache_dir.")
                except Exception as e:
                    error_str = str(e)
                    if "NoneType" in error_str or "expected str" in error_str or "Can't load tokenizer" in error_str or "tokenizer" in error_str.lower():
                        print(f"Error with explicit cache_dir: {e}")
                        print("Trying without cache_dir parameter...")
                        try:
                            processor = AutoProcessor.from_pretrained(MODEL_NAME, resume_download=True)
                            wavlm_model = AutoModel.from_pretrained(MODEL_NAME, resume_download=True).to(DEVICE)
                            print("Model loaded successfully using environment variables.")
                        except Exception as e2:
                            error_msg = (
                                f"\n{'='*60}\n"
                                f"CRITICAL ERROR: NoneType path issue in transformers library\n"
                                f"{'='*60}\n"
                                f"All loading attempts failed.\n"
                                f"Import error: {import_err}\n"
                                f"Direct load error: {e}\n"
                                f"Final error: {e2}\n\n"
                                f"This is a known bug in transformers 4.57.3.\n\n"
                                f"SOLUTIONS:\n"
                                f"1. Try: pip install transformers==4.40.0 (if compatible)\n"
                                f"2. Clear cache: Delete {hf_cache_dir} and retry\n"
                                f"3. Report this as a bug to transformers GitHub\n"
                                f"{'='*60}\n"
                            )
                            raise RuntimeError(error_msg)
                    else:
                        raise
            else:
                raise
    except Exception as e:
        error_str = str(e)
        if "tokenizer" in error_str.lower() or "Can't load" in error_str:
            error_msg = (
                f"\n{'='*60}\n"
                f"Error loading tokenizer/model\n"
                f"{'='*60}\n"
                f"Error: {e}\n\n"
                f"This usually means:\n"
                f"1. The model files are not fully downloaded\n"
                f"2. The cache is corrupted\n"
                f"3. Network connection issues\n\n"
                f"SOLUTIONS:\n"
                f"1. Clear the cache and retry:\n"
                f"   Delete: {hf_cache_dir}\n"
                f"   Then run this script again\n"
                f"2. Check your internet connection\n"
                f"3. Try manually downloading:\n"
                f"   from huggingface_hub import snapshot_download\n"
                f"   snapshot_download('{MODEL_NAME}')\n"
                f"{'='*60}\n"
            )
        else:
            error_msg = (
                f"\n{'='*60}\n"
                f"Unexpected error loading model\n"
                f"{'='*60}\n"
                f"Error: {e}\n"
                f"Please check your internet connection and try again.\n"
                f"{'='*60}\n"
            )
        raise RuntimeError(error_msg)

    for param in wavlm_model.parameters():
        param.requires_grad = False

    wavlm_hidden_dim = wavlm_model.config.hidden_size
    acoustic_dim = len(ACOUSTIC_FEATURE_COLS)

    classifier = WavLMHybridClassifier(wavlm_hidden_dim, acoustic_dim, num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(classifier, wavlm_model, processor, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = eval_model(classifier, wavlm_model, processor, val_loader, criterion)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = classifier.state_dict()

    if best_state is not None:
        classifier.load_state_dict(best_state)

    test_loss, test_acc, y_true, y_pred = eval_model(classifier, wavlm_model, processor, test_loader, criterion)
    print("\n=== HYBRID MODEL – TEST RESULTS ===")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.3f}")
    print("\nClassification report (0=neutral, 1=trustworthy):")
    print(classification_report(y_true, y_pred, target_names=["neutral", "trustworthy"]))

    for group_col in ["ethnicity", "age_group", "sex"]:
        group_metrics(test_df, y_true, y_pred, group_col)


if __name__ == "__main__":
    main()
