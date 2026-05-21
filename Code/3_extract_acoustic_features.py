import os
import numpy as np
import pandas as pd
import soundfile as sf
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm

# Paths
# Get the directory where the script is actually sitting
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current Working Directory: {current_dir}")

METADATA_IN = os.path.join(current_dir, "data/metadata.csv")            # your existing file
METADATA_OUT = os.path.join(current_dir, "data/metadata_acoustic.csv")    # new file with extra features

# Pitch settings (Praat-style)
PITCH_MIN_HZ = 75   # ok for mixed-sex dataset
PITCH_MAX_HZ = 500

def compute_features(wav_path: str):
    """
    Compute acoustic features relevant to trust from a .wav file.
    Returns a dict with:
      - duration_sec
      - f0_mean_hz
      - f0_std_hz
      - hnr_mean_db
    You can extend this with jitter, shimmer, CPP later if you wish.
    """
    try:
        snd = parselmouth.Sound(wav_path)
    except Exception as e:
        print(f"Error loading {wav_path}: {e}")
        return {
            "duration_sec": np.nan,
            "f0_mean_hz": np.nan,
            "f0_std_hz": np.nan,
            "hnr_mean_db": np.nan,
        }

    # Duration
    duration_sec = snd.duration

    # Pitch (F0)
    pitch = snd.to_pitch(time_step=None, pitch_floor=PITCH_MIN_HZ, pitch_ceiling=PITCH_MAX_HZ)
    f0_values = pitch.selected_array["frequency"]  # Hz
    # Keep only voiced frames
    f0_voiced = f0_values[f0_values > 0]
    if len(f0_voiced) > 0:
        f0_mean = float(np.mean(f0_voiced))
        f0_std = float(np.std(f0_voiced))
    else:
        f0_mean = np.nan
        f0_std = np.nan

    # Harmonics-to-Noise Ratio (HNR)
    try:
        harm = snd.to_harmonicity_cc(time_step=None, minimum_pitch=PITCH_MIN_HZ)
        hnr_mean_db = float(call(harm, "Get mean", 0, 0))  # over whole file
    except Exception:
        hnr_mean_db = np.nan

    return {
        "duration_sec": duration_sec,
        "f0_mean_hz": f0_mean,
        "f0_std_hz": f0_std,
        "hnr_mean_db": hnr_mean_db,
    }


def main():
    df = pd.read_csv(METADATA_IN)

    # Ensure wav_path column exists
    if "wav_path" not in df.columns:
        raise ValueError("metadata.csv must contain a 'wav_path' column with full paths to .wav files.")

    durations = []
    f0_means = []
    f0_stds = []
    hnrs = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting acoustic features"):
        wav_path = row["wav_path"]
        feats = compute_features(wav_path)
        durations.append(feats["duration_sec"])
        f0_means.append(feats["f0_mean_hz"])
        f0_stds.append(feats["f0_std_hz"])
        hnrs.append(feats["hnr_mean_db"])

    df["duration_sec"] = durations
    df["f0_mean_hz"] = f0_means
    df["f0_std_hz"] = f0_stds
    df["hnr_mean_db"] = hnrs

    df.to_csv(METADATA_OUT, index=False)
    print(f"Saved extended metadata to {METADATA_OUT}")


if __name__ == "__main__":
    main()
