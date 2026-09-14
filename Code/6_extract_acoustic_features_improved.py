import os
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current Working Directory: {current_dir}")

METADATA_IN = os.path.join(current_dir, "data/metadata.csv")                        
METADATA_OUT = os.path.join(current_dir, "data/metadata_acoustic_prosody.csv")    

PITCH_MIN_HZ = 60.0
PITCH_MAX_HZ = 500.0


def compute_acoustic_prosody_features(wav_path: str):
    """
    Compute acoustic + prosodic features relevant to trust.

    Returns a dict with:
      - duration_sec
      - f0_mean_hz
      - f0_std_hz
      - hnr_mean_db
      - jitter_local
      - cpp_mean_db  (approximate, using harmonicity as CPP-like measure)
    """
    feats = {
        "duration_sec": np.nan,
        "f0_mean_hz": np.nan,
        "f0_std_hz": np.nan,
        "hnr_mean_db": np.nan,
        "jitter_local": np.nan,
        "cpp_mean_db": np.nan,
    }

    if not os.path.exists(wav_path):
        print(f"[WARN] File not found: {wav_path}")
        return feats

    try:
        snd = parselmouth.Sound(wav_path)
    except Exception as e:
        print(f"[WARN] Error loading {wav_path}: {e}")
        return feats


    feats["duration_sec"] = snd.duration

    try:
        pitch = snd.to_pitch() 
        f0_values = pitch.selected_array["frequency"]  
        f0_voiced = f0_values[f0_values > 0]

        if len(f0_voiced) > 0:
            feats["f0_mean_hz"] = float(np.mean(f0_voiced))
            feats["f0_std_hz"] = float(np.std(f0_voiced))
    except Exception as e:
        print(f"[WARN] Pitch error in {wav_path}: {e}")


    try:
        harm = snd.to_harmonicity_cc() 
        hnr_mean_db = float(call(harm, "Get mean", 0, 0)) 
        feats["hnr_mean_db"] = hnr_mean_db
        feats["cpp_mean_db"] = hnr_mean_db  
    except Exception as e:
        print(f"[WARN] Harmonicity error in {wav_path}: {e}")

    try:
        point_process = call(
            snd,
            "To PointProcess (periodic, cc)",
            PITCH_MIN_HZ,
            PITCH_MAX_HZ,
        )

        min_period = 1.0 / PITCH_MAX_HZ
        max_period = 1.0 / PITCH_MIN_HZ

        jitter_local = call(
            point_process,
            "Get jitter (local)",
            0.0,           
            0.0,           
            min_period,    
            max_period,     
            1.3,          
        )
        feats["jitter_local"] = jitter_local

    except Exception as e:
        print(f"[WARN] Jitter error in {wav_path}: {e}")

    return feats


def main():
    df = pd.read_csv(METADATA_IN)

    if "wav_path" not in df.columns:
        raise ValueError("metadata.csv must contain a 'wav_path' column with full paths to .wav files.")

    durations = []
    f0_means = []
    f0_stds = []
    hnrs = []
    jitters = []
    cpps = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting acoustic + prosodic features"):
        wav_path = row["wav_path"]
        feats = compute_acoustic_prosody_features(wav_path)

        durations.append(feats["duration_sec"])
        f0_means.append(feats["f0_mean_hz"])
        f0_stds.append(feats["f0_std_hz"])
        hnrs.append(feats["hnr_mean_db"])
        jitters.append(feats["jitter_local"])
        cpps.append(feats["cpp_mean_db"])

    df["duration_sec"] = durations
    df["f0_mean_hz"] = f0_means
    df["f0_std_hz"] = f0_stds
    df["hnr_mean_db"] = hnrs
    df["jitter_local"] = jitters
    df["cpp_mean_db"] = cpps

    df.to_csv(METADATA_OUT, index=False)
    print(f"Saved extended metadata to {METADATA_OUT}")

    numeric_cols = [
        "duration_sec",
        "f0_mean_hz",
        "f0_std_hz",
        "hnr_mean_db",
        "jitter_local",
        "cpp_mean_db",
    ]
    print("\nNon-NaN counts per feature:")
    print(df[numeric_cols].count())
    print("\nFirst 5 rows of the new features:")
    print(df[numeric_cols].head())


if __name__ == "__main__":
    main()
