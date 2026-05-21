Nice choice — adding more detailed prosodic features is exactly what this project needs next. We’ll extend your **feature extraction step** to compute:

* Duration (already done)
* F0 mean (Hz)
* F0 std (Hz)
* HNR mean (dB)
* **Jitter (local)** – cycle-to-cycle pitch variation
* **Shimmer (local)** – cycle-to-cycle amplitude variation
* **CPP-like measure** – using Praat’s harmonicity as a proxy (simple, robust; good enough for ML)

---

## 1️⃣ Make sure dependencies are installed

If you haven’t already:

```bash
pip install praat-parselmouth soundfile librosa pandas tqdm
```

---

## 2️⃣ Updated feature extraction script

Save this as (or overwrite) `extract_acoustic_features.py`.
It reads your existing `metadata.csv` (with `wav_path`), computes **all features** for each file, and writes `metadata_acoustic_prosody.csv`.

```python
import os
import numpy as np
import pandas as pd
import soundfile as sf
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm

# ==== INPUT/OUTPUT PATHS ====
METADATA_IN = r"data/metadata.csv"                        # your existing metadata with wav_path, labels, etc.
METADATA_OUT = r"data/metadata_acoustic_prosody.csv"      # new file with extended features

# Praat-style pitch settings
PITCH_MIN_HZ = 75.0
PITCH_MAX_HZ = 500.0

def compute_acoustic_prosody_features(wav_path: str):
    """
    Compute acoustic + prosodic features relevant to trust:

    - duration_sec
    - f0_mean_hz
    - f0_std_hz
    - hnr_mean_db
    - jitter_local (absolute, seconds)
    - shimmer_local (in dB)
    - cpp_mean_db  (approximate, using harmonicity as a CPP-like measure)

    Returns a dict (values can be np.nan if extraction fails).
    """
    # Default values (in case of error)
    default = {
        "duration_sec": np.nan,
        "f0_mean_hz": np.nan,
        "f0_std_hz": np.nan,
        "hnr_mean_db": np.nan,
        "jitter_local": np.nan,
        "shimmer_local": np.nan,
        "cpp_mean_db": np.nan,
    }

    if not os.path.exists(wav_path):
        print(f"File not found: {wav_path}")
        return default

    try:
        snd = parselmouth.Sound(wav_path)
    except Exception as e:
        print(f"Error loading {wav_path}: {e}")
        return default

    # --- Duration ---
    duration_sec = snd.duration

    # --- Pitch (F0) ---
    try:
        pitch = snd.to_pitch(
            time_step=0.0,
            pitch_floor=PITCH_MIN_HZ,
            pitch_ceiling=PITCH_MAX_HZ,
        )
        f0_values = pitch.selected_array["frequency"]  # Hz
        f0_voiced = f0_values[f0_values > 0]
        if len(f0_voiced) > 0:
            f0_mean = float(np.mean(f0_voiced))
            f0_std = float(np.std(f0_voiced))
        else:
            f0_mean = np.nan
            f0_std = np.nan
    except Exception as e:
        print(f"Pitch error in {wav_path}: {e}")
        f0_mean = np.nan
        f0_std = np.nan

    # --- Harmonicity (used for HNR + CPP-like feature) ---
    try:
        # "To Harmonicity (cc)" ~ harmonicity in dB; often used as a CPP-like measure in practice
        harm = snd.to_harmonicity_cc(
            time_step=0.0,
            minimum_pitch=PITCH_MIN_HZ
        )
        hnr_mean_db = float(call(harm, "Get mean", 0, 0))      # over entire file
        cpp_mean_db = hnr_mean_db                              # we treat this as a CPP-like measure
    except Exception as e:
        print(f"Harmonicity error in {wav_path}: {e}")
        hnr_mean_db = np.nan
        cpp_mean_db = np.nan

    # --- PointProcess for jitter & shimmer ---
    try:
        # Periodic point process extraction
        point_process = call(
            snd,
            "To PointProcess (periodic, cc)",
            PITCH_MIN_HZ,
            PITCH_MAX_HZ
        )

        # Praat manual suggests:
        # min_pitch_period = 1 / PITCH_MAX
        # max_pitch_period = 1 / PITCH_MIN
        min_period = 1.0 / PITCH_MAX_HZ
        max_period = 1.0 / PITCH_MIN_HZ

        # Jitter (local): relative cycle-to-cycle F0 variation
        jitter_local = call(
            point_process,
            "Get jitter (local)",
            0.0,            # start time
            0.0,            # end time (0 = whole file)
            min_period,     # minimum period
            max_period,     # maximum period
            1.3             # maximum period factor
        )

        # Shimmer (local): relative cycle-to-cycle amplitude variation
        shimmer_local = call(
            [snd, point_process],
            "Get shimmer (local)",
            0.0,            # start time
            0.0,            # end time
            min_period,     # minimum period
            max_period,     # maximum period
            1.3,            # maximum period factor
            0.03,           # maximum amplitude factor (default from Praat)
            0.45            # silence threshold
        )
    except Exception as e:
        print(f"Jitter/Shimmer error in {wav_path}: {e}")
        jitter_local = np.nan
        shimmer_local = np.nan

    return {
        "duration_sec": duration_sec,
        "f0_mean_hz": f0_mean,
        "f0_std_hz": f0_std,
        "hnr_mean_db": hnr_mean_db,
        "jitter_local": jitter_local,
        "shimmer_local": shimmer_local,
        "cpp_mean_db": cpp_mean_db,
    }


def main():
    df = pd.read_csv(METADATA_IN)

    if "wav_path" not in df.columns:
        raise ValueError("metadata.csv must contain a 'wav_path' column with full paths to .wav files.")

    durations = []
    f0_means = []
    f0_stds = []
    hnrs = []
    jitters = []
    shimmers = []
    cpps = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting acoustic + prosodic features"):
        wav_path = row["wav_path"]
        feats = compute_acoustic_prosody_features(wav_path)

        durations.append(feats["duration_sec"])
        f0_means.append(feats["f0_mean_hz"])
        f0_stds.append(feats["f0_std_hz"])
        hnrs.append(feats["hnr_mean_db"])
        jitters.append(feats["jitter_local"])
        shimmers.append(feats["shimmer_local"])
        cpps.append(feats["cpp_mean_db"])

    df["duration_sec"] = durations
    df["f0_mean_hz"] = f0_means
    df["f0_std_hz"] = f0_stds
    df["hnr_mean_db"] = hnrs
    df["jitter_local"] = jitters
    df["shimmer_local"] = shimmers
    df["cpp_mean_db"] = cpps

    df.to_csv(METADATA_OUT, index=False)
    print(f"Saved extended metadata to {METADATA_OUT}")


if __name__ == "__main__":
    main()
```

Then run:

```bash
python extract_acoustic_features.py
```

You should now have **`data/metadata_acoustic_prosody.csv`** with these columns added:

* `duration_sec`
* `f0_mean_hz`
* `f0_std_hz`
* `hnr_mean_db`
* `jitter_local`
* `shimmer_local`
* `cpp_mean_db`

---

## 3️⃣ Plug these new features into your hybrid WavLM model

In your `train_hybrid_wavlm_acoustic.py` (or equivalent), do two small updates:

### 3.1 Point to the new CSV

```python
METADATA_CSV = r"data/metadata_acoustic_prosody.csv"
```

### 3.2 Expand the feature list

Change:

```python
ACOUSTIC_FEATURE_COLS = ["duration_sec", "f0_mean_hz", "f0_std_hz", "hnr_mean_db"]
```

to:

```python
ACOUSTIC_FEATURE_COLS = [
    "duration_sec",
    "f0_mean_hz",
    "f0_std_hz",
    "hnr_mean_db",
    "jitter_local",
    "shimmer_local",
    "cpp_mean_db",
]
```

Leave the rest of the training code as-is, then run your training script again.

---

## 4️⃣ What to look for in the new results

When you rerun training, check:

1. **Overall test accuracy** – ideally > 0.5 now
2. **Recall for “trustworthy”** – we want this to move away from 0.00
3. **Group metrics** – see if some groups improve more than others:

   * maybe older speakers show clearer trust cues
   * maybe male vs female patterns differ

Those changes will become **evidence in your ISEF story** that:

* Detailed prosody features (jitter, shimmer, CPP) carry extra signal about trust intention.
* Different demographic groups may or may not benefit equally from the same feature set.

---

If you share the **new classification report + group metrics**, I can:

* Help interpret them in “paper style” language,
* Suggest which **plots and tables** to make, and
* Draft the **Results & Discussion** section for the student’s ISEF paper/poster.
