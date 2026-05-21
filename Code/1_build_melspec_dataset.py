import os
import re
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

# ========= CONFIG =========
# Get the directory where the script is actually sitting
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current Working Directory: {current_dir}")

# Folder where all 1152 .wav files live (can contain subfolders)
DATA_DIR = os.path.join(current_dir, "data/wav")           # folder where all 1152 .wav files live (can contain subfolders)

# Where to save .npy spectrograms
OUT_SPEC_DIR = os.path.join(current_dir, "data/melspec")   # where to save .npy spectrograms

# Where to save labels/metadata
OUT_METADATA_CSV =os.path.join(current_dir, "data/metadata.csv")

# Audio / mel-spectrogram parameters
TARGET_SR = 16000        # resample to 16kHz
N_MELS = 64              # number of mel bands
N_FFT = 1024
HOP_LENGTH = 256

os.makedirs(OUT_SPEC_DIR, exist_ok=True)

# ========= UPDATED FILENAME PARSER =========
"""
We handle filenames such as:
- 1901_bof_n01c.wav          (black, older, female, neutral, sentence 01, suffix 'c')
- 1904_bym_n01.wav           (black, younger, male, neutral, sentence 01)
- 2010_aof_n01.wav           (Asian, older, female, neutral, sentence 01)
- 1896_ayf_n01.wav           (Asian, younger, female, neutral, sentence 01)
- 1893_wof_n01.wav           (white, older, female, neutral, sentence 01)
- 1898_wym_n01c.wav          (white, younger, male, neutral, sentence 01, suffix 'c')
- 1923_wom_t12c_version2.wav (white, older, male, trustworthy, sentence 12, suffix 'c_version2')

Pattern:
  speakerID_(ethnicity age sex)_(intent)(two-digit sentence)[optional extra...] .wav
"""

FNAME_RE = re.compile(
    r"""^
    (?P<speaker_id>\d+)_              # e.g. 1901
    (?P<demo>[abw][oy][mf])_          # e.g. bof, bym, aof, wym, wom
    (?P<intent>[nt])                  # n = neutral, t = trustworthy
    (?P<sent>\d{2})                   # two-digit sentence number
    (?P<suffix>.*)?                   # anything extra (e.g. 'c', '_version2')
    \.wav$
    """,
    re.VERBOSE | re.IGNORECASE,
)

ETH_MAP = {
    "w": "white",
    "b": "black",
    "a": "south_asian",  # or "asian" if you prefer that label
}

AGE_MAP = {
    "y": "younger",
    "o": "older",
}

SEX_MAP = {
    "m": "male",
    "f": "female",
}

INTENT_MAP = {
    "n": "neutral",
    "t": "trustworthy",
}


def parse_filename(filename: str):
    """
    Parse filename into labels based on the coding scheme.

    Expected base pattern:
      speakerID_ethnicityAgeSex_intentSentence[extra].wav

    Example:
      1901_bof_n01c.wav          -> speaker_id=1901, demo=bof, intent=n, sent=01
      1923_wom_t12c_version2.wav -> speaker_id=1923, demo=wom, intent=t, sent=12
    """
    m = FNAME_RE.match(filename)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {filename}")

    speaker_id = int(m.group("speaker_id"))
    demo = m.group("demo").lower()
    ethnicity_code = demo[0]
    age_code = demo[1]
    sex_code = demo[2]

    intent_code = m.group("intent").lower()
    sentence_id = int(m.group("sent"))

    return {
        "speaker_id": speaker_id,
        "ethnicity": ETH_MAP.get(ethnicity_code, "unknown"),
        "age_group": AGE_MAP.get(age_code, "unknown"),
        "sex":      SEX_MAP.get(sex_code, "unknown"),
        "intent":   INTENT_MAP.get(intent_code, "unknown"),
        "sentence_id": sentence_id,
        # You can store suffix if you want to inspect later:
        "suffix": m.group("suffix") or "",
    }


# ========= AUDIO HELPERS =========

def load_audio(path: str, target_sr: int = TARGET_SR):
    """
    Load audio and resample to target_sr.
    """
    y, sr = sf.read(path)  # soundfile handles 16-bit wav nicely
    if y.ndim > 1:
        # convert stereo to mono
        y = np.mean(y, axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y, target_sr


def audio_to_melspec(y, sr):
    """
    Convert waveform to log-mel spectrogram.
    Returns: np.ndarray [n_mels, time_frames]
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel + 1e-10)
    return mel_db


# ========= MAIN PIPELINE =========

def main():
    records = []

    # Find all wav files recursively
    wav_paths = []
    for root, dirs, files in os.walk(DATA_DIR):
        for fname in files:
            if fname.lower().endswith(".wav"):
                wav_paths.append(os.path.join(root, fname))

    print(f"Found {len(wav_paths)} .wav files")

    for wav_path in tqdm(wav_paths, desc="Processing audio"):
        fname = os.path.basename(wav_path)

        # 1) Parse filename → labels
        try:
            meta = parse_filename(fname)
        except ValueError as e:
            print(f"Skipping file (name issue): {fname} ({e})")
            continue

        # 2) Load & convert to mel-spectrogram
        try:
            y, sr = load_audio(wav_path, TARGET_SR)
            mel = audio_to_melspec(y, sr)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue

        # 3) Save spectrogram as .npy
        spec_fname = fname.replace(".wav", ".npy")
        spec_path = os.path.join(OUT_SPEC_DIR, spec_fname)
        np.save(spec_path, mel)

        # 4) Build record for metadata
        record = {
            "wav_path": wav_path,
            "spec_path": spec_path,
            "speaker_id": meta["speaker_id"],
            "ethnicity": meta["ethnicity"],
            "age_group": meta["age_group"],
            "sex": meta["sex"],
            "intent": meta["intent"],
            "sentence_id": meta["sentence_id"],
            "suffix": meta["suffix"],
        }
        records.append(record)

    # 5) Save metadata CSV
    df = pd.DataFrame(records)
    df.to_csv(OUT_METADATA_CSV, index=False)
    print(f"Saved metadata with {len(df)} examples to {OUT_METADATA_CSV}")


if __name__ == "__main__":
    main()
