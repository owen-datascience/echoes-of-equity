Awesome, that test result is *exactly* the signal we needed: the simple CNN on spectrograms isn’t really “hearing” trust. Time to do **C + D**: a **hybrid deep model (wav2vec2 + acoustic features)** and **fairness analysis**.

We’ll do it in **two concrete steps**, with full code:

1. **Extract acoustic trust features** (pitch, pitch variability, duration, HNR, etc.) for every file
2. **Train a hybrid model: wav2vec2 embeddings + acoustic features → trust vs neutral**, and analyze performance by **ethnicity, age_group, sex**

---

## Step 1 – Extract acoustic features into a new metadata file

We’ll use:

* `parselmouth` (Python → Praat) for prosodic features
* `soundfile` + `librosa` for audio loading & duration

### 1.1 Install the needed packages

```bash
pip install praat-parselmouth soundfile librosa pandas tqdm
```

### 1.2 Script: `extract_acoustic_features.py`

This reads your existing `metadata.csv`, computes features for each `wav_path`, and writes `metadata_acoustic.csv`.

Run:

```bash
python extract_acoustic_features.py
```

Now you have `data/metadata_acoustic.csv` with **wav paths + labels + acoustic trust features**.

