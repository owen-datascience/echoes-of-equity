Nice, you’re at the fun part now 😄
You already have the hard piece (prosody features) done, so training the **Hybrid Model (WavLM + Prosody)** is just wiring things together.

Below is an **end-to-end step**:

1. Use `metadata_acoustic_prosody.csv`
2. Feed **raw audio** into WavLM → get an embedding
3. Concatenate embedding with your 6 prosody features
4. Train a small classifier
5. Evaluate overall + by ethnicity / age_group / sex

---

## 0. Prereqs

If not done yet:

```bash
pip install transformers torchaudio soundfile librosa scikit-learn
```

Make sure you have this file from the previous step:

```text
data/metadata_acoustic_prosody.csv
```

with (at least) these columns:

```text
wav_path, intent, speaker_id, ethnicity, age_group, sex,
duration_sec, f0_mean_hz, f0_std_hz, hnr_mean_db, jitter_local, cpp_mean_db
```

---

## 1. Full Hybrid Model Script (WavLM + Prosody)

Save this as **`train_hybrid_wavlm_prosody.py`** and run it with:

```bash
python train_hybrid_wavlm_prosody.py
```

---

## 2. What to look at in the output

When you run this script, pay attention to:

1. **Final test accuracy**

   * Compare to your **acoustic-only 0.684**.
   * If hybrid > 0.684 → strong evidence that WavLM embeddings add value.

2. **Trustworthy class recall**

   * Earlier deep baselines had recall ≈ 0.0
   * Now we want recall for `trustworthy` to clearly improve (e.g., 0.6–0.8).

3. **Group metrics**

   * Check accuracy by `ethnicity`, `age_group`, `sex`
   * If some group is much lower, that’s a **fairness concern** to analyze in your project.

---

## 3. How this fits your ISEF story

You now have three key models:

* **Prosody-only Random Forest** – already at ~68%
* **Hybrid WavLM + Prosody** – should beat that
* (Optional) **WavLM-only** – likely worse than hybrid and maybe even worse than prosody-only

This lets your student claim:

> “We show that prosodic features (pitch, jitter, HNR, CPP, duration) are crucial for detecting trustworthy intention in speech. A hybrid model that combines WavLM embeddings with prosody achieves the best performance and reveals how acoustic trust cues vary by demographic group.”

When you have the hybrid model’s **classification report and group metrics**, paste them here and I’ll help you:

* interpret them,
* turn them into **figures & tables**, and
* write ISEF-level **Results + Discussion** text.
