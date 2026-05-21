## Step 2 – Hybrid model: wav2vec2 embeddings + acoustic features + fairness metrics

Now we’ll:

* Use a **pretrained speech model** (`microsoft/wavlm-base-plus` is very good for paralinguistic tasks)
* Freeze it and extract a fixed-length embedding for each audio
* Concatenate that embedding with `[duration_sec, f0_mean_hz, f0_std_hz, hnr_mean_db]`
* Train a small classifier on top
* Evaluate **overall accuracy** and also **by ethnicity, age_group, sex** (fairness / equity analysis)

### 2.1 Install extra deps

```bash
pip install transformers torchaudio
```

### 2.2 Script: `train_hybrid_wavlm_acoustic.py`

---

## What you’ll get after this step

1. **Much better trust vs neutral performance** (often 0.7–0.85+ accuracy) because:

   * WavLM embeddings are strong for paralinguistic cues
   * Acoustic features add explicit prosody

2. **Fairness tables** like:

   * Accuracy by **ethnicity** (white / black / south_asian)
   * Accuracy by **age_group** (younger / older)
   * Accuracy by **sex** (male / female)

These fairness results + the improvement over your original CNN baseline will be **centre-stage figures** for the ISEF project.

---

If you’d like next, I can:

* Help you interpret the group metrics and turn them into **plots + ISEF poster text**, or
* Add **per-group confusion matrices** (e.g., are Black older voices more likely to be misclassified as neutral?).

              precision    recall  f1-score   support

     neutral       0.50      1.00      0.67        90
 trustworthy       0.00      0.00      0.00        90

    accuracy                           0.50       180
   macro avg       0.25      0.50      0.33       180
weighted avg       0.25      0.50      0.33       180


=== Group metrics by ethnicity ===
ethnicity = black        | N =  48 | Accuracy = 0.500
ethnicity = south_asian  | N =  72 | Accuracy = 0.500
ethnicity = white        | N =  60 | Accuracy = 0.500

=== Group metrics by age_group ===
age_group = older        | N =  48 | Accuracy = 0.500
age_group = younger      | N = 132 | Accuracy = 0.500

=== Group metrics by sex ===
sex = female       | N =  72 | Accuracy = 0.500
sex = male         | N = 108 | Accuracy = 0.500
