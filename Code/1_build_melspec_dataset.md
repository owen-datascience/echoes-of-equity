**Based on the project outline (`Echoes of Equity`) and the dataset paper**, your **next step** is:

> **Build a data pipeline that:**
>
> 1. Loads all `.wav` files
> 2. Parses labels from the filenames (trustworthy vs neutral + demographics)
> 3. Converts each audio file into a **Mel-spectrogram**
> 4. Saves:
>
>    * `mel_spectrograms/*.npy`
>    * a `metadata.csv` with all labels + paths

Below is a **single Python script** that does all of that.

---

## 🧾 Assumptions

* Your audio files sit under a folder like:

```text
data/
  wav/
    1901_bof_t05.wav
    1901_bof_n05.wav
    ...
```

* Filenames follow the pattern described in the paper:
  `speakerID_ethnicityAgeSex_intentSentence.wav`, e.g. `1901_bof_t05.wav` where: 

  * `1901` → speaker ID
  * `b`   → Black
  * `o`   → Older
  * `f`   → Female
  * `t`   → Trustworthy intent (`n` = Neutral)
  * `05`  → Sentence #5

You can adjust the `DATA_DIR` path in the script if your structure is different.

---

## 🧪 Install dependencies (once)

From your terminal:

```bash
pip install librosa soundfile numpy pandas tqdm
```

---

## 🧠 Script: build Mel-spectrogram dataset from 1,152 wav files

Save this as `build_melspec_dataset.py` and run it:

---

## ✅ What you’ve accomplished after running this

1. **All 1,152 `.wav` files → Mel-spectrogram `.npy` files**
2. A clean `metadata.csv` with labels:

   * `intent` (neutral/trustworthy)
   * `ethnicity`, `age_group`, `sex`, `speaker_id`, `sentence_id`
3. You now have a **ready-to-train dataset** for the CNN + adversarial model described in *Echoes of Equity* .

