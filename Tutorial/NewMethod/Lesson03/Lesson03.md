# ⭐ Lesson 3 — Audio Processing & Mel-Spectrograms for Trustworthiness

---

## 1. Lesson Summary

This lesson teaches the student how to:

* Load **raw audio (.wav)** files.
* Understand basic audio concepts: **sampling rate, waveform, frequency**.
* Convert audio into **Mel-spectrograms**, which are image-like inputs for CNNs.
* Save and visualize spectrograms for later use in the deep learning model.

By the end, the student will be able to build a simple **audio → Mel-spectrogram pipeline**, a key building block for the project’s CNN and adversarial debiasing architecture.

---

## 2. Key Points

* Deep learning models work best on **structured numeric grids** → spectrograms.
* A **waveform** is sound amplitude over time; a **spectrogram** is energy over time & frequency.
* The **Mel scale** is a frequency scale aligned with how humans perceive pitch.
* **Short-Time Fourier Transform (STFT)** converts audio into time–frequency representation.
* A **Mel-spectrogram** is a transformed STFT summed into Mel frequency bands.
* Parameters like **sample rate, n_fft, hop_length, n_mels** affect resolution and model performance.
* Libraries like **librosa** or **torchaudio** make audio processing manageable.
* Properly saving and organizing spectrograms is essential for reproducible experiments.

---

## 3. Real-World Examples or Stories

* **Speech recognition (e.g., Google, Alexa)**: They feed spectrogram-like features into deep neural networks to recognize words and speakers.
* **Music genre classification**: Systems that guess if a track is rock, jazz, or classical almost always use spectrograms.
* **Birdsong recognition & bioacoustics**: Ecologists use spectrograms to detect specific bird calls or endangered species.

Your project does something similar but with an important twist: **predicting trustworthiness intent fairly across demographics.**

---

## 4. Terminology Explained

* **Sampling Rate (Hz)** – How many audio samples are taken per second (e.g., 16,000 Hz).
* **Waveform** – A 1D array of audio amplitudes over time.
* **Frequency (Hz)** – How fast a sound wave oscillates; higher frequency = higher pitch.
* **STFT (Short-Time Fourier Transform)** – A sliding window FFT that converts small chunks of audio from time domain to frequency domain.
* **Spectrogram** – A 2D matrix: time on x-axis, frequency on y-axis, intensity as color.
* **Mel Scale** – A non-linear scale where equal steps sound equally spaced in pitch to humans.
* **Mel-Spectrogram** – Spectrogram where frequencies are converted to the Mel scale.
* **Decibels (dB)** – Logarithmic scale of intensity; good for viewing spectrograms.
* **Log-Mel Spectrogram** – Mel-spectrogram converted to dB (log scale) for better visual and model performance.

---

## 5. How It Works (Step-by-Step)

### A. Loading Audio

Using `librosa`:

```python
import librosa

file_path = "data/example.wav"
y, sr = librosa.load(file_path, sr=16000)  # y: waveform, sr: sample rate
print("Waveform length:", len(y))
print("Sample rate:", sr)
```

* `y` is a numpy array of floats between -1 and 1.
* `sr` is the sampling rate (here we force 16 kHz for consistency).

---

### B. Computing a Mel-Spectrogram

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

n_fft = 1024      # FFT window size
hop_length = 256  # step between windows
n_mels = 64       # number of Mel bands

S = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_fft=n_fft,
    hop_length=hop_length,
    n_mels=n_mels,
    power=2.0
)

print("Mel-spectrogram shape:", S.shape)
```

* `S` has shape `(n_mels, time_frames)`.
* Each row is a Mel band; each column is a time slice.

---

### C. Converting to dB & Visualizing

```python
S_db = librosa.power_to_db(S, ref=S.max())

plt.figure(figsize=(6,4))
librosa.display.specshow(
    S_db,
    sr=sr,
    hop_length=hop_length,
    x_axis="time",
    y_axis="mel"
)
plt.title("Mel-Spectrogram")
plt.colorbar(format="%+2.0f dB")
plt.tight_layout()
plt.show()
```

This image is what your CNN will see later.

---

### D. Saving Spectrograms for CNN

Instead of plotting interactively, you can save to file:

```python
plt.savefig("data/example_melspec.png")
```

Later, your CNN can either:

* Load these PNGs as images, or
* Compute spectrograms on the fly directly from audio.

---

### E. Connecting to the Project

In *Echoes of Equity*, the plan is to:

1. Download the 1,152 .wav files.
2. Convert each into a **Mel-spectrogram**.
3. Feed these spectrograms to a **2D-CNN** that predicts `trustworthy` vs `neutral`.
4. Add an **adversarial demographic head** later. 

Lesson 3 sets up step 2 solidly.

---

## 6. Practice Exercises (5)

**Exercise 1 – Load Audio & Print Info**
Write a script that:

1. Loads `demo_tone.wav` (or any speech file you have).
2. Prints the sample rate and number of samples.
3. Computes the duration in seconds.

---

**Exercise 2 – Inspect Waveform Shape**
Using numpy, compute:

1. The minimum and maximum value of the waveform.
2. The mean amplitude.
3. A comment: does it look normalized (within [-1, 1])?

---

**Exercise 3 – Mel-Spectrogram Parameters**
Compute a Mel-spectrogram with:

* `n_fft=1024, hop_length=256, n_mels=64`

Then try `n_mels=128`.
Compare shapes and write down how the image changes.

---

**Exercise 4 – Save Spectrogram Image**
Modify your code to:

1. Save the Mel-spectrogram to `data/example_melspec.png`.
2. Open the image to visually inspect it.

---

**Exercise 5 – Time-Frequency Tradeoff**
Change `hop_length` from 256 to 512 and 128.
For each version, note:

* How does the time resolution change?
* How does the visual look differ (more squished or stretched in time)?

---

## 7. Solutions (Model Answers)

**Solution 1 – Load Audio & Print Info**

```python
import librosa

y, sr = librosa.load("data/demo_tone.wav", sr=None)
print("Sample rate:", sr)
print("Number of samples:", len(y))

duration = len(y) / sr
print("Duration (sec):", duration)
```

---

**Solution 2 – Inspect Waveform Shape**

```python
import numpy as np
import librosa

y, sr = librosa.load("data/demo_tone.wav", sr=None)
print("Min:", y.min())
print("Max:", y.max())
print("Mean:", y.mean())
```

Typically, `y.min()` and `y.max()` should be between -1 and 1; `mean` close to 0.

---

**Solution 3 – Mel-Spectrogram Parameters**

```python
import librosa

y, sr = librosa.load("data/demo_tone.wav", sr=None)

S1 = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=64)
S2 = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=128)

print("Shape with 64 mels:", S1.shape)   # (64, T)
print("Shape with 128 mels:", S2.shape)  # (128, T)
```

Explanation: More Mel bands → higher “vertical” resolution.

---

**Solution 4 – Save Spectrogram Image**

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt

y, sr = librosa.load("data/demo_tone.wav", sr=None)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=64)
S_db = librosa.power_to_db(S, ref=S.max())

plt.figure(figsize=(6,4))
librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis="time", y_axis="mel")
plt.title("Mel-Spectrogram")
plt.colorbar(format="%+2.0f dB")
plt.tight_layout()
plt.savefig("data/demo_tone_melspec.png")
print("Saved to data/demo_tone_melspec.png")
```

---

**Solution 5 – Time-Frequency Tradeoff**

Key idea:

* **Smaller hop_length** (e.g., 128) → more time frames → better time resolution, heavier computation.
* **Larger hop_length** (e.g., 512) → fewer time frames → coarser time resolution, lighter computation.

Students should describe how the spectrogram looks more stretched/compressed along the time axis.

---

## 8. Q&A (10 Frequently Asked Questions)

1. **Q:** Why Mel-spectrograms instead of raw waveforms?
   **A:** They encode frequency and time information in a way that aligns with human perception and are easier for CNNs to learn from.

2. **Q:** Can I change the sample rate?
   **A:** Yes, but keep it consistent across all samples (e.g., 16 kHz) to avoid mismatched feature shapes.

3. **Q:** What does `n_fft` do?
   **A:** It controls the window size of the STFT; larger `n_fft` → better frequency resolution, but more computation.

4. **Q:** Why convert to dB (`power_to_db`)?
   **A:** Log scale (dB) better reflects human hearing and spreads out low-intensity differences.

5. **Q:** What happens if `n_mels` is very small (e.g., 20)?
   **A:** You lose fine-grained frequency detail, which can hurt classification.

6. **Q:** Why do we need `hop_length`?
   **A:** It controls how much we slide the window; smaller hop = more overlapping windows = smoother but heavier.

7. **Q:** Is the color in the spectrogram important for the CNN?
   **A:** The CNN sees numbers, not “colors”; colormap is just for humans. The numeric values are what matter.

8. **Q:** Can I use torchaudio instead of librosa?
   **A:** Yes, especially if your training is in PyTorch; concepts are identical.

9. **Q:** Should I save spectrogram as PNG or as a numpy array?
   **A:** Either is fine; PNG is convenient for visualization, numpy arrays can be more precise and efficient for training.

10. **Q:** How does this relate to fairness?
    **A:** The same processing pipeline is applied to all speakers; later, bias comes from modeling and data distribution, not from unequal preprocessing.

---

## 9. Quiz (10 Questions)

1. **What does a waveform represent?**
   ➜ Amplitude of sound over time.

2. **What is the Mel scale?**
   ➜ A perceptual frequency scale aligned with human hearing.

3. **What function in librosa computes Mel-spectrograms?**
   ➜ `librosa.feature.melspectrogram`.

4. **What parameter controls the number of Mel frequency bands?**
   ➜ `n_mels`.

5. **Increasing `n_mels` does what to the spectrogram?**
   ➜ Increases vertical (frequency) resolution.

6. **What does STFT stand for?**
   ➜ Short-Time Fourier Transform.

7. **Why convert power spectrogram to dB?**
   ➜ To use a log scale like human perception and enhance low-energy details.

8. **What controls time resolution in a spectrogram?**
   ➜ `hop_length`.

9. **If you double `hop_length`, what happens to the number of time frames?**
   ➜ Roughly halves (coarser time resolution).

10. **Why are spectrograms useful for CNNs?**
    ➜ They transform audio into 2D grids (like images) that CNNs are good at processing.

---

## 10. Mini Practice Project – WAV → Mel-Spectrogram (with .zip)

I’ve created a ready-made mini project so the student can practice the full audio pipeline.

### Project Goal

* Load a simple audio file (`demo_tone.wav`).
* Compute its Mel-spectrogram with librosa.
* Save a PNG image of the spectrogram.

### What’s in the .zip

**`lesson3_miniproject_melspec.zip`** contains:

* `lesson3_miniproject_melspec/`

  * `README.md` – Setup and run instructions.
  * `data/demo_tone.wav` – A generated sine-wave audio file.
  * `src/make_melspec.py` – Script that:

    * Loads `demo_tone.wav`
    * Computes a Mel-spectrogram
    * Converts to dB
    * Saves it as `data/demo_tone_melspec.png`

### How to Use

1. Unzip the file.

2. In a terminal inside the unzipped folder:

   ```bash
   pip install librosa matplotlib
   python src/make_melspec.py
   ```

3. Open `data/demo_tone_melspec.png` and inspect the result.

4. Bonus: replace `demo_tone.wav` with your own short speech recording and rerun.

---

## 11. References

* Project description and milestones for audio processing and Mel-spectrograms. 
* Librosa documentation (audio & spectrograms): [https://librosa.org/doc/latest/](https://librosa.org/doc/latest/)
* Intro video on spectrograms (any YouTube “What is a spectrogram?” works well).
* PyTorch Audio Tutorial (if later using torchaudio): [https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html](https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html)

---

## 12. Additional Information (Competition-Focused Tips)

* Encourage the student to **save a few example spectrograms** from different speakers and intents. These can become **figures on the poster** showing qualitative differences.
* Have them note in their logbook which **parameters** they used (`n_fft`, `hop_length`, `n_mels`)—later they can mention a brief **hyperparameter exploration** section.
* A nice ISEF-style figure:

  * Panel A: waveform
  * Panel B: Mel-spectrogram (neutral)
  * Panel C: Mel-spectrogram (trustworthy)
    With captions discussing visible differences.
