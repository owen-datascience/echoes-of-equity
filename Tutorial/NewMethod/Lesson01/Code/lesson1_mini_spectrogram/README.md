# Lesson 1 Mini Project – Spectrogram Explorer

This mini project helps you practice working with audio and spectrograms.

## What you will do

1. Load a short example audio file (`audio/sample_voice.wav`).
2. Compute its spectrogram using `scipy.signal.spectrogram`.
3. Save the spectrogram as an image in the `output/` folder.
4. Open the image and describe what you see.

> Note: In the full ISEF project, you will use **Mel-spectrograms** with libraries like `librosa` or `torchaudio`. This mini project uses a standard spectrogram to keep the code simple and easy to run.

## Folder structure

- `audio/`
  - `sample_voice.wav` – synthetic "voice-like" audio file.
- `src/`
  - `plot_spectrogram.py` – script to compute and save the spectrogram.
- `output/`
  - `spectrogram.png` – generated when you run the script.

## How to run

1. Make sure you have Python installed (3.9+ is fine).
2. Install required packages (inside a virtual environment is recommended):

```bash
pip install numpy scipy matplotlib
```

3. Run the script from the project root:

```bash
python src/plot_spectrogram.py
```

4. After running, open `output/spectrogram.png` to see the spectrogram.

## Reflection questions

- Where are the brightest areas in the spectrogram?  
- How does the frequency content change over time?  
- How might this look different if it were real speech instead of a synthetic tone?  

Once you are comfortable with this, you will be ready to move on to **Mel-spectrograms** and then to **CNN models** for trustworthy intent recognition.
