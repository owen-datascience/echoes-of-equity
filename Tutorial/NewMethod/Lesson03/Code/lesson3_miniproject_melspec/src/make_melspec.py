"""
Lesson 3 Mini Project: From WAV to Mel-Spectrogram

This script:
1. Loads a WAV file from the data/ folder.
2. Computes a Mel-spectrogram using librosa.
3. Visualizes and saves the Mel-spectrogram as a PNG image.
"""

from pathlib import Path
import librosa
import librosa.display
import matplotlib.pyplot as plt

def main():
    # 1. Load the audio file
    base_dir = Path(__file__).resolve().parents[1]
    wav_path = base_dir / "data" / "demo_tone.wav"
    print(f"Loading audio from: {wav_path}")
    y, sr = librosa.load(wav_path, sr=None)  # keep original sample rate

    # 2. Compute Mel-spectrogram
    n_fft = 1024       # window size
    hop_length = 256   # step between windows
    n_mels = 64        # number of Mel bands

    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,   # power spectrogram
    )

    # Convert to decibels for better visualization
    S_db = librosa.power_to_db(S, ref=S.max())

    # 3. Plot and save
    fig, ax = plt.subplots(figsize=(6, 4))
    img = librosa.display.specshow(
        S_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=ax,
    )
    ax.set_title("Mel-Spectrogram of demo_tone.wav")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

    out_path = base_dir / "data" / "demo_tone_melspec.png"
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved Mel-spectrogram to: {out_path}")

if __name__ == "__main__":
    main()
