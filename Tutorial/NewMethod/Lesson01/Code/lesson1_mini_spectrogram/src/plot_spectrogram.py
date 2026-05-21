# Import the os library - helps us work with files and folders on the computer
import os
# Import numpy - a library for working with numbers and arrays (lists of numbers)
import numpy as np
# Import wavfile from scipy.io - helps us read audio .wav files
from scipy.io import wavfile
# Import spectrogram from scipy.signal - creates a spectrogram (visual representation of sound frequencies over time)
from scipy.signal import spectrogram
# Import matplotlib.pyplot - a library for creating graphs and charts
import matplotlib.pyplot as plt

# Define the main function that will run our program
def main():
    # SECTION: Set up file paths
    # Get the absolute path to this current file (the script we're running)
    # Then go up two directories to get the base project folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Create a path to the "audio" folder where our sound files are stored
    audio_dir = os.path.join(base_dir, "audio")
    # Create a path to the "output" folder where we'll save our spectrogram image
    output_dir = os.path.join(base_dir, "output")
    # Create the output folder if it doesn't exist yet (exist_ok=True means don't error if it already exists)
    os.makedirs(output_dir, exist_ok=True)

    # Create the full path to the specific audio file we want to analyze
    audio_path = os.path.join(audio_dir, "sample_voice.wav")
    # Check if the audio file actually exists at that location
    if not os.path.exists(audio_path):
        # If the file doesn't exist, stop the program and show an error message
        raise FileNotFoundError(f"Could not find audio file at {audio_path}")

    # SECTION 1: Load the audio file
    # Read the audio file and get two things:
    # sr = sample rate (how many times per second the audio was recorded)
    # data = the actual audio data as numbers
    sr, data = wavfile.read(audio_path)

    # Check if the audio has two channels (stereo, like left and right speakers)
    # If stereo, convert to mono (single channel)
    if len(data.shape) == 2:
        # Average the left and right channels together to make one channel (axis=1 means average across columns)
        data = data.mean(axis=1)

    # SECTION 2: Create a spectrogram from the audio data
    # This breaks down the sound into its different frequencies over time
    # f = frequencies (different pitches in the sound)
    # t = time points (when each frequency appears)
    # Sxx = the power/intensity of each frequency at each time
    # nperseg=512 means we look at 512 samples at a time
    # noverlap=256 means each window overlaps the previous by 256 samples (helps smooth the result)
    f, t, Sxx = spectrogram(data, fs=sr, nperseg=512, noverlap=256)

    # SECTION 3: Create a visual plot of the spectrogram
    # Create a new figure (blank canvas) with size 8 inches wide by 4 inches tall
    plt.figure(figsize=(8, 4))
    # Create a colored mesh plot where:
    # - x-axis is time (t)
    # - y-axis is frequency (f)
    # - colors represent the power in decibels (dB) - louder sounds are brighter
    # We convert to dB using 10 * log10, and add 1e-10 to avoid taking log of zero
    # shading='gouraud' makes the colors blend smoothly
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    # Label the y-axis to show it represents frequency in Hertz (Hz)
    plt.ylabel("Frequency (Hz)")
    # Label the x-axis to show it represents time in seconds
    plt.xlabel("Time (s)")
    # Add a title to the top of the plot
    plt.title("Spectrogram of sample_voice.wav")
    # Add a color bar on the side showing what each color means (power in decibels)
    plt.colorbar(label="Power (dB)")

    # SECTION 4: Save the plot as an image file
    # Create the full path where we want to save the image
    output_path = os.path.join(output_dir, "spectrogram.png")
    # Automatically adjust spacing so labels don't get cut off
    plt.tight_layout()
    # Save the plot as a PNG image file
    plt.savefig(output_path)
    # Print a message to let the user know where the file was saved
    print(f"Spectrogram saved to: {output_path}")

# This special line checks if this file is being run directly (not imported as a module)
if __name__ == "__main__":
    # If running directly, call the main function to start the program
    main()
