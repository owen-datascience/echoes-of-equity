# Lesson 4 Mini Project - Simple CNN on Synthetic Spectrograms

This mini project belongs to Lesson 4 of the Echoes of Equity course.

You will:
1. Load a small synthetic dataset of 32x32 "spectrogram" images.
2. Train a simple 2-layer CNN in PyTorch to classify them into
   two classes: 0 = neutral, 1 = trustworthy.
3. Observe training and validation accuracy printed to the console.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install torch torchvision numpy scikit-learn
```

## Run

```bash
python src/train_cnn.py
```

You should see validation accuracy quickly rise above chance (0.5),
showing that the CNN can learn patterns in the synthetic spectrograms.
