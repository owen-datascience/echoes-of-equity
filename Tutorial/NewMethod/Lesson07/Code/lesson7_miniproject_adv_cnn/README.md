# Lesson 7 Mini Project - Adversarial CNN on Synthetic Spectrograms

This mini project belongs to Lesson 7 of the Echoes of Equity course.

You will:
1. Load a synthetic dataset of 32x32 spectrogram-like "images" with a
   binary main label and a 3-class group label.
2. Train a small CNN feature extractor with:
   - Main head: binary classification (e.g., trustworthy vs neutral)
   - Adversarial group head: 3-class group prediction (e.g., ethnicity)
3. Use a Gradient Reversal Layer (GRL) so that the shared CNN features
   become good for the main task but worse at predicting the group.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install torch numpy scikit-learn
```

## Run

```bash
python src/train_adv_cnn.py
```

Then:
- Observe training and validation accuracy for both main and group tasks.
- Experiment by changing `lambda_grl` in `train()` to 0.0, 0.5, 1.0 and
  compare how main vs group accuracy changes.
