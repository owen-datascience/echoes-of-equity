# Lesson 9 Mini Project - Visualizing CNN Attention with Grad-CAM

This mini project belongs to Lesson 9 of the Echoes of Equity course.

You will:
1. Train a small CNN on a synthetic spectrogram-like dataset.
2. Use Grad-CAM to visualize which time-frequency regions the model
   focuses on when making predictions.
3. Save PNG images showing the original spectrogram next to the
   Grad-CAM heatmap overlay.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install torch numpy scikit-learn matplotlib
```

## Run

```bash
python src/gradcam_demo.py
```

Check the `outputs/` folder for PNG images such as
`example_0_gradcam.png`. These show the areas that contributed most
to the model's prediction, similar to how you will visualize what the
Echoes of Equity CNN is "listening" to on real Mel-spectrograms.
