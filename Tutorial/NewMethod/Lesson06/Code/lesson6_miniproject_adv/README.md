# Lesson 6 Mini Project - Adversarial Debiasing with Gradient Reversal

This mini project belongs to Lesson 6 of the Echoes of Equity course.

You will:
1. Load a synthetic dataset where group membership (e.g., ethnicity)
   is correlated with features and the main label.
2. Train a small neural network with:
   - A shared feature extractor
   - A main label head (binary classification)
   - An adversarial group head (3-class classification)
3. Use a Gradient Reversal Layer (GRL) so that the feature extractor
   learns to be *good* for the main task but *bad* at predicting the
   sensitive group, reducing encoded bias.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install torch numpy scikit-learn
```

## Run

```bash
python src/train_adv_debias.py
```

Watch the printed output:
- Main task accuracy should stay reasonably high.
- Group prediction accuracy should *decrease* on the validation set,
  indicating that group information is being removed from the features.

Experiment:
- Change `lambda_grl` in `train()` to 0.0, 0.5, 1.0 and compare how
  group accuracy changes. Higher lambda means stronger debiasing.
