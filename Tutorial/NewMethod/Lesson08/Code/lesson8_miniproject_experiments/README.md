# Lesson 8 Mini Project - Experiment Tracking & Hyperparameter Tuning

This mini project belongs to Lesson 8 of the Echoes of Equity course.

You will:
1. Load a synthetic spectrogram-like dataset with:
   - binary main labels
   - 3-class group labels
2. Train an adversarial CNN for several (learning_rate, lambda_grl) configs.
3. Log validation main accuracy and group accuracy to a CSV file.
4. Inspect the CSV and decide which config best balances performance and debiasing.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install torch numpy scikit-learn
```

## Run

```bash
python src/run_experiments.py
```

Then open `data/experiment_results.csv` in Excel, Google Sheets, or a text editor.
Compare how different learning rates and lambda_grl values affect
validation main accuracy vs group accuracy.
