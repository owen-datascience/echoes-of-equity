# Lesson 5 Mini Project - Evaluation & Fairness Metrics

This mini project belongs to Lesson 5 of the Echoes of Equity course.

You will:
1. Load a synthetic dataset with true labels, model scores, predictions, and ethnicity.
2. Compute confusion matrix, precision, recall, F1, and ROC-AUC overall.
3. Compute per-ethnicity metrics and a fairness gap in recall.
4. Save an ROC curve plot to the data/ folder.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Run

```bash
python src/analyze_metrics.py
```

Then:
- Inspect printed metrics.
- Open `data/roc_overall.png` to see the ROC curve.
- Answer the reflection questions from Lesson 5.
