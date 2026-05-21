# Lesson 10 Mini Project - Final Evaluation & Report Generator

This mini project belongs to Lesson 10 of the Echoes of Equity course.

You will:
1. Load a synthetic evaluation dataset that mimics your final test set
   (with groups, ground truth labels, and predictions from two models).
2. Compute overall accuracy, per-group recall, and the recall fairness
   gap for:
   - a baseline model
   - a debiased model
3. Print a concise comparison in the terminal.
4. Generate a markdown file `final_report.md` that you can adapt and
   copy into your ISEF paper/poster.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install pandas
```

## Run

```bash
python src/final_report.py
```

Then open `final_report.md` in any text editor or markdown viewer. Edit the
'Interpretation' section to practice writing your own scientific
conclusions based on the numbers.
