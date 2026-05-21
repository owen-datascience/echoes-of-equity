Lesson 5 Mini Practice Project: Training Baseline Classifiers

Files in this folder:
- synthetic_trust_baseline_dataset.csv
  A synthetic dataset where each row is one spoken utterance.
  Columns:
    * duration_seconds  - length of the utterance in seconds
    * mean_f0_hz        - average pitch (Hz)
    * sd_f0_hz          - pitch variability
    * hnr_db            - harmonic-to-noise ratio (voice clarity)
    * shimmer_db        - shimmer in decibels
    * cpp_db            - cepstral peak prominence (dB)
    * intent_label      - 0 = neutral, 1 = trustworthy

- train_baseline_classifiers.py
  A fully commented Python script that:
    * Loads the dataset using pandas
    * Splits data into training and test sets
    * Trains Logistic Regression and Random Forest classifiers
    * Evaluates each model with accuracy, precision, recall, and F1-score
    * Prints confusion matrices and classification reports
    * Provides interpretation hints

How to run:
1. Make sure you have Python 3 installed.
2. Install required packages (only once). In a terminal, run:
     pip install pandas scikit-learn
3. Open a terminal or command prompt in this folder.
4. Run:
     python train_baseline_classifiers.py
5. Read the printed metrics and compare the two baseline models.

Learning goal:
- Practice the supervised learning workflow for a binary classification task.
- See how to train and evaluate two simple baseline models.
- Prepare for later lessons that will apply similar code to the real trust dataset.
