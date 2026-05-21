Lesson 7 Mini Practice Project: Feature Importance for Trust Detection

Files in this folder:
- synthetic_trust_feature_importance_dataset.csv
  A synthetic dataset where each row is one spoken utterance.
  Columns:
    * duration_seconds  - length of the utterance in seconds
    * mean_f0_hz        - average pitch (Hz)
    * sd_f0_hz          - pitch variability
    * hnr_db            - harmonic-to-noise ratio (voice clarity)
    * shimmer_db        - shimmer in decibels
    * cpp_db            - cepstral peak prominence (dB)
    * intent_label      - 0 = neutral, 1 = trustworthy

- feature_importance_trust.py
  A fully commented Python script that:
    * Loads the dataset
    * Splits data into training and test sets
    * Trains a Random Forest classifier
    * Computes the test accuracy
    * Extracts and prints feature importances
    * Sorts features by importance
    * Plots a bar chart of feature importances using matplotlib
    * Provides interpretation hints

How to run:
1. Make sure you have Python 3 installed.
2. Install required packages (only once). In a terminal, run:
     pip install pandas scikit-learn numpy matplotlib
3. Open a terminal or command prompt in this folder.
4. Run:
     python feature_importance_trust.py
5. Read the printed outputs and look at the bar chart.
   Identify which features are most important for predicting trustworthy intent.

Learning goal:
- Understand how to inspect and interpret feature importances from a Random Forest.
- Connect model behavior back to domain knowledge about speech and trust cues.
