Lesson 6 Mini Practice Project: Leave-One-Speaker-Out Cross-Validation (LOSO)

Files in this folder:
- synthetic_trust_loso_dataset.csv
  A synthetic dataset where each row is one spoken utterance.
  Columns:
    * speaker_id        - ID of the speaker (e.g., S01, S02, ...)
    * duration_seconds  - length of the utterance in seconds
    * mean_f0_hz        - average pitch (Hz)
    * sd_f0_hz          - pitch variability
    * hnr_db            - harmonic-to-noise ratio (voice clarity)
    * shimmer_db        - shimmer in decibels
    * cpp_db            - cepstral peak prominence (dB)
    * intent_label      - 0 = neutral, 1 = trustworthy

- loso_cross_validation.py
  A fully commented Python script that:
    * Loads the dataset
    * Uses speaker_id as the grouping variable
    * Performs leave-one-speaker-out cross-validation using LeaveOneGroupOut
    * Trains Logistic Regression for each fold
    * Computes accuracy, precision, recall, and F1-score per speaker
    * Prints average and standard deviation of metrics across all speakers
    * Provides interpretation hints

How to run:
1. Make sure you have Python 3 installed.
2. Install required packages (only once). In a terminal, run:
     pip install pandas scikit-learn numpy
3. Open a terminal or command prompt in this folder.
4. Run:
     python loso_cross_validation.py
5. Read the printed metrics for each speaker and the final averages.
   Try to think about what they say about generalization to new speakers.

Learning goal:
- Understand the idea of leave-one-speaker-out cross-validation.
- Learn how to implement group-based cross-validation in scikit-learn.
- See why this evaluation is stricter and more realistic for voice-based models.
