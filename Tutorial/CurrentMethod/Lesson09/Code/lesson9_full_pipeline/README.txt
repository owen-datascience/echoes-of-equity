Lesson 9 Mini Practice Project: End-to-End Trust Classification Pipeline (LOSO)

Files in this folder:
- synthetic_trust_full_pipeline_dataset.csv
  A synthetic utterance-level dataset with:
    * utterance_id   - ID of the spoken clip
    * speaker_id     - ID of the speaker
    * duration_seconds, mean_f0_hz, sd_f0_hz, hnr_db, shimmer_db, cpp_db
    * mean_rating    - average trust rating (1-7 scale)
    * std_rating     - rating disagreement (standard deviation)
    * n_ratings      - number of ratings for this clip
    * intent_label   - 0 = neutral/low trust, 1 = trustworthy

- full_trust_pipeline_loso.py
  A fully commented Python script that:
    * Loads the dataset and prints basic information
    * Builds features (X), labels (y), and groups (speaker_id)
    * Performs Leave-One-Speaker-Out cross-validation
    * Trains Logistic Regression and Random Forest classifiers
    * Computes accuracy, precision, recall, F1 per fold and averages across speakers
    * Prints a short summary comparing the models

How to run:
1. Make sure you have Python 3 installed.
2. Install required packages (only once). In a terminal, run:
     pip install pandas numpy scikit-learn
3. Open a terminal or command prompt in this folder.
4. Run:
     python full_trust_pipeline_loso.py
5. Read the printed per-speaker and average metrics.
   Compare the performance of Logistic Regression vs Random Forest.

Learning goal:
- See how all the pieces (labels, features, LOSO, metrics, and models)
  come together into a single, research-style evaluation pipeline.
