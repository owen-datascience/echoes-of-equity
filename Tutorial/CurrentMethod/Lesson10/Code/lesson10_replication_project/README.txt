Lesson 10 Mini Practice Project: Mini Replication Study

Files in this folder:
- replication_trust_dataset.csv
  A synthetic utterance-level dataset with:
    * utterance_id, speaker_id
    * duration_seconds, mean_f0_hz, sd_f0_hz, hnr_db, shimmer_db, cpp_db
    * mean_rating, std_rating, n_ratings
    * intent_label (0 = neutral/low trust, 1 = trustworthy)

- replication_trust_experiment.py
  A fully commented Python script that:
    * Loads the dataset
    * Builds features (X), labels (y), and groups (speaker_id)
    * Runs Leave-One-Speaker-Out cross-validation
    * Trains Logistic Regression and Random Forest
    * Computes metrics (accuracy, precision, recall, F1) per speaker
    * Builds confusion matrices for both models
    * Writes a text summary to 'replication_report.txt'

- replication_report_template.txt
  A text template that helps you write your own mini replication report.

How to run:
1. Make sure you have Python 3 installed.
2. Install requirements (only once). In a terminal, run:
     pip install pandas numpy scikit-learn
3. Open a terminal or command prompt in this folder.
4. Run:
     python replication_trust_experiment.py
5. Open 'replication_report.txt' to read the automatically generated summary.
6. Use 'replication_report_template.txt' as a guide to write your own report.

Learning goal:
- Practice running a full trust-detection experiment and summarizing the results
  in a research-style report, similar to what you will do when reproducing
  the real paper on trust in voices.
