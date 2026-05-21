Lesson 1 Mini Practice Project: Simple Voice Trust Classifier

Files in this folder:
- mini_voice_trust_dataset.csv
  A small synthetic dataset with:
    * duration_seconds  - how long the sentence lasts (in seconds)
    * mean_pitch_hz     - average pitch of the voice (in Hertz)
    * hnr_db            - harmonic-to-noise ratio (voice quality, in decibels)
    * intent_label      - 0 = neutral intent, 1 = trustworthy intent

- train_logistic_regression.py
  A Python script that:
    * Loads the dataset
    * Splits it into training and test sets
    * Trains a Logistic Regression model
    * Evaluates accuracy on the test set
    * Prints the model coefficients

How to run:
1. Make sure you have Python 3 installed.
2. Install the dependencies (you can run this in a terminal):
     pip install pandas scikit-learn
3. Open a terminal or command prompt in this folder.
4. Run:
     python train_logistic_regression.py
5. Look at the printed accuracy and coefficients.

Learning goal:
- Understand how numerical acoustic features can be used
  to classify whether a voice is produced with neutral or
  trustworthy intent using a simple machine learning model.
