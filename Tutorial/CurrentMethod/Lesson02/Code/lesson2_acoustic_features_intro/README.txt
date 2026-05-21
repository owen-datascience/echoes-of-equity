Lesson 2 Mini Practice Project: Exploring Acoustic Features

Files in this folder:
- synthetic_acoustic_features.csv
  A small synthetic dataset where each row represents one spoken sentence.
  Columns:
    * mean_f0_hz        - average pitch of the voice (in Hertz)
    * sd_f0_hz          - how much the pitch varies (standard deviation of F0)
    * hnr_db            - harmonic-to-noise ratio (voice quality, in decibels)
    * shimmer_db        - shimmer in decibels (tiny loudness variations)
    * cpp_db            - cepstral peak prominence (strength of harmonic structure)
    * duration_seconds  - how long the sentence lasts (in seconds)
    * intent_label      - 0 = neutral intent, 1 = trustworthy intent

- analyze_acoustic_features.py
  A fully commented Python script that:
    * Loads the dataset
    * Prints the first rows and the shape of the data
    * Shows basic statistics (min, max, mean, etc.) for each feature
    * Computes mean feature values for neutral vs trustworthy sentences
    * Computes the difference in means (trustworthy - neutral)
    * Shows a correlation matrix of the features
    * Prints hints to help you interpret the results

How to run:
1. Make sure you have Python 3 installed.
2. Install the required Python packages (only once). In a terminal, run:
     pip install pandas numpy
3. Open a terminal or command prompt in this folder.
4. Run:
     python analyze_acoustic_features.py
5. Read the printed outputs carefully.
   Pay special attention to how the average feature values differ
   between neutral (label 0) and trustworthy (label 1) intent.

Learning goal:
- Understand what acoustic features look like as numbers in a table.
- See how average feature values can differ between two types of speech intent.
- Prepare for later lessons, where we will train full machine learning models
  using similar features extracted from real audio recordings.
