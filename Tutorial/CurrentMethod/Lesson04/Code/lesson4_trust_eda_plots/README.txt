Lesson 4 Mini Practice Project: Exploratory Data Analysis (EDA) and Simple Plots

Files in this folder:
- synthetic_trust_eda_dataset.csv
  A small synthetic dataset where each row is one spoken utterance.
  Columns include:
    * speaker_id        - ID of the speaker
    * age_group         - 'younger' or 'older'
    * sex               - 'female' or 'male'
    * ethnicity         - 'white', 'black', or 'south_asian'
    * sentence_id       - sentence number within a speaker
    * intent_label      - 0 = neutral, 1 = trustworthy
    * duration_seconds  - length of the utterance in seconds
    * mean_f0_hz        - average pitch (Hz)
    * sd_f0_hz          - pitch variability
    * hnr_db            - harmonic-to-noise ratio (voice clarity)

- trust_eda_plots.py
  A fully commented Python script that:
    * Loads the dataset
    * Prints basic info about the data
    * Shows class distribution of intent_label
    * Prints summary statistics for numeric features
    * Computes mean feature values by intent_label
    * Plots histograms of mean_f0_hz and hnr_db for neutral vs trustworthy
    * Provides hints on how to interpret the plots

How to run:
1. Make sure you have Python 3 installed.
2. Install required packages (only once). In a terminal, run:
     pip install pandas matplotlib
3. Open a terminal or command prompt in this folder.
4. Run:
     python trust_eda_plots.py
5. Two histogram windows should appear, one for mean_f0_hz and one for hnr_db.
   Study the plots to see how neutral and trustworthy distributions differ.

Learning goal:
- Practice basic Exploratory Data Analysis (EDA) with pandas.
- Learn how to inspect class balance and feature distributions.
- Learn how to create and interpret simple histograms for comparing groups.
