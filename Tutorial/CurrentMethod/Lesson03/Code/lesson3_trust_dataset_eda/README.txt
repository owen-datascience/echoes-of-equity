Lesson 3 Mini Practice Project: Understanding Dataset Structure

Files in this folder:
- synthetic_trust_dataset_structure.csv
  A small synthetic dataset that mimics the structure of
  the trustworthy-intent speech dataset at a basic level.
  Each row is one utterance and includes:
    * speaker_id        - an ID for the speaker (e.g., S01, S02, ...)
    * age_group         - 'younger' or 'older'
    * sex               - 'female' or 'male'
    * ethnicity         - 'white', 'black', or 'south_asian'
    * sentence_id       - which sentence number this is for the speaker
    * intent_label      - 0 = neutral intent, 1 = trustworthy intent
    * duration_seconds  - how long the utterance is (in seconds)
    * mean_f0_hz        - average pitch of the voice (in Hertz)

- explore_trust_dataset_structure.py
  A fully commented Python script that:
    * Loads the dataset with pandas
    * Prints the first few rows and the shape of the data
    * Shows how many utterances each speaker has
    * Shows the distribution of intent labels
    * Checks how many neutral vs trustworthy utterances each speaker has
    * Displays how speakers are distributed by age_group, sex, and ethnicity
    * Prints summary statistics for duration and mean_f0_hz
    * Provides interpretation hints

How to run:
1. Make sure you have Python 3 installed.
2. Install the required Python package (only once). In a terminal, run:
     pip install pandas
3. Open a terminal or command prompt in this folder.
4. Run:
     python explore_trust_dataset_structure.py
5. Read the printed outputs carefully.
   Try to imagine how a larger, real dataset used in research
   would look similar but with many more speakers and utterances.

Learning goal:
- Understand how a speech dataset is organized into rows and columns.
- See how speaker-level metadata (age_group, sex, ethnicity) is stored.
- See how intent labels and simple features are stored for each utterance.
- Prepare for later lessons that will load and analyze the real research dataset.
