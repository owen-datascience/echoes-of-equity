Lesson 8 Mini Practice Project: Aggregating Human Trust Ratings

Files in this folder:
- synthetic_trust_ratings_long.csv
  A synthetic dataset in 'long' format where each row is one rating.
  Columns:
    * utterance_id  - ID of the spoken clip (e.g., U001, U002, ...)
    * rater_id      - ID of the human rater (e.g., R01, R02, ...)
    * trust_rating  - rating on a 1-7 scale (1 = very untrustworthy, 7 = very trustworthy)

- aggregate_trust_ratings.py
  A fully commented Python script that:
    * Loads the ratings table
    * Shows basic info (number of utterances and raters)
    * Aggregates ratings per utterance (mean, standard deviation, count)
    * Creates a binary label based on a threshold on mean_rating
    * Plots the distribution of mean ratings
    * Plots mean_rating vs std_rating (disagreement)
    * Saves an utterance-level CSV (trust_utterance_level_labels.csv) for later modeling

How to run:
1. Make sure you have Python 3 installed.
2. Install required packages (only once). In a terminal, run:
     pip install pandas numpy matplotlib
3. Open a terminal or command prompt in this folder.
4. Run:
     python aggregate_trust_ratings.py
5. Inspect the printed outputs and generated plots.
   Think about how the mean and standard deviation of ratings relate
   to the idea of labels used for training a trust-detection model.

Learning goal:
- Understand how to go from many human ratings to a single label per audio clip.
- See how rating distributions and disagreement can be explored with simple plots.
