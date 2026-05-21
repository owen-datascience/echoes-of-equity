# ⭐ Lesson 8 — Experiment Design, Hyperparameter Tuning & Reproducibility

---

## 1. Lesson Summary

In this lesson, the student learns how to:

* Design **clean experiments** for the adversarial CNN (baseline vs different λ, learning rates, etc.).
* Tune **hyperparameters** in a structured way instead of guessing.
* Track experiments in a **results table / CSV**, so they can write a convincing **Results & Methods** section for ISEF.
* Think about **reproducibility**: seeds, fixed splits, saving configs and code.

By the end, the student should:

* Have a **small grid of experiments** (e.g., λ = 0, 0.5, 1.0; LR = 1e-3, 5e-4, etc.).
* Be able to answer: “Which configuration best balances performance and fairness, and why?”

---

## 2. Key Points

* Hyperparameters (learning rate, λ for GRL, batch size, epochs) can **dramatically** impact results.
* Good experiments vary **one or two hyperparameters at a time** while keeping everything else fixed.
* Use a **train/validation split** and keep it fixed (same seed) for fair comparisons.
* Log results in a **structured table** (CSV, Google Sheet): each row = one experiment.
* Track at least:

  * Main validation accuracy / recall (trustworthy).
  * Group validation accuracy (for adversarial head).
  * Possibly fairness gaps (from Lesson 5).
* Reproducibility:

  * Fix random seeds.
  * Record code version, dataset version, and hyperparameters.
  * Be able to rerun an experiment and get similar numbers.
* A well-organized experiment table is something judges and reviewers love.

---

## 3. Real-World Examples or Stories

* In research labs, ML engineers keep **experiment logs** with dozens or hundreds of runs. Papers often come from noticing a pattern in those logs.
* For fairness research, you might discover that λ = 0.7 gives almost the same accuracy as λ = 0.0 but significantly reduces recall gaps. That’s publishable insight.
* In ISEF papers that win top awards, students often show **ablation studies** (“without fairness head vs with fairness head”) and **hyperparameter sweeps**, not just a single model.

---

## 4. Terminology Explained

* **Hyperparameter** – A setting you choose before training (e.g., learning rate, batch size, number of epochs, λ for GRL).
* **Grid Search** – Trying all combinations from a small set of hyperparameter values (e.g., LR ∈ {1e-3, 5e-4}, λ ∈ {0, 0.5, 1.0}).
* **Random Seed** – A fixed value given to the random number generator to make experiments repeatable.
* **Reproducibility** – Ability to rerun code and get (roughly) the same results.
* **Experiment Config** – A specific combination of hyperparameters and settings used in one run.
* **Ablation Study** – Systematically removing or changing components (e.g., removing GRL) to see their effect.
* **Experiment Log / Results Table** – A structured record of each experiment’s config and outcome.

---

## 5. How It Works (Step-by-Step)

Here’s a simple plan for hyperparameter experiments on your adversarial CNN:

### Step 1 – Choose Hyperparameters to Explore

For example:

* Learning rate: `lr ∈ {1e-3, 5e-4}`
* GRL strength: `lambda_grl ∈ {0.0, 0.5, 1.0}`

That gives 2 × 3 = **6 experiments** (you can do fewer to start).

---

### Step 2 – Fix the Data Split & Random Seed

* Use `train_test_split(..., random_state=42, stratify=y_group)` and **don’t change it**.
* Set `torch.manual_seed(seed)` and `np.random.seed(seed)` for each run.

This ensures differences between runs come from hyperparameters, not random chance.

---

### Step 3 – Define Metrics to Record

For each experiment, record:

* `lr`
* `lambda_grl`
* Validation main accuracy (trustworthiness).
* Validation group accuracy (ethnicity prediction).

Optionally later: fairness metrics per group (from Lesson 5).

---

### Step 4 – Run Experiments in a Loop

You can write a script that:

* Loops over each `(lr, lambda_grl)` pair.
* Trains the adversarial CNN for a modest number of epochs (e.g., 8–10).
* Evaluates on validation set.
* Writes a row to `experiment_results.csv`.

You’ll see patterns like:

* λ = 0 → main acc high, group acc high (more bias risk).
* λ = 0.5 or 1.0 → group acc lower, main acc slightly lower but still good.

---

### Step 5 – Analyze the Table

After you have `experiment_results.csv`, you can:

* Open it in Excel or Google Sheets.
* Sort by main accuracy or group accuracy.
* Choose a configuration that **balances** goals:

  * Main performance high enough,
  * Group accuracy (and later fairness gaps) low enough.

This analysis step becomes a **Result subsection** in the final paper.

---

## 6. Practice Exercises (5)

**Exercise 1 – Identify Hyperparameters**

List at least five hyperparameters you might tune in the Echoes of Equity project and briefly explain what each controls.

---

**Exercise 2 – Design a Grid Search**

Suppose you can run exactly **6 experiments**.
Propose a grid of `(lr, lambda_grl)` values that makes sense and explain why.

---

**Exercise 3 – Seed & Split**

Explain why we keep the same validation split and random seed across all experiments.
What problem happens if we don’t?

---

**Exercise 4 – Results Table Sketch**

Draw a small table with columns and 3 example rows for:

* `lr`, `lambda_grl`, `val_main_acc`, `val_group_acc`.

Fill in fake numbers that illustrate the kind of pattern you expect (e.g., λ higher → group acc lower).

---

**Exercise 5 – Reproducibility Checklist**

Write a checklist of at least 6 items that you should track to make your experiments reproducible (for yourself and future readers).

---

## 7. Solutions (Model Answers)

**Solution 1 – Example Hyperparameters**

* Learning rate (`lr`) – How big each gradient step is.
* Batch size – How many samples per gradient update.
* Number of epochs – How many passes over the training set.
* `lambda_grl` – Strength of adversarial debiasing.
* CNN architecture choices – number of filters, kernel size, number of convolutional layers.
* Weight decay (L2 regularization) – How strongly we penalize large weights.

---

**Solution 2 – Grid Search Example**

With 6 experiments, one simple grid:

* `lr ∈ {1e-3, 5e-4}`
* `lambda_grl ∈ {0.0, 0.5, 1.0}`

This explores both a standard learning rate and a slightly smaller one, and three levels of debiasing (none, medium, strong). It’s small but covers key trade-offs.

---

**Solution 3 – Seed & Split**

We keep the seed and split fixed so:

* All models see **exactly the same training and validation data**.
* Differences in metrics are due mostly to hyperparameters, not random differences in which samples went to train vs validation.

If we don’t fix them:

* Some models might “get lucky” with an easier validation split, leading to **unfair comparisons**.

---

**Solution 4 – Results Table Sketch**

Example:

| lr   | lambda_grl | val_main_acc | val_group_acc |
| ---- | ---------- | ------------ | ------------- |
| 1e-3 | 0.0        | 0.88         | 0.80          |
| 1e-3 | 0.5        | 0.86         | 0.55          |
| 5e-4 | 1.0        | 0.84         | 0.40          |

This shows a reasonable pattern: higher λ (0 → 1.0) tends to lower group accuracy (less group info), with a small trade-off in main accuracy.

---

**Solution 5 – Reproducibility Checklist**

Example checklist:

1. Dataset version and preprocessing steps.
2. Train/validation split (including `random_state`).
3. Random seeds for numpy and PyTorch.
4. Model architecture (layers, sizes, activation functions).
5. Hyperparameters (lr, batch size, epochs, λ, weight decay, etc.).
6. Code version (Git commit hash, file version).
7. Exact command used to run the experiment.
8. Metrics recorded and where they’re stored (CSV path).

---

## 8. Q&A (10 Common Questions)

1. **Q:** How many experiments do I need for ISEF?
   **A:** Quality > quantity. Even 6–10 well-designed experiments with clear analysis is much better than 50 random runs with no structure.

2. **Q:** What if results change slightly even with fixed seed?
   **A:** Small variations are normal (e.g., due to GPU nondeterminism). You just want them to be **similar**, not identical.

3. **Q:** Should I always pick the model with the highest main accuracy?
   **A:** Not necessarily. In a fairness project, you may prefer a slightly lower accuracy with much better fairness metrics.

4. **Q:** Is grid search the only way?
   **A:** No. There’s random search, Bayesian optimization, etc. But grid search is simple and enough for this project.

5. **Q:** How do I show these experiments in my paper/poster?
   **A:** Use a table of configs vs metrics, and a short discussion of trends (e.g., “As λ increased, group accuracy decreased while main accuracy remained above 84%”).

6. **Q:** What if one experiment performs much worse?
   **A:** That’s still useful! You can explain why that configuration is not good and what it taught you.

7. **Q:** Do I need a separate test set beyond validation?
   **A:** Ideally yes for a final unbiased performance estimate, but for ISEF, a train/val split plus careful methodology is often acceptable.

8. **Q:** How long should each experiment run?
   **A:** Long enough to converge reasonably (e.g., 8–20 epochs), but not so long that you can’t iterate. Use a smaller model or dataset if needed.

9. **Q:** Can I reuse hyperparameters from other related papers?
   **A:** Yes, you can start from them and then do a small local search around those values.

10. **Q:** How do I keep my experiment logs organized over months?
    **A:** Use a consistent naming convention (experiment IDs), store CSVs in a dedicated folder, and consider a simple spreadsheet summarizing everything.

---

## 9. Quiz (10 Questions)

1. **What is a hyperparameter?**
   ➜ A configuration value set before training, like learning rate or λ, that the model does not learn automatically.

2. **Why do we use a validation set?**
   ➜ To evaluate and compare different hyperparameters without touching the eventual test set.

3. **What is grid search?**
   ➜ Systematically trying all combinations of a small set of hyperparameter values.

4. **Why is it important to fix random seeds?**
   ➜ To make results repeatable and ensure differences come from hyperparameters, not randomness.

5. **What is an ablation study?**
   ➜ Systematically removing or changing parts of the model (e.g., GRL) to see their effect on performance.

6. **Which of these is *not* a hyperparameter?**
   A) Learning rate
   B) Model weights
   C) Number of epochs
   D) λ for GRL
   ➜ Correct: **B) Model weights** (they are learned, not preset).

7. **Where should you store experiment results?**
   ➜ In a structured format like a CSV / spreadsheet with one row per experiment.

8. **What is one sign of poor experiment design?**
   ➜ Changing many things at once, so you can’t tell what caused differences in results.

9. **Why might you choose λ = 0.5 instead of λ = 0.0?**
   ➜ To introduce some debiasing while keeping main accuracy reasonably high.

10. **What does “reproducible experiment” mean?**
    ➜ Another person (or future you) can rerun it with the same code, data, and hyperparameters and get similar results.

---

## 10. Mini Practice Project – Experiment Tracking & Hyperparameter Tuning (.zip)

To make this practical, I created a mini project where the student actually **runs multiple experiments** and logs results.

### Project Goal

* Run several adversarial CNN experiments with different:

  * Learning rates (`lr`)
  * GRL strengths (`lambda_grl`)
* Track:

  * Validation main accuracy
  * Validation group accuracy
* Save everything to `experiment_results.csv` and interpret the trade-offs.

### What’s in the .zip

**`lesson8_miniproject_experiments.zip`** contains:

* `lesson8_miniproject_experiments/`

  * `README.md` – Instructions and reflection prompts.
  * `data/synthetic_experiment_data.npz` – Synthetic spectrogram-like dataset:

    * `X`: (N, 1, 32, 32)
    * `y_main`: binary labels
    * `y_group`: 3-group labels
  * `src/run_experiments.py` – Script that:

    * Defines an adversarial CNN (feature extractor + main + group heads with GRL).
    * Loops over several (lr, lambda_grl) configs.
    * Trains each for a few epochs.
    * Logs `lr`, `lambda_grl`, `val_main_acc`, `val_group_acc` to `data/experiment_results.csv`.

### How to Run

1. Unzip the archive.

2. Install dependencies:

   ```bash
   pip install torch numpy scikit-learn
   ```

3. Run:

   ```bash
   python src/run_experiments.py
   ```

4. Open `data/experiment_results.csv` in Excel / Sheets.

5. Answer questions like:

   * Which configuration has the **best main accuracy**?
   * Which configuration leads to the **lowest group accuracy**?
   * Which setup offers the **best balance**?

These answers can later be adapted into the “Hyperparameter Tuning” or “Model Selection” subsection of the final paper.

---

## 11. References

* General ML experiment design: any section on evaluation & tuning in an ML textbook (e.g., *Hands-On Machine Learning with Scikit-Learn & TensorFlow*).
* Reproducibility tips from ML blogs / talks (e.g., conference tutorials on reproducible ML).
* Fairness papers often include hyperparameter/ablation tables—great to skim for formatting ideas.

---

## 12. Additional Information (Mentor Tips)

* Encourage the student to **name experiments** (e.g., `cnn_lambda1_lr1e-3`) and maybe keep a simple Google Sheet summarizing the best ones.
* For the final ISEF report, I would recommend:

  * A table of 5–10 experiments with short captions.
  * A paragraph explaining the trend: “As λ increased, group accuracy decreased from X to Y, while main accuracy remained above Z.”
* This lesson also secretly trains them in **research discipline**, which will be incredibly valuable for future college research and internships.
