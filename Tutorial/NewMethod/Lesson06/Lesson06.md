# ⭐ Lesson 6 — Adversarial Debiasing & Gradient Reversal for Fair Representations

---

## 1. Lesson Summary

In this lesson, the student will:

* Understand the idea of **adversarial debiasing**: training a model that

  * predicts **trustworthiness intent** well, while
  * **unlearning demographic cues** (ethnicity).
* Learn how a **Gradient Reversal Layer (GRL)** works conceptually.
* See how to build a network with:

  * **Shared feature extractor** (from Mel-spectrograms),
  * **Main head** (trustworthy vs neutral),
  * **Adversarial head** (ethnicity).
* Train on a synthetic dataset and observe how increasing GRL strength lowers **group-prediction accuracy** (good) while keeping **main-task accuracy** reasonable.

This lesson directly corresponds to the “adversarial head” and “reduce WWIB bias” design in your project plan. 

---

## 2. Key Points

* Standard models may unintentionally encode **demographic information** in their internal representations.
* Adversarial debiasing adds an extra network that tries to **predict the sensitive attribute** (ethnicity), while the feature extractor tries to **fool** it.
* The **Gradient Reversal Layer (GRL)** multiplies gradients by **-λ**, turning a minimization objective into a **maximization** from the feature extractor’s perspective.
* The main task (trustworthy vs neutral) is trained **normally**, so we still want high performance.
* The adversarial head being **bad** at group prediction is a sign of **less encoded bias**.
* λ (lambda) controls how strongly we push the model to forget demographic information.
* We evaluate success by checking:

  * Main accuracy / recall, and
  * Group (ethnicity) accuracy from the adversarial head and fairness gaps.

---

## 3. Real-World Examples or Stories

* In hiring algorithms, we do not want internal features to strongly predict **race or gender**, even if those attributes are not explicitly in the input—because that leads to systemic discrimination.
* In your project, the goal is to build a trustworthiness model whose internal features **do not encode ethnicity**, reducing WWIB (White Western Individualist Bias). 
* Think of the adversarial head as a “mini-judge” trying to guess ethnicity from features; the feature extractor learns to “hide” ethnicity from that judge while still keeping signal for trustworthiness.

---

## 4. Terminology Explained

* **Sensitive Attribute** – A personal characteristic like ethnicity, gender, or age that we want the model to treat fairly.
* **Adversarial Head** – A classifier that predicts the sensitive attribute using the shared features.
* **Gradient Reversal Layer (GRL)** – A special layer that passes data unchanged in the forward pass but multiplies gradients by -λ in the backward pass.
* **λ (lambda)** – A hyperparameter controlling how strongly we push the model to remove sensitive information.
* **Shared Feature Extractor** – Network part that turns input (spectrogram) into a compact representation used by both heads.
* **Domain Adversarial Training** – General technique similar to what we’re doing; often used to make models robust across domains (e.g., different accents).
* **Representation Invariance** – The property that features are not predictive of certain attributes (like demographic group).

---

## 5. How It Works (Step-by-Step)

### A. Architecture Layout

We split the model into three pieces:

1. **FeatureExtractor (F)**:
   Takes spectrogram (or synthetic feature vector) → outputs `h` (hidden representation).

2. **MainHead (M)**:
   Takes `h` → predicts `y_main` (trustworthy vs neutral).

3. **GroupHead (G)**:
   Takes **gradient-reversed** `h` → predicts `y_group` (ethnicity).

Formally:

* Forward pass:

  * `h = F(x)`
  * `y_main_logits = M(h)`
  * `h_rev = GRL(h)`
  * `y_group_logits = G(h_rev)`

* Loss:

  * `L_total = L_main(y_main_logits, y_main) + L_group(y_group_logits, y_group)`

Because of GRL, `F` is pushed to **minimize** `L_main` but **maximize** `L_group`.

---

### B. Gradient Reversal Layer (GRL)

Conceptually:

* Forward: `GRL(h) = h` (do nothing).
* Backward: `∂L/∂h = -λ * ∂L/∂(GRL(h))`.

So when we backprop through the group head, the feature extractor’s gradients are flipped, meaning it **moves away** from representations that let the group head succeed.

In PyTorch, we define a custom autograd `Function`:

```python
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GRL.apply(x, lambda_)
```

---

### C. Training Loop

Per batch (x, y_main, y_group):

1. `h = F(x)`
2. `logits_main = M(h)` → main loss: `loss_main`
3. `h_rev = grad_reverse(h, lambda_grl)`
4. `logits_group = G(h_rev)` → group loss: `loss_group`
5. `loss = loss_main + loss_group`
6. Backprop: update F, M, G with optimizer.

If debiasing works, after training:

* **Main accuracy** on validation: still good.
* **Group accuracy** on validation: lower than without GRL (closer to random guess for 3 groups ≈ 33%).

---

### D. Connecting to the Project

For *Echoes of Equity*: 

* **F** will be a CNN over Mel-spectrograms.
* **M** predicts trustworthiness intent.
* **G** predicts ethnicity (3 groups).
* You will tune **λ** and compare:

  * Baseline CNN without GRL
  * CNN + GRL (various λ)
* You’ll measure:

  * Trustworthiness accuracy, recall per group
  * Fairness gap in recall across White/Black/South Asian
  * Group head accuracy (how predictable ethnicity still is).

---

## 6. Practice Exercises (5)

**Exercise 1 – GRL Concept Check**
Explain in your own words what GRL does in forward and backward passes, and why this helps debias representations.

---

**Exercise 2 – Architecture Diagram**
Draw (on paper or in a notebook) the three-part adversarial architecture:

* FeatureExtractor → MainHead
* FeatureExtractor → GRL → GroupHead

Label inputs, outputs, and loss flows.

---

**Exercise 3 – Compare λ Values (Concept)**
What do you expect to happen if:

* λ = 0.0
* λ = 0.5
* λ = 2.0
  to main accuracy and group accuracy?

---

**Exercise 4 – Baseline Without GRL**
Modify the training loop to disable GRL (i.e. don’t use gradient reversal, or set λ = 0).
Observe:

* Validation main accuracy
* Validation group accuracy

---

**Exercise 5 – With GRL**
Re-enable GRL with λ = 1.0 and train again.
Compare the validation main and group accuracies with Exercise 4 and write a short paragraph summarizing the differences.

---

## 7. Solutions (Model Answers)

**Solution 1 – GRL Concept**

* Forward: GRL just passes `h` through unchanged.
* Backward: GRL multiplies the gradient by `-λ`, so the feature extractor is updated in the **opposite direction** for the group loss.
* Result: The feature extractor learns features that make it **hard** for the group head to predict ethnicity (removing group information) while still learning features useful for the main task.

---

**Solution 2 – Architecture Diagram Description**

* Input: `x` (e.g., spectrogram).
* `x → F → h` (shared representation).
* `h → M → y_main_logits` (trustworthy vs neutral).
* `h → GRL → h_rev → G → y_group_logits` (ethnicity).
* Losses:

  * `L_main` backpropagates normally through M and F.
  * `L_group` backpropagates normally through G, but **reversed** through GRL into F.

---

**Solution 3 – λ Values (Expected Behavior)**

* **λ = 0.0**: no adversarial effect; F ignores group loss, so group head can easily predict ethnicity → high group accuracy, possibly more bias.
* **λ = 0.5**: moderate debiasing; group accuracy should drop compared to λ=0 while main accuracy stays close.
* **λ = 2.0**: very strong debiasing; group accuracy may drop close to random, but if too strong, main accuracy might also deteriorate.

---

**Solution 4 – Baseline Without GRL**

If you disable GRL, you have:

* `h_rev = h` (no gradient reversal).
* F is optimized to be good for both main and group tasks, meaning features explicitly encode ethnicity.
* Typically you see:

  * High main accuracy (good).
  * High group accuracy (bad for fairness).

---

**Solution 5 – With GRL**

With λ = 1.0:

* Group head accuracy on validation should be **lower** (approaching 1/3 ≈ 0.33 for 3 groups).
* Main accuracy might remain similar or slightly lower.
* Interpretation: features contain **less group information**, which is what we want for debiasing.

---

## 8. Q&A (10 Common Questions)

1. **Q:** Why don’t we just remove ethnicity from the input?
   **A:** Even if you remove explicit ethnicity labels, models can infer it from other cues (accent, pitch patterns, etc.). GRL helps remove this implicit encoding.

2. **Q:** Why not simply penalize group accuracy in the loss directly?
   **A:** That would still train the feature extractor to **help** the group head. GRL flips the gradient so F learns to **hurt** group prediction.

3. **Q:** Does lower group accuracy always mean better fairness?
   **A:** It usually means less group information in features, which is helpful, but we still need to check *subgroup performance metrics* (recall, gaps) to be sure.

4. **Q:** Can adversarial training completely remove bias?
   **A:** Not necessarily; it can reduce **one type** of bias (encoded demographic info), but data collection, labeling bias, and other factors still matter.

5. **Q:** Will GRL make my model worse overall?
   **A:** If λ is set well, main performance usually stays good while fairness improves. If λ is too large, performance may drop too much.

6. **Q:** Why do we use CrossEntropyLoss for the group head?
   **A:** Because ethnicity is a multi-class classification problem (3 groups).

7. **Q:** How many groups can the adversarial head handle?
   **A:** Any number; you just change the output size of the group head.

8. **Q:** Can this technique be applied to other tasks?
   **A:** Yes—any place you want features to be invariant to a specific attribute (e.g., domain, accent, device type).

9. **Q:** How do judges react to adversarial debiasing?
   **A:** Very positively, if explained clearly. It shows understanding of both deep learning and fairness.

10. **Q:** How will we use GRL with CNNs on spectrograms?
    **A:** Replace the tabular FeatureExtractor with your CNN (from Lesson 4) and attach main and group heads to the CNN’s final feature vector.

---

## 9. Quiz (10 Questions)

1. **What is the main purpose of adversarial debiasing?**
   ➜ To learn representations that perform well on the main task while being less predictive of sensitive attributes.

2. **What does GRL do during the forward pass?**
   ➜ Passes inputs through unchanged.

3. **What does GRL do during the backward pass?**
   ➜ Multiplies gradients by -λ, reversing their direction.

4. **What are the three main parts of the adversarial network?**
   ➜ FeatureExtractor, MainHead, GroupHead.

5. **If group accuracy stays very high after adversarial training, what does that suggest?**
   ➜ The features still encode strong group information; debiasing is weak.

6. **If λ is set to 0.0, what effect does GRL have?**
   ➜ No effect; no adversarial pressure.

7. **What might happen if λ is too large?**
   ➜ Main task performance may drop significantly.

8. **Why do we need a group label (`y_group`) at training time?**
   ➜ The adversarial head needs ground-truth group labels to learn what to predict (and what we want the features to hide).

9. **Is adversarial debiasing done at training time or inference time?**
   ➜ Training time; at inference, we only use the feature extractor + main head.

10. **How do we know if adversarial debiasing is working?**
    ➜ Group head accuracy goes down while main task accuracy and fairness metrics (like subgroup recall gaps) improve.

---

## 10. Mini Practice Project – Adversarial Debiasing with GRL (with .zip)

I created a mini project where the student can **see GRL in action** on a synthetic dataset.

### Project Goal

* Use synthetic tabular data where **group membership is correlated** with both features and labels.
* Train a small **FeatureExtractor + MainHead + GroupHead** network.
* Use a **Gradient Reversal Layer** so the feature extractor hides group information.
* Explore different λ values to see the trade-off between main accuracy and group accuracy.

### What’s in the .zip

**`lesson6_miniproject_adv_debias.zip`** contains:

* `lesson6_miniproject_adv/`

  * `README.md` – Instructions and experiment ideas.
  * `data/synthetic_adv_data.npz` – Numpy archive with:

    * `X`: (600, 16) feature vectors
    * `y_main`: (600,) main labels (0/1)
    * `y_group`: (600,) group labels (0,1,2 for 3 groups)
  * `src/train_adv_debias.py` – Script including:

    * `GRL` custom autograd function
    * `FeatureExtractor`, `MainHead`, `GroupHead`
    * Training loop with `lambda_grl` parameter
    * Prints train/val accuracy for main and group heads each epoch

### How to Run

1. Unzip the archive.

2. Install dependencies:

   ```bash
   pip install torch numpy scikit-learn
   ```

3. Run:

   ```bash
   python src/train_adv_debias.py
   ```

4. Watch printed lines like:

   > Epoch 05 | Train main acc: 0.88, Train group acc: 0.55 | Val main acc: 0.85, Val group acc: 0.38

5. Try changing `lambda_grl` in `train()` to `0.0`, `0.5`, `1.0` and compare:

   * How main accuracy changes.
   * How group accuracy changes.

This gives an intuitive feel for how strong debiasing affects performance.

---

## 11. References

* Adversarial fairness ideas from your project plan (adversarial head to reduce WWIB). 
* Domain-Adversarial Training of Neural Networks (DANN) — classic paper (for more advanced reading).
* PyTorch custom autograd functions: official docs for extending autograd.

---

## 12. Additional Information (Mentor Tips)

* Have the student keep a **small results table**: rows = λ values, columns = main accuracy, group accuracy, fairness gap (if they later add subgroup metrics).
* For the final project, they can say something like:

  > “We used an adversarial debiasing architecture with a gradient reversal layer to reduce encoded ethnicity information in the spectrogram representations, which reduced the recall gap between White and South Asian speakers from X% to Y% while maintaining overall accuracy.”
* That single sentence, backed by proper metrics and plots, will sound extremely strong to ISEF judges.