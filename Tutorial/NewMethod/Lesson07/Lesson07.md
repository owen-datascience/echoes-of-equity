# ⭐ Lesson 7 — Adversarial CNN on Spectrograms: From Idea to Full Architecture

---

## 1. Lesson Summary

In this lesson, the student will:

* Combine what they learned from:

  * Lesson 3 (Mel-spectrograms),
  * Lesson 4 (CNN classifier),
  * Lesson 6 (adversarial debiasing & GRL),
* Into **one coherent adversarial CNN architecture** for spectrogram-like inputs.
* Understand how to:

  * Use a **CNN feature extractor** on 2D inputs,
  * Attach **main** and **adversarial group** heads,
  * Train with a **Gradient Reversal Layer** to reduce encoded bias.
* Practice on a **synthetic spectrogram dataset** to safely debug the architecture before using the real trustworthiness corpus.

This architecture is a dry run of the real system described in your Echoes of Equity project plan. 

---

## 2. Key Points

* A **2D CNN** can serve as the shared feature extractor for both main and adversarial tasks.
* The **main head** predicts trustworthiness (binary).
* The **group head** predicts ethnicity (multi-class).
* A **Gradient Reversal Layer (GRL)** is inserted before the group head.
* During training, we minimize:

  * `L_main` for trustworthiness,
  * `L_group` for ethnicity,
    but GRL makes the feature extractor **maximize** `L_group`, erasing group information.
* We monitor **main accuracy** and **group accuracy** on validation to see the trade-off.
* Once the student is comfortable with this setup on toy data, they can switch to **real Mel-spectrograms** from the actual dataset.

---

## 3. Real-World Examples or Stories

* **Domain-adversarial CNNs** are used to make models robust across different accents, microphones, or languages by learning **domain-invariant** features.
* Your project adapts this idea for **fairness**: instead of “domains,” the adversary targets **ethnicity**, with the goal of making features **ethnicity-invariant**.
* For ISEF, being able to say, “We used an adversarial CNN with gradient reversal to minimize demographic information in the internal representation” is a big intellectual “wow” moment for judges.

---

## 4. Terminology Explained

* **2D CNN Feature Extractor** – The convolutional layers and pooling layers that turn a 2D spectrogram into a 1D feature vector.
* **Main Task Head** – A small network (usually linear layers) sitting on top of features to predict the primary label (trustworthiness).
* **Adversarial Group Head** – A small network predicting group label (ethnicity).
* **Shared Representation** – The output of the feature extractor that both heads use.
* **Synthetic Spectrograms** – Fake spectrogram-like images used for debugging and learning; they behave similarly to real ones but are easy to generate.
* **Joint Training** – Training feature extractor and both heads together in a single optimization loop.

---

## 5. How It Works (Step-by-Step)

Here we use a **toy setup**: 32×32 “spectrogram-like” images with a binary main label and 3-group demographic label.

### A. Data

* `X`: shape `(N, 1, 32, 32)` – synthetic spectrogram images.
* `y_main`: shape `(N,)` – binary labels (0/1).
* `y_group`: shape `(N,)` – group labels (0, 1, 2 for three ethnicities).

In the mini-project data, group-dependent patterns are injected so the network *could* learn ethnicity if we let it.

---

### B. Dataset & DataLoader

```python
from torch.utils.data import Dataset, DataLoader
import torch

class AdvCnnDataset(Dataset):
    def __init__(self, X, y_main, y_group):
        self.X = torch.from_numpy(X)
        self.y_main = torch.from_numpy(y_main)
        self.y_group = torch.from_numpy(y_group)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y_main[idx], self.y_group[idx]
```

Then split into train/val and wrap with `DataLoader`.

---

### C. CNN Feature Extractor

```python
import torch.nn as nn

class CnnFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)  # shape: (batch, 16*8*8)
        return x
```

This structure mirrors Lesson 4’s CNN, but now used as a **shared** extractor.

---

### D. Heads & GRL

**Main Head:**

```python
class MainHead(nn.Module):
    def __init__(self, in_dim=16*8*8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, h):
        return self.fc(h)
```

**Group Head:**

```python
class GroupHead(nn.Module):
    def __init__(self, in_dim=16*8*8, n_groups=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_groups),
        )

    def forward(self, h):
        return self.fc(h)
```

**GRL:**

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

### E. Training Loop (High-Level)

For each batch `(xb, yb_main, yb_group)`:

1. Compute features: `h = feat(xb)`
2. Main path:

   * `logits_main = main_head(h)`
   * `loss_main = CrossEntropy(logits_main, yb_main)`
3. Group path:

   * `h_rev = grad_reverse(h, lambda_grl)`
   * `logits_group = group_head(h_rev)`
   * `loss_group = CrossEntropy(logits_group, yb_group)`
4. Total loss:

   * `loss = loss_main + loss_group`
5. Backprop and optimizer step.
6. Track training & validation accuracy for both heads.

Interpretation:

* If `lambda_grl > 0`, the network is encouraged to **hide group information** while still doing the main task well.

---

## 6. Practice Exercises (5)

**Exercise 1 – Forward Path Explanation**
Explain step-by-step what happens when you call:

```python
h = feat(xb)
logits_main = main_head(h)
h_rev = grad_reverse(h, lambda_grl)
logits_group = group_head(h_rev)
```

---

**Exercise 2 – Shapes Check**
Given input batch `xb` with shape `(32, 1, 32, 32)`:

* What are the shapes after:

  * First conv + pool
  * Second conv + pool
  * Flatten
  * Main head output
  * Group head output

---

**Exercise 3 – λ Experiment (Paper Plan)**
Design a small experiment plan where you will run the model with λ = 0.0, 0.5, 1.0.
For each λ, list the metrics you will record (e.g., main val accuracy, group val accuracy) and what you expect.

---

**Exercise 4 – Turn Off Adversarial Head**
Conceptually, what changes if you train only the main head (no group head / GRL)?
How might this affect fairness compared to the adversarial version?

---

**Exercise 5 – Link to Real Spectrograms**
Write a short paragraph describing how you would replace the synthetic 32×32 spectrograms with real Mel-spectrograms from the trustworthiness dataset (paths, preprocessing, shapes).

---

## 7. Solutions (Model Answers)

**Solution 1 – Forward Path Explanation**

* `h = feat(xb)` – CNN feature extractor converts each spectrogram into a feature vector.
* `logits_main = main_head(h)` – Main head outputs logits for trustworthy vs neutral.
* `h_rev = grad_reverse(h, lambda_grl)` – Same features in forward pass, but will reverse gradient during backprop.
* `logits_group = group_head(h_rev)` – Adversarial head predicts group (ethnicity) from the gradient-reversed features.

---

**Solution 2 – Shapes**

Starting with `(batch=32, channels=1, H=32, W=32)`:

* After Conv1 + ReLU + MaxPool2d(2):

  * Channels: 8, H and W halved: `(32, 8, 16, 16)`
* After Conv2 + ReLU + MaxPool2d(2):

  * Channels: 16, H and W halved again: `(32, 16, 8, 8)`
* After Flatten:

  * Each sample becomes length `16*8*8 = 1024`: `(32, 1024)`
* Main head output:

  * `(32, 2)` (two classes).
* Group head output:

  * `(32, 3)` (three groups).

---

**Solution 3 – λ Experiment Plan**

For each λ ∈ {0.0, 0.5, 1.0}:

* Train for 15 epochs with the same random seed.
* Record:

  * Validation main accuracy (binary).
  * Validation group accuracy (multi-class).
* Expectations:

  * λ = 0.0 → high main accuracy, high group accuracy (lots of group info).
  * λ = 0.5 → similar main accuracy, lower group accuracy.
  * λ = 1.0 → slightly lower main accuracy, group accuracy closer to random (1/3).

---

**Solution 4 – Only Main Head**

If you train only the main head:

* There is no pressure to hide group information.
* The CNN is free to encode whatever helps main accuracy, including group-specific patterns.
* This may increase fairness issues because internal features could strongly encode ethnicity.

---

**Solution 5 – Linking to Real Spectrograms**

Example paragraph:

> To move from synthetic 32×32 spectrogram images to real Mel-spectrograms, I would first compute Mel-spectrograms from the audio files using librosa, then normalize and resize them (e.g., to 64×64 or 128×128). I would store them as numpy arrays or images, with corresponding labels for trustworthiness and ethnicity. The CNN feature extractor would be updated to handle the new input size (e.g., 1×64×64). The training loop would remain the same, except that the DataLoader would now load real spectrogram tensors instead of synthetic ones.

---

## 8. Q&A (10 Common Questions)

1. **Q:** Why do we use a CNN instead of a fully connected network here?
   **A:** Because spectrograms have spatial (time-frequency) structure; CNNs exploit local patterns efficiently.

2. **Q:** How is this different from Lesson 6’s adversarial model?
   **A:** Lesson 6 used a tabular feature extractor; Lesson 7 uses a **CNN** on 2D inputs, which is what we’ll use for real spectrograms.

3. **Q:** Do we need to use GRL when evaluating the trained model?
   **A:** GRL only affects training. At test time, it just passes features through.

4. **Q:** Can we remove the adversarial head at inference time?
   **A:** Yes. For deployment, you only need the feature extractor + main head.

5. **Q:** How do we know if the adversarial CNN is “better” than a baseline?
   **A:** Compare main metrics (accuracy, recall) and fairness metrics (e.g., recall per group, fairness gap) against the baseline CNN without GRL.

6. **Q:** Is this architecture heavy for a high schooler’s computer?
   **A:** The synthetic version is tiny. The real version with modest CNN size should still be fine on CPU or Colab.

7. **Q:** Can we use more convolutional layers?
   **A:** Yes, but start simple to avoid debugging complexity. You can gradually deepen the model later.

8. **Q:** Should we apply data augmentation to spectrograms?
   **A:** Potentially yes (time/frequency masking), but that’s an advanced extension for later lessons.

9. **Q:** Does adversarial training always guarantee fairness?
   **A:** No guarantee; it reduces one type of encoded bias. You still must evaluate subgroup metrics carefully.

10. **Q:** How do we explain this architecture to non-technical judges?
    **A:** Something like: “We trained the network to detect trustworthiness while also training a second network that tries to guess the speaker’s ethnicity. We then taught the shared representation to hide information from this second network, so the main model focuses on speech patterns rather than demographics.”

---

## 9. Quiz (10 Questions)

1. **What type of data does the adversarial CNN in this lesson consume?**
   ➜ 2D spectrogram-like images.

2. **What are the three main modules of the adversarial CNN?**
   ➜ CNN feature extractor, main head, group head.

3. **What does the main head predict?**
   ➜ Trustworthiness (binary: trustworthy vs neutral).

4. **What does the group head predict?**
   ➜ Group/ethnicity (multi-class).

5. **Where is GRL placed in the network?**
   ➜ Between the feature extractor and the group head.

6. **What happens to gradients when passing through GRL?**
   ➜ They are multiplied by -λ (reversed and scaled).

7. **What is the effect of increasing λ (if not too big)?**
   ➜ Group accuracy usually drops (less group info), while main accuracy can stay reasonably high.

8. **Why use MaxPool2d in the CNN?**
   ➜ To reduce spatial resolution and help the network focus on high-level patterns.

9. **At deployment, which parts of the model are necessary?**
   ➜ Feature extractor + main head.

10. **How does this lesson prepare for using the real dataset?**
    ➜ It validates the full adversarial CNN architecture on toy data, so we can later swap in real spectrograms and trustworthiness/ethnicity labels with confidence.

---

## 10. Mini Practice Project – Adversarial CNN on Synthetic Spectrograms (.zip)

To make all this concrete, I’ve created a mini project that trains an **adversarial CNN** on synthetic spectrogram-like images.

### Project Goal

* Train a CNN with:

  * Shared convolutional feature extractor,
  * Main classification head,
  * Adversarial group head with GRL.
* Observe how changing `lambda_grl` affects:

  * Main validation accuracy,
  * Group validation accuracy.

### What’s in the .zip

**`lesson7_miniproject_adv_cnn.zip`** contains:

* `lesson7_miniproject_adv_cnn/`

  * `README.md` – Instructions and experiment ideas.
  * `data/synthetic_adv_cnn_data.npz` – Numpy archive with:

    * `X`: (N, 1, 32, 32) synthetic spectrogram images
    * `y_main`: (N,) main labels (0/1)
    * `y_group`: (N,) group labels (0,1,2)
  * `src/train_adv_cnn.py` – Script that:

    * Defines `CnnFeatureExtractor`, `MainHead`, `GroupHead`, and `GRL`
    * Trains the model with `lambda_grl`
    * Prints train/validation main & group accuracy each epoch

### How to Run

1. Unzip the archive.

2. Install dependencies:

   ```bash
   pip install torch numpy scikit-learn
   ```

3. Run:

   ```bash
   python src/train_adv_cnn.py
   ```

4. Then:

   * Note main and group accuracy on validation.
   * Change `lambda_grl` in `train()` to `0.0`, `0.5`, `1.0`, re-run, and record results.
   * Write a short reflection: which λ seems to best balance main accuracy and reducing group predictability?

---

## 11. References

* Echoes of Equity project design sections on **adversarial head and fairness**, and CNN-based models. 
* Domain-Adversarial Training literature (for deeper background): “Domain-Adversarial Training of Neural Networks” (Ganin et al.).
* PyTorch tutorial on building CNNs for classification.

---

## 12. Additional Information (Mentor Tips)

* Encourage the student to **save tables** of (λ, main acc, group acc) as this will directly support a figure or table in the final paper.
* In the actual project, after plugging in real spectrograms, they should:

  * Compare baseline CNN vs adversarial CNN on **subgroup recall**.
  * Show that the adversarial CNN reduces recall gaps while maintaining strong overall performance.
* For competition judging, this lesson’s content is where the student can confidently talk about:

  * Architecture innovation,
  * Fairness-aware design,
  * Systematic experimentation with hyperparameters (λ).