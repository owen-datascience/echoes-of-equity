# ⭐ Lesson 9 — Explainability: What Is the Model “Listening To”?

---

## 1. Lesson Summary

In this lesson, the student will:

* Learn why **explainability** is critical for a fairness project like *Echoes of Equity*. 
* Understand how to use **Grad-CAM** (Gradient-weighted Class Activation Mapping) to highlight **which parts of a spectrogram** influenced a prediction.
* Connect explainability to **fairness**:

  * Are we focusing on “trustworthiness prosody” or demographic cues?
* Practice on a **synthetic spectrogram dataset**:

  * Train a small CNN,
  * Generate Grad-CAM heatmaps,
  * Interpret them qualitatively.

These skills will later be applied to the real Mel-spectrograms for the ISEF paper/poster.

---

## 2. Key Points

* Explainability helps answer: **“Why did the model call this sample trustworthy?”**
* **Grad-CAM** uses gradients of the target class with respect to feature maps in a convolutional layer to produce a **heatmap of importance**.
* For spectrograms, Grad-CAM shows **time–frequency regions** the model relies on.
* In a fairness project, we care if heatmaps differ **systematically** between groups in suspicious ways (e.g., focusing mainly on portions correlated with accent or background noise).
* Combining:

  * **Quantitative metrics** (accuracy, fairness gaps), and
  * **Qualitative visualizations** (Grad-CAM)
    makes the scientific argument much stronger. 
* Judges love clear visuals: “Before debiasing vs after debiasing” heatmaps for the same utterance.

---

## 3. Real-World Examples or Stories

* In medical imaging, Grad-CAM heatmaps show radiologists whether a model is focusing on the actual lesion or irrelevant artifacts.
* In speech emotion recognition, Grad-CAM on spectrograms can reveal which **pitch or energy patterns** matter most for detecting anger, happiness, etc.
* For *Echoes of Equity*, you can show heatmaps for:

  * Baseline CNN (no debiasing) vs
  * Adversarially debiased CNN,
    on the **same audio sample**, and discuss if the attention shifts toward more “universal” trust cues.

---

## 4. Terminology Explained

* **Explainability / Interpretability** – Methods to understand how a model makes its decisions.
* **Grad-CAM** – A technique that uses gradients backpropagated from a target class to create a **class-specific saliency map** over the feature maps of a CNN.
* **Activation Maps / Feature Maps** – Outputs of convolution layers that capture local patterns in the input.
* **Heatmap** – A color-coded image where bright/high values represent important regions and dark/low values less important regions.
* **Saliency** – How much each part of the input contributes to the output.
* **Overlay** – Plotting the heatmap on top of the original spectrogram to make interpretation easier.

---

## 5. How It Works (Step-by-Step)

We’ll use a tiny CNN on 32×32 spectrogram-like images (synthetic) to demonstrate Grad-CAM.

### A. Model Structure

```python
class SimpleCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(16 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # 2 classes
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
```

We will compute Grad-CAM with respect to the **last conv layer** (`self.conv[-2]`, the second `Conv2d`).

---

### B. Intuition of Grad-CAM

1. Pick an input `x` and a **target class** (e.g., predicted class).
2. Run forward pass → get feature maps `A` from chosen conv layer and logits `y`.
3. Compute gradients of `y[target_class]` w.r.t. `A`.
4. Average gradients over spatial dimensions → “importance weights” for each channel.
5. Compute weighted sum of feature maps using these weights → class activation map.
6. Apply ReLU, normalize, and upsample to input resolution.
7. Overlay on original spectrogram.

---

### C. Implementation Sketch

**Hooks to capture activations and gradients:**

```python
conv_layer = model.conv[-2]
activations = []
gradients = []

def forward_hook(module, inp, out):
    activations.append(out.detach())

def backward_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0].detach())

handle_f = conv_layer.register_forward_hook(forward_hook)
handle_b = conv_layer.register_backward_hook(backward_hook)
```

**Compute Grad-CAM:**

```python
x = x.to(device)
x.requires_grad_(True)

logits = model(x)
loss = logits[0, target_class]
model.zero_grad()
loss.backward()

act = activations[0]     # (1, C, Hc, Wc)
grad = gradients[0]      # (1, C, Hc, Wc)

weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
cam = (weights * act).sum(dim=1)              # (1, Hc, Wc)
cam = torch.relu(cam[0])

cam -= cam.min()
if cam.max() > 0:
    cam /= cam.max()
```

**Upsample and overlay:**

```python
cam_upsampled = torch.nn.functional.interpolate(
    cam.unsqueeze(0).unsqueeze(0),
    size=x.shape[2:], mode="bilinear", align_corners=False
)[0, 0]

heatmap = cam_upsampled.detach().cpu().numpy()
```

Then plot original + overlay:

```python
axes[0].imshow(x_np[0, 0], aspect="auto", origin="lower")
axes[1].imshow(x_np[0, 0], cmap="gray", origin="lower", aspect="auto")
axes[1].imshow(heatmap, cmap="jet", alpha=0.5, origin="lower", aspect="auto")
```

---

### D. Connecting to Echoes of Equity

On the **real project**: 

* Compute Grad-CAM for:

  * Baseline CNN (no adversarial head), and
  * Debiased CNN (with GRL),
    on the **same audio**.
* Ask:

  * Are we focusing on similar regions?
  * Does the debiased network move away from obviously group-specific artifacts?
* Show these pairs of heatmaps on the ISEF board under **“Model Interpretation & Fairness Evidence”**.

---

## 6. Practice Exercises (5)

**Exercise 1 – Explainability Motivation**
In your own words, explain why visual explainability is especially important in a fairness-focused project like Echoes of Equity.

---

**Exercise 2 – Grad-CAM Intuition**
Without math, describe what Grad-CAM tells you about a model’s prediction on a spectrogram.

---

**Exercise 3 – Where to Hook**
Which layer of the CNN is usually a good choice for Grad-CAM, and why not the very first or very last layer?

---

**Exercise 4 – Misclassification Analysis**
Suppose the model misclassifies a sample (predicts trustworthy but it’s actually neutral). How could a Grad-CAM heatmap help you analyze the error?

---

**Exercise 5 – Fairness Hypothesis**
Write a short hypothesis: how you would expect Grad-CAM heatmaps to **change** before vs after adversarial debiasing for the same speaker.

---

## 7. Solutions (Model Answers)

**Solution 1 – Explainability Motivation**

Explainability lets us see *how* the model is using the audio. In a fairness project, we want confidence that the model focuses on **trust-related prosody** instead of unwanted demographic cues. Grad-CAM visualizations help judges and users trust that your fairness methods actually changed what the model listens to.

---

**Solution 2 – Grad-CAM Intuition**

Grad-CAM highlights which parts of the spectrogram had the **largest influence** on a specific prediction. Bright areas in the heatmap are where the model was “paying attention” when deciding “trustworthy” vs “neutral.”

---

**Solution 3 – Where to Hook**

A late convolutional layer (e.g., the last conv block) is usually best:

* Early layers detect very low-level patterns (like edges/noise), which are too fine-grained.
* The very last fully connected layer loses spatial structure.
* Late conv layers still have spatial layout but represent higher-level patterns.

---

**Solution 4 – Misclassification Analysis**

Grad-CAM can show that the model focused on an irrelevant burst of noise or a short segment where the speaker raised their voice, instead of the overall steady tone. This can suggest ideas for data cleaning, augmentation, or architecture changes.

---

**Solution 5 – Fairness Hypothesis**

Example hypothesis:

> Before adversarial debiasing, Grad-CAM heatmaps might focus on regions that correlate with certain accents or speaking styles typical of one group. After debiasing, heatmaps should become more similar across groups, emphasizing core trust cues (steady pitch, moderate intensity) rather than group-specific quirks.

---

## 8. Q&A (10 Common Questions)

1. **Q:** Is Grad-CAM the only explainability method?
   **A:** No. There are others like vanilla saliency, Integrated Gradients, LIME, SHAP, etc. Grad-CAM is convenient for CNNs on images/spectrograms.

2. **Q:** Does Grad-CAM change how the model behaves?
   **A:** No, it just analyzes the model after training; it doesn’t alter weights.

3. **Q:** Can Grad-CAM prove the model is fair?
   **A:** Not by itself. It provides qualitative evidence that supports quantitative fairness metrics.

4. **Q:** What if the heatmaps look noisy?
   **A:** That may mean the model is not very confident or the architecture is too small. Training longer or smoothing can help.

5. **Q:** Can we use Grad-CAM on the adversarial head?
   **A:** Yes, in theory you could visualize what features help the group classifier and compare them to the main task’s focus.

6. **Q:** Do we need GPU for Grad-CAM?
   **A:** No, CPU is fine for small models, just slower.

7. **Q:** How many example heatmaps should we show in the paper/poster?
   **A:** A handful of **carefully chosen cases** (e.g., one per group, before/after debiasing) is better than dozens of similar images.

8. **Q:** Can Grad-CAM be wrong?
   **A:** It’s an approximation, so don’t over-interpret single pixels. Look for broader patterns and validate with domain knowledge.

9. **Q:** How do we choose which samples to visualize?
   **A:** Good choices: typical correct predictions, interesting edge cases, and misclassified examples.

10. **Q:** Does Grad-CAM work for non-CNN models?
    **A:** It’s primarily designed for CNNs. Other methods are more suitable for transformers or tabular models.

---

## 9. Quiz (10 Questions)

1. **What is the main purpose of Grad-CAM?**
   ➜ To highlight which regions of an input (image/spectrogram) are most important for a specific class prediction.

2. **Which part of the model do we typically apply Grad-CAM to?**
   ➜ A late convolutional layer.

3. **What information does Grad-CAM use to compute the heatmap?**
   ➜ Gradients of the target class score with respect to the feature maps.

4. **Why is Grad-CAM especially suitable for spectrograms?**
   ➜ Spectrograms are 2D like images, and Grad-CAM produces 2D heatmaps aligned with them.

5. **Does Grad-CAM require retraining the model?**
   ➜ No, it only requires access to gradients during evaluation.

6. **How can Grad-CAM support fairness analysis?**
   ➜ By showing whether the model is focusing on similar prosodic regions across different demographic groups.

7. **What does a brighter region in a Grad-CAM heatmap indicate?**
   ➜ A region that strongly contributed to the model’s decision for that class.

8. **What is the role of ReLU in Grad-CAM?**
   ➜ It keeps only positive influences for the target class, making the heatmap easier to interpret.

9. **Why do we normalize the heatmap to [0, 1]?**
   ➜ To map it cleanly into color scales and compare across samples.

10. **What is a good way to present Grad-CAM results in your ISEF poster?**
    ➜ Side-by-side panels: original spectrogram vs Grad-CAM overlay, with clear labels and a short caption explaining what the model is focusing on.

---

## 10. Mini Practice Project – Grad-CAM on Synthetic Spectrograms (.zip)

To make this hands-on, I prepared a mini project where the student:

* Trains a small CNN on synthetic spectrogram-like images, and
* Uses Grad-CAM to visualize what the model is “listening to.”

### Project Goal

* Learn how to:

  * Attach hooks to CNN layers,
  * Compute Grad-CAM heatmaps,
  * Save PNG visualizations,
  * Interpret them.

### What’s in the .zip

**`lesson9_miniproject_explain.zip`** contains:

* `lesson9_miniproject_explain/`

  * `README.md` – Step-by-step instructions.
  * `data/synthetic_explain_data.npz` – Synthetic spectrogram-like dataset:

    * `X`: (N, 1, 32, 32)
    * `y_main`: (N,) binary labels (class 0 vs 1)
    * `y_group`: (N,) group labels (not used here but included for realism)
  * `src/gradcam_demo.py` – Script that:

    * Trains `SimpleCnn` for a few epochs.
    * Computes Grad-CAM for a few validation samples.
    * Saves side-by-side original + Grad-CAM images in `outputs/`.

### How to Run

1. Unzip the archive.

2. Install dependencies:

   ```bash
   pip install torch numpy scikit-learn matplotlib
   ```

3. Run:

   ```bash
   python src/gradcam_demo.py
   ```

4. Open the `outputs/` folder and view images like `example_0_gradcam.png`.

Prompt the student:

* Describe where the red/yellow regions are.
* Are they aligned with the obvious “class pattern” in the synthetic spectrogram?
* How might this look different for real trustworthiness spectrograms?

---

## 11. References

* Echoes of Equity project outline: fairness-aware CNN and visualization deliverables.
* Original Grad-CAM paper (for advanced reading): “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.”
* Example tutorials: “Grad-CAM with PyTorch” (various blog posts and GitHub examples).

---

## 12. Additional Information (Mentor Tips)

* For the real project, I’d recommend:

  * Choose **3–6 representative samples** (across groups) and show heatmaps for:

    * Random Forest (no visualization, just for contrast),
    * Baseline CNN,
    * Adversarially debiased CNN.
* For the paper/poster, include a small **“Methods: Explainability”** section explaining Grad-CAM in 3–4 sentences plus a diagram.
* Encourage the student to keep a **notebook of interesting Grad-CAM cases** (especially misclassifications) and reflect on whether they align with their fairness story.
