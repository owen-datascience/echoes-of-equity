# ⭐ Lesson 4 — Building Your First CNN for Trustworthiness Classification

---

## 1. Lesson Summary

In this lesson, the student learns how to:

* Treat Mel-spectrograms as **image-like inputs**.
* Build a small **Convolutional Neural Network (CNN)** in PyTorch.
* Train the CNN on a simple dataset and monitor **training/validation accuracy**.
* Understand the basic building blocks: convolution, ReLU, pooling, flatten, dense layers.

This CNN is a **non-adversarial baseline**: later, you will modify and extend it with an adversarial head to reduce demographic bias as described in the project plan. 

---

## 2. Key Points

* CNNs are ideal for **2D grid data** like images and spectrograms.
* A CNN learns **local patterns** (edges, shapes, textures) with convolution filters.
* **Pooling** layers reduce spatial size, making the model faster and more robust.
* **ReLU** introduces nonlinearity so the network can learn complex relationships.
* A typical architecture for spectrograms: Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → Dense → Output.
* We split data into **train** and **validation** sets to measure generalization.
* We use a **loss function** (CrossEntropyLoss) and an **optimizer** (Adam/SGD) to update weights.
* This simple CNN becomes the “baseline deep model” that your fairness-aware model will later improve upon.

---

## 3. Real-World Examples or Stories

* **Speech command recognition (e.g., “OK Google”)** uses CNN-like models on short spectrogram snippets.
* **Environmental sound classification** (sirens, dog barking, door closing) relies heavily on CNNs on audio spectrograms.
* **Emotion recognition from voice**: CNNs learn patterns of prosody associated with emotions like anger or joy.

Your project is similar: a CNN on spectrograms learns acoustic patterns associated with “trustworthy” vs “neutral” intent.

---

## 4. Terminology Explained

* **Convolutional Layer (Conv2d)** – A layer that slides small filters across the input to detect local patterns (like edges, frequency bands, etc.).
* **Filter / Kernel** – A small matrix (e.g., 3×3) that is multiplied with local patches of the input.
* **Feature Map** – The output of one filter; each filter learns to detect a specific pattern.
* **ReLU (Rectified Linear Unit)** – Activation function `f(x) = max(0, x)` that introduces nonlinearity.
* **Max Pooling** – Operation that takes the maximum value in a small window (e.g., 2×2), reducing resolution while keeping important features.
* **Flatten** – Reshapes 2D feature maps into a 1D vector so it can be fed to fully connected layers.
* **Fully Connected (Dense) Layer** – Classic neural layer where each input connects to each output.
* **Epoch** – One full pass through the training dataset.
* **Batch** – A subset of the dataset processed in one gradient update.
* **Loss Function** – Measures how wrong the model is; training aims to minimize it.

---

## 5. How It Works (Step-by-Step)

Here’s a conceptual pipeline for this lesson (with synthetic data for now):

1. **Prepare input data**

   * Use 2D spectrograms (or synthetic “spectrogram-like” arrays) of shape `(1, H, W)`.
   * Labels: `0` for neutral, `1` for trustworthy.

2. **Create a PyTorch Dataset**

```python
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)  # shape: (N, 1, H, W)
        self.y = torch.from_numpy(y)  # shape: (N,)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

3. **Build a SimpleCNN**

```python
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # halves H and W
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 2)      # 2 classes
        )

    def forward(self, x):
        return self.net(x)
```

4. **Training Loop**

* Use `CrossEntropyLoss` to compare predictions vs labels.
* Use `Adam` optimizer to update weights.
* For each epoch:

  * Loop over batches, compute loss, backprop, optimizer step.
  * Track training accuracy and loss.
  * Evaluate on validation set after each epoch.

5. **Result**

You should see validation accuracy go above 0.5 (better than chance). This proves the CNN can learn patterns from spectrogram-like data. Later you’ll swap in real Mel-spectrograms from the trustworthiness dataset.

---

## 6. Practice Exercises (5)

**Exercise 1 – CNN Architecture Sketch**
On paper or in a markdown cell, sketch a tiny CNN architecture for spectrograms of size `1×32×32`:

* Two conv layers, two max-pool layers, one hidden dense layer, one output layer.
  Write down the shape of the data after each layer.

---

**Exercise 2 – Dataset Class**
Write a PyTorch `Dataset` class that:

* Accepts numpy arrays `X` (N×1×32×32) and `y` (N,).
* Returns `(spectrogram, label)` pairs.

---

**Exercise 3 – DataLoader Setup**
Using your `Dataset`, create a PyTorch `DataLoader` for:

* Batch size = 16
* Shuffle = True for training set

---

**Exercise 4 – Forward Pass Test**
Instantiate `SimpleCNN`, create a fake batch of data with shape `(4, 1, 32, 32)`, and:

* Run it through the model.
* Print the output shape.

---

**Exercise 5 – Training Epoch Skeleton**
Write code to perform **one epoch** of training:

* Loop over `train_loader`.
* Compute logits, loss, backprop, and optimizer step.
* Track batch accuracy and print average at the end.

---

## 7. Solutions (Model Answers)

**Solution 1 – CNN Architecture Sketch (Shapes)**
Input: `(1, 32, 32)`

* Conv1 (1→8, kernel 3, padding 1): `(8, 32, 32)`
* MaxPool1 (2×2): `(8, 16, 16)`
* Conv2 (8→16, kernel 3, padding 1): `(16, 16, 16)`
* MaxPool2 (2×2): `(16, 8, 8)`
* Flatten: `16 * 8 * 8 = 1024`
* Dense1: 1024 → 32
* Dense2: 32 → 2

---

**Solution 2 – Dataset Class**

```python
from torch.utils.data import Dataset
import torch

class SpectrogramDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)        # float32
        self.y = torch.from_numpy(y)        # int64

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

---

**Solution 3 – DataLoader Setup**

```python
from torch.utils.data import DataLoader

train_ds = SpectrogramDataset(X_train, y_train)
val_ds = SpectrogramDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
```

---

**Solution 4 – Forward Pass Test**

```python
import torch

model = SimpleCNN()
fake_batch = torch.randn(4, 1, 32, 32)  # 4 spectrograms
logits = model(fake_batch)
print("Output shape:", logits.shape)  # should be (4, 2)
```

---

**Solution 5 – Training Epoch Skeleton**

```python
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
running_loss = 0.0
correct = 0
total = 0

for Xb, yb in train_loader:
    Xb, yb = Xb.to(device), yb.to(device)

    optimizer.zero_grad()
    logits = model(Xb)
    loss = criterion(logits, yb)
    loss.backward()
    optimizer.step()

    running_loss += loss.item() * Xb.size(0)
    preds = logits.argmax(dim=1)
    correct += (preds == yb).sum().item()
    total += yb.size(0)

epoch_loss = running_loss / total
epoch_acc = correct / total
print(f"Train loss: {epoch_loss:.4f}, Train acc: {epoch_acc:.3f}")
```

---

## 8. Q&A (10 Common Questions)

1. **Q:** Why use a CNN instead of a regular dense network on flattened pixels?
   **A:** CNNs exploit local structure and share weights, making them more efficient and better at learning spatial patterns.

2. **Q:** How many layers should my CNN have?
   **A:** Start small (2 conv layers) so it’s easy to debug; you can add more later if needed.

3. **Q:** What if my model overfits?
   **A:** Use regularization (dropout, weight decay), data augmentation, or smaller models.

4. **Q:** Should I use ReLU or another activation?
   **A:** ReLU is standard and works well. You can experiment with LeakyReLU or others later.

5. **Q:** Why do we split into train and validation sets?
   **A:** To measure how well the model generalizes to unseen data.

6. **Q:** What is CrossEntropyLoss?
   **A:** A loss function for multi-class classification that compares predicted probabilities to true class labels.

7. **Q:** What does the optimizer do?
   **A:** It updates model weights using gradients from backpropagation to reduce the loss.

8. **Q:** How do I know if I’m using the GPU?
   **A:** Check `torch.cuda.is_available()` and move model and tensors to `device`.

9. **Q:** What is a “good” validation accuracy?
   **A:** For synthetic data, > 0.9 is possible; for real data, you compare against your baseline (e.g., > 71% from the Random Forest in the paper). 

10. **Q:** When do we bring in fairness/adversarial training?
    **A:** After we have a strong CNN baseline; this lesson is about building that foundation.

---

## 9. Quiz (10 Questions)

1. **What type of data are spectrograms?**
   ➜ 2D grid data (like images).

2. **Which layer in a CNN detects local patterns?**
   ➜ Convolutional layer (Conv2d).

3. **What does ReLU stand for, and what is its formula?**
   ➜ Rectified Linear Unit; `f(x) = max(0, x)`.

4. **Why do we use MaxPool2d?**
   ➜ To reduce spatial size and focus on the most important features.

5. **After two 2×2 max-pooling layers, how much are H and W reduced?**
   ➜ Reduced by a factor of 4 (half twice).

6. **Which loss function is standard for multi-class classification in PyTorch?**
   ➜ `nn.CrossEntropyLoss`.

7. **What does `.argmax(dim=1)` do on model outputs?**
   ➜ Picks the index of the largest logit, i.e., the predicted class.

8. **Why do we shuffle the training data each epoch?**
   ➜ To reduce learning order bias and improve generalization.

9. **What is an epoch?**
   ➜ One full pass through the entire training dataset.

10. **How do you move a model to GPU in PyTorch?**
    ➜ `model.to(device)` where `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`.

---

## 10. Mini Practice Project – Simple CNN on Synthetic Spectrograms (with .zip)

To make this very hands-on, I’ve prepared a **mini project** that trains a CNN on synthetic spectrogram-like data.

### Project Goal

* Load a synthetic dataset of `(1, 32, 32)` “spectrograms” with 2 classes.
* Train a small CNN in PyTorch.
* Observe training and validation accuracy over 10 epochs.

### What’s in the .zip

**`lesson4_miniproject_cnn.zip`** contains:

* `lesson4_miniproject_cnn/`

  * `README.md` – Instructions.
  * `data/synthetic_spectrograms.npz` – Numpy archive with:

    * `X`: `(120, 1, 32, 32)` float32 arrays
    * `y`: `(120,)` int64 labels (0 = neutral, 1 = trustworthy)
  * `src/train_cnn.py` – Script that:

    * Loads the dataset
    * Splits into train/validation sets
    * Defines `SpectrogramDataset` and `SimpleCNN`
    * Trains for 10 epochs and prints train/val accuracy

### How to Run

1. Unzip the archive.

2. Create a Python environment and install dependencies:

   ```bash
   pip install torch torchvision numpy scikit-learn
   ```

3. Run:

   ```bash
   python src/train_cnn.py
   ```

4. Watch the printed training/validation accuracy.

5. Optional: modify the architecture (more filters, different hidden size) and see how performance changes.

---

## 11. References

* Project plan sections on **Model Innovation – Train CNN** and **Deep Learning Model Training**. 
* PyTorch Tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)
* “Convolutional Neural Networks” chapter in *Dive into Deep Learning* (free online textbook).
* Any intro CNN explanation videos (e.g., 3Blue1Brown’s neural network series).

---

## 12. Additional Information (Coach Notes for ISEF)

* Have the student treat this mini CNN as a **sandbox**: try different `n_filters`, kernel sizes, and compare metrics.
* Encourage them to document **hyperparameters** and results in a table—this becomes part of the “Model Architecture & Hyperparameter Tuning” section in the paper.
* Later, when they move from synthetic data to real Mel-spectrograms from the OSF dataset, they’ll reuse the **same code structure**, just changing the data loader.
