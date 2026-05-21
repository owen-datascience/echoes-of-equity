# Tutorial 05 — Deep Learning Intro

Pause before writing any more code. The next two tutorials use **deep learning** — neural networks built with TensorFlow — and you need the vocabulary before the code will mean anything.

This tutorial has no code to copy. Read it twice if you have to. Take notes.

---

## Deep learning = neural networks with many layers

A **neural network** is a function that takes some numbers in, runs them through a sequence of "layers", and outputs other numbers. "Deep" just means there are many layers.

The diagram everyone draws looks like this:

```
       Inputs           Hidden layers              Output
   (features)
                        ●──●──●
        ●──┐         ╱     ╲   ╲
            ╲       ●       ●──●──● ──▶ probability
        ●──┐╲─╲   ╱   ╲   ╱       ╲
            ╲ ●─●     ●─●           ●──▶ "Trustworthy"
        ●──┐╱       ╲   ╱   ╲    ╱
            ╱       ●       ●──●──●
        ●──┘         ╲     ╱   ╱
                        ●──●──●
```

Each circle is a **neuron**. Each line is a **connection**, with its own learnable weight.

---

## What is a neuron, really?

A single neuron is just this tiny formula:

```
output = activation_function( w1*x1 + w2*x2 + ... + wn*xn + b )
                              └──────── a weighted sum ─────┘
```

- The `x`s are the inputs from the previous layer.
- The `w`s are weights — the things the network learns.
- `b` is a bias — a small adjustment per neuron, also learned.
- `activation_function` adds the *nonlinear* twist that makes neural networks powerful.

If you stacked neurons without activation functions, the whole network would mathematically collapse into one big linear function — exactly equivalent to Logistic Regression. The activation is what gives neural nets their flexibility.

---

## Common activation functions

You only need to know three.

### 1. ReLU (`Rectified Linear Unit`)

```
relu(x) = max(0, x)
```

```
  output ▲
         │       ╱
         │     ╱
         │   ╱
         │ ╱
       0 │╱_________________▶ input
            0
```

- Negative input? Output 0. Positive input? Pass it through.
- Fast and simple — the default choice for hidden layers in 2026.

### 2. Sigmoid

You met this in Tutorial 03. It squishes any number into 0–1, so it's perfect for the **final layer** of a binary classifier:

```
sigmoid(x) = 1 / (1 + e^(-x))
```

### 3. Softmax

For multi-class outputs (which we don't use in this project). It turns several raw scores into probabilities that sum to 1. Mentioned for completeness.

---

## How does a neural network learn?

This is the magic part — and surprisingly, it's not that complicated.

The recipe is called **backpropagation + gradient descent**, but in plain English:

1. **Forward pass.** Feed an example through the network. Get an output.
2. **Compute loss.** Compare the output to the correct answer using a **loss function** (a single number measuring "how wrong"). For binary classification we use `binary_crossentropy`.
3. **Backward pass.** Use calculus to figure out, for every weight in the network, "if I nudged this weight a little bigger, would the loss get better or worse, and by how much?" That's the gradient.
4. **Update.** Nudge every weight a tiny amount in the direction that reduces loss.
5. **Repeat** for the next example. And the next. And the next.

```
                    ┌───────────┐
                    │  Network  │
   Inputs ─────────▶│           │─────▶ Output
                    │           │            │
                    └───────────┘            ▼
                         ▲              Compare to truth
                         │                   │
                Adjust weights ◀─── Gradients (calculus magic)
                                            from loss
```

The `optimizer` is the algorithm that controls the "nudge" step. The most popular is **Adam**. You'll see `optimizer='adam'` in the code. Just know it's a good default.

---

## Epochs, batches, and steps — the loop vocabulary

| Term | Meaning |
|------|---------|
| **Batch** | A small group of examples processed together (e.g. 32 rows). |
| **Step** | One forward + backward pass on a single batch. |
| **Epoch** | One full sweep through the training set. |

If you have 1000 training rows and `batch_size=32`, then one epoch is `ceil(1000/32) = 32` steps.

We train for many epochs (e.g. 50) so the network sees each example many times and gradually settles into good weights.

---

## Why we still split, scale, and impute

Everything you learned in Tutorial 02 still applies. Neural nets are *more* sensitive to scale than classical models, not less. They are also more prone to overfitting on small datasets, so they need carefully held-out test data.

---

## Why neural networks need help to not overfit

Neural networks are extremely flexible. Give them enough neurons and they will memorize the training set perfectly — and then completely fail on test data. To prevent this, we use **regularization** tricks. You will see three of them in the next tutorial:

### Dropout

```python
Dropout(0.3)
```

During each training step, randomly "turn off" 30% of the neurons in this layer. This forces the rest of the network to not rely too heavily on any single neuron. Result: less memorization, better generalization.

(At prediction time, dropout is automatically disabled.)

### Batch Normalization

```python
BatchNormalization()
```

Rescales the activations going through this layer so they have roughly mean 0 and std 1. This makes training faster and more stable. It also acts as a mild regularizer.

### Early stopping (not used in our scripts, but worth knowing)

Watch a held-out validation set's loss during training. The moment it starts going *up* (even while training loss keeps going down), stop training. That gap is the model starting to memorize.

---

## A vocabulary cheat sheet

You'll meet all of these in the next tutorial. Read once now; you'll look back.

| Term | One-sentence meaning |
|------|----------------------|
| Tensor | A multi-dimensional array (like a NumPy array). TensorFlow's basic data type. |
| Layer | A reusable block of neurons. |
| `Dense` | A "fully connected" layer where every input talks to every neuron. |
| `Sequential` | A model that stacks layers in a straight line. |
| Input shape | The shape of a single sample (e.g. `(58,)` for our 58 features). |
| Loss | Number measuring how wrong the model currently is. |
| Optimizer | Algorithm that updates weights (e.g. Adam). |
| Compile | The step where you wire together loss, optimizer, and metrics. |
| Fit | The training loop. |
| Evaluate | Compute metrics on test data. |

---

## ANN vs CNN vs RNN — three flavors you'll hear about

| Network type | What it's good at | Used in this project? |
|--------------|-------------------|------------------------|
| **ANN** (a.k.a. MLP, fully-connected) | General-purpose tabular data | ✅ Tutorial 06 |
| **CNN** (Convolutional) | Spatial / signal patterns — images, audio, sequences | ✅ Tutorial 07 (1D version) |
| **RNN / LSTM / Transformer** | Long sequences with order — text, time series | Out of scope |

---

## Check yourself

1. What does an activation function do?
2. What is one epoch?
3. Why do we use dropout?
4. Why does a single neuron's formula look so much like Logistic Regression's?

When this all sinks in, head to **[Tutorial 06 — Building an ANN](06_building_ann.md)** and put it into code.
