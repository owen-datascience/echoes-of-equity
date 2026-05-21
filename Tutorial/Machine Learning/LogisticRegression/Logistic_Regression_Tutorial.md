# Tutorial: Logistic Regression for High School Students

## 1. What Is Logistic Regression?

**Logistic Regression** is a machine learning algorithm used for **classification**.

It answers questions like:

| Question                       | Answer Type        |
| ------------------------------ | ------------------ |
| Will a student pass the exam?  | Yes / No           |
| Is an email spam?              | Spam / Not spam    |
| Does a tumor look dangerous?   | Benign / Malignant |
| Will a customer buy a product? | Buy / Not buy      |

Even though the name contains **“regression,”** logistic regression is mainly used for **classification**, especially **binary classification**, where there are only two possible outcomes. Google’s Machine Learning Crash Course explains logistic regression as a model designed to predict the **probability** of an outcome. ([Google for Developers][1])

---

## 2. Main Idea

Imagine we want to predict whether a student will pass an exam based on study hours.

| Study Hours | Result |
| ----------: | ------ |
|           1 | Fail   |
|           2 | Fail   |
|           3 | Fail   |
|           5 | Pass   |
|           7 | Pass   |
|           9 | Pass   |

A normal line from linear regression might predict values like:

```text
-0.2, 0.3, 0.8, 1.4
```

But for classification, we want a probability between:

```text
0 and 1
```

So logistic regression uses an **S-shaped curve** called the **sigmoid function**.

---

## 3. The Sigmoid Function

The sigmoid function converts any number into a value between 0 and 1.

```text
sigmoid(z) = 1 / (1 + e^(-z))
```

Example outputs:

|  z | sigmoid(z) | Meaning       |
| -: | ---------: | ------------- |
| -5 |     0.0067 | Very unlikely |
| -1 |      0.269 | Unlikely      |
|  0 |        0.5 | Unsure        |
|  1 |      0.731 | Likely        |
|  5 |      0.993 | Very likely   |

So if the model returns:

```text
0.82
```

That means:

```text
82% chance the answer is class 1
```

For example:

```text
82% chance the student passes
```

---

## 4. How Logistic Regression Makes a Decision

Logistic regression first calculates a score:

```text
z = w1x1 + w2x2 + b
```

Where:

| Symbol | Meaning                      |
| ------ | ---------------------------- |
| x1, x2 | Input features               |
| w1, w2 | Weights learned by the model |
| b      | Bias                         |
| z      | Raw score                    |

Then it applies sigmoid:

```text
probability = sigmoid(z)
```

Finally, it uses a threshold:

```text
If probability >= 0.5 → predict 1
If probability < 0.5 → predict 0
```

Example:

| Probability | Prediction |
| ----------: | ---------- |
|        0.91 | Pass       |
|        0.73 | Pass       |
|        0.49 | Fail       |
|        0.12 | Fail       |

---

## 5. Simple Real-Life Example

Suppose we want to predict whether a student will pass based on:

```text
x = hours studied
```

The model might learn:

```text
z = 1.2 × hours - 4
```

Now test different study hours.

### Student A: 2 hours

```text
z = 1.2 × 2 - 4 = -1.6
sigmoid(-1.6) ≈ 0.17
```

Prediction:

```text
17% chance of passing → Fail
```

### Student B: 6 hours

```text
z = 1.2 × 6 - 4 = 3.2
sigmoid(3.2) ≈ 0.96
```

Prediction:

```text
96% chance of passing → Pass
```

---

# 6. Python Example Using scikit-learn

The official scikit-learn documentation provides a `LogisticRegression` class for training logistic regression models, and it supports regularization by default. ([Scikit-learn][2])

## Step 1: Import Libraries

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
```

## Step 2: Create Training Data

```python
# X = study hours
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
])

# y = exam result
# 0 = fail, 1 = pass
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
```

## Step 3: Train the Model

```python
model = LogisticRegression()
model.fit(X, y)
```

## Step 4: Make Predictions

```python
new_students = np.array([
    [2],
    [4],
    [5],
    [7],
    [9]
])

predictions = model.predict(new_students)
probabilities = model.predict_proba(new_students)

print("Predictions:", predictions)
print("Probabilities:")
print(probabilities)
```

Example output may look like:

```text
Predictions: [0 0 1 1 1]

Probabilities:
[[0.95 0.05]
 [0.66 0.34]
 [0.45 0.55]
 [0.14 0.86]
 [0.04 0.96]]
```

Each probability row means:

```text
[probability of class 0, probability of class 1]
```

So:

```text
[0.14, 0.86]
```

means:

```text
14% chance fail, 86% chance pass
```

---

# 7. Full Beginner-Friendly Code

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# Training data
# Each row contains one feature: number of study hours
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [9],
    [10]
])

# Labels
# 0 means fail, 1 means pass
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Create the model
model = LogisticRegression()

# Train the model
model.fit(X, y)

# New students
new_students = np.array([
    [2],
    [4],
    [5],
    [7],
    [9]
])

# Predict class labels
predictions = model.predict(new_students)

# Predict probabilities
probabilities = model.predict_proba(new_students)

# Display results
for i in range(len(new_students)):
    hours = new_students[i][0]
    prediction = predictions[i]
    pass_probability = probabilities[i][1]

    result = "Pass" if prediction == 1 else "Fail"

    print(f"Study hours: {hours}")
    print(f"Predicted result: {result}")
    print(f"Probability of passing: {pass_probability:.2f}")
    print()
```

---

# 8. Example Output

```text
Study hours: 2
Predicted result: Fail
Probability of passing: 0.05

Study hours: 4
Predicted result: Fail
Probability of passing: 0.34

Study hours: 5
Predicted result: Pass
Probability of passing: 0.55

Study hours: 7
Predicted result: Pass
Probability of passing: 0.86

Study hours: 9
Predicted result: Pass
Probability of passing: 0.96
```

---

# 9. Important Vocabulary

| Term                  | Simple Explanation                                        |
| --------------------- | --------------------------------------------------------- |
| Classification        | Predicting a category                                     |
| Binary classification | Classification with two choices                           |
| Feature               | Input used by the model                                   |
| Label                 | Correct answer used during training                       |
| Probability           | Chance that something will happen                         |
| Sigmoid function      | Function that turns a number into a value between 0 and 1 |
| Threshold             | Cutoff used to make a final decision                      |
| Weight                | Number learned by the model showing feature importance    |
| Bias                  | Extra number that shifts the model                        |
| Training              | Process where the model learns from examples              |

---

# 10. Logistic Regression vs Linear Regression

| Linear Regression            | Logistic Regression         |
| ---------------------------- | --------------------------- |
| Predicts a number            | Predicts a probability      |
| Used for regression          | Used for classification     |
| Example: predict house price | Example: predict spam email |
| Output can be any number     | Output is between 0 and 1   |
| Uses a straight line         | Uses an S-shaped curve      |

---

# 11. How to Evaluate Logistic Regression

After training a model, we need to check if it works well.

Common evaluation metrics include:

| Metric           | Meaning                                         |
| ---------------- | ----------------------------------------------- |
| Accuracy         | How many predictions were correct               |
| Precision        | Of predicted positives, how many were correct   |
| Recall           | Of actual positives, how many were found        |
| F1 Score         | Balance between precision and recall            |
| Confusion Matrix | Table showing correct and incorrect predictions |

Google’s classification module introduces thresholding, confusion matrices, accuracy, precision, recall, and AUC as important tools for evaluating classification models. ([Google for Developers][3])

---

# 12. Example: Confusion Matrix

Suppose the model predicts whether emails are spam.

|                   | Predicted Spam | Predicted Not Spam |
| ----------------- | -------------: | -----------------: |
| Actually Spam     |             45 |                  5 |
| Actually Not Spam |              8 |                 42 |

Explanation:

| Result | Meaning                             |
| ------ | ----------------------------------- |
| 45     | Correctly detected spam             |
| 42     | Correctly detected not spam         |
| 5      | Missed spam                         |
| 8      | Wrongly marked normal email as spam |

---

# 13. Student Exercise

## Exercise 1: Concept Check

Answer these questions:

1. Is logistic regression used mainly for regression or classification?
2. Why do we need the sigmoid function?
3. What does a probability of `0.87` mean?
4. What happens if the probability is `0.42` and the threshold is `0.5`?
5. Give three real-world examples where logistic regression can be used.

## Sample Answers

1. Classification.
2. To convert any number into a probability between 0 and 1.
3. There is an 87% chance the example belongs to class 1.
4. The model predicts class 0.
5. Spam detection, disease prediction, student pass/fail prediction.

---

# 14. Coding Exercise

Modify the previous code so the model predicts whether a student will be admitted to a summer AI program.

Use two features:

```text
hours studied per week
project score
```

Example data:

```python
X = [
    [2, 60],
    [3, 65],
    [4, 70],
    [5, 75],
    [6, 80],
    [7, 85],
    [8, 90],
    [9, 95]
]

y = [0, 0, 0, 1, 1, 1, 1, 1]
```

Then predict for:

```python
[4, 78]
[6, 82]
[9, 88]
```

---

# 15. Mini Project Ideas

## Project 1: Student Success Predictor

Build a model that predicts whether a student will pass based on:

```text
study hours
homework completion
quiz average
attendance
```

## Project 2: Email Spam Classifier

Use logistic regression to classify emails as:

```text
spam or not spam
```

Features could include:

```text
number of links
number of capital letters
contains words like “free,” “winner,” “urgent”
```

## Project 3: Sports Win Predictor

Predict whether a team will win based on:

```text
past win rate
average points
home game or away game
number of injured players
```

---

# 16. Recommended Online Articles

## Best beginner-friendly resources

1. **Google Machine Learning Crash Course — Logistic Regression**
   Good for understanding logistic regression as probability prediction and classification. ([Google for Developers][1])

2. **Google Machine Learning Crash Course — Classification**
   Useful after logistic regression because students need to understand threshold, confusion matrix, precision, recall, and AUC. ([Google for Developers][3])

3. **scikit-learn LogisticRegression Documentation**
   Best official reference for using logistic regression in Python. ([Scikit-learn][2])

4. **DataCamp Logistic Regression with Python Tutorial**
   Practical Python tutorial using scikit-learn, updated in 2024. ([DataCamp][4])

5. **DigitalOcean Logistic Regression with Scikit-learn Tutorial**
   A newer practical tutorial that covers theory and Python implementation. ([DigitalOcean][5])

---

# 17. Recommended Videos

1. **StatQuest: Logistic Regression**
   One of the clearest visual explanations for beginners. ([YouTube][6])

2. **StatQuest Logistic Regression Playlist**
   Good for students who want to go deeper into odds, log-odds, coefficients, and maximum likelihood. ([YouTube][7])

3. **Google Machine Learning Crash Course YouTube Playlist**
   Includes short videos on logistic regression and classification. ([YouTube][8])

4. **Khan Academy Linear Regression Review**
   Useful prerequisite if students need to review what linear regression means before learning logistic regression. ([Khan Academy][9])

---

# 18. Simple Teaching Plan for a 90-Minute Class

## Part 1: Motivation — 10 minutes

Ask students:

```text
Can we predict whether a student will pass based on study hours?
Can we predict whether an email is spam?
Can we predict whether a patient has a disease?
```

Explain that these are classification problems.

## Part 2: Linear Regression Problem — 10 minutes

Show why linear regression is not ideal for classification because it can output values below 0 or above 1.

## Part 3: Sigmoid Function — 15 minutes

Draw the S-shaped curve.

Explain:

```text
large negative number → close to 0
0 → 0.5
large positive number → close to 1
```

## Part 4: Python Demo — 25 minutes

Use the study-hours pass/fail example.

## Part 5: Evaluation — 15 minutes

Introduce:

```text
accuracy
confusion matrix
precision
recall
```

## Part 6: Student Practice — 15 minutes

Students modify the dataset and test new predictions.

---

# 19. Key Takeaway

Logistic regression is one of the most important beginner machine learning algorithms. It is simple, fast, and powerful for problems where we need to predict **yes/no**, **true/false**, or **class 0/class 1** outcomes. The key idea is:

```text
Use a linear equation to calculate a score,
use the sigmoid function to turn the score into a probability,
then use a threshold to make a classification decision.
```

[1]: https://developers.google.com/machine-learning/crash-course/logistic-regression?utm_source=chatgpt.com "Logistic Regression | Machine Learning"
[2]: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?utm_source=chatgpt.com "LogisticRegression — scikit-learn 1.8.0 documentation"
[3]: https://developers.google.com/machine-learning/crash-course?utm_source=chatgpt.com "Google's Machine Learning Crash Course"
[4]: https://www.datacamp.com/tutorial/understanding-logistic-regression-python?utm_source=chatgpt.com "Python Logistic Regression Tutorial with Sklearn & Scikit"
[5]: https://www.digitalocean.com/community/tutorials/logistic-regression-with-scikit-learn?utm_source=chatgpt.com "Mastering Logistic Regression with Scikit-Learn"
[6]: https://www.youtube.com/watch?v=yIYKR4sgzI8&utm_source=chatgpt.com "StatQuest: Logistic Regression"
[7]: https://www.youtube.com/playlist?list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe&utm_source=chatgpt.com "Logistic Regression"
[8]: https://www.youtube.com/playlist?list=PLOU2XLYxmsILTKLltkh859KJ9BizDvd_S&utm_source=chatgpt.com "Machine Learning Crash Course"
[9]: https://www.khanacademy.org/math/statistics-probability/describing-relationships-quantitative-data/introduction-to-trend-lines/a/linear-regression-review?utm_source=chatgpt.com "Linear regression review (article)"
