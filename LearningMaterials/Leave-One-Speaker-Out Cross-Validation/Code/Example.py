from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# ------------------------------------------------------
# STEP 1: Create fake sample data for 4 speakers
# ------------------------------------------------------

# Suppose we have 12 samples, 3 samples per speaker
# Features (X) could be anything — here just random numbers
X = np.array([
    [1, 2], [2, 1], [1, 1],     # Speaker 1
    [3, 3], [3, 4], [4, 3],     # Speaker 2
    [5, 5], [6, 5], [5, 6],     # Speaker 3
    [8, 8], [9, 8], [8, 9]      # Speaker 4
])

# Labels (y): pretend this is emotion (0 = sad, 1 = happy)
y = np.array([
    0, 0, 1,   # Speaker 1
    1, 1, 0,   # Speaker 2
    0, 1, 1,   # Speaker 3
    1, 0, 1    # Speaker 4
])

# Speaker groups: tells us which speaker each sample came from
speakers = np.array([
    1, 1, 1,   # samples 1-3
    2, 2, 2,   # samples 4-6
    3, 3, 3,   # samples 7-9
    4, 4, 4    # samples 10-12
])

# ------------------------------------------------------
# STEP 2: Set up Leave-One-Speaker-Out Cross Validation
# ------------------------------------------------------
logo = LeaveOneGroupOut()

accuracies = []

# ------------------------------------------------------
# STEP 3: Run LOSO
# ------------------------------------------------------
for train_idx, test_idx in logo.split(X, y, groups=speakers):

    # Get training data (all speakers except one)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train a simple classifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Test on the left-out speaker
    y_pred = clf.predict(X_test)

    # Measure accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    print(f"Left-out speaker: {speakers[test_idx][0]}")
    print(f"Test accuracy: {acc:.2f}\n")

# ------------------------------------------------------
# STEP 4: Final average accuracy
# ------------------------------------------------------
print("Average LOSO accuracy:", np.mean(accuracies))
