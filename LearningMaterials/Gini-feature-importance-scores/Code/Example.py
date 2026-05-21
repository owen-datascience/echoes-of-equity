# ---------------------------------------------
# Gini Feature Importance Example
# ---------------------------------------------

# 1. Import libraries
from sklearn.datasets import load_breast_cancer   # sample dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 2. Load a sample dataset
#    This dataset predicts whether a tumor is cancerous (1) or not (0)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)  # features
y = data.target                                          # labels

# 3. Train a Decision Tree model
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X, y)

# 4. Get Gini Feature Importance from Decision Tree
tree_importance = tree.feature_importances_

# Put results into a nice table
tree_results = pd.DataFrame({
    'Feature': X.columns,
    'Gini Importance': tree_importance
}).sort_values(by='Gini Importance', ascending=False)

print("=== Decision Tree Gini Feature Importance ===")
print(tree_results)

# 5. Train a Random Forest model
forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(X, y)

# 6. Get Gini Feature Importance from Random Forest
forest_importance = forest.feature_importances_

forest_results = pd.DataFrame({
    'Feature': X.columns,
    'Gini Importance': forest_importance
}).sort_values(by='Gini Importance', ascending=False)

print("\n=== Random Forest Gini Feature Importance ===")
print(forest_results)
