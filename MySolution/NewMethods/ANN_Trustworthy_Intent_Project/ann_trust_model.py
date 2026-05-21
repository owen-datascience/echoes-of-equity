
# Project: Deep Neural Network (ANN) for Trustworthy Intent Classification
# This script uses a Neural Network to identify if a voice is 'Neutral' or 'Trustworthy'.

# 1. IMPORT LIBRARIES: These are tools that help us handle data and build the AI.
import pandas as pd # Used to read and organize data (like Excel)
import numpy as np # Used for math calculations
import tensorflow as tf # The main library for building Deep Learning models
from tensorflow.keras.models import Sequential # A way to stack layers in our brain-like model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # Different types of 'neurons' and layers
from sklearn.model_selection import train_test_split # Tools to split data into training and testing sets
from sklearn.preprocessing import StandardScaler, LabelEncoder # Tools to clean and scale the numbers
from sklearn.metrics import accuracy_score, roc_auc_score # Tools to see how well our AI is doing
import os

# 2. LOAD THE DATA: We read the file containing voice characteristics.
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Speech_dataset_characteristics.csv')
df = pd.read_csv(csv_path)

# 3. PREPARE THE TARGET: The 'Speaker_Intent' column has words like 'Neutral'. 
# AI needs numbers, so we change 'Neutral' and 'Trustworthy' to 0 and 1.
encoder = LabelEncoder()
df['Speaker_Intent'] = encoder.fit_transform(df['Speaker_Intent'])

# 4. SELECT FEATURES: We choose the columns with acoustic measurements (like pitch).
# We remove names and IDs because they don't help the AI understand voice patterns.
X = df.drop(columns=['Audio_Filename', 'Speaker_ID', 'Speaker_Ethnicity', 
                    'Speaker_AgeGroup', 'Speaker_Sex', 'Speaker_Intent', 'Sentence_Num'])
y = df['Speaker_Intent']

# 5. CLEAN THE DATA: If any numbers are missing, we fill them with the average value.
X = X.fillna(X.mean())

# 6. SPLIT THE DATA: We use 80% to 'teach' the AI and 20% to 'test' if it actually learned.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 7. SCALE THE NUMBERS: AI works best when numbers are in the same range (like 0 to 1).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. BUILD THE ARTIFICIAL BRAIN (ANN): We create a model with multiple layers of 'neurons'.
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)), # First layer: 128 neurons
    BatchNormalization(), # Keeps the training stable
    Dropout(0.3), # Randomly turns off some neurons to prevent the AI from 'memorizing' (overfitting)
    Dense(64, activation='relu'), # Second layer: 64 neurons
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'), # Third layer: 32 neurons
    Dense(1, activation='sigmoid') # Final layer: outputs a probability between 0 and 1
])

# 9. COMPILE THE MODEL: We tell the AI how to learn (optimizer) and how to measure mistakes (loss).
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# 10. TRAIN THE AI: The AI looks at the training data 50 times (epochs) to find patterns.
print("Training the Deep Neural Network... Please wait.")
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

# 11. EVALUATE THE RESULTS: We check how accurate the AI is on the test data it has never seen.
loss, accuracy, auc = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"--- RESULTS ---")
print(f"Accuracy: {accuracy*100:.2f}% (Target: >70%)")
print(f"AUC Score: {auc:.2f} (Target: >0.78)")
