
# Project: 1D Convolutional Neural Network (1D-CNN) for Trustworthy Intent Classification
# This script uses a more advanced AI architecture often used for patterns in signals like sound.

# 1. IMPORT LIBRARIES
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score
import os

# 2. DATA LOADING AND PREPARATION
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Speech_dataset_characteristics.csv')
df = pd.read_csv(csv_path)

encoder = LabelEncoder()
df['Speaker_Intent'] = encoder.fit_transform(df['Speaker_Intent'])

# 3. FEATURE SELECTION (Using acoustic features only)
X = df.drop(columns=['Audio_Filename', 'Speaker_ID', 'Speaker_Ethnicity', 
                    'Speaker_AgeGroup', 'Speaker_Sex', 'Speaker_Intent', 'Sentence_Num'])
y = df['Speaker_Intent']
X = X.fillna(X.mean())

# 4. DATA SPLITTING
# Joint stratification on (intent, ethnicity) keeps the SAME ratio of every
# (Neutral/Trustworthy x White/Black/SouthAsian) cell in both halves, so the
# per-ethnicity accuracy we report later is not skewed by a lucky split.
_eth_codes = LabelEncoder().fit_transform(df['Speaker_Ethnicity'])
_strata = y.values * 10 + _eth_codes  # 6 cells: 2 intents x 3 ethnicities
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=_strata, random_state=42)

# 5. DATA SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. RESHAPE FOR CNN: CNNs expect 3D data. We reshape our 2D data to add a third 'signal' dimension.
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# 7. BUILD THE CNN MODEL: This model 'scans' across features to find local patterns.
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)), # Scans features
    BatchNormalization(),
    Dropout(0.2),
    Conv1D(32, kernel_size=3, activation='relu'), # Another scan for complex patterns
    Flatten(), # Flattens the scans into a single list of numbers
    Dense(64, activation='relu'), # A regular hidden layer
    Dense(1, activation='sigmoid') # Final prediction
])

# 8. COMPILE THE MODEL
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# 9. TRAIN THE MODEL
print("Training the 1D Convolutional Neural Network...")
model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, verbose=0)

# 10. RESULTS
loss, accuracy, auc = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"--- RESULTS ---")
print(f"Accuracy: {accuracy*100:.2f}% (Target: >70%)")
print(f"AUC Score: {auc:.2f} (Target: >0.78)")
