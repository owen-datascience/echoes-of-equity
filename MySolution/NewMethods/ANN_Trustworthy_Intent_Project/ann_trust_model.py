import pandas as pd 
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder 
from sklearn.metrics import accuracy_score, roc_auc_score 
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Speech_dataset_characteristics.csv')
df = pd.read_csv(csv_path)


encoder = LabelEncoder()
df['Speaker_Intent'] = encoder.fit_transform(df['Speaker_Intent'])


X = df.drop(columns=['Audio_Filename', 'Speaker_ID', 'Speaker_Ethnicity', 
                    'Speaker_AgeGroup', 'Speaker_Sex', 'Speaker_Intent', 'Sentence_Num'])
y = df['Speaker_Intent']

X = X.fillna(X.mean())


_eth_codes = LabelEncoder().fit_transform(df['Speaker_Ethnicity'])
_strata = y.values * 10 + _eth_codes  # 6 cells: 2 intents x 3 ethnicities
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=_strata, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)), 
    BatchNormalization(), 
    Dropout(0.3), 
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') 
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])


print("Training the Deep Neural Network... Please wait.")
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)


loss, accuracy, auc = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"--- RESULTS ---")
print(f"Accuracy: {accuracy*100:.2f}% (Target: >70%)")
print(f"AUC Score: {auc:.2f} (Target: >0.78)")
