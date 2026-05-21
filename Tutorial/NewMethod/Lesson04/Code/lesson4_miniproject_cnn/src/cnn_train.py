
"""
Lesson 4 Mini Project: Simple CNN Baseline
Loads tiny synthetic spectrogram arrays (64x64) and trains a small CNN.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models

base = Path(__file__).resolve().parents[1]
data_dir = base/"data"
labels = pd.read_csv(data_dir/"labels.csv")

X=[]
y=[]
for _,row in labels.iterrows():
    arr=np.load(data_dir/row["filename"])
    X.append(arr)
    y.append(1 if row["label"]=="trustworthy" else 0)

X=np.array(X)[...,None]
y=np.array(y)

model=models.Sequential([
    layers.Conv2D(16,(3,3),activation='relu',input_shape=(64,64,1)),
    layers.MaxPooling2D(),
    layers.Conv2D(32,(3,3),activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(32,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(X,y,epochs=3,batch_size=8,verbose=1)
print("Training complete.")
