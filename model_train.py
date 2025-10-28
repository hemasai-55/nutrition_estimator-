# model_train.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Step 1: Generate Dummy Data (Replace with real dataset later)
num_samples = 1000
X = np.random.rand(num_samples, 64, 64, 3)
y = np.random.rand(num_samples, 4) * [700, 50, 30, 100]  # [Calories, Protein, Fat, Carbs]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(4, activation='linear')  # [Calories, Protein, Fat, Carbs]
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Step 3: Train
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Step 4: Evaluate
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Model trained successfully! Test MAE: {mae:.2f}")

# Step 5: Save model
os.makedirs('saved_model', exist_ok=True)
model.save('saved_model/nutrition_model.h5')
print("💾 Model saved to saved_model/nutrition_model.h5")

