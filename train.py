import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

print("Training model...")

# Load training data
train_data = pd.read_csv("data/processed/train.csv")

# Split features and target
X_train = train_data.drop("Survived", axis=1)
y_train = train_data["Survived"]

# Create model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Save model
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to models/model.pkl")