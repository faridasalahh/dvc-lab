# train.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Paths
DATA_DIR = "data/processed"
MODEL_DIR = "models"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "model.pkl")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)


# Load training data
train_data = pd.read_csv("data/processed/train.csv")
X_train = train_data.drop("Survived", axis=1)
y_train = train_data["Survived"]

# Train model
print("Training model...")
model = LogisticRegression(max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to models/model.pkl")