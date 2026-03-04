import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import os

print("Starting validation...")

os.makedirs("reports", exist_ok=True)

test = pd.read_csv("data/processed/test.csv")

X = test.drop("Survived", axis=1)
y = test["Survived"]

model = joblib.load("models/model.pkl")

preds = model.predict(X)

accuracy = accuracy_score(y, preds)

# Save metrics
with open("reports/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy}, f)

# Confusion matrix
cm = confusion_matrix(y, preds)
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.savefig("reports/confusion_matrix.png")

print("Validation finished.")