import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression

print("Starting training...")

os.makedirs("models", exist_ok=True)

train = pd.read_csv("data/processed/train.csv")

X = train.drop("Survived", axis=1)
y = train["Survived"]

model = LogisticRegression(max_iter=200)
model.fit(X, y)

joblib.dump(model, "models/model.pkl")

print("Training finished.")