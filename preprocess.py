import pandas as pd
from sklearn.model_selection import train_test_split
import os

print("Starting preprocessing...")

# Create folders if they don't exist
os.makedirs("data/processed", exist_ok=True)

# Load raw dataset
df = pd.read_csv("data/titanic.csv")

# Simple cleaning (drop missing values)
df = df.dropna()

# Split dataset
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Save processed files
train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("Preprocessing finished.")