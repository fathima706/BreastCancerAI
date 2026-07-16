import pandas as pd

# Load the dataset
df = pd.read_csv("dataset/breast_cancer_data.csv")

print("=" * 50)
print("Dataset Information")
print("=" * 50)

print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nTarget Variable (Relapse Free Status):")
print(df["Relapse Free Status"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())