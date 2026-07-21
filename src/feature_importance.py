import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocess import load_data, preprocess_data

# =====================================================
# Create Output Folder
# =====================================================

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("Generating Feature Importance")
print("=" * 60)

# =====================================================
# Load Dataset
# =====================================================

df = load_data()

X, y, numerical_features, categorical_features = preprocess_data(df)

# =====================================================
# Load Model
# =====================================================

model = joblib.load("models/recurrence_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# =====================================================
# Feature Importance
# =====================================================

feature_names = preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# =====================================================
# Save CSV
# =====================================================

importance_df.to_csv(
    "models/feature_importance.csv",
    index=False
)

# =====================================================
# Plot
# =====================================================

plt.figure(figsize=(12, 8))

top20 = importance_df.head(20)

plt.barh(
    top20["Feature"],
    top20["Importance"]
)

plt.gca().invert_yaxis()

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 20 Feature Importance")

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png",
    dpi=300
)

plt.close()

print("\nTop 20 Features\n")
print(top20)

print("\nFeature importance saved successfully!")
print("=" * 60)