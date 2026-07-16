import joblib
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocess import load_data, preprocess_data, split_data

# Load data
df = load_data()

X, y = preprocess_data(df)

X_train, X_test, y_train, y_test = split_data(X, y)

# Load model
pipeline = joblib.load("saved_models/random_forest_pipeline.pkl")

# Get trained classifier
rf = pipeline.named_steps["classifier"]

# Get transformed feature names
preprocessor = pipeline.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()

importance = rf.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df.head(20))

plt.figure(figsize=(10,8))

plt.barh(
    importance_df["Feature"][:20],
    importance_df["Importance"][:20]
)

plt.gca().invert_yaxis()

plt.title("Top 20 Important Features")

plt.tight_layout()

plt.savefig("outputs/feature_importance.png", dpi=300)

plt.show()