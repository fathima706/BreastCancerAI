import os
import pickle
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.preprocess import (
    load_data,
    preprocess_data,
    split_data
)

# ======================================================
# Create Models Folder
# ======================================================

os.makedirs("models", exist_ok=True)

# ======================================================
# Load Dataset
# ======================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = load_data()

print(f"Dataset Shape : {df.shape}")

# ======================================================
# Preprocess
# ======================================================

X, y, numerical_features, categorical_features = preprocess_data(df)

print(f"Features : {len(X.columns)}")
print(f"Samples  : {len(X)}")

# ======================================================
# Train Test Split
# ======================================================

X_train, X_test, y_train, y_test = split_data(X, y)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ======================================================
# Preprocessing Pipeline
# ======================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ======================================================
# Transform Data
# ======================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

# ======================================================
# Random Forest
# ======================================================

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=12,

    min_samples_split=5,

    min_samples_leaf=2,

    random_state=42,

    class_weight="balanced"

)

print("\nTraining Random Forest...\n")

model.fit(

    X_train_processed,

    y_train

)

print("Training Completed.")

# ======================================================
# Prediction
# ======================================================

y_pred = model.predict(X_test_processed)

y_prob = model.predict_proba(X_test_processed)[:, 1]

# ======================================================
# Metrics
# ======================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")

# ======================================================
# Save Model
# ======================================================

joblib.dump(

    model,

    "models/recurrence_model.pkl"

)

joblib.dump(

    preprocessor,

    "models/preprocessor.pkl"

)

# ======================================================
# Save Feature Names
# ======================================================

feature_names = preprocessor.get_feature_names_out()

with open(

    "models/feature_names.pkl",

    "wb"

) as f:

    pickle.dump(

        feature_names,

        f

    )

# ======================================================
# Save Metrics
# ======================================================

metrics = {

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1": f1,

    "roc_auc": roc_auc,

    "confusion_matrix": cm,

    "classification_report": report

}

joblib.dump(

    metrics,

    "models/metrics.pkl"

)

# ======================================================
# Save Feature Importance
# ======================================================

importance = pd.DataFrame({

    "Feature": feature_names,

    "Importance": model.feature_importances_

})

importance = importance.sort_values(

    by="Importance",

    ascending=False

)

importance.to_csv(

    "models/feature_importance.csv",

    index=False

)

print("\nTop Features\n")

print(importance.head(15))

print("\n")

print("=" * 60)

print("Everything Saved Successfully")

print("=" * 60)