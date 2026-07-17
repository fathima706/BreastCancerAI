import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

# =====================================================
# Create Required Folders
# =====================================================

os.makedirs("saved_models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# =====================================================
# Load Dataset
# =====================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = load_data()

X, y, numerical_features, categorical_features = preprocess_data(df)

X_train, X_test, y_train, y_test = split_data(X, y)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# =====================================================
# Numerical Pipeline
# =====================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

# =====================================================
# Categorical Pipeline
# =====================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

# =====================================================
# Combine Preprocessing
# =====================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numerical_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)

# =====================================================
# Random Forest Classifier
# =====================================================

classifier = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# =====================================================
# Full Machine Learning Pipeline
# =====================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)

# =====================================================
# Train Model
# =====================================================

print("\nTraining Random Forest...\n")

model.fit(X_train, y_train)

# =====================================================
# Predictions
# =====================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# =====================================================
# Evaluation
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc = roc_auc_score(
    y_test,
    y_prob
)

# =====================================================
# Print Results
# =====================================================

print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc:.4f}")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

# =====================================================
# Save Metrics
# =====================================================

metrics = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC"
    ],

    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        roc
    ]

})

metrics.to_csv(
    "outputs/model_metrics.csv",
    index=False
)

# =====================================================
# Feature Importance
# =====================================================

feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

importance = model.named_steps[
    "classifier"
].feature_importances_

importance_df = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

print("\nTop 10 Important Features\n")
print(importance_df.head(10))

# =====================================================
# Save Model
# =====================================================

joblib.dump(
    model,
    "saved_models/random_forest_pipeline.pkl"
)

print("\nPipeline Saved Successfully!")

print("Model Saved Successfully!")

print("Metrics Saved!")

print("Feature Importance Saved!")

print("=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)