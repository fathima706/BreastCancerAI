import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from src.preprocess import load_data, preprocess_data, split_data


# -------------------------
# Load dataset
# -------------------------
df = load_data()

X, y = preprocess_data(df)

X_train, X_test, y_train, y_test = split_data(X, y)


# -------------------------
# Identify column types
# -------------------------
numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns


# -------------------------
# Numeric pipeline
# -------------------------
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


# -------------------------
# Categorical pipeline
# -------------------------
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# -------------------------
# Combine preprocessing
# -------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# -------------------------
# Full ML Pipeline
# -------------------------
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]
)


# -------------------------
# Train model
# -------------------------
model.fit(X_train, y_train)


# -------------------------
# Save model
# -------------------------
joblib.dump(model, "saved_models/random_forest_pipeline.pkl")

print("Model trained successfully!")
print("Pipeline saved successfully!")