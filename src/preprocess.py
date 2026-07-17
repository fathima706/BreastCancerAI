import pandas as pd
from sklearn.model_selection import train_test_split

# =====================================================
# Dataset Loader
# =====================================================

DATASET_PATH = "dataset/breast_cancer_data.csv"


def load_data():
    """
    Load the breast cancer dataset.
    """

    df = pd.read_csv(DATASET_PATH)

    return df


# =====================================================
# Feature Selection
# =====================================================

def preprocess_data(df):
    """
    Select model features and target.
    """

    # Remove rows where target is missing
    df = df.dropna(subset=["Relapse Free Status"]).copy()

    numerical_features = [
        "Age at Diagnosis",
        "Tumor Size",
        "Tumor Stage",
        "Lymph nodes examined positive",
        "Neoplasm Histologic Grade"
    ]

    categorical_features = [
        "ER Status",
        "PR Status",
        "HER2 Status",
        "Chemotherapy",
        "Hormone Therapy"
    ]

    selected_features = numerical_features + categorical_features

    X = df[selected_features].copy()

    y = df["Relapse Free Status"].replace({
    "0:Not Recurred": 0,
    "1:Recurred": 1
    }).astype(int)

    # Fill missing numerical values
    for col in numerical_features:
        X[col] = X[col].fillna(X[col].median())

    # Fill missing categorical values
    for col in categorical_features:
        X[col] = X[col].fillna("Unknown")

    return (
        X,
        y,
        numerical_features,
        categorical_features
    )


# =====================================================
# Train Test Split
# =====================================================

def split_data(X, y):
    """
    Split data into train and test sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test