import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    """
    Load the breast cancer dataset.
    """
    df = pd.read_csv("dataset/breast_cancer_data.csv")
    return df


def preprocess_data(df):
    """
    Select important features and target.
    """

    # Remove rows where target is missing
    df = df.dropna(subset=["Relapse Free Status"])

    selected_features = [
        "Age at Diagnosis",
        "Tumor Size",
        "Tumor Stage",
        "Lymph nodes examined positive",
        "Neoplasm Histologic Grade",
        "ER Status",
        "PR Status",
        "HER2 Status",
        "Chemotherapy",
        "Hormone Therapy"
    ]

    X = df[selected_features]
    y = df["Relapse Free Status"]

    return X, y


def split_data(X, y):
    """
    Split the dataset into training and testing sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test