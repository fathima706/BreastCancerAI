import os
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from src.preprocess import load_data, preprocess_data, split_data

# Create outputs folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Load data
df = load_data()

X, y = preprocess_data(df)

X_train, X_test, y_train, y_test = split_data(X, y)

# Load trained model
model = joblib.load("saved_models/random_forest_pipeline.pkl")

# Predict
y_pred = model.predict(X_test)

# Create confusion matrix plot
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.title("Confusion Matrix")

plt.savefig("outputs/confusion_matrix.png", dpi=300)

plt.show()

print("Confusion matrix saved successfully!")