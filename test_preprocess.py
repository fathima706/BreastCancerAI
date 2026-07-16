from src.preprocess import load_data, preprocess_data, split_data

# Load dataset
df = load_data()

# Preprocess
X, y = preprocess_data(df)

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

# Split dataset
X_train, X_test, y_train, y_test = split_data(X, y)

print("\nTraining Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("\nTraining Labels:", y_train.shape)
print("Testing Labels:", y_test.shape)