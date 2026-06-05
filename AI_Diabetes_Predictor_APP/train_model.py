import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load your dataset (adjust the path if needed)
try:
    df = pd.read_csv('diabetes.csv')  # Change this to your dataset path
    print("✓ Dataset loaded successfully")
    print(f"Dataset shape: {df.shape}")
except:
    print("Error: diabetes.csv not found!")
    print("Please provide your dataset or adjust the path")
    exit()

# Prepare features and target
X = df.drop('Outcome', axis=1)  # Adjust 'Outcome' if your target column has a different name
y = df['Outcome']

print(f"\nFeatures: {list(X.columns)}")
print(f"Target distribution:\n{y.value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Test accuracy
accuracy = model.score(X_test_scaled, y_test)
print(f"✓ Model trained successfully!")
print(f"Test Accuracy: {accuracy:.2%}")

# Save model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ model.pkl saved")

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ scaler.pkl saved")

print("\nDone! You can now run your Flask app.")