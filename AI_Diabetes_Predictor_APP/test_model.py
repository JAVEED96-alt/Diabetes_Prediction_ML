import pickle
import numpy as np
import os

print("=" * 50)
print("TESTING MODEL AND SCALER")
print("=" * 50)

# Check if files exist
print("\n1. Checking if files exist...")
if os.path.exists('model.pkl'):
    print("   ✓ model.pkl found")
else:
    print("   ✗ model.pkl NOT FOUND")

if os.path.exists('scaler.pkl'):
    print("   ✓ scaler.pkl found")
else:
    print("   ✗ scaler.pkl NOT FOUND")

# Try loading model
print("\n2. Loading model...")
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("   ✓ Model loaded successfully")
    print(f"   Model type: {type(model)}")
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    model = None

# Try loading scaler
print("\n3. Loading scaler...")
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("   ✓ Scaler loaded successfully")
    print(f"   Scaler type: {type(scaler)}")
except Exception as e:
    print(f"   ✗ Error loading scaler: {e}")
    scaler = None

# Try a test prediction
print("\n4. Testing prediction...")
if model is not None and scaler is not None:
    try:
        # Test data: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigree, Age]
        test_data = np.array([[0, 120, 80, 25, 100, 25.5, 0.5, 35]])
        print(f"   Test input shape: {test_data.shape}")
        
        scaled_data = scaler.transform(test_data)
        print(f"   ✓ Scaler transform successful")
        print(f"   Scaled data shape: {scaled_data.shape}")
        
        prediction = model.predict_proba(scaled_data)
        print(f"   ✓ Model prediction successful")
        print(f"   Prediction: {prediction}")
        print(f"   Diabetes probability: {prediction[0][1]:.1%}")
        
    except Exception as e:
        print(f"   ✗ Error during prediction: {e}")
else:
    print("   ✗ Cannot test - model or scaler not loaded")

print("\n" + "=" * 50)
print("TEST COMPLETE")
print("=" * 50)