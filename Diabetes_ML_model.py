import pandas as pd
import joblib
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load your training data (replace with your actual data path)
data = pd.read_csv('diabetes.csv')  # Make sure this CSV exists

# Get feature columns (adjust these based on your dataset)
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Create and fit the scaler
scaler = StandardScaler()
scaler.fit(data[features])

# Save the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("Scaler saved successfully as 'scaler.pkl'")


model_file = "model.pkl"
Pipeline_file = "pipeline.pkl"


def build_pipline(num_attribute):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scalar", StandardScaler())
    ])


    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribute),
    ])
    return full_pipeline

if not os.path.exists(model_file):
    dabatic = pd.read_csv("Diabetes.csv")

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)


    for train_index, test_index in split.split(dabatic, dabatic["Outcome"]):
        train =  dabatic.loc[train_index]
        test = dabatic.loc[test_index].to_csv("input.csv", index=False)
        # Split features and labels like the housing example
        dabatic_labels = dabatic["Outcome"].copy()
        dabatic_features = dabatic.drop("Outcome", axis=1)

        

    target = "Outcome"
    num_attribute = dabatic.drop(columns=[target]).columns.tolist()
    cat_attribute = []  # No categorical features

    Pipeline = build_pipline(num_attribute)



    dabatic_preperd = Pipeline.fit_transform(train)
    model = RandomForestClassifier(random_state=42)
    model.fit(dabatic_preperd, train[target])

    joblib.dump(model, model_file)
    joblib.dump(Pipeline, Pipeline_file)
    print("congratulations! Model and pipeline saved successfully.")
    
else:
    
    model = joblib.load(model_file)
    Pipeline = joblib.load(Pipeline_file)
    Pipeline = joblib.load(Pipeline_file)

    input_data = pd.read_csv("input.csv")
    transformed_data = Pipeline.transform(input_data)
    predictions = model.predict(transformed_data).round(0)
    input_data["Outcome"] = predictions

    input_data.to_csv("Classifier.csv", index=False)
    print("inference is complete, result saved to output.csv enjoy!")