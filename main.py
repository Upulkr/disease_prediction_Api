from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS 
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# Load models dynamically
models = {
    "diabetes": {
        "path": "models/diabetic_final_model.sav",
        "features": ["pregnencies", "glucose", "bloodPressure", "skinThickness", "insulin", "bmi", "diabetesPedigreeFunction", "age"],
        "message": {1: "You might have diabetes. Please consult a doctor.", 0: "You are not likely to have diabetes. Stay healthy!"}
    },
    "breast_cancer": {
        "path": "models/breast_cancer_pediction_final_model.sav",
        "features": ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension'],
        "message": {0: "You might have breast cancer. Please consult a doctor.", 1: "No signs of breast cancer detected."}
    },
    "heart_disease": {
        "path": "models/heart_pediction_final_model.sav",
        "features": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
        "message": {1: "You might have heart disease. Please consult a doctor.", 0: "No signs of heart disease detected."}
    }
}

# Load all models into memory
for disease, info in models.items():
    model_path = info["path"]
    if os.path.exists(model_path):
        models[disease]["model"] = pickle.load(open(model_path, "rb"))
    else:
        models[disease]["model"] = None  # Handle missing models

@app.route('/')
def index():
    return jsonify({'message': 'Health Prediction API - Available diseases: diabetes, breast_cancer, heart_disease'})





@app.route('/predict/<disease>', methods=['POST'])
def predict(disease):
    try:
        if disease not in models:
            return jsonify({"error": f"Invalid disease type. Available options: {list(models.keys())}"}), 400

        model_info = models[disease]
        model = model_info["model"]
        
        if model is None:
            return jsonify({"error": f"Model for {disease} not found."}), 500

        data = request.json  
        print("Received data:", data)  

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        required_keys = model_info["features"]
        print("Required keys:", required_keys)

        # Check for missing keys
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing key: {key}"}), 400
        
         # Extract features dynamically
        features = [data[key] for key in required_keys]
        input_array = np.array([features]).astype(float)
        prediction = model.predict(input_array)[0]

        # Get the message for this disease
        message = model_info["message"].get(int(prediction), "No message available.")

        return jsonify({"prediction": str(prediction), "message": message}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Sending 500 status for errors


if __name__ == '__main__':
    app.run(debug=True, port=8000)
