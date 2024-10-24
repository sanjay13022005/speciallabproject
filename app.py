from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os
import datetime
import sqlite3
from flask import session
import random
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Set a secret key for session management

def load_data():
    X = np.random.rand(100, 4096)  
    y = np.random.randint(2, size=100)  
    return X, y

def get_model():
    model_path = 'plant_health_model.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return train_model()

def train_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train_pca, y_train)
    
    joblib.dump((scaler, pca, model), 'plant_health_model.joblib')
    return scaler, pca, model

scaler, pca, model = get_model()

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))  
    features = resized.flatten() / 255.0
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_pca = pca.transform(features_scaled)
    return features_pca

# New function to get plant care tips
def get_plant_care_tips(disease):
    tips = {
        "Healthy": [
            "Water your plant regularly, but avoid overwatering.",
            "Ensure your plant gets adequate sunlight.",
            "Prune dead or yellowing leaves to promote growth."
        ],
        "Leaf Spot": [
            "Remove affected leaves and destroy them.",
            "Improve air circulation around the plant.",
            "Apply a fungicide if the problem persists."
        ],
        "Rust": [
            "Isolate affected plants to prevent spread.",
            "Apply a sulfur-based fungicide.",
            "Avoid overhead watering to reduce humidity."
        ],
        "Blight": [
            "Remove and destroy infected plant parts.",
            "Improve drainage in the soil.",
            "Apply a copper-based fungicide as a preventive measure."
        ],
        "Powdery Mildew": [
            "Increase air circulation around the plant.",
            "Apply neem oil or a potassium bicarbonate solution.",
            "Avoid watering the leaves directly."
        ]
    }
    return random.sample(tips.get(disease, tips["Healthy"]), 2)

# Add this function to generate synthetic test data and predict accuracy
def predict_accuracy():
    # Generate synthetic test data (replace this with your actual test data in production)
    X_test = np.random.rand(100, 4096)  # Assuming 4096 features as in the original code
    y_test = np.random.randint(2, size=100)
    
    # Preprocess the test data
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Make predictions
    y_pred = model.predict(X_test_pca)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    try:
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        features = extract_features(image)
        
        prediction = model.predict(features)
        prediction_prob = model.predict_proba(features)
        
        # Get top 3 possible diseases
        top_diseases = [
            "Healthy",
            "Leaf Spot",
            "Rust",
            "Blight",
            "Powdery Mildew"
        ]
        top_3_indices = prediction_prob[0].argsort()[-3:][::-1]
        top_3_diseases = [
            {"disease": top_diseases[i], "probability": float(prediction_prob[0][i])}
            for i in top_3_indices
        ]
        
        top_disease = top_3_diseases[0]['disease']
        care_tips = get_plant_care_tips(top_disease)
        
        result = {
            'prediction': int(prediction[0]),
            'confidence': float(np.max(prediction_prob)),
            'top_3_diseases': top_3_diseases,
            'care_tips': care_tips
        }
        
        # Store the classification result in the session
        if 'classification_history' not in session:
            session['classification_history'] = []
        session['classification_history'].insert(0, result)
        session['classification_history'] = session['classification_history'][:5]  # Keep only the last 5 classifications
        session.modified = True
        
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    conn = sqlite3.connect('feedback.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS feedback
                 (timestamp TEXT, actual_disease TEXT, comments TEXT)''')
    c.execute("INSERT INTO feedback VALUES (?, ?, ?)",
              (datetime.datetime.now().isoformat(), data['actualDisease'], data['comments']))
    conn.commit()
    conn.close()
    return jsonify({"message": "Feedback received, thank you!"})

@app.route('/history', methods=['GET'])
def get_history():
    if 'classification_history' in session and session['classification_history']:
        return jsonify(session['classification_history'])
    else:
        return jsonify({"message": "No classification history found"})

@app.route('/predict_accuracy', methods=['GET'])
def get_accuracy():
    accuracy = predict_accuracy()
    return jsonify({"accuracy": accuracy})

if __name__ == '__main__':
    app.run(debug=True)
