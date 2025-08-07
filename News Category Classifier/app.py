# File: app.py

import os
from flask import Flask, render_template, request
import joblib

# --- CONFIGURATION ---
MODEL_DIR = 'models'
MODEL_FILE = os.path.join(MODEL_DIR, 'news_classifier_pipeline.pkl')

app = Flask(__name__)

# --- LOAD PRE-TRAINED MODEL ---
try:
    classifier_pipeline = joblib.load(MODEL_FILE)
    print("✅ Pre-trained classification model loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Model file not found. Please run 'python train_model.py' first.")
    classifier_pipeline = None

# --- HELPER FUNCTION FOR PREDICTION ---
def classify_news(text):
    if classifier_pipeline:
        prediction = classifier_pipeline.predict([text])[0]
        return prediction
    return "Model not loaded."

# --- FLASK ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    input_text = ""
    if request.method == 'POST':
        input_text = request.form.get('news_text')
        if input_text:
            prediction = classify_news(input_text)
            
    return render_template('index.html', prediction=prediction, input_text=input_text)

# --- RUN THE APP ---
if __name__ == '__main__':
    if classifier_pipeline is None:
        print("*********************************************************")
        print("STOP: Application cannot start due to missing model file.")
        print("Please run 'python train_model.py' first to prepare data.")
        print("*********************************************************")
    else:
        app.run(debug=True)