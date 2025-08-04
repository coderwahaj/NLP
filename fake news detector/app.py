import os
import joblib
from flask import Flask, render_template, request
from preprocessing import preprocess # Import the function

# Initialize the Flask application
app = Flask(__name__)

# --- Load Model and Vectorizer ---
# We load them once when the app starts for efficiency.
MODEL_PATH = os.path.join(os.getcwd(), "model.pkl")
VECTORIZER_PATH = os.path.join(os.getcwd(), "vectorizer.pkl")

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or vectorizer: {e}")
    model = None
    vectorizer = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    input_text = ""

    if request.method == 'POST':
        # Check if the model and vectorizer were loaded
        if not model or not vectorizer:
            prediction_result = "Error: Model or vectorizer not loaded. Please check server logs."
            return render_template('index.html', prediction_result=prediction_result)

        input_text = request.form.get('news_content', '')

        if input_text.strip():
            # 1. Preprocess the input text
            processed_text = preprocess(input_text)
            
            # 2. Vectorize the processed text
            vectorized_text = vectorizer.transform([processed_text])
            
            # 3. Predict using the model
            prediction = model.predict(vectorized_text)[0] # Get the first element
            
            # 4. Set the result text
            if prediction == 1:
                prediction_result = "🟢 This appears to be REAL News."
            else:
                prediction_result = "🔴 This appears to be FAKE News."
        else:
            prediction_result = "Please enter some text to analyze."
            
    return render_template('index.html', prediction_result=prediction_result, input_text=input_text)


if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on your network
    app.run(host='0.0.0.0', port=5000, debug=True)