from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import os

# Ensure NLTK packages are downloaded
nltk_packages = ['punkt', 'stopwords', 'wordnet',
                 'omw-1.4', 'averaged_perceptron_tagger']
for package in nltk_packages:
    try:
        nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
    except LookupError:
        nltk.download(package, quiet=True)

app = Flask(__name__)

# --- Load the trained models and vectorizer ---
try:
    lr_model = joblib.load(os.path.join(os.getcwd(), 'lr_model.pkl'))
    nb_model = joblib.load(os.path.join(os.getcwd(), 'nb_model.pkl'))
    tfidf_vectorizer = joblib.load(os.path.join(os.getcwd(), 'tfidf_vectorizer.pkl'))
    print("✅ Models and vectorizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models/vectorizer: {e}")
    lr_model = None
    nb_model = None
    tfidf_vectorizer = None

# --- Preprocessing Function (must match the one in main.py) ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    tag_map = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }
    return tag_map.get(tag[0], wordnet.NOUN)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        for word, pos in pos_tag(tokens)
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_result = None
    input_text = None

    if request.method == 'POST':
        input_text = request.form['review_content']
        if input_text:
            # Preprocess the input text
            processed_text = preprocess_text(input_text)
            
            # Vectorize the text using the trained TF-IDF vectorizer
            text_vector = tfidf_vectorizer.transform([processed_text])
            
            # Make a prediction with the Logistic Regression model
            lr_prediction = lr_model.predict(text_vector)
            
            # You can also get a probability score
            # lr_proba = lr_model.predict_proba(text_vector)[0]
            
            if lr_prediction[0] == 1:
                prediction_result = "SENTIMENT: POSITIVE"
            else:
                prediction_result = "SENTIMENT: NEGATIVE"
    
    return render_template('index.html', prediction_result=prediction_result, input_text=input_text)

if __name__ == '__main__':
    app.run(debug=True)