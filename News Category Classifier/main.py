# File: train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import joblib
import os
import ssl

# Fix for NLTK data download issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- CONFIGURATION ---
MODEL_DIR = 'models'
MODEL_FILE = os.path.join(MODEL_DIR, 'news_classifier_pipeline.pkl')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)

# --- TEXT PREPROCESSING FUNCTIONS ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """
    Cleans and preprocesses text for classification.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# --- MAIN TRAINING FUNCTION ---


def train_and_save_model():
    print("⏳ Starting model training and saving...")

    # 1. Download and load the AG News dataset
    try:
        from datasets import load_dataset
        dataset = load_dataset("ag_news")

        # --- CORRECTED LINE: Using .to_pandas() for stable conversion ---
        train_df = dataset['train'].to_pandas()
        test_df = dataset['test'].to_pandas()

        print("✅ AG News dataset loaded successfully.")
    except ImportError:
        print(
            "❌ 'datasets' library not found. Please install it with 'pip install datasets'.")
        return
    except Exception as e:
        print(f"❌ An error occurred during dataset loading: {e}")
        return

    # The dataset has 'text' and 'label' columns. Labels are 0, 1, 2, 3
    # Mapping the labels to human-readable categories
    label_map = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
    train_df['label'] = train_df['label'].map(label_map)
    test_df['label'] = test_df['label'].map(label_map)

    # 2. Preprocess the text data
    print("✨ Preprocessing text data...")
    train_df['text'] = train_df['text'].apply(preprocess_text)
    test_df['text'] = test_df['text'].apply(preprocess_text)
    print("✅ Text preprocessing complete.")

    # 3. Define the training pipeline
    # The pipeline will chain a TF-IDF vectorizer and a Logistic Regression model
    print("🧠 Defining model pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(random_state=42, C=1.0, solver='liblinear'))
    ])

    # 4. Train the model
    print("⚙️ Training the model (this may take a few minutes)...")
    pipeline.fit(train_df['text'], train_df['label'])
    print("✅ Model training complete.")

    # 5. Evaluate the model on the test set
    print("\n📈 Evaluating model performance...")
    predictions = pipeline.predict(test_df['text'])
    print(classification_report(
        test_df['label'], predictions, zero_division=0))

    # 6. Save the entire pipeline for later use in the Flask app
    print(f"💾 Saving the trained model pipeline to '{MODEL_FILE}'...")
    joblib.dump(pipeline, MODEL_FILE)
    print("✅ Model saved successfully. You can now run 'python app.py'.")


if __name__ == '__main__':
    train_and_save_model()
