import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import re
import os
from tqdm import tqdm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# Add joblib to save the models
import joblib

# Ensure required NLTK packages are downloaded
nltk_packages = ['punkt', 'stopwords', 'wordnet',
                 'omw-1.4', 'averaged_perceptron_tagger']
for package in nltk_packages:
    try:
        nltk.data.find(
            f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
    except LookupError:
        nltk.download(package, quiet=True)

# --- 1. Load and Preprocess Data ---
print("📄 Loading and preprocessing the dataset...")

# Assume your dataset is named 'IMDB_Dataset.csv' and has 'review' and 'sentiment' columns
# The 'sentiment' column should contain 'positive' and 'negative' values
try:
    df = pd.read_csv('dataset/IMDB_Dataset.csv')
except FileNotFoundError:
    print("❌ Error: 'IMDB_Dataset.csv' not found. Please place your dataset in the 'dataset' folder.")
    exit()

# Convert sentiment to numerical labels (0 for negative, 1 for positive)
df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Preprocessing functions
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


print("⏳ Applying text preprocessing to all reviews...")
tqdm.pandas()
df['cleaned_review'] = df['review'].progress_apply(preprocess_text)

# --- 2. Feature Extraction ---
print("\n⚙️ Converting text to numerical features using TF-IDF...")
X = df['cleaned_review'].fillna('')  # Handle potential NaN values
y = df['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# --- 3. Train and Evaluate Models ---
print("\n🤖 Training and evaluating Logistic Regression model...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_test_tfidf)

print("\n✅ Logistic Regression Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
print(classification_report(y_test, lr_pred))

print("\n🤖 Training and evaluating Naive Bayes classifier...")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)

print("\n✅ Naive Bayes Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, nb_pred):.4f}")
print(classification_report(y_test, nb_pred))

# --- 4. Bonus Task: Visualize with Word Clouds ---
def generate_wordcloud(text, title):
    wordcloud = WordCloud(
        width=800, height=400, background_color='white', colormap='viridis').generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=18)
    plt.tight_layout(pad=0)
    plt.show()

print("\n🌩️ Generating WordClouds for positive and negative reviews...")
positive_text = " ".join(df[df['label'] == 1]['cleaned_review'].fillna(''))
negative_text = " ".join(df[df['label'] == 0]['cleaned_review'].fillna(''))

generate_wordcloud(positive_text, "Most Frequent Words in Positive Reviews")
generate_wordcloud(negative_text, "Most Frequent Words in Negative Reviews")


# --- 5. Save the Models and Vectorizer ---
model_dir = os.path.join(os.getcwd())

try:
    joblib.dump(lr_model, os.path.join(model_dir, 'lr_model.pkl'))
    joblib.dump(nb_model, os.path.join(model_dir, 'nb_model.pkl'))
    joblib.dump(tfidf_vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    print("\n💾 Models and vectorizer saved successfully:")
    print("   • lr_model.pkl")
    print("   • nb_model.pkl")
    print("   • tfidf_vectorizer.pkl")
except Exception as e:
    print("❌ Failed to save models/vectorizer:", e)