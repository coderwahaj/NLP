import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import re
import nltk
import joblib
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Ensure required NLTK packages are downloaded
nltk_packages = ['punkt', 'stopwords', 'wordnet',
                 'omw-1.4', 'averaged_perceptron_tagger']
for package in nltk_packages:
    try:
        nltk.data.find(
            f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
    except LookupError:
        nltk.download(package, quiet=True)

# Load or preprocess dataset
cleaned_csv_path = os.path.join(os.getcwd(), "cleaned_news.csv")
if os.path.exists(cleaned_csv_path):
    print("✅ Found cleaned data. Loading from CSV...")
    df = pd.read_csv(cleaned_csv_path)
else:
    print("📄 Cleaned data not found. Loading raw CSVs...")
    try:
        fake_df = pd.read_csv("dataset/Fake.csv")
        true_df = pd.read_csv("dataset/True.csv")
    except Exception as e:
        print(f"❌ Failed to load raw CSVs: {e}")
        exit()

    # Add labels
    fake_df['label'] = 0
    true_df['label'] = 1

    # Combine, shuffle, and reset index
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Handle missing values
    df['text'] = df['text'].fillna('')
    df['title'] = df['title'].fillna('')

    def clean_source(text):
        dash_pos = text.find(' - ')
        if dash_pos != -1 and dash_pos < 100:
            return text[dash_pos + 3:]
        return text

    df['text'] = df['text'].apply(clean_source)
    df['content'] = df['title'] + " " + df['text']

    # Preprocessing
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

    def preprocess(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'\@\w+|\#', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(pos))
                  for word, pos in pos_tag(tokens)
                  if word not in stop_words and len(word) > 2]
        return " ".join(tokens)

    print("⏳ Preprocessing text...")
    tqdm.pandas()
    df['cleaned_content'] = df['content'].progress_apply(preprocess)

    # Save cleaned dataset
    try:
        df[['title', 'text', 'label', 'cleaned_content']].to_csv(
            cleaned_csv_path, index=False)
        print(f"🧼 Cleaned dataset saved to: {cleaned_csv_path}")
    except Exception as e:
        print(f"❌ Failed to save cleaned data: {e}")

# Proceed with vectorization and model training
print(df['label'].value_counts())

X = df['cleaned_content'].fillna('')  # ADD THIS LINE
y = df['label']

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("\n✅ Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
model_path = os.path.join(os.getcwd(), "model.pkl")
vectorizer_path = os.path.join(os.getcwd(), "vectorizer.pkl")

try:
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print("\n💾 Model and vectorizer saved successfully:")
    print("   •", model_path)
    print("   •", vectorizer_path)
except Exception as e:
    print("❌ Failed to save model/vectorizer:", e)


def generate_wordcloud(text, title):
    wordcloud = WordCloud(
        width=800, height=400, background_color='white', colormap='viridis').generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=18)
    plt.tight_layout(pad=0)
    plt.show()


# Join all cleaned content by label
fake_text = " ".join(df[df['label'] == 0]['cleaned_content'].fillna(''))
real_text = " ".join(df[df['label'] == 1]['cleaned_content'].fillna(''))

print("\n🌩️ Generating WordClouds...")
generate_wordcloud(fake_text, "Common Terms in Fake News")
generate_wordcloud(real_text, "Common Terms in Real News")
