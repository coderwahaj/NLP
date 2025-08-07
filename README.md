# 🧠 NLP Applications: Resume Screening, Sentiment Analysis, News Classification & Fake News Detection

This repository contains a collection of Natural Language Processing (NLP) mini-projects built using **Flask** as a front-end interface and **scikit-learn** (with other tools) for backend ML logic. Each project is fully functional and features a clean, responsive UI for interaction.

---

## 📁 Projects Included

### 📝 Task 1: Sentiment Analysis on Product Reviews
- Dataset: IMDb / Amazon Product Reviews (Kaggle)
- Binary classification of product reviews as **positive** or **negative**
- Preprocessing: Lowercasing, stopword removal
- Vectorization: TF-IDF
- Classifier: Logistic Regression & Naive Bayes (compared)
- **Bonus**: Word cloud of most common positive & negative words

### 📰 Task 2: News Category Classification
- Dataset: AG News (Kaggle)
- Multiclass classification: business, sports, politics, technology
- Preprocessing: Lemmatization, stopword removal, tokenization
- Vectorization: TF-IDF
- Classifiers: Logistic Regression, Random Forest, SVM
- **Bonus**: Word cloud per category
- Optional: Feedforward Neural Network (Keras)

### 🕵️‍♂️ Task 3: Fake News Detection
- Dataset: Fake and Real News Dataset (Kaggle)
- Binary classification: Real vs. Fake news articles
- Preprocessing: Tokenization, lemmatization, stopwords
- Model: Logistic Regression & SVM
- Evaluation: Accuracy, F1-Score
- **Bonus**: Word clouds for Fake vs. Real news

### 📄 Task 4: Resume Screening using NLP
- Dataset: Resume Dataset + Job Dataset (Kaggle)
- Embedding-based resume matcher using **cosine similarity**
- Used **Sentence Transformers** for semantic search
- Output includes:
  - Match score (e.g., 85%)
  - Matched job title
  - Highlighted skills
- **Bonus**: 
  - Entity extraction (skills, experience)
  - Beautiful upload panel and result UI

---

## 🛠️ Tools & Libraries Used

- **Python 3.x**
- **Flask** (for web front-end)
- **Scikit-learn** (ML modeling)
- **Pandas** & **NumPy** (data handling)
- **NLTK** / **spaCy** (text preprocessing)
- **Sentence Transformers** (for embeddings)
- **Keras** (basic neural network)
- **Matplotlib** / **WordCloud** (visualizations)

---

## 🧪 How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/nlp-mini-projects.git
   cd nlp-mini-projects
2. Create virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run a project:

   ```bash
   python app.py
   ```

Each task has a separate `app.py`—run the one you want to use.

---

## 🎯 Results & Performance

* All models were evaluated using accuracy and F1-score.
* Resume Screening used **semantic similarity** to deliver intuitive match scores.
* Visual feedback (word clouds, match highlighting) improves explainability.

---

## 🌐 UI Snapshots

> Include screenshots here (upload them in `images/` folder and link with markdown)

---

## 💡 Future Enhancements

* Deploy to Heroku or Render
* Add login system to save past results
* Expand resume parsing with docx/pdf parsing libraries

---

## 📬 Contact

Created by [Wahaj Asif](https://github.com/coderwahaj)
Feel free to reach out or contribute via issues or pull requests.




