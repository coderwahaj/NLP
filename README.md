# 🧠 Natural Language Processing Projects using Flask

This repository contains multiple complete NLP applications, all built using **Flask** with clean and responsive **custom UIs**.

## 🚀 Projects Included

### ✅ 1. Sentiment Analysis on Product Reviews
- **Dataset**: IMDb or Amazon Product Reviews
- **Model**: Logistic Regression, Naive Bayes
- **Bonus**: Visualization of positive/negative frequent words

### ✅ 2. News Category Classification
- **Dataset**: AG News Dataset
- **Model**: Logistic Regression, Random Forest, SVM, Neural Network
- **Bonus**: Word clouds, bar charts

### ✅ 3. Fake News Detection
- **Dataset**: Fake and Real News Dataset
- **Model**: Logistic Regression, SVM
- **Bonus**: Visualization with Word Cloud

### ✅ 4. Resume Screening using NLP
- **Dataset**: Resume + Job Descriptions Dataset
- **Model**: Semantic Search using Sentence Transformers
- **Bonus**: Matching percentage, named entity extraction, frontend resume upload


## 💡 Features

- Clean and **fully functional frontend UIs** for each app
- Complete **backend logic using Flask**
- Interactive model results with **visualizations**
- Upload resume and view job matches in real-time
- Modularized code for easy extension

---

## 🖥️ Folder Structure

```

nlp-projects/
│
├── task1\_sentiment\_analysis/
│   ├── app.py
│   ├── static/
│   ├── templates/
│   └── model.pkl
│
├── task2\_news\_classification/
│   ├── app.py
│   ├── static/
│   ├── templates/
│   └── model.pkl
│
├── task3\_fake\_news\_detection/
│   ├── app.py
│   ├── static/
│   ├── templates/
│   └── model.pkl
│
├── task4\_resume\_screening/
│   ├── app.py
│   ├── static/
│   ├── templates/
│   └── resume\_matcher.py
│
└── requirements.txt

```

## ⚙️ How to Run

1. Clone the repository  
   `git clone https://github.com/yourusername/nlp-projects.git`

2. Navigate to a task directory  
   `cd task1_sentiment_analysis`

3. Install dependencies  
   `pip install -r requirements.txt`

4. Run the Flask server  
   `python app.py`

5. Open in browser:  
   `http://localhost:5000`

---

## 📈 What I Learned

- NLP techniques like **tokenization, lemmatization, stopword removal**
- Feature extraction using **TF-IDF, CountVectorizer**
- **Multiclass and binary classification** (Logistic Regression, Naive Bayes, SVM, Random Forest, Neural Network)
- **Text similarity and semantic matching** using Sentence Transformers
- Real-time resume screening and **cosine similarity**
- Designing beautiful **Flask UIs** with HTML/CSS
- Visualizing word frequencies using **WordCloud and matplotlib**
- Serving ML models via **Flask APIs**
- Full-stack development (frontend + backend) for **NLP systems**

---

Each task has a separate `app.py`—run the one you want to use.

---

## 🎯 Results & Performance

* All models were evaluated using accuracy and F1-score.
* Resume Screening used **semantic similarity** to deliver intuitive match scores.
* Visual feedback (word clouds, match highlighting) improves explainability.

---

## 💡 Future Enhancements

* Deploy to Heroku or Render
* Add login system to save past results
* Expand resume parsing with docx/pdf parsing libraries

---

## 📬 Contact

Created by [Wahaj Asif](https://github.com/coderwahaj)
Feel free to reach out or contribute via issues or pull requests.




