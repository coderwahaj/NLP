# File: app.py

import os
import re
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sentence_transformers import SentenceTransformer, util
from werkzeug.utils import secure_filename
import joblib
import warnings



from docx import Document
from pdfminer.high_level import extract_text as extract_pdf_text

# --- CONFIGURATION & SETUP ---
warnings.simplefilter(action='ignore', category=FutureWarning)
UPLOAD_FOLDER = 'uploads'
MODEL_DIR = 'models'
EMBEDDINGS_FILE = os.path.join(MODEL_DIR, 'jobs_embeddings.pkl')
JOB_DATA_FILE = os.path.join(MODEL_DIR, 'jobs_dataframe.pkl')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- HARDCODED SKILL LIST (Must match the list in setup_embeddings.py) ---
SKILLS_LIST = [
    'python', 'pandas', 'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'sql', 'java', 'javascript',
    'jquery', 'machine learning', 'regression', 'svm', 'naive bayes', 'knn', 'random forest',
    'decision trees', 'boosting', 'cluster analysis', 'word embedding', 'sentiment analysis',
    'natural language processing', 'nlp', 'dimensionality reduction', 'topic modelling', 'lda',
    'nmf', 'pca', 'neural nets', 'mysql', 'sqlserver', 'cassandra', 'hbase', 'elasticsearch',
    'd3.js', 'dc.js', 'plotly', 'kibana', 'ggplot', 'tableau', 'regular expression', 'html',
    'css', 'angular', 'logstash', 'kafka', 'flask', 'git', 'docker', 'computer vision', 'opencv',
    'deep learning', 'testing', 'windows xp', 'database testing', 'aws', 'django', 'selenium',
    'jira', 'c++', 'r', 'excel', 'power bi', 'gcp', 'azure',
    'mern', 'nextjs', 'react', 'nodejs', 'express', 'mongodb'
]

# --- LOAD PRE-COMPUTED DATA ---
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    jobs_df = joblib.load(JOB_DATA_FILE)
    job_embeddings = joblib.load(EMBEDDINGS_FILE)
    print("✅ Models and pre-computed data loaded successfully.")
except Exception as e:
    print(f"❌ Critical Error loading model files: {e}")
    jobs_df = None
    job_embeddings = None

# --- HELPER FUNCTIONS ---


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def extract_text_from_file(filepath):
    """Extracts text from a .pdf or .docx file using robust libraries."""
    file_extension = filepath.split('.')[-1].lower()
    text = ""
    try:
        if file_extension == "pdf":
            text = extract_pdf_text(filepath)
        elif file_extension == "docx":
            doc = Document(filepath)
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            return "Error: Unsupported file format."
        return text.strip() if text.strip() else "Error: Extracted text is empty."
    except Exception as e:
        print(f"Error during text extraction from {filepath}: {e}")
        return "Error: Could not read file."


def extract_skills_from_text(text):
    """Extracts skills from a given text using the predefined SKILLS_LIST."""
    if not isinstance(text, str):
        return []
    skill_pattern = r'\b(' + '|'.join(re.escape(skill)
                                      for skill in SKILLS_LIST) + r')\b'
    found_skills = re.findall(skill_pattern, text.lower())
    return set(found_skills)

# ▼▼▼ THIS ENTIRE FUNCTION HAS BEEN REPLACED TO HANDLE DUPLICATE CATEGORIES ▼▼▼


def rank_jobs(resume_text, top_n=5, semantic_weight=0.7, skill_weight=0.3):
    """
    Ranks jobs by finding the best-matching profile within each unique job category,
    then ranking these categories to provide diverse and relevant recommendations.
    """
    if jobs_df is None or job_embeddings is None or not isinstance(resume_text, str) or not resume_text:
        return []

    # Get skills from the user's resume
    user_skills = extract_skills_from_text(resume_text)

    # Calculate the base semantic scores for all job profiles
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(resume_embedding, job_embeddings)[0]

    # --- Stage 1: Find the best job profile within each category ---
    best_jobs_by_category = {}

    for i, job_row in jobs_df.iterrows():
        # Calculate the hybrid score for the current job profile
        semantic_score = cosine_scores[i].item()
        job_skills = set(job_row['skills'])
        matching_skills = user_skills.intersection(job_skills)

        skill_score = len(matching_skills) / \
            len(job_skills) if len(job_skills) > 0 else 0

        weighted_score = (semantic_score * semantic_weight) + \
            (skill_score * skill_weight)

        current_job_result = {
            'score': round(weighted_score * 100, 2),
            'title': job_row['job_title'],
            'matching_skills': sorted(list(matching_skills)),
            'description_snippet': job_row['job_description'][:200] + '...'
        }

        # If this category is new, or if this job has a higher score than the
        # previous best for this category, update it.
        category_title = current_job_result['title']
        if category_title not in best_jobs_by_category or \
           current_job_result['score'] > best_jobs_by_category[category_title]['score']:
            best_jobs_by_category[category_title] = current_job_result

    # --- Stage 2: Rank the best profiles from each category ---
    # Convert the dictionary of best jobs into a list
    final_recommendations = list(best_jobs_by_category.values())

    # Sort the list by score to get the final ranked recommendations
    final_recommendations.sort(key=lambda x: x['score'], reverse=True)

    return final_recommendations[:top_n]


# --- FLASK ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'resume_file' not in request.files or not request.files['resume_file'].filename:
            return redirect(request.url)

        file = request.files['resume_file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            resume_text = extract_text_from_file(filepath)

            MIN_TEXT_LENGTH = 50
            if "Error" in resume_text or len(resume_text) < MIN_TEXT_LENGTH:
                error_msg = f"Failed to extract sufficient text from {filename}. The file might be empty, corrupted, or a scanned/image-based PDF. Please use a text-based file."
                return render_template('index.html', error=error_msg)

            ranked_results = rank_jobs(resume_text)
            return render_template('index.html', results=ranked_results, resume_filename=filename)

    return render_template('index.html', results=None)


# --- RUN THE APP ---
if __name__ == '__main__':
    if jobs_df is None:
        print("*********************************************************")
        print("STOP: Application cannot start due to data loading errors.")
        print("Please run 'python create_embeddings.py' first to prepare data.")
        print("*********************************************************")
    else:
        app.run(debug=True)
