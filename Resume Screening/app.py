import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sentence_transformers import SentenceTransformer, util
from werkzeug.utils import secure_filename
import docx2txt
import pdfplumber
import joblib
import warnings

# Ignore the FutureWarning from PyTorch
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
MODEL_NAME = 'all-MiniLM-L6-v2'
MODEL_DIR = 'models'
EMBEDDINGS_FILE = os.path.join(MODEL_DIR, 'jobs_embeddings.pkl')
JOB_DATA_FILE = os.path.join(MODEL_DIR, 'jobs_dataframe.pkl')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx'}

# Ensure the uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- NLP SETUP ---
try:
    model = SentenceTransformer(MODEL_NAME)
    jobs_df = joblib.load(JOB_DATA_FILE)
    job_embeddings = joblib.load(EMBEDDINGS_FILE)
    print("✅ Models and embeddings loaded successfully.")
except (FileNotFoundError, joblib.JomError) as e:
    print(f"❌ Error loading pre-computed data: {e}")
    print("Please run 'python main.py' first to set up the embeddings.")
    jobs_df = pd.DataFrame()
    job_embeddings = None

# --- HELPER FUNCTIONS ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_file(filepath, file_type):
    """Extracts text from different file types, with a fallback for PDFs."""
    if file_type == 'docx':
        try:
            return docx2txt.process(filepath)
        except Exception:
            return "Error reading DOCX file."
    elif file_type == 'pdf':
        try:
            # Method 1: Using pdfplumber
            text = ""
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            if text:
                return text
        except Exception as e:
            print(f"pdfplumber failed: {e}. Attempting fallback method.")
        
        # Method 2: Fallback using PyPDF2
        try:
            import PyPDF2
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text
        except Exception as e:
            print(f"PyPDF2 fallback also failed: {e}")
            return "Error reading PDF file."
            
    return ""

def rank_jobs_by_similarity(resume_text, top_n=5):
    """Calculates cosine similarity and ranks jobs."""
    if job_embeddings is None or jobs_df.empty:
        return []
    
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    
    cosine_scores = util.pytorch_cos_sim(resume_embedding, job_embeddings)
    
    top_results_indices = sorted(range(len(cosine_scores[0])), key=lambda k: cosine_scores[0][k], reverse=True)[:top_n]
    
    ranked_jobs = []
    for i in top_results_indices:
        job_info = {
            'score': round(cosine_scores[0][i].item() * 100, 2),
            'category': jobs_df.loc[i, 'job_title'],
            'description_snippet': jobs_df.loc[i, 'job_description'][:200] + '...'
        }
        ranked_jobs.append(job_info)
        
    return ranked_jobs

# --- FLASK ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'resume_file' not in request.files:
            return redirect(request.url)
        
        file = request.files['resume_file']
        
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            file_type = filename.rsplit('.', 1)[1].lower()
            resume_text = extract_text_from_file(filepath, file_type)
            
            # This print statement is for debugging and will show what was extracted
            print("--- Extracted Resume Text (first 500 characters) ---")
            print(resume_text[:500])
            print("---------------------------------------------------")
            
            if not resume_text or "Error" in resume_text:
                error_message = f"Failed to extract text from your file: {filename}. Please try another file or format."
                return render_template('index.html', error=error_message)

            ranked_results = rank_jobs_by_similarity(resume_text)
            
            return render_template('index.html', results=ranked_results, resume_filename=filename)
            
    return render_template('index.html', results=None)

# --- RUN THE APP ---
if __name__ == '__main__':
    app.run(debug=True)