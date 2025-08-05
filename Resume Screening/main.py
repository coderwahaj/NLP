import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import joblib

# --- CONFIGURATION ---
JOBS_CSV_PATH = 'dataset/jobs.csv'
MODEL_NAME = 'all-MiniLM-L6-v2'
MODEL_DIR = 'models'
EMBEDDINGS_FILE = os.path.join(MODEL_DIR, 'jobs_embeddings.pkl')
JOB_DATA_FILE = os.path.join(MODEL_DIR, 'jobs_dataframe.pkl')

# Create the models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

def setup_embeddings():
    """Downloads model, processes job data, and creates embeddings for job descriptions."""
    print("⏳ Loading or downloading the Sentence Transformer model...")
    try:
        model = SentenceTransformer(MODEL_NAME)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Critical Error: Failed to load the Sentence Transformer model: {e}")
        print("Please check your internet connection and that 'sentence-transformers' is installed correctly.")
        return

    try:
        print(f"📄 Loading job data from {JOBS_CSV_PATH}...")
        # Use a more robust encoding like 'latin1' to handle weird characters
        jobs_df = pd.read_csv(JOBS_CSV_PATH, encoding='latin1')
        print("✅ Job data loaded.")
        
        # IMPROVEMENT: Robust column handling and data cleaning
        jobs_df.columns = jobs_df.columns.str.strip()
        jobs_df = jobs_df.rename(columns={
            'Category': 'job_title',  
            'Resume': 'job_description'
        })
        
        # Clean up garbled text and combine columns for better context
        jobs_df['job_description'] = jobs_df['job_description'].astype(str).str.replace(r'[^A-Za-z0-9,.\s]+', ' ', regex=True)
        jobs_df['job_text'] = jobs_df['job_title'].astype(str) + ". " + jobs_df['job_description'].astype(str)
        
        job_descriptions = jobs_df['job_text'].tolist()
        
        print("⚙️ Generating embeddings for all job descriptions...")
        job_embeddings = model.encode(job_descriptions, convert_to_tensor=True)
        
        print(f"💾 Saving embeddings to {EMBEDDINGS_FILE}...")
        joblib.dump(job_embeddings, EMBEDDINGS_FILE)
        
        # Save the processed dataframe as well to avoid re-loading CSV later
        print(f"💾 Saving processed job dataframe to {JOB_DATA_FILE}...")
        joblib.dump(jobs_df, JOB_DATA_FILE)
        
        print("✅ Setup complete. Embeddings and data saved.")
        print("\nSetup complete. You can now run 'python app.py' to start the web server.")
        
    except FileNotFoundError:
        print(f"❌ Error: {JOBS_CSV_PATH} not found. Please place your job data there.")
        
if __name__ == '__main__':
    setup_embeddings()