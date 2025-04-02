import os
import logging
import joblib
import pandas as pd
from google.cloud import storage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Google Cloud Configuration
GCP_CREDENTIALS_PATH = "C:\\Users\\wafaa\\gcp-creds\\finalproject-1234567-e5617b2836cb.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS_PATH
PROJECT_ID = "finalproject-1234567"
BUCKET_NAME = "groupfinal-central"
FILE_NAME = "spam.csv"
MODEL_NAME = "spam_classifier"

# Get Vertex AI model directory
MODEL_DIR = os.getenv("AIP_MODEL_DIR", "output_model")
os.makedirs(MODEL_DIR, exist_ok=True)

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Download file from GCS."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        logging.info(f"Successfully downloaded {source_blob_name} to {destination_file_name}.")
        return True
    except Exception as e:
        logging.error(f"Error downloading file from GCS: {e}")
        return False

def load_data(file_path):
    """Load the dataset."""
    try:
        data = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        logging.info(f"Data successfully loaded with shape {data.shape}.")
        return data
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return None

def preprocess_data(df):
    """Preprocess the dataset for training."""
    df.columns = ['label', 'email']
    if 'label' not in df or 'email' not in df:
        logging.error("Missing required columns ('label' or 'email') in dataset.")
        return None
    df = df.dropna(subset=['label', 'email'])
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    return df

def train_naive_bayes(df):
    """Train a Naive Bayes classifier."""
    df['email'] = df['email'].str.lower().str.strip()
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['email'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logging.info(f"Precision: {precision_score(y_test, y_pred):.4f}")
    logging.info(f"Recall: {recall_score(y_test, y_pred):.4f}")
    logging.info(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    return model, vectorizer

def save_model_to_gcs(model, vectorizer, bucket_name, model_name):
    """Save trained model and vectorizer to Google Cloud Storage."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        if not bucket.exists():
            logging.error(f"Bucket '{bucket_name}' does not exist.")
            return
        
        # Save and upload model
        model_file = os.path.join(MODEL_DIR, "model.pkl")
        joblib.dump(model, model_file)
        bucket.blob(f"{model_name}/model.pkl").upload_from_filename(model_file)
        logging.info(f"Model uploaded to gs://{bucket_name}/{model_name}/model.pkl")
        
        # Save and upload vectorizer
        vectorizer_file = os.path.join(MODEL_DIR, "vectorizer.pkl")
        joblib.dump(vectorizer, vectorizer_file)
        bucket.blob(f"{model_name}/vectorizer.pkl").upload_from_filename(vectorizer_file)
        logging.info(f"Vectorizer uploaded to gs://{bucket_name}/{model_name}/vectorizer.pkl")
    except Exception as e:
        logging.error(f"Failed to save model to GCS: {e}")

def main():
    """Main function to execute the training pipeline."""
    if download_from_gcs(BUCKET_NAME, FILE_NAME, FILE_NAME):
        data = load_data(FILE_NAME)
        if data is not None:
            df = preprocess_data(data)
            if df is not None:
                model, vectorizer = train_naive_bayes(df)
                if model and vectorizer:
                    save_model_to_gcs(model, vectorizer, BUCKET_NAME, MODEL_NAME)

if __name__ == "__main__":
    main()