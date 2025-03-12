import argparse
import os
import logging
import pandas as pd
import joblib
from google.cloud import storage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train a Naive Bayes model for spam classification.")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training (not used in Naive Bayes).")
args = parser.parse_args()
print(f"Training started with epochs: {args.epochs}")


# Load data from Google Cloud Storage (GCS)
def load_data_from_gcs(bucket_name, file_name):
    """Load data from a file in Google Cloud Storage."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        temp_file_path = "temp_data.csv"
        blob.download_to_filename(temp_file_path)
        logging.info(f"Data downloaded from GCS: {file_name}")

        df = pd.read_csv(temp_file_path)
        os.remove(temp_file_path)  # Clean up temporary file
        return df
    except Exception as e:
        logging.error(f"Failed to load data from GCS: {e}")
        raise

# Preprocess the dataset
def preprocess_data(df):
    """Preprocess the data for training."""
    if 'label' not in df or 'email' not in df:
        raise ValueError("Dataset must contain 'label' and 'email' columns.")
    
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})  # Convert labels to binary
    return df

# Train the Naive Bayes model
def train_model(df):
    """Train a Naive Bayes model."""
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['email']).toarray()
    y = df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    logging.info(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logging.info(f"Model Precision: {precision_score(y_test, y_pred):.4f}")
    logging.info(f"Model Recall: {recall_score(y_test, y_pred):.4f}")
    logging.info(f"Model F1-Score: {f1_score(y_test, y_pred):.4f}")

    return model, vectorizer

# Save model and vectorizer to GCS
def save_model_to_gcs(model, vectorizer, bucket_name, model_name):
    """Save trained model and vectorizer to Google Cloud Storage."""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Save model
        model_file = "model.pkl"
        joblib.dump(model, model_file)
        bucket.blob(f"{model_name}/{model_file}").upload_from_filename(model_file)
        os.remove(model_file)  # Clean up local file
        logging.info(f"Model saved to GCS: {model_name}/{model_file}")

        # Save vectorizer
        vectorizer_file = "vectorizer.pkl"
        joblib.dump(vectorizer, vectorizer_file)
        bucket.blob(f"{model_name}/{vectorizer_file}").upload_from_filename(vectorizer_file)
        os.remove(vectorizer_file)  # Clean up local file
        logging.info(f"Vectorizer saved to GCS: {model_name}/{vectorizer_file}")

    except Exception as e:
        logging.error(f"Failed to save model to GCS: {e}")
        raise

if __name__ == "__main__":
    # GCS configuration
    bucket_name = os.getenv("GCS_BUCKET_NAME", "groupfinal")  # Set bucket name via env variable or use default
    file_name = "processed_data.txt"
    model_name = "spam_classifier"

    # Load and preprocess data
    df = load_data_from_gcs(bucket_name, file_name)
    df = preprocess_data(df)

    # Train model
    model, vectorizer = train_model(df)

    # Save model to GCS
    save_model_to_gcs(model, vectorizer, bucket_name, model_name)
