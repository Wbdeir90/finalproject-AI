import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from google.cloud import storage

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training.")
args = parser.parse_args()

# Load data from GCS
def load_data_from_gcs(bucket_name, file_name):
    """Load data from a file in Google Cloud Storage."""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    temp_file_path = "temp_data.csv"
    blob.download_to_filename(temp_file_path)
    return pd.read_csv(temp_file_path)

# Preprocess data
def preprocess_data(df):
    """Preprocess the data for training."""
    # Convert labels to binary (spam: 1, ham: 0)
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    return df

# Train model
def train_model(df):
    """Train a Naive Bayes model."""
    # Vectorize text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['email']).toarray()
    y = df['label']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

    return model, vectorizer

# Save model to GCS
def save_model_to_gcs(model, vectorizer, bucket_name, model_name):
    """Save the trained model and vectorizer to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # Save the model
    model_file = "model.pkl"
    joblib.dump(model, model_file)
    blob = bucket.blob(f"{model_name}/{model_file}")
    blob.upload_from_filename(model_file)

    # Save the vectorizer
    vectorizer_file = "vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_file)
    blob = bucket.blob(f"{model_name}/{vectorizer_file}")
    blob.upload_from_filename(vectorizer_file)

if __name__ == "__main__":
    # Load data
    bucket_name = "groupfinal"  # Replace with your GCS bucket name
    file_name = "processed_data.txt"  # Replace with your processed data file name
    df = load_data_from_gcs(bucket_name, file_name)

    # Preprocess data
    df = preprocess_data(df)

    # Train model
    model, vectorizer = train_model(df)

    # Save model to GCS
    model_name = "spam_classifier"  # Replace with your model name
    save_model_to_gcs(model, vectorizer, bucket_name, model_name)