import os
import re
import logging
import argparse
import pandas as pd
import chardet
import sys
import apache_beam as beam
from google.cloud import storage, bigquery, pubsub_v1
from apache_beam.options.pipeline_options import PipelineOptions
from datetime import datetime
import tempfile
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk

# Configure logging
logging.basicConfig(level=logging.WARNING)

# Constants
PROJECT_ID = "finalproject-1234567"
GCP_CREDENTIALS_PATH = "C:\\Users\\wafaa\\gcp-creds\\finalproject-1234567-e5617b2836cb.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS_PATH

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

PROJECT_ID = "finalproject-1234567"
bucket_name = 'groupfinal-central'
file_name = "spam.csv"
REGION = "us-central1"
STAGING_LOCATION = f"gs://{bucket_name}/staging"
TEMP_LOCATION = f"gs://{bucket_name}/temp"
INPUT_FILE = f"gs://groupfinal-central/spam.csv"
OUTPUT_FILE = f"gs://groupfinal-central/output/result"
SERVICE_ACCOUNT_EMAIL = "dataflow-service-account-306@finalproject-1234567.iam.gserviceaccount.com"
TOPIC_ID = "dataflow-status"
TEMP_FILE_PATH = f"gs://{bucket_name}/temp/tempfile.csv"

# Pub/Sub functions
def publish_message(topic_id, message):
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, topic_id)
    future = publisher.publish(topic_path, message.encode("utf-8"))
    print(f"Published message: {message}")
    return future.result()

# Encoding detection
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding'] if result['encoding'] else 'utf-8'
    if encoding.lower() not in ['utf-8', 'ascii', 'latin-1']:
        encoding = 'utf-8'
    return encoding

# Data loading functions
def load_data_from_gcs(bucket_name, file_name):
    temp_file_path = None
    try:
        client = storage.Client()
        bucket = client.bucket("groupfinal-central")
        blob = bucket.blob(file_name)
        temp_file_path = os.path.join(tempfile.gettempdir(), "temp_data.csv")
        blob.download_to_filename(temp_file_path)
        encoding = detect_encoding(temp_file_path)
        df = pd.read_csv(temp_file_path, encoding=encoding, usecols=[0, 1], 
                        names=["label", "email"], skiprows=1)
        df.to_csv('spam.csv', encoding='utf-8', index=False)
        return df, temp_file_path
    except Exception as e:
        logging.error(f"Error loading data from GCS: {e}")
        return None, None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Pipeline components
class PreprocessData(beam.DoFn):
    def process(self, element):
        if not element or 'email' not in element or 'label' not in element:
            return
        email = element['email'].lower()
        email = re.sub(r'[^\w\s]', '', email)
        yield {'email': email, 'label': element['label']}

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize words
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    return " ".join(filtered_tokens)


# Function to parse CSV line (assuming CSV format: label,text)
def parse_csv_line(line):
    try:
        parts = line.split(",", 1)  # Split only on first comma
        label, text = parts[0].strip(), parts[1].strip()
        return {"label": label, "text": preprocess_text(text)}
    except Exception as e:
        return None  # Ignore malformed rows

#  Dataflow pipeline
def run_dataflow_pipeline(input_file, output_file, temp_file_path):

    job_name = f"dataflow-job-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    options = PipelineOptions(
        runner="DataflowRunner",
        project=PROJECT_ID,
        region=REGION,
        staging_location=STAGING_LOCATION,
        temp_location=TEMP_LOCATION,
        job_name=job_name,
        service_account_email=SERVICE_ACCOUNT_EMAIL,
    )

    with beam.Pipeline(options=options) as p:
        # Read input data from GCS
        lines = (p 
                 | "Read from GCS" >> beam.io.ReadFromText(INPUT_FILE, skip_header_lines=1)
                 | "Parse CSV Lines" >> beam.Map(parse_csv_line)
                 | "Filter Valid Rows" >> beam.Filter(lambda x: x is not None))

        # Process text
        processed_text = (lines 
                          | "Preprocess Text" >> beam.Map(lambda x: {
                                "label": x["label"], 
                                "text": preprocess_text(x["text"])
                            }))

        # Write output to GCS
        processed_text | "Write to GCS" >> beam.io.WriteToText(OUTPUT_FILE, file_name_suffix=".csv")

    print("Pipeline execution completed successfully.")

# Function to load data into BigQuery
def load_data_to_bigquery(df, dataset_id, table_id):
    try:
        client = bigquery.Client(project="finalproject-1234567")
        dataset_ref = client.dataset(dataset_id)
        table_ref = dataset_ref.table(table_id)
        
        schema = [
            bigquery.SchemaField("label", "STRING"),
            bigquery.SchemaField("email", "STRING"),
        ]

        job = client.load_table_from_dataframe(df, table_ref)
        job.result()

        logging.info(f"Data successfully loaded into BigQuery table {dataset_id}.{table_id}")

    except Exception as e:
        logging.error(f"Error loading data into BigQuery: {e}")
        
def train_naive_bayes(df):
    try:
        df['email'] = df['email'].apply(preprocess_text)
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(df['email'])
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='spam')
        recall = recall_score(y_test, y_pred, pos_label='spam')
        f1 = f1_score(y_test, y_pred, pos_label='spam')
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Log metrics
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1 Score: {f1}")
        logging.info(f"Confusion Matrix:\n{conf_matrix}")

        # Create results string
        result_str = f"""
        Accuracy: {accuracy}
        Precision: {precision}
        Recall: {recall}
        F1 Score: {f1}
        Confusion Matrix: {conf_matrix}
        """.strip()

        # Convert to native bytes and print
        result_bytes = result_str.encode('utf-8')
        sys.stdout.buffer.write(result_bytes)
        sys.stdout.buffer.write(b"\n")

        logging.info("Naive Bayes training results printed in native bytes.")

    except Exception as e:
        logging.error(f"Error training model: {e}")

# Main execution
if __name__ == "__main__":

    bucket_name = 'groupfinal-central'
    file_name = 'spam.csv'

    logging.info("Loading data from GCS...")
    df, temp_file_path = load_data_from_gcs(bucket_name, file_name)

    if df is not None:
        logging.info("Running Dataflow pipeline...")
        input_file = f"gs://{bucket_name}/{file_name}"
        output_file = f"gs://{bucket_name}/processed_data.txt"
        run_dataflow_pipeline(input_file, output_file, temp_file_path)

        logging.info("Training Naive Bayes classifier...")
        train_naive_bayes(df)

        logging.info("Loading data into BigQuery...")
        dataset_id = "final"
        table_id = "spam"
        load_data_to_bigquery(df, dataset_id, table_id)

        logging.info("Process completed!")
    else:
        logging.error("Failed to load data.")