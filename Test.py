import os
import re
import logging
import pandas as pd
import nltk
import apache_beam as beam
from google.cloud import storage, bigquery
from google.cloud import aiplatform
from apache_beam.options.pipeline_options import PipelineOptions
from datetime import datetime
import chardet
import tempfile
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from google.cloud import pubsub_v1
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configure logging
logging.basicConfig(level=logging.INFO)

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set the Google Cloud service account key file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\wafaa\OneDrive\Desktop\Final Project\finalproject-1234567-e5617b2836cb.json"
logging.info(f"GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")

# Pub/Sub Configuration
PROJECT_ID = "finalproject-1234567"

def publish_message(topic_id, message):
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, topic_id)
    future = publisher.publish(topic_path, message.encode("utf-8"))
    print(f"Published message: {message}")
    return future.result()

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding'] if result['encoding'] else 'utf-8'
    if encoding.lower() not in ['utf-8', 'ascii', 'latin-1']:
        encoding = 'utf-8'
    return encoding

def load_data_from_gcs(bucket_name, file_name):
    temp_file_path = None
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        temp_file_path = os.path.join(tempfile.gettempdir(), "temp_data.csv")
        blob.download_to_filename(temp_file_path)
        encoding = detect_encoding(temp_file_path)
        df = pd.read_csv(temp_file_path, encoding=encoding, usecols=[0, 1], names=["label", "email"], skiprows=1)
        df.to_csv('spam.csv', encoding='utf-8', index=False)
        return df, temp_file_path  # Return both the DataFrame and temp_file_path
    except Exception as e:
        logging.error(f"Error loading data from GCS: {e}")
        return None, None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def run_dataflow_pipeline(input_file, output_file, temp_file_path):
    job_name = f"dataflow-job-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    options = PipelineOptions(
        runner="DataflowRunner",
        project="finalproject-1234567",
        region="us-central1",
        job_name=job_name,
        temp_location=f"gs://{bucket_name}/temp",
        staging_location=f"gs://{bucket_name}/staging"
    )
    with beam.Pipeline(options=options) as p:
        (
            p
            | 'Read from GCS' >> beam.io.ReadFromText(input_file, skip_header_lines=1, coder=beam.coders.BytesCoder())
            | 'Decode Bytes' >> beam.Map(lambda x: x.decode('latin-1'))
            | 'Parse CSV' >> beam.Map(parse_csv_line)
            | 'Filter Invalid Rows' >> beam.Filter(lambda x: x is not None)
            | 'Preprocess Data' >> beam.ParDo(PreprocessData())
            | 'Write to GCS' >> beam.io.WriteToText(output_file)
        )
    if temp_file_path and os.path.exists(temp_file_path):
        os.remove(temp_file_path)

# Apache Beam DoFn for preprocessing data
class PreprocessData(beam.DoFn):
    def process(self, element):
        if not element or 'email' not in element or 'label' not in element:
            logging.warning(f"Invalid Element: {element}")
            return
        email = element['email'].lower()
        email = re.sub(r'[^\w\s]', '', email)
        logging.info(f"Preprocessed Email: {email}")
        yield {'email': email, 'label': element['label']}

# Function to parse a CSV line
def parse_csv_line(line):
    import csv
    try:
        reader = csv.reader([line])
        values = next(reader)
        if len(values) >= 2:
            logging.info(f"Parsed Line: {values}")
            return {'label': values[0].strip(), 'email': values[1].strip()}
    except Exception as e:
        logging.error(f"Error parsing line: {e}")
    return None

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

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

# Function to train Naive Bayes classifier
def train_naive_bayes(df):
    try:
        logging.info("Preprocessing text data...")
        df['email'] = df['email'].apply(preprocess_text)
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(df['email'])
        y = df['label']

        logging.info("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        logging.info("Training Naive Bayes classifier...")
        model = MultinomialNB()
        model.fit(X_train, y_train)

        logging.info("Evaluating model...")
        y_pred = model.predict(X_test)

        logging.info(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        logging.info(f"Precision: {precision_score(y_test, y_pred, pos_label='spam')}")
        logging.info(f"Recall: {recall_score(y_test, y_pred, pos_label='spam')}")
        logging.info(f"F1 Score: {f1_score(y_test, y_pred, pos_label='spam')}")
        logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    except Exception as e:
        logging.error(f"Error training model: {e}")

# Main execution
if __name__ == "__main__":
    bucket_name = 'groupfinal'
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