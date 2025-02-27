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
import csv
from nltk.corpus import stopwords
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

# Function to detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding'] if result['encoding'] else 'utf-8'

# Function to load CSV from Google Cloud Storage
def load_data_from_gcs(bucket_name, file_name):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)

        # Temporary file storage
        temp_file_path = os.path.join(tempfile.gettempdir(), "temp_data.csv")
        blob.download_to_filename(temp_file_path)

        encoding = detect_encoding(temp_file_path)
        df = pd.read_csv(temp_file_path, encoding=encoding, usecols=[0, 1], names=["label", "email"], skiprows=1)

        return df
    except Exception as e:
        logging.error(f"Error loading data from GCS: {e}")
        return None
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Function to parse a CSV line
def parse_csv_line(line):
    try:
        reader = csv.reader([line])
        values = next(reader)
        if len(values) >= 2:
            return {'label': values[0].strip(), 'email': values[1].strip()}
    except Exception as e:
        logging.error(f"Error parsing line: {e}")
    return None

# Apache Beam DoFn for preprocessing data
class PreprocessData(beam.DoFn):
    def process(self, element):
        if not element or 'email' not in element or 'label' not in element:
            return
        email = element['email'].lower()
        email = re.sub(r'[^\w\s]', '', email)
        yield {'email': email, 'label': element['label']}

# Function to run the Apache Beam Dataflow pipeline
def run_dataflow_pipeline(input_file, output_file):
    job_name = f"dataflow-job-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    
    options = PipelineOptions(
        runner="DataflowRunner",
        project="finalproject-1234567",
        region="us-central1",
        job_name=job_name,
        temp_location=f"gs://groupfinal/temp",
        staging_location=f"gs://groupfinal/staging"
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
        logging.info(f"Data loaded into BigQuery table {dataset_id}.{table_id}")
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
        df['email'] = df['email'].apply(preprocess_text)
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(df['email'])
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = MultinomialNB()
        model.fit(X_train, y_train)

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
    df = load_data_from_gcs(bucket_name, file_name)

    if df is not None:
        logging.info("Data loaded successfully!")
        logging.info(df.head())

        logging.info("Running Dataflow pipeline...")
        input_file = f"gs://{bucket_name}/{file_name}"
        output_file = f"gs://{bucket_name}/processed_data.txt"
        run_dataflow_pipeline(input_file, output_file)

        logging.info("Training Naive Bayes classifier...")
        train_naive_bayes(df)

        logging.info("Loading data into BigQuery...")
        dataset_id = "final"
        table_id = "spam"
        load_data_to_bigquery(df, dataset_id, table_id)

        logging.info("Process completed!")
    else:
        logging.error("Failed to load data.")
