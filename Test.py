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
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO)

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set the Google Cloud service account key file path

# Set the Google Cloud service account key file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\wafaa\OneDrive\Desktop\Final Project\finalproject-1234567-e5617b2836cb.json"

# Debug: Print the value of GOOGLE_APPLICATION_CREDENTIALS
logging.info(f"GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")

# Set Google Cloud service account key file path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\wafaa\OneDrive\Desktop\FinalProject\gcp-creds\finalproject-1234567-e5617b2836cb.json"

# Pub/Sub Configuration
PROJECT_ID = "finalproject-1234567"

def publish_message(topic_id, message):
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, topic_id)
    
    future = publisher.publish(topic_path, message.encode("utf-8"))
    print(f"Published message: {message}")
    
    return future.result()

# Call the function with the correct arguments
publish_message("pubsubfinal", "Pub/Sub is working correctly")


def detect_encoding(file_path):
    """
    Detect the encoding of a file.
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    result = chardet.detect(raw_data)
    return result['encoding'] if result['encoding'] else 'utf-8'

# Function to download a file from GCS
def download_from_gcs(bucket_name, file_name, local_path):
    """
    Download a file from GCS to a local path.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.download_to_filename(local_path)
    logging.info(f"Downloaded {file_name} from GCS to {local_path}")

# Function to load CSV from Google Cloud Storage
def load_data_from_gcs(bucket_name, file_name):
    temp_file_path = None  # Initialize the variable before the try block
    try:
        logging.info("Creating storage client...")
        client = storage.Client()
        logging.info(f"Accessing bucket: {bucket_name}")
        bucket = client.bucket(bucket_name)
        logging.info(f"Accessing blob: {file_name}")
        blob = bucket.blob(file_name)

        # Temporary file storage
        temp_file_path = os.path.join(tempfile.gettempdir(), "temp_data.csv")
        logging.info(f"Downloading blob to: {temp_file_path}")
        blob.download_to_filename(temp_file_path)

        encoding = detect_encoding(temp_file_path)
        logging.info(f"Detected encoding: {encoding}")
        df = pd.read_csv(temp_file_path, encoding=encoding, usecols=[0, 1], names=["label", "email"], skiprows=1)

        # Save a copy locally to debug issues
        df.to_csv('spam.csv', encoding='utf-8', index=False)
        logging.info("Saved local copy: spam.csv")

        return df
    except Exception as e:
        logging.error(f"Error loading data from GCS: {e}")
        return None
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            logging.info(f"Removing temporary file: {temp_file_path}")
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

# Custom DoFn to decode text
class DecodeText(beam.DoFn):
    def __init__(self, encoding):
        self.encoding = encoding

    def process(self, element):
        try:
            # Decode the element using the specified encoding
            decoded_element = element.decode(self.encoding, errors='replace')
            yield decoded_element
        except Exception as e:
            logging.error(f"Error decoding element: {e}")

# Function to parse a CSV line
def parse_csv_line(line):
    import csv  # Add this inside the function
    try:
        reader = csv.reader([line])
        values = next(reader)
        if len(values) >= 2:
            logging.info(f"Parsed Line: {values}")
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
    # Generate a unique job name based on the current UTC timestamp
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

    # Clean up the temporary file after pipeline completion
    os.remove(temp_file_path)
    logging.info(f"Removed temporary file: {temp_file_path}")

    # Clean up the temporary file
    os.remove(temp_file_path)
    logging.info(f"Removed temporary file: {temp_file_path}")

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

        

def upload_to_gcs(local_file, bucket_name, gcs_path):
    """Uploads a file to Google Cloud Storage.

    Args:
        local_file (str): Path to the local file.
        bucket_name (str): Name of the GCS bucket.
        gcs_path (str): Destination path in GCS.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(local_file)
        print(f"File {local_file} uploaded to gs://{bucket_name}/{gcs_path}")
    except Exception as e:
        print(f"Failed to upload {local_file} to GCS: {e}")


# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize and remove stopwords
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
    df = load_data_from_gcs(bucket_name, file_name)

    if df is not None:
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