from google.cloud import storage
import pandas as pd
from io import StringIO

# Define Google Cloud project and bucket details
project_name = "finalproject"
bucket_name = "groupfinal"
file_name = "spam.csv"

client = storage.Client(project=project_name)
bucket = client.bucket(bucket_name)
blob = bucket.blob(file_name)

# Download the CSV file as a string
content = blob.download_as_text()

df = pd.read_csv(StringIO(content), sep="\t", header=None, names=["label", "email"])

print(df.head(10))
