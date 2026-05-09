from google.cloud import storage
import pandas as pd
from io import StringIO
import json



class GCSHandler:
    def __init__(self, bucket_name: str):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def download_csv(self, blob_name: str) -> pd.DataFrame:
        blob = self.bucket.blob(blob_name)
        data = blob.download_as_text()
        return pd.read_csv(StringIO(data))

    def upload_csv(self, dataframe: pd.DataFrame, blob_name: str):
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(
            dataframe.to_csv(index=False),
            content_type='text/csv'
        )
    
    def download_json(self, blob_name: str):

        blob = self.bucket.blob(blob_name)
        data = blob.download_as_text()
        return json.loads(data)


    def upload_json(self, data: dict, blob_name: str):

        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(
            json.dumps(data, indent=4),
            content_type='application/json'
        )
        print(
            f'Uploaded JSON to gs://{self.bucket.name}/{blob_name}'
        )