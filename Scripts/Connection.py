import boto3
import pandas as pd
import logging
from src.utils.logger import logging
from io import StringIO


class S3_connection:
    def __init__(self, bucket, access_key, secret_key):
        self.bucket = bucket
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )

    def fetch_data(self,file_name):
        obj = self.s3_client.get_object(Bucket=self.bucket, Key=file_name)
        df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        return df