import boto3
import pandas as pd
from src.utils.logger import logging
from io import StringIO
import os


class S3_connection:
    def __init__(self, bucket, access_key, secret_key):
        self.bucket = bucket  # only bucket name
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )

    def fetch_data(self, folder_path, file_name):
        """
        Fetch CSV from S3 and return DataFrame.
        """
        key = f"{folder_path}/{file_name}" if folder_path else file_name

        logging.info(f"Fetching file from S3: s3://{self.bucket}/{key}")

        obj = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

        logging.info("File successfully fetched from S3")
        return df

    def download_file(self, folder_path, file_name, local_save_path):
        """
        Download S3 file and save it locally at given path.
        
        Example:
        local_save_path = "RAW_DATA/Dataset.csv"
        """
        key = f"{folder_path}/{file_name}" if folder_path else file_name

        logging.info(f"Downloading file from S3: s3://{self.bucket}/{key}")

        # Create local directory if not exists
        local_dir = os.path.dirname(local_save_path)
        if local_dir and not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)

        # Download and save
        self.s3_client.download_file(self.bucket, key, local_save_path)

        logging.info(f"File saved locally at: {local_save_path}")
        return local_save_path
