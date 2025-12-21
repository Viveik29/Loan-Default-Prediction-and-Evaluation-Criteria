import os
import boto3
import pandas as pd
from botocore.config import Config
from dotenv import load_dotenv
from src.utils.logger import logging

# Load .env variables
load_dotenv()


class S3_connection:
    def __init__(self, bucket: str, region: str):
        """
        Initialize S3 connection using environment variables.
        """
        self.bucket = bucket

        config = Config(
            retries={"max_attempts": 10, "mode": "standard"},
            read_timeout=120
        )

        self.s3_client = boto3.client(
            "s3",
            region_name=region,
            config=config
        )

    def fetch_data(self, folder_path: str, file_name: str, local_save_path: str):
        """
        Download CSV from S3 and return DataFrame.
        """
        key = f"{folder_path}/{file_name}" if folder_path else file_name

        logging.info(f"Downloading file from S3: s3://{self.bucket}/{key}")

        # Create local directory if not exists
        os.makedirs(os.path.dirname(local_save_path), exist_ok=True)

        # Reliable S3 download
        self.s3_client.download_file(
            Bucket=self.bucket,
            Key=key,
            Filename=local_save_path
        )

        logging.info(f"File downloaded successfully at {local_save_path}")

        # Read locally (safe)
        df = pd.read_csv(local_save_path)

        logging.info("CSV loaded into DataFrame successfully")
        return df
