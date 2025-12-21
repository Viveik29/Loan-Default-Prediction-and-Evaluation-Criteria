
import boto3
import yaml
import os
from Scripts.Connection import S3_connection
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
# Load parameters from params.yaml
def params_load(file_path):
    with open(file_path, "r") as f:
        params = yaml.safe_load(f)
        return params
    





def main():
    params_path= 'params.yaml'
    params = params_load(params_path)
    bucket_name = params["data_ingestion"]["bucket_name"]
    file_name = params["data_ingestion"]["file_name"]
    folder_path = params['data_ingestion']['folder_path']
    region = params['data_ingestion']['region_name']
    # aws_access_key = params["data_ingestion"]["aws_access_key_id"]
    # aws_secrets = params["data_ingestion"]["aws_secret_access_key"]
    test_size = params["data_ingestion"]["test_size"]
    local_save_path = "RAW_DATA/Downloaded_Data/Dataset.csv"
    # version_id = params["data_ingestion"]["version_id"]
    # output_path = params["data_ingestion"]["output_path"]

    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
    s3_obj = S3_connection(bucket_name,region)
    s3_obj.fetch_data(folder_path,file_name,local_save_path)
    #s3_obj.download_file(folder_path, file_name, local_save_path)
    #df= data_balancing(df)
    df=pd.read_csv(local_save_path)
    df_train, df_test = train_test_split(df,test_size= test_size,random_state=42)
    output_path = os.path.join("RAW_DATA", "Raw")
    os.makedirs(output_path, exist_ok=True)
    df_train_data = os.path.join('RAW_DATA','Raw','df_train.csv')
    df_test_data = os.path.join('RAW_DATA','Raw','df_test.csv')
    print(df_train)
    df_train.to_csv(df_train_data,index = False)
    df_test.to_csv(df_test_data,index = False)

if __name__ == "__main__":
    main()