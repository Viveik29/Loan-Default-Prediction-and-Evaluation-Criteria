
import boto3
import yaml
import os
from Scripts.Connection import S3_connection
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
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
    aws_access_key = params["data_ingestion"]["aws_access_key_id"]
    aws_secrets = params["data_ingestion"]["aws_secret_access_key"]
    test_size = params["data_ingestion"]["test_size"]
    local_save_path = "RAW_DATA/Downloaded_Data/Dataset.csv"
    # version_id = params["data_ingestion"]["version_id"]
    # output_path = params["data_ingestion"]["output_path"]


    s3_obj = S3_connection(bucket_name,aws_access_key,aws_secrets)
    df = s3_obj.fetch_data(folder_path,file_name)
    s3_obj.download_file(folder_path, file_name, local_save_path)
    #df= data_balancing(df)
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