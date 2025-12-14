from sklearn.ensemble import RandomForestClassifier
import yaml
import pickle
import pandas as pd
import os


def params_load(file_path):
    with open(file_path, "r") as f:
        params = yaml.safe_load(f)
        return params
def model_training(df1,df2, n_estimators, random_state):

    model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    class_weight="balanced",   # handle imbalance
    n_jobs=-1
    )
    model.fit(df1, df2)
    #pickle.dump(model,'Models/model.pkl')
    # Ensure folder exists
    os.makedirs("Models", exist_ok=True)

    # Save model properly
    with open("Models/model.pkl", "wb") as f:
        pickle.dump(model, f)



def main():
    params_path = 'params.yaml'
    params = params_load(params_path)
    n_estimators = params["model_trainning"]["n_estimators"]
    random_state = params["model_trainning"]["random_state"]
    #file_path1 = "RAW_DATA/clearn_data/X_train.csv"
    file_path=os.path.join("RAW_DATA","Clearn_data")
    os.makedirs(file_path,exist_ok=True)
    file_path1 = os.path.join(file_path,"X_train.csv")
    #file_path2 = "RAW_DATA/clearn_data/y_train.csv"
    file_path2 = os.path.join(file_path,"y_train.csv")
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)
    y = df2.values.ravel()
    #print(df1)
    #print(y)
    model_training(df1,y,n_estimators,random_state)





if __name__ == '__main__':
    main()
