import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score
from sklearn.model_selection import train_test_split

def load_data(data_path: str) ->pd.DataFrame:
    try:
        data = pd.read_csv(data_path)
        return data
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        return pd.DataFrame()
    
def model_evaluation(model, x_test = pd.DataFrame, y_test= pd.DataFrame) -> dict:
    try:
        y_pred= model.predict(x_test)

        acc =accuracy_score(y_test, y_pred)
        pre =precision_score(y_test, y_pred)
        recall =recall_score(y_test, y_pred)
        f1score =f1_score(y_test, y_pred)

        metrics_dict={
             'acc': acc,
             'precision':pre,
             'recall':recall,
             'f1_score':f1score
             }
        return metrics_dict
    except Exception as e:
        print(f"The Error while model evaluation is occured: {e}")
        

def load_model(model):
    try:
        with open(model, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"The Error while model loading is occured: {e}")

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame,pd.Series]:
    try:
        X = data.drop(columns=['Potability'],axis=1)
        y = data['Potability']
        return X,y
    except Exception as e:
        raise Exception(f"Error Preparing data:{e}")

def save_metrics(metrics_dict: dict, metrics_path: str)->None:
    try:
         with open(metrics_path, "w") as f:
             json.dump(metrics_dict, f, indent=4)
    except Exception as e:
        print(f"The Error while saving metrics is occured: {e}")


def main():
    test_data_path = "./data/processed/test_processed_mean.csv"
    model_path = "models/model.pkl"
    metrics_path = "reports/metrics.json"
    test_data = load_data(test_data_path)
    x_test, y_test = prepare_data(test_data)
    model = load_model(model_path)
    metrics_dict = model_evaluation(model, x_test, y_test)
    save_metrics(metrics_dict, metrics_path)




if __name__ == "__main__":
    main()