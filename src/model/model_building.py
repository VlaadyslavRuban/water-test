import pickle
import numpy as np
import os
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier



def load_params(params_file: str) -> int:
    try:
        with open(params_file, 'r') as file:
            params = yaml.safe_load(file)["model_building"]["n_estimators"]
        return params
    except FileNotFoundError:
        print(f"Error: The file {params_file} does not exist.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
    

def load_data(data_path: str) -> pd.DataFrame:
    try: 
        data = pd.read_csv(data_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()
#Pandas's array
#x_train=train_data.iloc[:,0:-1].values
#y_train = train_data.iloc[:,-1].values

#

def prepare_data(train_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    try:
        x_train = train_data.drop(columns=['Potability'], axis=1)
        y_train=train_data['Potability']
        return x_train.values, y_train.values
    except Exception as e:
        print(f"Error preparing data: {e}")
        return np.array([]), np.array([])

def train_model(x_train: np.ndarray, y_train: np.ndarray, n_estimators: int) -> RandomForestClassifier:
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(x_train, y_train)
        return clf
    except Exception as e:
        raise Exception(f"Error training model: {e}")    

def save_model(model: RandomForestClassifier, model_name: str) -> None:
    try:
        with open(model_name, "wb") as file:
            pickle.dump(model, file)
    except Exception as e:
        raise Exception(f"Error saving model to {model_name}: {e}")






def main():
    try:
        data_path = "./data/processed/train_processed_mean.csv"
        model_path= "models/model.pkl"
        params_file_path = "params.yaml"
        print("Model building is started!!!")

        train_data = load_data(data_path)
        n_estimators = load_params(params_file_path)
        train_data = prepare_data(train_data)
        x_train, y_train = train_data
        model = train_model(x_train, y_train, n_estimators)
        save_model(model, model_path)
        print("Model training and saving completed successfully.")

    except Exception as e:
        print(f"An error occurred during model building: {e}")


if __name__ == "__main__":
    main()