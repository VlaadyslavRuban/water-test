import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml



def load_data(url:str) -> pd.DataFrame:
    try:
        data = pd.read_csv(url)
        return data

    except Exception as e:
        print(f"Error loading data from {url}: {e}")
        return pd.DataFrame()

def load_params(params_file: str)->float:
    try:
        with open(params_file, 'r') as file:
            params = yaml.safe_load(file)["data_collection"]["test_size"]
        return params
    except FileNotFoundError:
        print(f"Error: The file {params_file} does not exist.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
    


#yaml_file = 'params.yaml'
#with open(yaml_file, 'r') as file:
 #   test_size_yaml = yaml.safe_load(file)["data_collection"]["test_size"]




def split_data(data: pd.DataFrame, test_size_from: float) ->tuple[pd.DataFrame,pd.DataFrame]:
    try:
        return train_test_split(data, test_size=test_size_from, random_state=42) 
    except Exception as e:
        print(f"Error splitting data: {e}")
        return pd.DataFrame(), pd.DataFrame()   




def save_data(df:pd.DataFrame, file_path: str)->None:
    try:
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")




def main():
    datafile_path = "https://raw.githubusercontent.com/DataThinkers/Datasets/main/DS/water_potability.csv"
    params_filepath = "params.yaml"
    raw_data_path=os.path.join("data", "raw")
    try:
        print("Data collection is started!!!")
        data = load_data(datafile_path)
        test_size_yaml = load_params(params_filepath)
        train_data, test_data = split_data(data, test_size_yaml)
        os.makedirs(raw_data_path)
        save_data(test_data, os.path.join(raw_data_path, "test.csv"))
        save_data(train_data, os.path.join(raw_data_path, "train.csv"))
        print("Data collection is completed!!!")
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == "__main__":
    main()