import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split



def data_load(data_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(data_path)
        return data
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        return pd.DataFrame()





def fill_missing_with_mean(df):
    try:
         for column in df.columns:
            if df[column].isnull().any():
                mean_value = df[column].mean()
                df[column].fillna(mean_value,inplace=True)
                return df
    except Exception as e:
        print(f"The error during the filling missed.. values::: {e}")



def save_data(pd: pd.DataFrame, file_path: str) -> None:
    try:
        pd.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")











def main():
    try:
        print("Data preparation is Started!!!.")
        raw_data_path = "./data/raw"
        processed_data_path = "./data/processed"
   
        train_data = data_load(os.path.join(raw_data_path, "train.csv"))
        test_data =data_load(os.path.join(raw_data_path, "test.csv"))

        os.makedirs(processed_data_path)
        train_processed_data = fill_missing_with_mean(train_data)
        test_processed_data = fill_missing_with_mean(test_data)
        save_data(train_processed_data, os.path.join(processed_data_path, "train_processed_mean.csv"))
        save_data(test_processed_data, os.path.join(processed_data_path, "test_processed_mean.csv"))

        print("Data preparation is completed!!!.")
   # data_path = os.path.join("data", "processed")
    except Exception as e:
        raise Exception(f"An error occurred :{e}")



if __name__ == "__main__":
    main()

    



