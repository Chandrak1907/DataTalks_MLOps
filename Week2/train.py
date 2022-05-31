import argparse
import os
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run(data_path):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))

    mlflow.sklearn.autolog()
    rf = RandomForestRegressor(max_depth=10, random_state=0)
    rf.fit(X_train, y_train)
    with open('models/rf_reg.bin', 'wb') as f_out:
#         pickle.dump((dv, rf), f_out)
        pickle.dump(( rf), f_out)
    
    y_pred = rf.predict(X_valid)
    
    mlflow.log_param("train-data-path", "./week2_data/green_tripdata_2021-01.parquet")
    mlflow.log_param("valid-data-path", "./week2_data/green_tripdata_2021-02.parquet")
    rmse = mean_squared_error(y_valid, y_pred, squared=False)
    mlflow.log_metric('rmse',rmse)
    mlflow.sklearn.log_model(rf, artifact_path="models_rf")    
    mlflow.log_artifact(local_path="models/rf_reg.bin", artifact_path="models_pickle")

    

    rmse = mean_squared_error(y_valid, y_pred, squared=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="the location where the processed NYC taxi trip data was saved."
    )
    args = parser.parse_args()
    with mlflow.start_run():
        
        mlflow.set_tag("developer", "Chandra") 
        
        run(args.data_path)
