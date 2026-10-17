"""Replace this template with the code used to train and save best_model/model.pkl.

The final script should reproduce raw-data loading, chronological splitting,
lag construction, preprocessing, model selection, final fitting, and model
serialization. sklearn.preprocessing tools may be fitted on input features
using training data only and then saved for prediction. Keep all supporting
Python modules in training_code/. Do not use library pipelines, composition
wrappers, imputers, feature selectors, or output transformers. Target
transformation and output post-processing must remain explicit.
"""

from __future__ import annotations
import csv
import pandas as pd
import json

from build_lagged_features import build_lagged_frame

import argparse
from pathlib import Path
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error
import pickle
import matplotlib.pyplot as plt  # Plot data


def plot_errors(predictions,Yval):
    fig,ax = plt.subplots(figsize=(12,5))

    ax.plot(Yval.to_numpy(), label="Actual chlorophyll")
    ax.plot(predictions, label="Predicted chlorophyll", alpha=0.7)

    ax.set_xlabel("Validation day index")
    ax.set_ylabel("Chlorophyll-a (mg/m³)")
    ax.legend()
    ax.grid(True)

    plt.show()

def standardize_inputs(x_Train,x_val):

    feature_means = x_Train.mean(axis=0)
    feature_std = x_Train.std(axis=0, ddof=0)

    x_Train = (x_Train-feature_means)/ feature_std
    x_val = (x_val-feature_means)/ feature_std

    return x_Train,x_val



def saveResults(model,r2,mae,rmse,feature_names,out_dir):
    metadata = {
        "model_type": type(model).__name__,
        "feature_columns": feature_names,
        "validation_metrics": {
            "r2": float(r2),
            "mae": float(mae),
            "rmse": float(rmse),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, allow_nan=False)
    with (out_dir / "model.pkl").open("wb") as file:
        pickle.dump(model, file)

def validator(model,x,y):
    predictions = model.predict(x)
    
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    rmse = root_mean_squared_error(y, predictions)
    return r2,mae,rmse,predictions

def Linear_Regression_training(x,y):
    model = Ridge(alpha=1100)
    model.fit(x,y)

    return model

def pre_processing(data,lag_spec):

    raw = pd.read_csv(data, parse_dates=["date"])
    raw = raw.sort_values("date").reset_index(drop=True)
  
    table = build_lagged_frame(raw,lag_spec)
    #print(table)

    for column in table.columns:
        if column!= "date":
            table[column] = pd.to_numeric(table[column])


    Features = table.drop(columns=["date", "chlorophyll_a_mg_m3"])
    Output = table["chlorophyll_a_mg_m3"]
    #print(Features)
    X_train,Y_train,X_val,Y_val=training_val_split(Features,Output,table)
    #print(X_train)


    return X_train,Y_train,X_val,Y_val

def training_val_split(x,y,table):
    training_rows = table["date"] < "2022-01-01"
    validation_rows = ((table["date"] >= "2022-01-01") & (table["date"] < "2023-01-01"))
    #print(table)

    X_train = x.loc[training_rows]
    Y_train = y.loc[training_rows]

    X_val = x.loc[validation_rows]
    Y_val = y.loc[validation_rows]
    #print(X_val)

    return X_train,Y_train,X_val,Y_val
    

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--days", required=False, default=3287,type=int)
    parser.add_argument("--plot_predictions", required = False,action="store_true")

    lag_spec = {
        "chlorophyll_a_mg_m3": list(range(1, 15)),
        "sst_c": list(range(0, 5)),
        "par_umol_m2_s": list(range(0, 2)),
        "nitrate_umol_l": list(range(0, 5)),
        "wind_speed_m_s": list(range(0, 5)),
        "upwelling_index": list(range(0, 5)),
        "mixed_layer_depth_m": list(range(0, 4)),
        "current_speed_m_s": list(range(0, 5)),
        "river_discharge_index": list(range(0, 5)),
        "cloud_fraction": list(range(0, 5)),
        "turbidity_ntu": list(range(0, 5)),
    }

    #returns training and validation splits for model training
    x_T,y_T,x_V,y_V=pre_processing(parser.parse_args().data,lag_spec)
    x_T,x_V = standardize_inputs(x_T,x_V)
    model=Linear_Regression_training(x_T,y_T)
    r2,mae,rmse,predictions=validator(model,x_V,y_V)
    print(r2)
    print(mae)
    print(rmse)
    if parser.parse_args().plot_predictions:
        plot_errors(predictions,y_V)
    #print(x_V)
    #feature_columns = x_V.columns.tolist()
    saveResults(model,r2,mae,rmse,x_V.columns.tolist(),parser.parse_args().output)



if __name__ == "__main__":
    main()
