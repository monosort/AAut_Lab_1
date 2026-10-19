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

#change these whenever different training/validation periods are wanted
# Training range format: 2015-01-01 -> TRAIN_DATES
# Validation range format: TRAIN_DATES -> VALIDATION_DATES
TRAIN_DATES = "2023-01-01"
VALIDATION_DATES = "2024-01-01"

def plot_errors(predictions,Yval):
    """Plots the predicted chlorophyl values and the real chlorophyl values to see errors"""

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

    return feature_means,feature_std,x_Train,x_val

def next_experiment_directory():
    # Resolve models/ relative to train.py, not the terminal directory.
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    experiment_numbers = []

    for folder in models_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("experiment_"):
            number = folder.name.removeprefix("experiment_")

            if number.isdigit():
                experiment_numbers.append(int(number))

    next_number = max(experiment_numbers, default=0) + 1
    output_dir = models_dir / f"experiment_{next_number}"

    # Default exist_ok=False prevents reusing an existing experiment folder.
    output_dir.mkdir()

    return output_dir

def saveResults(model,r2,mae,rmse,lag_spec,means,stds,alpha):
    """saves model and metadata to experiment folder and checks the """

    # metadada is used to be saved into  JSON file so that 
    # parameters and training conditions acan be reproduced and compared
    metadata = {
        "model_type": type(model).__name__,
        "lag_spec": lag_spec,
        "validation_metrics": {
            "r2": float(r2),
            "mae": float(mae),
            "rmse": float(rmse),
        },
        "training_date_range" : f"2015-01-01 -> {TRAIN_DATES}",
        "validation_date_range" : f"{TRAIN_DATES} -> {VALIDATION_DATES}",
        "alpha": alpha,
    }

    # Here only data required by my_model.py and utils.py is saved into model.pkl
    artifact = {
        "model": model,
        "lag_spec": lag_spec,
        "feature_means": means,
        "feature_stds": stds,
        "validation_metrics": {
            "r2": float(r2),
            "mae": float(mae),
            "rmse": float(rmse),
        },
    }

    out_dir=next_experiment_directory()

    overide=False

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, allow_nan=False)

    with (out_dir / "model.pkl").open("wb") as file:
        pickle.dump(artifact, file)

    with (Path("../best_model/model.pkl")).open("rb") as file:
        if artifact["validation_metrics"]["r2"] > pickle.load(file)["validation_metrics"]["r2"]:
            overide = True

    if overide == True:
        with Path("../best_model/model.pkl").open("wb") as file:
            pickle.dump(artifact, file)

    

def validator(model,x,y):
    predictions = model.predict(x)
    
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    rmse = root_mean_squared_error(y, predictions)
    return r2,mae,rmse,predictions

def Linear_Regression_training(x,y, estimator):
    """uses festures x and output chlorophyl values y as well as desired estimator to find the coeficients"""

    if estimator == "Ridge":
        alpha=1100
        model = Ridge(alpha)
        model.fit(x,y)

    elif estimator == "Lasso":
        alpha=0.021
        model = Lasso(alpha)
        model.fit(x,y) 

    # only want to see coef values for Lasso to see if any have been set to 0
    if type(model).__name__ == "Lasso":
        for feature, coefficient in zip(x.columns, model.coef_):
            print(f"{feature}: {coefficient:.6f}")
    return model,alpha

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
    training_rows = table["date"] < TRAIN_DATES
    validation_rows = ((table["date"] >= TRAIN_DATES) & (table["date"] < VALIDATION_DATES))
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
    parser.add_argument("--output", required=False, type=Path)
    parser.add_argument("--days", required=False, default=3287,type=int)
    parser.add_argument("--plot_predictions", required = False,action="store_true")
    parser.add_argument("--estimator", required=False, default="Ridge",type=str)
    parser.add_argument("--no_seasonality",required=False,default=False,action="store_true")

    lag_spec = {
        "chlorophyll_a_mg_m3": [1,2,7,8],
        "nitrate_umol_l": [3,4],
        "wind_speed_m_s": [1,2],
        "upwelling_index": [2,3],
        "salinity_psu": [5],
    }

    #returns training and validation splits for model training
    x_T,y_T,x_V,y_V=pre_processing(parser.parse_args().data,lag_spec)

    if parser.parse_args().no_seasonality:
        x_T = x_T.drop(columns=["annual_sin", "annual_cos"])
        x_V = x_V.drop(columns=["annual_sin", "annual_cos"])

    means,stds,x_T,x_V = standardize_inputs(x_T,x_V)
    model,alpha=Linear_Regression_training(x_T,y_T,parser.parse_args().estimator)
    r2,mae,rmse,predictions=validator(model,x_V,y_V)
    print(r2)
    print(mae)
    print(rmse)
    if parser.parse_args().plot_predictions:
        plot_errors(predictions,y_V)
    #print(x_V)
    #feature_columns = x_V.columns.tolist()
    saveResults(model,r2,mae,rmse,lag_spec,means,stds,alpha)



if __name__ == "__main__":
    main()
