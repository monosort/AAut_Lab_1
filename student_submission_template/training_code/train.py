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
from sklearn.linear_model import LinearRegression,Ridge,Lasso,HuberRegressor
from sklearn.metrics import r2_score,mean_absolute_error,root_mean_squared_error
import pickle
import matplotlib.pyplot as plt  # Plot data

#change these whenever different training/validation periods are wanted
# Training range format: 2015-01-01 -> TRAIN_DATES
# Validation range format: TRAIN_DATES -> VALIDATION_DATES
TRAIN_DATES = "2022-01-01"
VALIDATION_DATES = "2023-01-01"

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
    """carries out standardization on features, does this using: (x-mean(x))/x_std
    this is so that units of each feature dont bias the alpha coeficient in Ridge and Lasso"""

    feature_means = x_Train.mean(axis=0)
    feature_std = x_Train.std(axis=0, ddof=0)

    x_Train = (x_Train-feature_means)/ feature_std
    x_val = (x_val-feature_means)/ feature_std

    return feature_means,feature_std,x_Train,x_val

def next_experiment_directory():
    """automatically resolves saving path and saving folder name"""
    # Resolve models/ relative to train.py, not the terminal directory.
    models_dir = Path(__file__).resolve().parents[2] / "models"
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

def saveResults(model,r2,mae,rmse,lag_spec,means,stds,alpha,epsilon,estimator):
    """saves model and metadata to experiment folder"""

    # metadada is used to be saved into  JSON file so that 
    # parameters and training conditions acan be reproduced and compared
    if estimator == "Ridge" or estimator == "Lasso":

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

    elif estimator == "Huber":
        metadata = {
            "model_type": type(model).__name__,
            "lag_spec": lag_spec,
            "validation_metrics": {
                "r2": float(r2),
                "mae": float(mae),
                "rmse": float(rmse),
            },
            "feature_names":model.feature_names_in_.tolist(),
            "training_date_range" : f"2015-01-01 -> {TRAIN_DATES}",
            "validation_date_range" : f"{TRAIN_DATES} -> {VALIDATION_DATES}",
            "alpha": alpha,
            "epsilon":epsilon,
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
    """carries out prediction on Validation set and calculates prediction statistics"""

    predictions = model.predict(x)
    
    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)
    rmse = root_mean_squared_error(y, predictions)
    return r2,mae,rmse,predictions

def Linear_Regression_training(x,y, estimator):
    """uses festures x and output chlorophyl values y as well as desired estimator to find the coeficients"""

    params =[]

    if estimator == "Ridge":
        alpha=1100
        model = Ridge(alpha)
        model.fit(x,y)
        params.append(alpha)
        params.append(0)

    elif estimator == "Lasso":
        alpha=0.001
        model = Lasso(alpha)
        model.fit(x,y) 
        params.append(alpha)
        params.append(0)


    elif estimator == "Huber":
        alpha = 0
        epsilon = 1.51
        model = HuberRegressor(alpha=alpha,epsilon=epsilon,max_iter=20000)
        model.fit(x,y)
        params.append(alpha)
        params.append(epsilon)



    # only want to see coef values for Lasso to see if any have been set to 0
    if type(model).__name__ == "Lasso" or type(model).__name__ == "HuberRegressor":
        for feature, coefficient in zip(x.columns, model.coef_):
            print(f"{feature}: {coefficient:.6f}")



    return model,params

def pre_processing(data,lag_spec):
    """reads data in from csv file, into a panda structure then builds the lagged
      frame using the script provided by the Proffesor team
      Lastly the training_val_split() function is called to
      split the data into training and validation Pandas"""

    raw = pd.read_csv(data, parse_dates=["date"])
    raw = raw.sort_values("date").reset_index(drop=True)
  
    table = build_lagged_frame(raw,lag_spec)

    for column in table.columns:
        if column!= "date":
            table[column] = pd.to_numeric(table[column])


    Features = table.drop(columns=["date", "chlorophyll_a_mg_m3"])
    Output = table["chlorophyll_a_mg_m3"]
    X_train,Y_train,X_val,Y_val=training_val_split(Features,Output,table)

    return X_train,Y_train,X_val,Y_val

def add_polynomial_feature(features_T,features_V):
    """adding non linear interaction feature, model coeficients remain linear.
    Lots of combinations where tried of those shown below for the best model we found that 
    only these two where necessary"""
    new_column1 = "upwhelling_lag2_squared"
    for feature in (features_T,features_V):
        feature[new_column1] = (feature["upwelling_index__lag_2"] * feature["upwelling_index__lag_2"])

    # new_column2 = "Nitrate_lag3_squared"
    # for feature2 in (features_T,features_V):
    #     feature2[new_column2] = (feature2["nitrate_umol_l__lag_3"] * feature2["nitrate_umol_l__lag_3"])

    new_column3 = "Upwelling_lag_2_x_nitrate_lag_3"
    for feature3 in (features_T,features_V):
        feature3[new_column3] = (feature3["upwelling_index__lag_2"] * feature3["nitrate_umol_l__lag_3"])

    # new_column4 = "nitrate_lag_3_x_par_umol_lag_1"
    # for feature4 in (features_T,features_V):
    #     feature4[new_column4] = (feature4["nitrate_umol_l__lag_3"] * feature4["par_umol_m2_s__lag_1"])

    # new_column6 = "par_umol_m2_s__lag_1_x_mixed_layer_depth_m__lag_1"
    # for feature6 in (features_T,features_V):
    #     feature6[new_column6] = (feature6["par_umol_m2_s__lag_1"] * feature6["mixed_layer_depth_m__lag_1"])

    # new_column7 = "par_umol_m2_s_lag_1_squared"
    # for feature7 in (features_T,features_V):
    #     feature7[new_column7] = (feature7["par_umol_m2_s__lag_1"] * feature7["par_umol_m2_s__lag_1"])

    # new_column8 = "upwelling_index__lag_2_x_current_speed_m_s__lag_0"
    # for feature8 in (features_T,features_V):
    #     feature8[new_column8] = (feature8["upwelling_index__lag_2"] * feature8["current_speed_m_s__lag_0"])

    return features_T,features_V

def add_seasonal_features(features, seasonal_sources):
    """ adds seasonal relationships to features themselves by multiplication"""

    features = features.copy()

    for source in seasonal_sources:
        for basis in ("annual_sin", "annual_cos"):
            new_column = f"{source}__x_{basis}"
            features[new_column] = features[source] * features[basis]

    return features

def training_val_split(x,y,table):
    """splits the dataset into training and validation Pandas as specfied by the date range"""

    training_rows = table["date"] < TRAIN_DATES
    validation_rows = ((table["date"] >= TRAIN_DATES) & (table["date"] < VALIDATION_DATES))

    X_train = x.loc[training_rows]
    Y_train = y.loc[training_rows]

    X_val = x.loc[validation_rows]
    Y_val = y.loc[validation_rows]

    return X_train,Y_train,X_val,Y_val
    

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=False, type=Path)
    parser.add_argument("--days", required=False, default=3287,type=int)
    parser.add_argument("--plot_predictions", required = False,action="store_true")
    parser.add_argument("--estimator", required=False, default="Ridge",type=str)
    parser.add_argument("--no_seasonality",required=False,default=False,action="store_true")
    parser.add_argument("--no_save",required=False,default=False,action="store_true")
    parser.add_argument
    # lag_spec = {
    #     "chlorophyll_a_mg_m3": [1,2,7,8],
    #     "nitrate_umol_l": [3,4],
    #     "wind_speed_m_s": [1,2],
    #     "upwelling_index": [2,3],
    #     "salinity_psu": [5],
    # }

    # more broad lag spec
    # ctrl + K, then ctrl + U to uncomment
    # ctrl + K, then ctrl + C to comment
    lag_spec = {
        "chlorophyll_a_mg_m3": list(range(1, 14)),
        "sst_c": list(range(0, 8)),
        "par_umol_m2_s": list(range(0, 8)),
        "nitrate_umol_l": [3],
        "wind_speed_m_s": [1],
        "upwelling_index": [2],
        "mixed_layer_depth_m": list(range(0, 8)),
        "current_speed_m_s": [11],
        "river_discharge_index": [7],
        "cloud_fraction": [0],
        "surface_pressure_hpa": list(range(0, 4)),
        "turbidity_ntu": [13],
        "salinity_psu":[11],
        }

    seasonal_sources = [
        "current_speed_m_s__lag_11",
        "mixed_layer_depth_m__lag_0",
        "par_umol_m2_s__lag_1",
        "par_umol_m2_s__lag_6",
        "sst_c__lag_0",
        "surface_pressure_hpa__lag_0",
        "upwelling_index__lag_2",
        "wind_speed_m_s__lag_1",
    ]

    #returns training and validation splits for model training
    x_T,y_T,x_V,y_V=pre_processing(parser.parse_args().data,lag_spec)

    x_T,x_V = add_polynomial_feature(x_T,x_V)

    if parser.parse_args().no_seasonality:
        x_T = add_seasonal_features(x_T,seasonal_sources)
        x_V = add_seasonal_features(x_V,seasonal_sources)
        x_T = x_T.drop(columns=["annual_sin", "annual_cos"])
        x_V = x_V.drop(columns=["annual_sin", "annual_cos"])

    means,stds,x_T,x_V = standardize_inputs(x_T,x_V)
    model,params=Linear_Regression_training(x_T,y_T,parser.parse_args().estimator)
    r2,mae,rmse,predictions=validator(model,x_V,y_V)

    print(f"R^2 = {r2}")
    print(f"{mae}")
    print(rmse)

    if parser.parse_args().plot_predictions:
        plot_errors(predictions,y_V)
    #print(x_V)
    #feature_columns = x_V.columns.tolist()
    if parser.parse_args().no_save == False:
        saveResults(model,r2,mae,rmse,lag_spec,means,stds,params[0],params[1],parser.parse_args().estimator)



if __name__ == "__main__":
    main()
