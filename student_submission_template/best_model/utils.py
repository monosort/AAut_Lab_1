"""Auxiliary prediction functions imported by my_model.py."""

from __future__ import annotations

from typing import Any
import pandas as pd
import numpy as np

def add_polynomial_feature(features):
    """computes and adds polynomial interaction features"""

    features["upwhelling_lag2_squared"] = (features["upwelling_index__lag_2"] ** 2)
    features["Upwelling_lag_2_x_nitrate_lag_3"] = (features["upwelling_index__lag_2"] * features["nitrate_umol_l__lag_3"])

    return features

def add_seasonal_features(features, seasonal_sources):
    """ adds seasonal relationships to features themselves by multiplication"""

    features = features.copy()

    for source in seasonal_sources:
        for basis in ("annual_sin", "annual_cos"):
            new_column = f"{source}__x_{basis}"
            features[new_column] = features[source] * features[basis]

    return features
POLY_FEATURES = [
    ("sst_c__lag_0", "mixed_layer_depth_m__lag_2"),
    ("nitrate_umol_l__lag_3", "current_speed_m_s__lag_1"),
    ("nitrate_umol_l__lag_0", "nitrate_umol_l__lag_7"),
]

def add_poly(df):
    df = df.copy()
    for a, b in POLY_FEATURES:
        df[f"{a}*{b}"] = df[a] * df[b]
    return df

def extract_lag_spec(model):
    #here we must extract the lag spec from the model dictionary
    lag_spec = model["lag_spec"]
    return lag_spec

def standardize_inputs(feature_row,model):

    feature_row = feature_row.loc[:, model["feature_means"].index]

    scaled_row =  (feature_row-(model["feature_means"]))/model["feature_stds"]

    return scaled_row

def pre_processing(history,current_day,lag_spec):

    features_row = one_day_row(history,current_day,lag_spec)

    features_row = add_polynomial_feature(features_row)

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

    features_row = add_seasonal_features(features_row, seasonal_sources)
    

    return features_row

def one_day_row(history: pd.DataFrame, day_features: pd.DataFrame,
                lag_spec: dict[str, list[int]] | None = None) -> pd.DataFrame:
    """Construct exactly one feature row from observed history plus today's exogenous data."""
    if len(day_features) != 1:
        raise ValueError("day_features must contain exactly one row")
    today = day_features.copy()
    if "chlorophyll_a_mg_m3" not in today:
        today["chlorophyll_a_mg_m3"] = np.nan
    combined = pd.concat([history, today], ignore_index=True, sort=False)
    lagged = build_lagged_frame(combined, lag_spec=lag_spec, require_target=False)
    row = lagged.tail(1).drop(columns=["date", "chlorophyll_a_mg_m3"], errors="ignore")
    if len(row) != 1:
        raise ValueError("insufficient history to construct the requested lags")
    return row

def pre_processing(history, current_day, lag_spec):
    features_row = one_day_row(history, current_day, lag_spec)
    features_row = add_poly(features_row)
    return features_row

def build_lagged_frame(raw: pd.DataFrame, lag_spec: dict[str, list[int]] | None = None,
                       require_target: bool = True) -> pd.DataFrame:
    """Return rows indexed by prediction date, never using target lag 0."""
    if "date" not in raw:
        raise ValueError(f"missing required column: date")
    if require_target and "chlorophyll_a_mg_m3" not in raw:
        raise ValueError(f"missing required column: chlorophyll_a_mg_m3")
    frame = raw.copy().sort_values("date").reset_index(drop=True)
    columns: dict[str, pd.Series | np.ndarray] = {"date": pd.to_datetime(frame["date"])}
    if "chlorophyll_a_mg_m3" in frame:
        columns["chlorophyll_a_mg_m3"] = frame["chlorophyll_a_mg_m3"].astype(float)
    for col, lags in lag_spec.items():
        if col not in frame:
            raise ValueError(f"lag specification refers to missing column: {col}")
        for lag in lags:
            if col == "chlorophyll_a_mg_m3" and lag == 0:
                raise ValueError("target lag 0 is forbidden: it leaks the answer")
            columns[f"{col}__lag_{lag}"] = pd.to_numeric(frame[col], errors="coerce").shift(lag)
    day = pd.to_datetime(frame["date"]).dt.dayofyear.to_numpy()
    columns["annual_sin"] = np.sin(2 * np.pi * day / 365.25)
    columns["annual_cos"] = np.cos(2 * np.pi * day / 365.25)
    out = pd.DataFrame(columns)
    if require_target:
        out = out.dropna(subset=["chlorophyll_a_mg_m3"])
    max_lag = max(max(v) for v in lag_spec.values())
    return out.iloc[max_lag:].reset_index(drop=True)