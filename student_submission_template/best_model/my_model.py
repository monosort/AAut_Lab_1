"""Structurally valid zero-prediction example; replace it with the final model."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd

from utils import pre_processing,extract_lag_spec,standardize_inputs


def load_model(model_path: str | Path) -> Any:
    """Load and return the submitted fitted model artifact."""
    with Path(model_path).open("rb") as handle:
        return pickle.load(handle)


def predict(model: Any, history_df: pd.DataFrame, day_features_df: pd.DataFrame) -> np.ndarray:
    """Return the dummy model's constant as a NumPy array with shape (1,).

    history_df contains all observations strictly before the prediction date,
    including past chlorophyll. day_features_df contains exactly the current
    day's exogenous variables and date, but never current-day chlorophyll.

    This example deliberately does not use either DataFrame. Students must
    replace it with their lag construction, saved input preprocessing, and
    prediction procedure. Fitted sklearn.preprocessing input objects are
    allowed; library pipelines and output-processing wrappers are not.
    """
    #first we get the lag_spec from the model.pkl
    lag_spec = extract_lag_spec(model)
    # we then build the lagged feature row for target days prediction
    feature_row = pre_processing(history_df,day_features_df,lag_spec)
    # standardize the inputs so that ridge/Lasso has fair comparison 
    # and units do not impact the algorithm in n undesired way
    scaled_row = standardize_inputs(feature_row,model)
    #now we can finaly carry out a prediction
    prediction = model["model"].predict(scaled_row)
    # explicit no-negative chlorophyl values rule
    prediction = np.maximum(prediction, 0.0)
    

    return np.asarray(prediction, dtype=float)
