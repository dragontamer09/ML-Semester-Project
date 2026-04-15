"""
preprocess.py
-------------
Cleans and transforms raw feature tuples into sklearn-ready numpy arrays.
All column lists are derived from feature_metadata.py — nothing is
hardcoded here.
 
Pipeline
--------
1. Unpack tuples into a DataFrame
2. Parse temporal features from Transaction_Time / Transaction_Date
3. Drop identifier and raw temporal columns
4. Encode binary Yes/No columns as 0/1
5. Apply a ColumnTransformer (impute + scale numerics, impute + OHE categoricals)
6. Return X (float numpy array) and y (int array or None)
"""
 
from matplotlib import dates
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Optional
 
from feature_metadata import (
    BINARY_FEATURES,
    CAT_FEATURES,
    DROP_COLS,
    NUMERIC_FEATURES,
)

# --Column name lists
# -------------------------------------------------------------------------------
# Built once at import time
_NUMERIC_COLS = [f.name for f in NUMERIC_FEATURES]
_CATEGORICAL_COLS = [f.name for f in CAT_FEATURES]
_BINARY_COLS = [f.name for f in BINARY_FEATURES]

_ALL_NUMERIC_COLS = _NUMERIC_COLS + _BINARY_COLS

# temporal parsing
# --------------------------------------------------------------------------------
def _parse_temporal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Transaction_Time" in df.columns:
        df["Transaction_Hour"] = (pd.to_datetime(df["Transaction_Time"], format="%H:%M", errors="coerce").dt.hour)

    if "Transaction_Date" in df.columns:
        df["Transaction_Month"] = (pd.to_datetime(df["Transaction_Date"], errors="coerce").dt.month)
        df["Transaction_DayOfWeek"] = dates.dt.dayofweek
        df["Transaction_Month"] = dates.dt.month
    return df

# -- Drop identifier and raw temporal columns
# --------------------------------------------------------------------------------
def _drop_cols(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=to_drop)

# -- Encode binary Yes/No flags
# --------------------------------------------------------------------------------
def _encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in _BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})
    return df

# -- Build the sklearn ColumnTransformer for imputation, scaling, and OHE
# --------------------------------------------------------------------------------
def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    present = set(df.columns)
    num_cols = [c for c in _ALL_NUMERIC_COLS if c in present]
    cat_cols = [c for c in _CATEGORICAL_COLS if c in present]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols)
        ],
        remainder = "drop",
    )

# -- internal functions
# --------------------------------------------------------------------------------
def tuples_to_df(tuples: list[tuple],) -> tuple[pd.DataFrame, np.ndarray | None]:

    feature_dicts = [t[0] for t in tuples]
    labels = [t[1] for t in tuples]
    df = pd.DataFrame(feature_dicts)
    y = np.array(labels, dtype=float) if any(l is not None for l in labels) else None
    return df, y

def _manual_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = _parse_temporal(df)
    df = _drop_cols(df)
    df = _encode_binary(df)
    return df

# -- Public API
# --------------------------------------------------------------------------------
def prepare_X_y(tuples: list[tuple], preprocessor: Optional[ColumnTransformer] = None, fit: bool = True,) -> tuple[np.ndarray, np.ndarray | None, ColumnTransformer]:
    df, y = tuples_to_df(tuples)
    df = _manual_transforms(df)

    if preprocessor is None:
        preprocessor = build_preprocessor(df)
    
    X = preprocessor.fit_transform(df) if fit else preprocessor.transform(df)


    return X, y, preprocessor


# Smoke Test
if __name__ == "__main__":
    import sys
    from load_data import load_pipeline
 
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/dragon/Documents/Repos/ML-Semester-Project/src/FraudShield_Banking_Data.csv"
    labeled, unlabeled = load_pipeline(path)
 
    X, y, prep = prepare_X_y(labeled, fit=True)
    print(f"X shape (labeled)   : {X.shape}")
    print(f"y shape             : {y.shape}")
    print(f"dtype               : {X.dtype}")
    print(f"Class balance  0={int((y == 0).sum())}  1={int((y == 1).sum())}")
 
    if unlabeled:
        X_inf, _, _ = prepare_X_y(unlabeled, preprocessor=prep, fit=False)
        print(f"X shape (unlabeled) : {X_inf.shape}")