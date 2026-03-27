# =============================================================================
# Project  : Data Pipleline Implementation
# File     : load_data.py
# Author   : John Panzer, with some help from Claude AI
# Course   : COMP4330 - Intro to Machine Learning
# Professor: Dr. Hamilton
# Due      : 2026-03-29
#------------------------------------------------------------------------------
# Loads the raw CSV file and separates it into labeled and unlabeled target
# variable tuples to prepare them for preprocessing
# =============================================================================

import pandas as pd
from pathlib import Path
from typing import Optional
import feature_metadata

# load the features of the CSV file, columns, etc.

# read CSV
def load_raw(filepath: str | Path) -> pd.DataFrame:
    return pd.read_csv(filepath)

# split the labeled entries and the unlabled entries into two dataframes
def split_labeled_unlabeled(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled_df = df[df[feature_metadata.TARGET_COL].notna()].copy
    unlabeled_df = df[df[feature_metadata.TARGET_COL].isna()].copy
    return labeled_df, unlabeled_df

# encode each entry with 1 or 0, 1 when the target variable is fraud and 0 otherwise (normal)
def encode_label(label: str) -> int:
    return 1 if str(label).strip().lower() == "fraud" else 0

def df_to_tuples (df: pd.DataFrame, feature_names: Optional[list[str]] = None, include_label: bool = True) -> list[tuple[dict, Optional[int]]]:
    #convert list to feature_dict, label tuples
    if feature_names is None:
        feature_names = [f.name for f in feature_metadata.MODEL_FEATURES]
    
    cols_present = [c for c in feature_names if c in df.columns]

    records = df[cols_present].to_dict(orient="records")
    labels = (
        df[feature_metadata.TARGET_COL].apply(encode_label).tolist()
        if include_label
        else [None] * len(df)
    )
    return list(zip(records, labels))

# setup the pipeline for preprocessing
def load_pipeline(filepath: str | Path, feature_names: Optional[list[str]] = None) -> tuple[list[tuple], list[tuple]]:
    df = load_raw(filepath)
    feature_metadata.validate_dataframe(df)

    labeled_df, unlabeled_df = split_labeled_unlabeled(df)

    labeled_tuples = df_to_tuples(labeled_df, feature_names, include_label=True)
    unlabeled_tuples = df_to_tuples(unlabeled_df, feature_names, include_label=False)

    return labeled_tuples, unlabeled_tuples


#----------------------test output----------------------
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "/home/dragon/Documents/Repos/ML-Semester-Project/src/FraudShield_Banking_Data.csv"
    labeled, unlabeled =load_pipeline(path)

    print(f"Labeled tuples   : {len(labeled):,}")
    print(f"Unlabeled tuples : {len(unlabeled):,}")
    print(f"\nSample labeled tuple:")
    print(f"  features : {labeled[0][0]}")
    print(f"  label    : {labeled[0][1]}")

    fraud_count = sum(1 for _, lbl in labeled if lbl == 1)
    normal_count = sum(1 for _, lbl in unlabeled if lbl ==0)

    print(f"\nClass distribution:")
    print(f"  Normal : {normal_count:,} ({normal_count / len(labeled):.1%})")
    print(f"  Fraud  : {fraud_count:,}  ({fraud_count  / len(labeled):.1%})")
