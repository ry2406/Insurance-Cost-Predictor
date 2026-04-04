"""
preprocess.py
Medical Insurance Cost Predictor — Data Preprocessing Module

Input:  data/insurance.csv (1338 rows, 7 columns: age, sex, bmi, children, smoker, region, charges)
Tasks:  encoding, feature engineering, train/test split, scaling
Output: DataFrames or numpy arrays ready for downstream models

Author: Ruide Yin
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Private Helpers 


def _load_raw(path="../data/insurance.csv"):
    """Load CSV and return raw DataFrame."""
    return pd.read_csv(path)


def _encode(df, method="onehot"):
    """
    Encode categorical columns.

    method="onehot":
        sex:    female=0, male=1
        smoker: no=0, yes=1
        region: one-hot with drop_first → 3 columns
                (region_northwest, region_southeast, region_southwest)
    method="label":
        sex:    female=0, male=1
        smoker: no=0, yes=1
        region: LabelEncoder → single integer column
    """
    df = df.copy()

    # Binary columns (same for both methods)
    df["sex"] = df["sex"].map({"female": 0, "male": 1})
    df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})

    if method == "onehot":
        df = pd.get_dummies(df, columns=["region"], drop_first=True, dtype=int)
    elif method == "label":
        le = LabelEncoder()
        df["region"] = le.fit_transform(df["region"])
    else:
        raise ValueError(f"Unknown encoding method: {method}")

    return df


def _add_features(df, include_bmi_smoker=True):
    """
    Feature engineering:
      - Add bmi_smoker = bmi * smoker (interaction term), unless include_bmi_smoker=False
      - Add log_charges = log1p(charges)
      - Drop age_group column if it exists
    """
    df = df.copy()
    if include_bmi_smoker:
        df["bmi_smoker"] = df["bmi"] * df["smoker"]
    df["log_charges"] = np.log1p(df["charges"])

    if "age_group" in df.columns:
        df.drop(columns=["age_group"], inplace=True)

    return df


# Data Preparation Interface 
# Return one fully processed DataFrame.


def get_regressor_data_linear(path="../data/insurance.csv"):
    """Pipeline: load → one-hot encode → add features.
    For: Linear Regression, Quantile Regression.  Target: log_charges."""
    df = _load_raw(path)
    df = _encode(df, method="onehot")
    df = _add_features(df)
    return df


def get_regressor_data_tree(path="../data/insurance.csv"):
    """Pipeline: load → label encode → add features.
    For: Random Forest, XGBoost.  Target: charges."""
    df = _load_raw(path)
    df = _encode(df, method="label")
    df = _add_features(df)
    return df


def get_regressor_data_torch(path="../data/insurance.csv"):
    """Pipeline: load → one-hot encode → add features.
    For: MLP.  Target: charges."""
    df = _load_raw(path)
    df = _encode(df, method="onehot")
    df = _add_features(df)
    return df


def get_regressor_data_mdn(path="../data/insurance.csv"):
    """Wrapper around get_regressor_data_torch.
    For: MDN.  Target: charges."""
    return get_regressor_data_torch(path)


def get_classifier_data_logistic(path="../data/insurance.csv"):
    """Pipeline: load → one-hot encode → add features (no bmi_smoker).
    For: Logistic Regression.  Target: smoker."""
    df = _load_raw(path)
    df = _encode(df, method="onehot")
    df = _add_features(df, include_bmi_smoker=False)
    return df


def get_classifier_data_tree(path="../data/insurance.csv"):
    """Pipeline: load → label encode → add features (no bmi_smoker).
    For: RF / XGB classifier.  Target: smoker."""
    df = _load_raw(path)
    df = _encode(df, method="label")
    df = _add_features(df, include_bmi_smoker=False)
    return df


def get_classifier_data_torch(path="../data/insurance.csv"):
    """Pipeline: load → one-hot encode → add features (no bmi_smoker).
    For: MLP classifier.  Target: smoker."""
    df = _load_raw(path)
    df = _encode(df, method="onehot")
    df = _add_features(df, include_bmi_smoker=False)
    return df


# Split Interface 
# Accept a processed DataFrame, perform 80/20 split with optional scaling.
# Stratify by smoker column to preserve the ~20/80 class ratio.
# random_state = 12138


def _build_feature_matrix(df, target_col):
    """
    Separate feature matrix X from target y.
    Drops: target_col, charges, log_charges, and any other non-feature columns.
    If target_col is not 'smoker', also drops smoker from X
    (smoker is only kept for stratification, not as a feature for regressors).
    Returns X (DataFrame), y (Series), stratify_col (Series).
    """
    # Columns that are never features
    cols_to_drop = {"charges", "log_charges"}
    cols_to_drop.add(target_col)

    # For regression tasks, remove smoker from features (used only for stratify)
    if target_col != "smoker":
        cols_to_drop.add("smoker")

    # For classification on smoker, drop bmi_smoker to avoid data leakage
    if target_col == "smoker" and "bmi_smoker" in df.columns:
        cols_to_drop.add("bmi_smoker")

    # Only drop columns that actually exist in the DataFrame
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]

    X = df.drop(columns=cols_to_drop)
    y = df[target_col]

    # Stratify column: use smoker if present and has >1 unique value, else skip
    if "smoker" in df.columns and df["smoker"].nunique() > 1:
        stratify_col = df["smoker"]
    else:
        stratify_col = None

    return X, y, stratify_col


def split_scaled(df, target_col):
    """
    Split and scale data.

    Steps:
      1. Separate X / y (see _build_feature_matrix)
      2. train_test_split (stratify=smoker, test_size=0.2, random_state=12138)
      3. StandardScaler fit on X_train, transform both X_train and X_test
      4. Return X_train, X_test, y_train, y_test, scaler (all numpy arrays except scaler)

    For: Linear / Quantile / MLP / MDN / Logistic
    """
    X, y, stratify_col = _build_feature_matrix(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=12138,
        stratify=stratify_col,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = y_train.values
    y_test = y_test.values

    return X_train, X_test, y_train, y_test, scaler


def split_unscaled(df, target_col):
    """
    Split without scaling.

    Same as split_scaled but skips the StandardScaler step.
    Returns X_train, X_test, y_train, y_test (all numpy arrays).

    For: RF / XGB (regression and classification)
    """
    X, y, stratify_col = _build_feature_matrix(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=12138,
        stratify=stratify_col,
    )

    return X_train.values, X_test.values, y_train.values, y_test.values


# ===================== Usage Examples =====================
#
# --- Block 1: Regression ---
# Linear:   df = get_regressor_data_linear(); X_tr, X_te, y_tr, y_te, scaler = split_scaled(df, "log_charges")
# Quantile: df = get_regressor_data_linear(); X_tr, X_te, y_tr, y_te, scaler = split_scaled(df, "log_charges")
# RF:       df = get_regressor_data_tree();   X_tr, X_te, y_tr, y_te = split_unscaled(df, "charges")
# XGBoost:  df = get_regressor_data_tree();   X_tr, X_te, y_tr, y_te = split_unscaled(df, "charges")
# MLP:      df = get_regressor_data_torch();  X_tr, X_te, y_tr, y_te, scaler = split_scaled(df, "charges")
# MDN:      df = get_regressor_data_mdn();    X_tr, X_te, y_tr, y_te, scaler = split_scaled(df, "charges")
#
# --- Block 2: Classification ---
# Logistic: df = get_classifier_data_logistic(); X_tr, X_te, y_tr, y_te, scaler = split_scaled(df, "smoker")
# RF clf:   df = get_classifier_data_tree();     X_tr, X_te, y_tr, y_te = split_unscaled(df, "smoker")
# XGB clf:  df = get_classifier_data_tree();     X_tr, X_te, y_tr, y_te = split_unscaled(df, "smoker")
# MLP clf:  df = get_classifier_data_torch();    X_tr, X_te, y_tr, y_te, scaler = split_scaled(df, "smoker")
#
# --- Block 2: Stratified Regression ---
# Smoker subset:     df = get_regressor_data_torch(); s_df = df[df["smoker"]==1]; X_tr, X_te, y_tr, y_te, scaler = split_scaled(s_df, "charges")
# Non-smoker subset: df = get_regressor_data_torch(); n_df = df[df["smoker"]==0]; X_tr, X_te, y_tr, y_te, scaler = split_scaled(n_df, "charges")