
from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def build_preprocess(train: pd.DataFrame, target: str, id_col: str | None = None):
    X = train.drop(columns=[c for c in [target, id_col] if c in train.columns])
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler(with_mean=False))])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])

    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
    return pre, num_cols, cat_cols
