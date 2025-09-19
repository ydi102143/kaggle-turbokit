
from __future__ import annotations
import pandas as pd

def add_time_features(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[dt_col])
    df["year"] = dt.dt.year; df["month"] = dt.dt.month; df["day"] = dt.dt.day
    df["dow"] = dt.dt.weekday; df["week"] = dt.dt.isocalendar().week.astype(int)
    return df
