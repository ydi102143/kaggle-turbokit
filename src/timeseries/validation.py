
from __future__ import annotations
from typing import Iterator, Tuple
import numpy as np
import pandas as pd

class PurgedGroupTimeSeriesSplit:
    def __init__(self, n_splits: int = 5, embargo: int = 0):
        assert n_splits >= 2
        self.n_splits = n_splits; self.embargo = embargo

    def split(self, X: pd.DataFrame, time_col: str, group_col: str | None = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        df = X[[time_col] + ([group_col] if group_col else [])].copy()
        if group_col:
            df = df.reset_index().sort_values([group_col, time_col]); idx = df["index"].values
        else:
            df = X[[time_col]].reset_index().sort_values(time_col); idx = df["index"].values
        n = len(idx)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int); fold_sizes[: n % self.n_splits] += 1
        current = 0; borders = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size; borders.append((start, stop)); current = stop
        for i in range(self.n_splits):
            tr_end = borders[i][0]; va_start, va_end = borders[i]
            tr_end = max(0, tr_end - self.embargo)
            tr_idx = idx[:tr_end]; va_idx = idx[va_start:va_end]
            if len(tr_idx) == 0 or len(va_idx) == 0: continue
            yield tr_idx, va_idx
