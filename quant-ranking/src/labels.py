"""Target: forward return a N giorni, e label binaria vs mediana cross-sectional."""

import pandas as pd

HORIZON_DAYS = 5


def compute_labels(panel: pd.DataFrame, horizon: int = HORIZON_DAYS) -> pd.DataFrame:
    close = panel["close"].unstack("ticker").sort_index()

    forward_return = close.shift(-horizon) / close - 1.0
    median = forward_return.median(axis=1)
    label = forward_return.gt(median, axis=0).astype(int)

    fwd_long = forward_return.stack().rename("forward_return")
    label_long = label.stack().rename("label")
    fwd_long.index.names = ["date", "ticker"]
    label_long.index.names = ["date", "ticker"]

    out = pd.concat([fwd_long, label_long], axis=1).dropna(subset=["forward_return"])
    return out
