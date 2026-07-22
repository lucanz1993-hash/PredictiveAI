"""Fattori cross-sectional: momentum, reversal, volatilita', volume relativo.

Ogni fattore viene calcolato per ticker (serie storica propria) e poi
standardizzato cross-sectionally (z-score fra i ticker, alla stessa data)
in modo da essere comparabile e stazionario nel tempo.
"""

import numpy as np
import pandas as pd

FACTOR_COLUMNS = ["mom_21", "mom_63", "mom_126", "rev_5", "vol_21", "rel_volume_21"]


def _cross_sectional_zscore(wide: pd.DataFrame) -> pd.DataFrame:
    mean = wide.mean(axis=1)
    std = wide.std(axis=1)
    return wide.sub(mean, axis=0).div(std, axis=0)


def compute_features(panel: pd.DataFrame) -> pd.DataFrame:
    close = panel["close"].unstack("ticker").sort_index()
    volume = panel["volume"].unstack("ticker").sort_index()

    raw = {
        "mom_21": close.pct_change(21),
        "mom_63": close.pct_change(63),
        "mom_126": close.pct_change(126),
        "rev_5": close.pct_change(5),
        "vol_21": close.pct_change().rolling(21).std(),
        "rel_volume_21": volume / volume.rolling(21).mean(),
    }

    zscored = {name: _cross_sectional_zscore(wide) for name, wide in raw.items()}

    frames = []
    for name, wide in zscored.items():
        long = wide.stack().rename(name)
        long.index.names = ["date", "ticker"]
        frames.append(long)

    features = pd.concat(frames, axis=1)
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features
