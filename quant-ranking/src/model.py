"""Training walk-forward di un LGBMClassifier sulle feature cross-sectional.

Ogni fold: train su tutto lo storico fino a T (espandendo), purge di
`horizon` giorni per evitare che la label (che guarda in avanti) del
train sconfini nel periodo di test, poi predict su un blocco di test
di `test_size` giorni. Non c'e' mai shuffle temporale.
"""

import pandas as pd
from lightgbm import LGBMClassifier

from features import FACTOR_COLUMNS

INITIAL_TRAIN_DAYS = 756  # ~3 anni di trading
TEST_BLOCK_DAYS = 21  # retrain circa mensile


def _model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=50,
        verbosity=-1,
    )


def walk_forward_splits(dates: pd.DatetimeIndex, horizon: int, initial_train: int, test_size: int):
    start = initial_train
    while start + test_size <= len(dates):
        train_end_idx = max(start - horizon, 1)  # purge: rimuove le ultime `horizon` date dal train
        train_dates = dates[:train_end_idx]
        test_dates = dates[start : start + test_size]
        yield train_dates, test_dates
        start += test_size


def run_walk_forward(
    dataset: pd.DataFrame,
    horizon: int,
    initial_train: int = INITIAL_TRAIN_DAYS,
    test_size: int = TEST_BLOCK_DAYS,
) -> pd.DataFrame:
    dates = dataset.index.get_level_values("date").unique().sort_values()
    results = []

    for train_dates, test_dates in walk_forward_splits(dates, horizon, initial_train, test_size):
        train = dataset.loc[dataset.index.get_level_values("date").isin(train_dates)]
        test = dataset.loc[dataset.index.get_level_values("date").isin(test_dates)]
        if train.empty or test.empty:
            continue

        clf = _model()
        clf.fit(train[FACTOR_COLUMNS], train["label"])
        scores = clf.predict_proba(test[FACTOR_COLUMNS])[:, 1]

        fold = test[["forward_return", "label"]].copy()
        fold["score"] = scores
        results.append(fold)

    if not results:
        raise ValueError(
            "Nessun fold walk-forward prodotto: storico troppo corto per "
            f"initial_train={initial_train} + test_size={test_size}"
        )

    return pd.concat(results).sort_index()
