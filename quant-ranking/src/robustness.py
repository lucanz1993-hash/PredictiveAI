"""Approfondimento del segnale long/short small/mid-cap a 21gg (il piu'
promettente della griglia esperimenti): bootstrap dell'incertezza
sullo Sharpe (solo 70 periodi non sovrapposti disponibili) e verifica di
tenuta su due sotto-periodi cronologici distinti.
"""

import numpy as np
import pandas as pd

from report import OUTPUT_DIR, summarize

N_BOOTSTRAP = 10000
HORIZON = 21
TAG = "small_mid_cap_h21"


def _periods_per_year(horizon: int) -> float:
    return 252 / horizon


def bootstrap_sharpe(returns: pd.Series, horizon: int, n_boot: int = N_BOOTSTRAP, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    values = returns.to_numpy()
    n = len(values)
    ppy = _periods_per_year(horizon)

    sharpes = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        std = sample.std()
        sharpes[i] = (sample.mean() / std * np.sqrt(ppy)) if std > 0 else 0.0

    point_std = values.std()
    return {
        "point_estimate": (values.mean() / point_std * np.sqrt(ppy)) if point_std > 0 else 0.0,
        "ci_5": np.percentile(sharpes, 5),
        "ci_95": np.percentile(sharpes, 95),
        "prob_positive": (sharpes > 0).mean(),
    }


def split_sub_periods(bt: pd.DataFrame, horizon: int, n_splits: int = 2) -> list[dict]:
    chunk_positions = np.array_split(np.arange(len(bt)), n_splits)
    results = []
    for positions in chunk_positions:
        chunk = bt.iloc[positions]
        metrics = summarize(chunk, horizon)
        metrics["period"] = f"{chunk.index[0].date()} -> {chunk.index[-1].date()}"
        metrics["n_periods"] = len(chunk)
        results.append(metrics)
    return results


def main():
    path = OUTPUT_DIR / f"backtest_long_short_{TAG}_returns.csv"
    bt = pd.read_csv(path, index_col="date")
    bt.index = pd.to_datetime(bt.index, utc=True)

    print(f"=== Bootstrap Sharpe long/short ({TAG}, n={len(bt)} periodi, {N_BOOTSTRAP} resample) ===")
    boot = bootstrap_sharpe(bt["portfolio_return"], HORIZON)
    print(f"Sharpe puntuale: {boot['point_estimate']:.3f}")
    print(f"Intervallo di confidenza 90%: [{boot['ci_5']:.3f}, {boot['ci_95']:.3f}]")
    print(f"P(Sharpe bootstrap > 0): {boot['prob_positive']:.1%}")

    print(f"\n=== Tenuta su sotto-periodi ({TAG}) ===")
    for metrics in split_sub_periods(bt, HORIZON, n_splits=2):
        print(f"\n{metrics['period']} (n={metrics['n_periods']} periodi)")
        print(f"  total_return: {metrics['total_return']:.4f}")
        print(f"  sharpe: {metrics['sharpe']:.4f}")
        print(f"  max_drawdown: {metrics['max_drawdown']:.4f}")
        print(f"  benchmark_sharpe: {metrics['benchmark_sharpe']:.4f}")
        print(f"  benchmark_total_return: {metrics['benchmark_total_return']:.4f}")


if __name__ == "__main__":
    main()
