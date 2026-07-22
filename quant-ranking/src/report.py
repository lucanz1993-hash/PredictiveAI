"""Metriche di performance ed equity curve, salvate in outputs/."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def _periods_per_year(horizon: int) -> float:
    return 252 / horizon


def _sharpe(returns: pd.Series, horizon: int) -> float:
    if returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(_periods_per_year(horizon))


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return drawdown.min()


def summarize(bt: pd.DataFrame, horizon: int) -> dict:
    equity = (1 + bt["portfolio_return"]).cumprod()
    bench_equity = (1 + bt["benchmark_return"]).cumprod()
    n_years = len(bt) / _periods_per_year(horizon)

    return {
        "n_periods": len(bt),
        "total_return": equity.iloc[-1] - 1,
        "annualized_return": equity.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else float("nan"),
        "sharpe": _sharpe(bt["portfolio_return"], horizon),
        "max_drawdown": _max_drawdown(equity),
        "benchmark_total_return": bench_equity.iloc[-1] - 1,
        "benchmark_sharpe": _sharpe(bt["benchmark_return"], horizon),
    }


def save_report(bt: pd.DataFrame, horizon: int, name: str = "backtest") -> dict:
    OUTPUT_DIR.mkdir(exist_ok=True)
    metrics = summarize(bt, horizon)

    print("\n--- Risultati backtest ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    equity = (1 + bt["portfolio_return"]).cumprod()
    bench_equity = (1 + bt["benchmark_return"]).cumprod()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity.index, equity.values, label="Modello (top decile)")
    ax.plot(bench_equity.index, bench_equity.values, label="Benchmark equal-weight universo")
    ax.set_title("Equity curve")
    ax.set_ylabel("Valore (base 1.0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_equity_curve.png")
    plt.close(fig)

    bt.to_csv(OUTPUT_DIR / f"{name}_returns.csv")
    print(f"\nSalvati risultati in: {OUTPUT_DIR}")
    return metrics
