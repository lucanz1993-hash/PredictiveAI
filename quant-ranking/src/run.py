"""Entrypoint: fetch dati -> feature -> label -> walk-forward -> backtest -> report.

`run_pipeline` e' parametrizzata per universo e orizzonte, cosi' da poter
essere richiamata piu' volte con configurazioni diverse (vedi `experiments.py`).
"""

from backtest import run_backtest
from data_fetch import fetch_universe
from edgar_fetch import fetch_universe_facts
from features import FACTOR_COLUMNS, compute_features
from fundamentals import FUNDAMENTAL_COLUMNS, compute_fundamental_features
from labels import HORIZON_DAYS, compute_labels
from model import run_walk_forward
from panel import build_panel, load_benchmark_returns
from report import OUTPUT_DIR, save_report
from universe import TICKERS


def compute_oos(tickers: list[str], horizon: int, tag: str) -> tuple:
    """Esegue fetch -> feature -> label -> training walk-forward, senza
    backtest. Riusabile per rilanciare il backtest con parametri diversi
    (es. costi di transazione) senza ripetere fetch/training."""
    print(f"\n=== Esperimento '{tag}': {len(tickers)} ticker, orizzonte {horizon}gg ===")

    print("1/5 - Download/caching dati storici (prezzi + bilanci SEC EDGAR)...")
    fetch_universe(tickers)
    facts_by_ticker = fetch_universe_facts(tickers)
    print(f"   Copertura EDGAR: {len(facts_by_ticker)}/{len(tickers)} ticker")

    print("2/5 - Costruzione panel e feature tecniche cross-sectional...")
    panel = build_panel(tickers)
    benchmark_returns = load_benchmark_returns()
    tech_features = compute_features(panel, benchmark_returns)

    print("3/5 - Costruzione feature value/quality (SEC EDGAR)...")
    fund_features = compute_fundamental_features(panel, facts_by_ticker)
    features = tech_features.join(fund_features, how="left")

    print("4/5 - Costruzione label...")
    labels = compute_labels(panel, horizon=horizon)

    dataset = features.join(labels, how="inner").dropna(subset=["forward_return", "label"])
    print(f"   Dataset: {len(dataset)} osservazioni, {dataset.index.get_level_values('date').nunique()} date")

    feature_columns = FACTOR_COLUMNS + FUNDAMENTAL_COLUMNS

    print("5/5 - Training walk-forward LightGBM...")
    oos, importance = run_walk_forward(dataset, feature_columns, horizon=horizon)

    print("\n--- Feature importance media (gain, walk-forward) ---")
    print(importance)
    OUTPUT_DIR.mkdir(exist_ok=True)
    importance.to_csv(OUTPUT_DIR / f"feature_importance_{tag}.csv", header=["importance"])

    return oos


def run_pipeline(tickers: list[str], horizon: int, tag: str) -> dict:
    oos = compute_oos(tickers, horizon, tag)

    print("\nBacktest e report...")
    print("\n== Long-only top decile ==")
    bt_long = run_backtest(oos, horizon=horizon, long_short=False)
    long_metrics = save_report(bt_long, horizon=horizon, name=f"backtest_long_{tag}")

    print("\n== Long/short (top decile vs bottom decile) ==")
    bt_long_short = run_backtest(oos, horizon=horizon, long_short=True)
    long_short_metrics = save_report(bt_long_short, horizon=horizon, name=f"backtest_long_short_{tag}")

    return {
        "tag": tag,
        "universe_size": len(tickers),
        "horizon": horizon,
        "long_sharpe": long_metrics["sharpe"],
        "long_total_return": long_metrics["total_return"],
        "long_short_sharpe": long_short_metrics["sharpe"],
        "long_short_total_return": long_short_metrics["total_return"],
        "benchmark_sharpe": long_metrics["benchmark_sharpe"],
        "benchmark_total_return": long_metrics["benchmark_total_return"],
    }


if __name__ == "__main__":
    run_pipeline(TICKERS, HORIZON_DAYS, tag="large_cap_h5")
