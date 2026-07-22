"""Entrypoint: fetch dati -> feature -> label -> walk-forward -> backtest -> report."""

from backtest import run_backtest
from data_fetch import fetch_universe
from features import compute_features
from labels import HORIZON_DAYS, compute_labels
from model import run_walk_forward
from panel import build_panel
from report import save_report
from universe import TICKERS


def main():
    print("1/5 - Download/caching dati storici...")
    fetch_universe()

    print("2/5 - Costruzione panel e feature cross-sectional...")
    panel = build_panel(TICKERS)
    features = compute_features(panel)

    print("3/5 - Costruzione label...")
    labels = compute_labels(panel, horizon=HORIZON_DAYS)

    dataset = features.join(labels, how="inner").dropna()
    print(f"   Dataset: {len(dataset)} osservazioni, {dataset.index.get_level_values('date').nunique()} date")

    print("4/5 - Training walk-forward LightGBM...")
    oos = run_walk_forward(dataset, horizon=HORIZON_DAYS)

    print("5/5 - Backtest e report...")
    bt = run_backtest(oos, horizon=HORIZON_DAYS)
    save_report(bt, horizon=HORIZON_DAYS)


if __name__ == "__main__":
    main()
