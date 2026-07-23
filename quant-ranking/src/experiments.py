"""Griglia di esperimenti: universo (large-cap vs small/mid-cap) x orizzonte
di previsione (5/10/21 giorni). Rilancia `run_pipeline` per ogni combinazione
e salva una tabella comparativa in outputs/.
"""

import pandas as pd

from report import OUTPUT_DIR
from run import run_pipeline
from universe import UNIVERSES

HORIZONS = [5, 10, 21]


def main():
    results = []
    for universe_name, tickers in UNIVERSES.items():
        for horizon in HORIZONS:
            tag = f"{universe_name}_h{horizon}"
            metrics = run_pipeline(tickers, horizon, tag=tag)
            results.append(metrics)

    comparison = pd.DataFrame(results).set_index("tag")
    OUTPUT_DIR.mkdir(exist_ok=True)
    comparison.to_csv(OUTPUT_DIR / "experiments_comparison.csv")

    print("\n\n=== Confronto esperimenti ===")
    print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
