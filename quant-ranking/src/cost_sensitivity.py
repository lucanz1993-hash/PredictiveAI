"""Sensibilita' del segnale long/short small/mid-cap a 21gg ai costi di
transazione: rilancia il backtest con diversi cost_bps riusando lo stesso
`oos` (nessun retraining), per isolare l'effetto dei soli costi assunti.

Nota: modella solo costi di trading (spread/impact stimato via bps sul
turnover), non costi di prestito titoli per la gamba short -- per una
strategia long/short reale andrebbero aggiunti separatamente.
"""

import pandas as pd

from backtest import run_backtest
from report import OUTPUT_DIR, summarize
from run import compute_oos
from universe import UNIVERSES

TAG = "small_mid_cap_h21"
HORIZON = 21
COST_BPS_GRID = [7.5, 15, 25, 50]


def main():
    tickers = UNIVERSES["small_mid_cap"]
    oos = compute_oos(tickers, HORIZON, tag=f"{TAG}_cost_sensitivity")

    rows = []
    for cost_bps in COST_BPS_GRID:
        bt = run_backtest(oos, horizon=HORIZON, long_short=True, cost_bps=cost_bps)
        metrics = summarize(bt, HORIZON)
        rows.append(
            {
                "cost_bps": cost_bps,
                "sharpe": metrics["sharpe"],
                "total_return": metrics["total_return"],
                "annualized_return": metrics["annualized_return"],
                "max_drawdown": metrics["max_drawdown"],
            }
        )

    table = pd.DataFrame(rows).set_index("cost_bps")
    print("\n=== Sensibilita' ai costi di transazione (long/short, small_mid_cap_h21) ===")
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))

    OUTPUT_DIR.mkdir(exist_ok=True)
    table.to_csv(OUTPUT_DIR / "cost_sensitivity_small_mid_cap_h21.csv")


if __name__ == "__main__":
    main()
