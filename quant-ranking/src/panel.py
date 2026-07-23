"""Costruzione del panel long-format (date, ticker) -> OHLCV a partire dalla cache locale."""

import pandas as pd

from data_fetch import fetch_ticker
from universe import BENCHMARK_TICKER, TICKERS


def build_panel(tickers: list[str] = TICKERS) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        try:
            df = fetch_ticker(ticker)
        except Exception as exc:
            print(f"  ...{ticker} non disponibile per il panel, saltato ({exc})")
            continue
        if df.empty:
            continue
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                                 "Volume": "volume", "Stock Splits": "stock_splits"})
        df["ticker"] = ticker
        frames.append(df)

    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.reset_index().set_index(["date", "ticker"]).sort_index()
    return panel


def load_benchmark_returns(ticker: str = BENCHMARK_TICKER) -> pd.Series:
    df = fetch_ticker(ticker)
    close = df["Close"].rename("close")
    close.index.name = "date"
    return close.pct_change()
