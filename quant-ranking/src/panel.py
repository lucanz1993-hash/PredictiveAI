"""Costruzione del panel long-format (date, ticker) -> OHLCV a partire dalla cache locale."""

import pandas as pd

from data_fetch import fetch_ticker
from universe import TICKERS


def build_panel(tickers: list[str] = TICKERS) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        df = fetch_ticker(ticker)
        if df.empty:
            continue
        df = df.rename(columns=str.lower)
        df["ticker"] = ticker
        frames.append(df)

    panel = pd.concat(frames)
    panel.index.name = "date"
    panel = panel.reset_index().set_index(["date", "ticker"]).sort_index()
    return panel
