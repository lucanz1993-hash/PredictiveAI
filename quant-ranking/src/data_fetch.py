"""Download e caching locale dell'OHLCV giornaliero per l'universo, via yfinance."""

from pathlib import Path

import pandas as pd
import yfinance as yf

from universe import BENCHMARK_TICKER, TICKERS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HISTORY_PERIOD = "10y"


def _cache_path(ticker: str) -> Path:
    return DATA_DIR / f"{ticker}.parquet"


def fetch_ticker(ticker: str, force: bool = False) -> pd.DataFrame:
    cache_path = _cache_path(ticker)
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    df = yf.Ticker(ticker).history(period=HISTORY_PERIOD, auto_adjust=True)
    df = df[["Open", "High", "Low", "Close", "Volume", "Stock Splits"]]
    df.index.name = "date"
    df.to_parquet(cache_path)
    return df


def fetch_universe(tickers: list[str] = TICKERS, force: bool = False) -> dict[str, pd.DataFrame]:
    DATA_DIR.mkdir(exist_ok=True)
    data = {}
    all_tickers = tickers + [BENCHMARK_TICKER]
    for i, ticker in enumerate(all_tickers, start=1):
        print(f"[{i}/{len(all_tickers)}] {ticker}")
        try:
            df = fetch_ticker(ticker, force=force)
            if df.empty:
                print(f"  ...nessun dato per {ticker}, saltato")
                continue
            data[ticker] = df
        except Exception as exc:
            print(f"  ...errore su {ticker}: {exc}")
    return data


if __name__ == "__main__":
    fetched = fetch_universe()
    print(f"\nScaricati {len(fetched)} ticker su {len(TICKERS) + 1} in {DATA_DIR}")
