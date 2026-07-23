"""Fattori value/quality point-in-time, costruiti dai bilanci storici SEC EDGAR.

Ogni fact XBRL diventa "noto" solo a partire dalla sua data di deposito
(`filed`, +1 giorno di margine), non dalla data di fine periodo contabile
(`end`): questo evita di applicare retroattivamente informazioni che il
mercato non aveva ancora al tempo T (lookahead bias).

Per le voci di conto economico (flusso: utile, ricavi, utile lordo) si usa
solo la cadenza annuale (10-K/10-K/A), per evitare di dover ricostruire il
TTM da trimestrali con logiche di stitching aggiuntive. Le voci di stato
patrimoniale (equity, liabilities) e le azioni in circolazione vengono
aggiornate ad ogni deposito disponibile (10-K o 10-Q).
"""

import numpy as np
import pandas as pd

from features import cross_sectional_zscore

FUNDAMENTAL_COLUMNS = ["earnings_yield", "book_to_market", "roe", "debt_to_equity", "gross_margin"]

_FLOW_FORMS = {"10-K", "10-K/A"}


def _point_in_time_series(entries: list[dict], annual_only: bool) -> pd.Series:
    if not entries:
        return pd.Series(dtype=float)

    df = pd.DataFrame(entries)
    if annual_only:
        df = df[df["form"].isin(_FLOW_FORMS)]
        if {"start", "end"}.issubset(df.columns):
            duration = (pd.to_datetime(df["end"]) - pd.to_datetime(df["start"])).dt.days
            df = df[duration.between(300, 400)]

    if df.empty:
        return pd.Series(dtype=float)

    df = df.copy()
    df["filed"] = pd.to_datetime(df["filed"])
    df = df.sort_values("filed").drop_duplicates(subset="filed", keep="last")
    return df.set_index("filed")["val"].astype(float)


def _get_tag_series(facts: dict, taxonomy: str, tag_candidates: list[str], annual_only: bool) -> pd.Series:
    tags = facts.get("facts", {}).get(taxonomy, {})
    for tag in tag_candidates:
        entry = tags.get(tag)
        if entry is None:
            continue
        units = entry.get("units", {})
        raw_entries = next(iter(units.values()), [])
        series = _point_in_time_series(raw_entries, annual_only=annual_only)
        if not series.empty:
            return series
    return pd.Series(dtype=float)


def _forward_fill_to_dates(series: pd.Series, dates: pd.DatetimeIndex, lag_days: int = 1) -> pd.Series:
    if series.empty:
        return pd.Series(index=dates, dtype=float)

    # I prezzi hanno un indice timezone-aware (yfinance), i depositi SEC no:
    # si allinea tutto in spazio tz-naive e si rimappa l'indice originale alla fine.
    dates_naive = dates.tz_localize(None) if dates.tz is not None else dates
    shifted = series.copy()
    shifted.index = shifted.index + pd.Timedelta(days=lag_days)

    combined_index = dates_naive.union(shifted.index)
    filled = shifted.reindex(combined_index).sort_index().ffill().reindex(dates_naive)
    filled.index = dates
    return filled


def _split_adjustment_factor(stock_splits: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Fattore per convertire un numero di azioni "grezzo" (alla data di
    deposito) nel suo equivalente split-adjusted coerente con `close`
    (yfinance restituisce sempre prezzi gia' split-adjusted, a prescindere
    da `auto_adjust`, ma i valori SEC EDGAR sono le azioni realmente in
    circolazione a quella data, non aggiustate)."""
    splits = stock_splits[stock_splits != 0].sort_index()
    if splits.empty:
        return pd.Series(1.0, index=dates)

    dates_naive = dates.tz_localize(None) if dates.tz is not None else dates
    splits_naive = splits.copy()
    splits_naive.index = splits_naive.index.tz_localize(None) if splits_naive.index.tz is not None else splits_naive.index

    cum_upto = splits_naive.cumprod()
    total = cum_upto.iloc[-1]

    combined_index = dates_naive.union(cum_upto.index)
    filled = cum_upto.reindex(combined_index).sort_index().ffill().reindex(dates_naive).fillna(1.0)
    factor = total / filled
    factor.index = dates
    return factor


def _ticker_ratios(facts: dict, close: pd.Series, stock_splits: pd.Series) -> pd.DataFrame:
    dates = close.index

    net_income = _forward_fill_to_dates(
        _get_tag_series(facts, "us-gaap", ["NetIncomeLoss"], annual_only=True), dates
    )
    revenues = _forward_fill_to_dates(
        _get_tag_series(
            facts, "us-gaap",
            ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
            annual_only=True,
        ),
        dates,
    )
    gross_profit = _forward_fill_to_dates(
        _get_tag_series(facts, "us-gaap", ["GrossProfit"], annual_only=True), dates
    )
    equity = _forward_fill_to_dates(
        _get_tag_series(
            facts, "us-gaap",
            ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
            annual_only=False,
        ),
        dates,
    )
    liabilities = _forward_fill_to_dates(
        _get_tag_series(facts, "us-gaap", ["Liabilities"], annual_only=False), dates
    )
    # Il fattore di aggiustamento va calcolato alle date di deposito grezze
    # (poche, sparse) PRIMA del forward-fill: se venisse applicato ogni
    # giorno dopo aver "spalmato" lo stesso valore grezzo su molti giorni,
    # il conteggio azioni verrebbe ri-scalato ogni giorno rispetto a quanti
    # split restano "nel futuro" da quel giorno, invece che una volta sola
    # rispetto alla data del deposito — causando un salto artificiale
    # esattamente nel giorno dello split.
    shares_raw_at_filing = _get_tag_series(facts, "dei", ["EntityCommonStockSharesOutstanding"], annual_only=False)
    shares_adjusted_at_filing = shares_raw_at_filing * _split_adjustment_factor(stock_splits, shares_raw_at_filing.index)
    shares = _forward_fill_to_dates(shares_adjusted_at_filing, dates)

    market_cap = shares * close

    return pd.DataFrame(
        {
            "earnings_yield": net_income / market_cap,
            "book_to_market": equity / market_cap,
            "roe": net_income / equity,
            "debt_to_equity": liabilities / equity,
            "gross_margin": gross_profit / revenues,
        },
        index=dates,
    )


def compute_fundamental_features(panel: pd.DataFrame, facts_by_ticker: dict[str, dict]) -> pd.DataFrame:
    close = panel["close"].unstack("ticker").sort_index()
    stock_splits = panel["stock_splits"].unstack("ticker").sort_index()

    frames = []
    for ticker in close.columns:
        facts = facts_by_ticker.get(ticker)
        if facts is None:
            continue
        ticker_close = close[ticker].dropna()
        ratios = _ticker_ratios(facts, ticker_close, stock_splits[ticker].reindex(ticker_close.index).fillna(0))
        ratios["ticker"] = ticker
        frames.append(ratios)

    if not frames:
        return pd.DataFrame(columns=FUNDAMENTAL_COLUMNS)

    raw = pd.concat(frames)
    raw.index.name = "date"
    raw = raw.reset_index().set_index(["date", "ticker"])

    zscored_frames = []
    for col in FUNDAMENTAL_COLUMNS:
        wide = cross_sectional_zscore(raw[col].unstack("ticker"))
        long = wide.stack().rename(col)
        long.index.names = ["date", "ticker"]
        zscored_frames.append(long)

    # Niente dropna() qui: alcuni settori (es. banche) non riportano
    # GrossProfit e resterebbero NaN su gross_margin. Scartare l'intera riga
    # per un singolo fattore mancante eliminerebbe interi settori dal
    # dataset (verificato: solo 22/76 ticker avrebbero tutte e 5 le ratio
    # complete). LightGBM gestisce nativamente i NaN nelle feature.
    features = pd.concat(zscored_frames, axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features
