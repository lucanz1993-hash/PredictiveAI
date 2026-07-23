"""Panieri statici di titoli USA liquidi, usati come universo cross-sectional."""

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ORCL", "CRM",
    "ADBE", "AMD", "CSCO", "INTC", "IBM", "QCOM", "TXN", "INTU", "NOW", "AMAT",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "SCHW", "BLK", "SPGI",
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    "PG", "KO", "PEP", "COST", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "CAT", "BA", "HON", "UPS", "GE", "LMT", "RTX", "DE",
    "DIS", "NFLX", "CMCSA", "T", "VZ",
    "LIN", "APD", "NEE", "DUK", "SO",
    "V", "MA", "PYPL",
]

# Paniere small/mid-cap: societa' USA quotate da tempo (storico decennale),
# liquide ma con capitalizzazione tipicamente 2-20 mld$, quindi con minore
# copertura di analisti/algoritmi rispetto alle large-cap sopra -- terreno
# dove un edge factor-based ha statisticamente piu' probabilita' di esistere.
SMALL_MID_CAP_TICKERS = [
    "RF", "KEY", "CMA", "ZION", "HBAN", "CFG", "FITB", "SNV", "WTFC", "RJF",
    "TXT", "SNA", "DOV", "XYL", "PNR", "WAB", "PWR", "IEX", "CR", "ITT",
    "ZBRA", "TER", "TRMB", "JNPR", "NTAP", "WDC", "STX", "FFIV", "JBL", "CDW",
    "MOH", "DVA", "HOLX", "TFX", "CHE", "PODD", "ICUI", "MASI", "STE", "BAX",
    "RL", "DECK", "CROX", "BURL", "WSM", "DKS", "TXRH", "ULTA", "AEO",
    "MOS", "CF", "STLD", "RS", "CLF", "DVN", "MRO", "APA", "ALB", "OVV",
]

BENCHMARK_TICKER = "SPY"

UNIVERSES = {
    "large_cap": TICKERS,
    "small_mid_cap": SMALL_MID_CAP_TICKERS,
}
