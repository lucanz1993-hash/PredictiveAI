"""Download e caching locale dei bilanci storici via SEC EDGAR XBRL company facts API.

Nessuna API key richiesta; la SEC chiede solo uno User-Agent identificativo
e un uso ragionevole della propria API pubblica (piccola pausa fra le
richieste).
"""

import json
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "edgar"
USER_AGENT = "quant-ranking-research luca.nz1993@gmail.com"
HEADERS = {"User-Agent": USER_AGENT}
REQUEST_DELAY_SECONDS = 0.2

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def get_cik_map(force: bool = False) -> dict[str, str]:
    cache_path = DATA_DIR / "company_tickers.json"
    if cache_path.exists() and not force:
        raw = json.loads(cache_path.read_text())
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        resp = requests.get(TICKERS_URL, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        raw = resp.json()
        cache_path.write_text(json.dumps(raw))

    return {entry["ticker"]: str(entry["cik_str"]).zfill(10) for entry in raw.values()}


def fetch_company_facts(ticker: str, cik_map: dict[str, str] | None = None, force: bool = False) -> dict | None:
    cache_path = DATA_DIR / f"{ticker}.json"
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text())

    cik_map = cik_map or get_cik_map()
    cik = cik_map.get(ticker)
    if cik is None:
        print(f"  ...nessun CIK trovato per {ticker}, saltato")
        return None

    resp = requests.get(COMPANY_FACTS_URL.format(cik=cik), headers=HEADERS, timeout=30)
    time.sleep(REQUEST_DELAY_SECONDS)
    if resp.status_code == 404:
        print(f"  ...nessun XBRL company facts per {ticker} (CIK {cik})")
        return None
    resp.raise_for_status()

    facts = resp.json()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(facts))
    return facts


def fetch_universe_facts(tickers: list[str], force: bool = False) -> dict[str, dict]:
    cik_map = get_cik_map(force=force)
    facts_by_ticker = {}
    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] {ticker} (EDGAR)")
        try:
            facts = fetch_company_facts(ticker, cik_map=cik_map, force=force)
            if facts is not None:
                facts_by_ticker[ticker] = facts
        except Exception as exc:
            print(f"  ...errore su {ticker}: {exc}")
    return facts_by_ticker


if __name__ == "__main__":
    from universe import TICKERS

    fetched = fetch_universe_facts(TICKERS)
    print(f"\nScaricati company facts per {len(fetched)}/{len(TICKERS)} ticker in {DATA_DIR}")
