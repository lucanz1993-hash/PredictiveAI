"""Simulazione 'azienda': paper trading giornaliero del segnale small/mid-cap
a 21gg (il piu' robusto trovato in fase di ricerca), con libro contabile
virtuale persistito in coin.

Convenzioni:
- Nessun denaro reale: solo prezzi di mercato reali usati per calcolare
  P&L virtuale (coin).
- Mark-to-market GIORNALIERO (prezzo corrente vs prezzo di ingresso), ma le
  decisioni di trading (apertura/chiusura posizioni) avvengono solo ogni 21
  sessioni di borsa, per restare fedeli alla metodologia validata nel
  backtest -- ribilanciare piu' spesso "per avere qualcosa da riportare"
  romperebbe il motivo per cui il segnale funziona.
- Long e short pesano il 50% del capitale ciascuno (dollar-neutral),
  equal-weight sulle posizioni di ciascuna gamba, stessa costruzione usata
  in backtest.py -- cosi' la simulazione live e' metodologicamente identica
  a quanto validato, non un'invenzione diversa.
- Stipendio (20 coin/mese) e bonus (2% sul profitto di trading nei mesi in
  cui supera 1000 coin) sono tracciati come contatori a parte: non vengono
  sottratti dal capitale investito, per tenere pulito il segnale di
  performance del trading.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd

from backtest import COST_BPS, TOP_FRACTION
from data_fetch import fetch_universe
from edgar_fetch import fetch_universe_facts
from features import FACTOR_COLUMNS, compute_features
from fundamentals import FUNDAMENTAL_COLUMNS, compute_fundamental_features
from labels import compute_labels
from model import train_live_model
from panel import build_panel, load_benchmark_returns
from universe import UNIVERSES

COMPANY_DIR = Path(__file__).resolve().parent.parent / "company"
LEDGER_PATH = COMPANY_DIR / "ledger.json"
REPORT_LOG_PATH = COMPANY_DIR / "reports.log"

TICKERS = UNIVERSES["small_mid_cap"]
HORIZON = 21
INITIAL_CAPITAL = 5000.0
MONTHLY_SALARY = 20.0
MONTHLY_TARGET = 1000.0
BONUS_RATE = 0.02
FEATURE_COLUMNS = FACTOR_COLUMNS + FUNDAMENTAL_COLUMNS


def _today_str() -> str:
    return date.today().isoformat()


def _month_key(d: str) -> str:
    return d[:7]


def load_ledger() -> dict:
    if LEDGER_PATH.exists():
        return json.loads(LEDGER_PATH.read_text())
    return {
        "inception_date": _today_str(),
        "cash": INITIAL_CAPITAL,
        "positions": [],
        "sessions_since_rebalance": 999,  # forza un ribilanciamento al primo avvio
        "last_mark_date": None,
        "history": [],
        "salary_accrued": 0.0,
        "bonus_accrued": 0.0,
        "last_salary_month": None,
        "rebalance_log": [],
    }


def save_ledger(ledger: dict) -> None:
    COMPANY_DIR.mkdir(exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2, default=str))


def _position_value(position: dict, current_price: float) -> float:
    sign = 1.0 if position["side"] == "long" else -1.0
    return position["notional"] * (1.0 + sign * (current_price / position["entry_price"] - 1.0))


def _portfolio_value(ledger: dict, prices: pd.Series) -> float:
    value = ledger["cash"]
    for position in ledger["positions"]:
        price = prices.get(position["ticker"])
        if price is not None:
            value += _position_value(position, price)
        else:
            value += position["notional"]  # prezzo non disponibile oggi: valore invariato
    return value


def get_live_scores(panel: pd.DataFrame, facts_by_ticker: dict) -> tuple[pd.Series, pd.Timestamp]:
    benchmark_returns = load_benchmark_returns()
    tech_features = compute_features(panel, benchmark_returns)
    fund_features = compute_fundamental_features(panel, facts_by_ticker)
    features = tech_features.join(fund_features, how="left")

    labels = compute_labels(panel, horizon=HORIZON)
    dataset = features.join(labels, how="inner").dropna(subset=["forward_return", "label"])

    clf = train_live_model(dataset, FEATURE_COLUMNS)

    latest_date = features.index.get_level_values("date").max()
    latest_cross = features.xs(latest_date, level="date").dropna(subset=FEATURE_COLUMNS, how="all")
    scores = pd.Series(
        clf.predict_proba(latest_cross[FEATURE_COLUMNS])[:, 1],
        index=latest_cross.index,
    )
    return scores, latest_date


def rebalance(ledger: dict, scores: pd.Series, prices: pd.Series, rebalance_date: pd.Timestamp) -> str:
    capital = _portfolio_value(ledger, prices)

    prev_long = {p["ticker"] for p in ledger["positions"] if p["side"] == "long"}
    prev_short = {p["ticker"] for p in ledger["positions"] if p["side"] == "short"}

    n = len(scores)
    k = max(1, int(n * TOP_FRACTION))
    ranked = scores.sort_values(ascending=False)
    new_long = set(ranked.head(k).index)
    new_short = set(ranked.tail(k).index)

    long_turnover = len(new_long - prev_long) / k
    short_turnover = len(new_short - prev_short) / k
    cost = (capital * 0.5) * (COST_BPS / 10000) * (long_turnover + short_turnover)

    leg_capital = (capital - cost) * 0.5
    new_positions = []
    for ticker in new_long:
        new_positions.append({
            "ticker": ticker, "side": "long",
            "entry_price": float(prices[ticker]), "notional": leg_capital / k,
            "entry_date": rebalance_date.strftime("%Y-%m-%d"),
        })
    for ticker in new_short:
        new_positions.append({
            "ticker": ticker, "side": "short",
            "entry_price": float(prices[ticker]), "notional": leg_capital / k,
            "entry_date": rebalance_date.strftime("%Y-%m-%d"),
        })

    ledger["positions"] = new_positions
    ledger["cash"] = 0.0  # capitale interamente dispiegato long+short
    ledger["sessions_since_rebalance"] = 0
    ledger["rebalance_log"].append({
        "date": rebalance_date.strftime("%Y-%m-%d"),
        "capital_before": capital,
        "cost": cost,
        "long": sorted(new_long),
        "short": sorted(new_short),
    })

    return (
        f"RIBILANCIAMENTO ({rebalance_date.date()}): capitale {capital:.2f} coin, "
        f"costo transazione {cost:.2f} coin.\n"
        f"  Long ({k}): {', '.join(sorted(new_long))}\n"
        f"  Short ({k}): {', '.join(sorted(new_short))}"
    )


def completed_month_pnl(ledger: dict, month: str) -> float:
    entries = [e for e in ledger["history"] if _month_key(e["date"]) == month]
    if not entries:
        return 0.0
    end_value = entries[-1]["portfolio_value"]
    prior_entries = [e for e in ledger["history"] if _month_key(e["date"]) < month]
    start_value = prior_entries[-1]["portfolio_value"] if prior_entries else INITIAL_CAPITAL
    return end_value - start_value


def accrue_salary_and_bonus(ledger: dict, today: str) -> str:
    month = _month_key(today)
    if ledger["last_salary_month"] == month:
        return ""

    lines = []
    if ledger["last_salary_month"] is not None:
        prev_month = ledger["last_salary_month"]
        month_pnl = completed_month_pnl(ledger, prev_month)
        ledger["salary_accrued"] += MONTHLY_SALARY
        lines.append(f"Stipendio {prev_month}: +{MONTHLY_SALARY:.2f} coin accreditati (contatore separato).")
        if month_pnl > MONTHLY_TARGET:
            bonus = BONUS_RATE * month_pnl
            ledger["bonus_accrued"] += bonus
            lines.append(
                f"Target superato per {prev_month} (P&L {month_pnl:.2f} > {MONTHLY_TARGET:.0f}): "
                f"bonus +{bonus:.2f} coin (2% del P&L)."
            )
        else:
            lines.append(f"Target NON raggiunto per {prev_month} (P&L {month_pnl:.2f} < {MONTHLY_TARGET:.0f}): nessun bonus.")

    ledger["last_salary_month"] = month
    return "\n".join(lines)


def month_to_date_pnl(ledger: dict, today: str, current_value: float) -> float:
    month = _month_key(today)
    baseline = INITIAL_CAPITAL
    for entry in ledger["history"]:
        if _month_key(entry["date"]) == month:
            baseline = entry["portfolio_value"]
            break
    else:
        if ledger["history"]:
            baseline = ledger["history"][-1]["portfolio_value"]
    return current_value - baseline


def build_report(ledger: dict, today: str, current_value: float, rebalance_note: str, salary_note: str) -> str:
    total_pnl = current_value - INITIAL_CAPITAL
    mtd_pnl = month_to_date_pnl(ledger, today, current_value)
    prev_value = ledger["history"][-1]["portfolio_value"] if ledger["history"] else INITIAL_CAPITAL
    daily_pnl = current_value - prev_value

    lines = [
        f"=== Report {today} ===",
        f"Valore portafoglio: {current_value:.2f} coin (capitale iniziale {INITIAL_CAPITAL:.0f})",
        f"P&L da inizio: {total_pnl:+.2f} coin",
        f"P&L oggi: {daily_pnl:+.2f} coin",
        f"P&L mese in corso: {mtd_pnl:+.2f} coin (target {MONTHLY_TARGET:.0f})",
        f"Stipendio accumulato: {ledger['salary_accrued']:.2f} coin | Bonus accumulato: {ledger['bonus_accrued']:.2f} coin",
        f"Posizioni aperte: {len(ledger['positions'])} "
        f"({sum(1 for p in ledger['positions'] if p['side']=='long')} long / "
        f"{sum(1 for p in ledger['positions'] if p['side']=='short')} short)",
        f"Prossimo ribilanciamento tra {max(0, 21 - ledger['sessions_since_rebalance'])} sessioni di borsa",
    ]
    if salary_note:
        lines.append(salary_note)
    if rebalance_note:
        lines.append(rebalance_note)
    return "\n".join(lines)


def run_once() -> str:
    ledger = load_ledger()
    today = _today_str()
    is_first_run = ledger["last_mark_date"] is None

    # Passo 1 (economico): controlla solo i ticker gia' in portafoglio (o
    # l'intero universo al primo avvio, visto che non ci sono ancora
    # posizioni) per scoprire se c'e' una nuova sessione di borsa.
    probe_tickers = sorted({p["ticker"] for p in ledger["positions"]}) or TICKERS
    fetch_universe(probe_tickers, force=True)
    probe_panel = build_panel(probe_tickers)
    data_date = probe_panel.index.get_level_values("date").max()
    data_date_str = data_date.strftime("%Y-%m-%d")

    if ledger["last_mark_date"] == data_date_str:
        return f"Nessuna nuova sessione di borsa da {data_date_str}: nessun aggiornamento."

    if not is_first_run:
        ledger["sessions_since_rebalance"] += 1

    due_for_rebalance = is_first_run or ledger["sessions_since_rebalance"] >= 21

    rebalance_note = ""
    if due_for_rebalance:
        # Passo 2 (costoso, solo se serve davvero ribilanciare): rifetch
        # dell'intero universo + bilanci SEC EDGAR per generare i punteggi.
        fetch_universe(TICKERS, force=True)
        facts_by_ticker = fetch_universe_facts(TICKERS)
        panel = build_panel(TICKERS)
        scores, data_date = get_live_scores(panel, facts_by_ticker)
        prices = panel["close"].xs(data_date, level="date")
        rebalance_note = rebalance(ledger, scores, prices, data_date)
    else:
        prices = probe_panel["close"].xs(data_date, level="date")

    current_value = _portfolio_value(ledger, prices)
    salary_note = accrue_salary_and_bonus(ledger, today)

    ledger["history"].append({"date": today, "portfolio_value": current_value})
    ledger["last_mark_date"] = data_date_str

    report = build_report(ledger, today, current_value, rebalance_note, salary_note)
    save_ledger(ledger)

    COMPANY_DIR.mkdir(exist_ok=True)
    with open(REPORT_LOG_PATH, "a") as f:
        f.write(report + "\n\n")

    return report


if __name__ == "__main__":
    print(run_once())
