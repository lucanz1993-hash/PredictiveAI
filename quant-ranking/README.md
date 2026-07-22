# Quant Ranking

Progetto di ricerca quantitativa: modello di **ranking cross-sectional** su un
paniere di titoli large-cap USA. Invece di prevedere il prezzo assoluto di un
singolo titolo (segnale/rumore troppo basso), il modello stima quali titoli
del paniere sovraperformeranno la mediana del paniere nei prossimi N giorni,
usando fattori quantitativi classici (momentum, reversal, volatilita', volume)
e un classificatore LightGBM, validato con walk-forward temporale rigoroso.

**Nota**: progetto di ricerca/educativo. Il backtest storico non garantisce
risultati futuri; nessuna raccomandazione di investimento personalizzata.

## Setup

```
cd quant-ranking
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```
cd src
python run.py
```

Esegue in sequenza: download/caching prezzi (yfinance) -> feature engineering
-> costruzione label -> training walk-forward LightGBM -> backtest -> report
(metriche a console + equity curve ed export CSV in `../outputs/`).

## Struttura

- `src/universe.py` - paniere di titoli (statico, editabile)
- `src/data_fetch.py` - download/caching OHLCV in `data/` (parquet)
- `src/panel.py` - costruzione panel long-format (date, ticker)
- `src/features.py` - fattori cross-sectional (z-score per data)
- `src/labels.py` - target: forward return a N giorni vs mediana del paniere
- `src/model.py` - training walk-forward LightGBM (expanding window + purge)
- `src/backtest.py` - simulazione portafoglio long top-decile + costi
- `src/report.py` - metriche (Sharpe, drawdown, return) ed equity curve

## Prossimi passi possibili

- Ampliare/rivedere l'universo di titoli
- Aggiungere fattori (es. qualita', value, sentiment)
- Provare `long_short=True` in `backtest.py` (long top-decile / short bottom-decile)
- Confrontare con un vero buy-&-hold SPY invece del solo benchmark equal-weight
- Grid search leggero su iperparametri LightGBM, sempre dentro il walk-forward
