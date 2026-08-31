# Probability Flow Momentum

An independent, causal implementation of a distribution-based momentum model,
designed for exploratory quantitative research in Python and Spyder.

The project studies how the empirical distribution of log returns changes over
time. It separates the **direction** of quantile transport from its
**intensity**, then evaluates raw and filtered signals through an explicit
multi-asset walk-forward procedure.

> This is an original research implementation. It is not the proprietary source
> code of the invite-only TradingView indicator STARK sigma.3, is not affiliated
> with its publisher, and does not claim to reproduce it exactly.

## Research components

- **Wasserstein distance** — magnitude of the change between a lagged reference
  return distribution and the current distribution.
- **Quantile Transport Direction (QTD)** — breadth of upward versus downward
  displacements along a symmetric quantile grid. QTD lies in `[-1, +1]` and is
  not multiplied by Wasserstein distance.
- **Tail asymmetry** — relative displacement of the upper and lower tails.
- **Transport velocity** — rate of change between consecutive empirical
  distributions.
- **RAW signals** — causal zero-crossings of QTD.
- **CONFIRMED signals** — RAW candidates filtered within a finite causal window.
- **STABLE / TRANSITION state** — a diagnostic hysteresis state, evaluated
  separately from directional signal performance.

## Confirmation policies

| Policy | Rule |
|---|---|
| `STRICT` | Direction + Wasserstein + aligned tail asymmetry + velocity |
| `TAIL_VETO` | Direction + Wasserstein + velocity; an opposing tail can veto |
| `SUPPORT_OR` | Direction + at least one supporting component; used as a permissive stress test |

## Walk-forward validation

The validation engine:

- imports the indicator engine instead of duplicating it;
- uses four expanding-window out-of-sample folds;
- selects threshold severity using training data only;
- evaluates SPY, QQQ, BTC-USD and EURUSD=X;
- enters at the next observed Open after a signal;
- measures directional returns at 1, 5, 10 and 20 bars;
- reports MAE, MFE, win rate, confirmation rate and confirmation lag;
- separates return availability from High/Low availability for MAE and MFE;
- retains all results in memory and does not export files automatically.

No signal is retroactively moved to an earlier bar. The last Yahoo Finance bar
is dropped by default because it may still be incomplete.

## Validation snapshot

The following pooled out-of-sample means were observed at the 20-bar horizon
using data available through August 2026 and **zero transaction costs**:

| Asset | RAW | STRICT | TAIL_VETO | SUPPORT_OR |
|---|---:|---:|---:|---:|
| SPY | +0.003% | -0.533% | -0.507% | +0.017% |
| QQQ | -0.044% | -1.960% | -1.868% | +0.064% |
| BTC-USD | -0.598% | +1.343% | +0.462% | -1.175% |
| EURUSD=X | -0.088% | +0.072% | +0.078% | -0.032% |

These results do **not** establish a general trading edge:

- severe confirmation policies deteriorated performance on SPY and QQQ;
- BTC-USD with `STRICT` was the most interesting hypothesis, but used a small
  and heterogeneous fold sample;
- small positive ETF/FX results may disappear after realistic costs;
- 94–95% of operational bars were classified as `TRANSITION`, making the state
  label weakly discriminative under the current thresholds;
- multi-bar outcomes can overlap, so this is an event study rather than the
  equity curve of a self-financing portfolio.

The negative and mixed findings are intentionally retained: the purpose of the
project is transparent validation, not optimisation toward an attractive
backtest.

## Repository files

- `probability_flow_spyder.py` — indicator engine, signals and Spyder chart.
- `probability_flow_validation_spyder.py` — multi-asset walk-forward validator.
- `requirements.txt` — tested Python dependencies.
- `.gitignore` — excludes local caches, environments and research datasets.

## Running the indicator in Spyder

1. Keep both Python files in the same directory.
2. Open `probability_flow_spyder.py` in Spyder.
3. In **CONFIGURAZIONE UTENTE**, select `demo`, `csv` or `yfinance`.
4. Press **F5**.

`demo` requires no internet connection. For a CSV input, provide genuine dates
and at least a `Close` column. The standalone chart can display close-only data,
but executable validation requires observed OHLC prices.

## Running the validation

1. Keep `probability_flow_validation_spyder.py` beside the engine file.
2. Review the tickers, horizons and cost assumption at the top of the file.
3. Press **F5** in Spyder.
4. Inspect the tables in the console and the returned objects in Variable
   Explorer.

Important returned objects include `SIGNAL_SUMMARY`, `PERFORMANCE_SUMMARY`,
`FOLD_PERFORMANCE`, `PERFORMANCE_BY_SIDE`, `STATE_SUMMARY` and
`VALIDATION_FIGURES`.

## Environment

Tested with Python 3.13.9, NumPy 2.3.5, pandas 2.3.3, Matplotlib 3.10.7 and
yfinance 1.5.2.

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Limitations

- Results depend on the data vendor and download date.
- Transaction costs default to zero and must be configured for economic tests.
- Multiple asset, policy and horizon comparisons increase selection risk.
- Formal inference should account for overlapping outcomes, for example with a
  block bootstrap or another dependence-robust method.
- Any policy selected after reviewing these results requires a new untouched
  holdout or a nested walk-forward design.
- This project is for research and educational purposes only and is not
  investment advice.

