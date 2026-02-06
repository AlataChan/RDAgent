# US equities: fundamentals + options implied volatility (RD-Agent framing)

This is a good fit for `rdagent data_science` because you can define your own dataset + scoring.

## Recommended task definition

Pick **one** clear prediction target:

- Cross-sectional next-`N`-day return (excess vs SPY) regression
- Cross-sectional next-`N`-day volatility regression
- Direction (up/down) classification

Keep the first iteration simple and tabular (one row per `date,ticker`).

## Avoid lookahead (critical)

Fundamentals are point-in-time:

- Use **filing/publication timestamps** (e.g., 10-Q/10-K filing date), not quarter-end values.
- When joining fundamentals to daily rows, use an `asof` join so each `date` only sees data available **on or before** that date.

Options IV is also point-in-time:

- Use the chain snapshot from the same day (or prior day) only.
- If you compute “IV rank over 1y”, ensure the window only uses historical days.

## Data sources (practical)

### Yahoo end-to-end (prototype)

Yahoo Finance (via `yfinance`) can cover the whole prototype loop:

- **Prices**:
  - `yf.download([...], start=..., end=...)` or `Ticker.history(...)`
- **Fundamentals snapshot** (convenient but not point-in-time):
  - `Ticker.get_info()` / `Ticker.info` (fields like `marketCap`, `trailingPE`, `priceToBook`, margins, growth, etc.)
- **Options chain IV snapshot**:
  - `Ticker.options` + `Ticker.option_chain(expiry)` (use `impliedVolatility`)

Important limitations:

- Yahoo options data is effectively a **current snapshot**. There is no robust, free historical US per-stock options chain via Yahoo; to get history you must **collect snapshots daily going forward**.
- Yahoo fundamentals via `info` are also a **current snapshot** (not filing-timestamped). For strict point-in-time backtests, replace fundamentals with sources that include **publication timestamps**.

### Upgrade paths (when you need point-in-time and stability)

- **Point-in-time fundamentals**:
  - SEC EDGAR (XBRL + filing dates), or
  - AkShare US financial indicators that include `NOTICE_DATE` for as-of joins
- **Options IV history / reliable chains**:
  - A paid options API/data vendor (Polygon/Intrinio/Tradier/ORATS/ThetaData/OptionMetrics, etc.)

## Feature suggestions (starter set)

Fundamentals (examples):

- Value: `P/E`, `P/B`, `EV/EBITDA`
- Quality: `ROE`, `gross_margin`, `operating_margin`
- Growth: `revenue_growth`, `earnings_growth`
- Leverage/liquidity: `debt_to_equity`, `current_ratio`

Options IV (examples):

- ATM IV (e.g., 30D)
- Term structure: `IV_30D - IV_90D`
- Skew: `IV_put_25d - IV_call_25d`
- IV rank / percentile (rolling)
- Realized vs implied spread: `IV_30D - RV_20D`

Controls:

- Sector/industry (one-hot or embeddings)
- Market cap, price, turnover

## Data packaging (practical)

Treat this as a “competition” named `us-fundamentals-iv`:

- `DS_LOCAL_DATA_PATH/us-fundamentals-iv/description.md`: objective, leakage rules, columns
- `train.csv`: includes `target` column (e.g., `fwd_return_21d`)
- `test.csv`: same columns without `target`
- `sample_submission.csv`: `id` + `target` columns (or whatever you define)
- `eval/us-fundamentals-iv/submission_test.csv`: test labels for offline scoring (optional but recommended)

Then run:

```bash
export DS_LOCAL_DATA_PATH="/abs/path/to/datasets-root"
export DS_SCEN="rdagent.scenarios.data_science.scen.DataScienceScen"
rdagent data_science --competition us-fundamentals-iv --loop-n 1
```
