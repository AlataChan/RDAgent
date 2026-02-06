---
name: rdagent
description: Run Microsoft RD-Agent (rdagent) on Linux: health_check, data_science (custom datasets), fin_factor_report, and UI/log/session operations.
homepage: https://github.com/microsoft/RD-Agent
metadata:
  {
    "openclaw":
      { "emoji": "🧰", "os": ["linux"], "requires": { "bins": ["rdagent"] } },
  }
---

# RD-Agent (rdagent)

Use this skill when you need to run or operate Microsoft RD-Agent on a Linux host.

This skill focuses on:

- Safe setup checks (`rdagent health_check`)
- Working-directory rules (`.env` + `./log` live under the current directory)
- Running the most relevant scenarios (`data_science`, `fin_factor_report`)
- Monitoring runs (`rdagent ui`) and resuming sessions

## Non-negotiables

- **Always** `cd` into a dedicated run directory before invoking `rdagent`.
  - RD-Agent loads `.env` from the current directory (`load_dotenv(".env")`).
  - RD-Agent writes logs under `./log/<timestamp>/...` (relative to the current directory).
- Never print or commit API keys. Keep them only in the Linux host’s `.env` (or your secret manager).

## Quick start (safe)

```bash
mkdir -p ~/rdagent-runs/us-fundamentals-iv
cd ~/rdagent-runs/us-fundamentals-iv

# Create .env in *this* directory (templates: references/env.md)
$EDITOR .env

# Sanity checks
rdagent health_check --no-check-env
rdagent health_check
```

## Scenario recommendation (US fundamentals + options IV)

Prefer **`rdagent data_science`** with a custom dataset folder (see `references/data_science-custom-dataset.md` and `references/us-equities-fundamentals-options-iv.md`).

Reason: the built-in Qlib loops (`fin_quant`, `fin_factor`, `fin_model`, `fin_factor_report`) are wired to Qlib demo data (by default it initializes Qlib with `~/.qlib/qlib_data/cn_data`), so US equities + options signals usually require custom data prep.

## Data pipeline (Yahoo fundamentals + Yahoo options IV)

Use the scripts in `scripts/` to build a real (non-demo) dataset from:

- **Yahoo Finance** (via `yfinance`): price history, a fundamentals snapshot, and options chain implied volatility

Install deps in the same Python environment where you run `rdagent`:

```bash
python -m pip install -U yfinance pandas numpy
```

### Cloud persistence (recommended)

Pick a **persistent, writable** datasets root (use an **absolute** path on the Linux host). Then set skill env overrides in `~/.openclaw/openclaw.json` so OpenClaw runs always write/read the same files:

```json
{
  "skills": {
    "rdagent": {
      "env": {
        "DS_LOCAL_DATA_PATH": "/srv/openclaw/datasets/rdagent",
        "RDAGENT_COMPETITION": "us-fundamentals-iv",
        "RDAGENT_TICKERS_FILE": "/srv/openclaw/datasets/rdagent/source_data/us-fundamentals-iv/tickers.txt",
        "RDAGENT_FEATURES_FILE": "/srv/openclaw/datasets/rdagent/source_data/us-fundamentals-iv/features.csv"
      }
    }
  }
}
```

Create the tickers file once (one ticker per line):

```bash
mkdir -p /srv/openclaw/datasets/rdagent/source_data/us-fundamentals-iv
$EDITOR /srv/openclaw/datasets/rdagent/source_data/us-fundamentals-iv/tickers.txt
```

Collect daily features (run this every trading day; it appends rows):

```bash
python {baseDir}/scripts/collect_us_fundamentals_iv_features.py --tickers-file /srv/openclaw/datasets/rdagent/source_data/us-fundamentals-iv/tickers.txt --overwrite-date
```

Build a `data_science` “competition” folder (train/test + eval labels):

```bash
python {baseDir}/scripts/build_us_fundamentals_iv_competition.py --competition us-fundamentals-iv
```

Then run RD-Agent:

```bash
export DS_LOCAL_DATA_PATH="/srv/openclaw/datasets/rdagent"
export DS_SCEN="rdagent.scenarios.data_science.scen.DataScienceScen"
rdagent data_science --competition us-fundamentals-iv --loop-n 1
```

## Run: data_science (custom dataset)

Before running:

- Confirm dataset folder exists under `DS_LOCAL_DATA_PATH/<task-name>/`.
- Set `DS_SCEN` to `rdagent.scenarios.data_science.scen.DataScienceScen` (not `KaggleScen`) for custom datasets.

Typical run (limit cost/time early):

```bash
export DS_LOCAL_DATA_PATH="/abs/path/to/datasets-root"
export DS_SCEN="rdagent.scenarios.data_science.scen.DataScienceScen"

rdagent data_science --competition us-fundamentals-iv --loop-n 1
```

Monitor:

```bash
rdagent ui --port 19899 --data-science
```

## Run: fin_factor_report (PDF filings -> factor ideas)

If you have a folder of PDFs (e.g. SEC filings) and want the agent to extract factor ideas from them:

```bash
rdagent fin_factor_report --report-folder /path/to/reports --all-duration 2h
rdagent ui --port 19899 --log-dir ./log
```

Notes:

- This is Qlib-based and typically needs Docker.
- For US equities you’ll likely need to adapt the Qlib data/provider used by the RD-Agent environment.

## Resume / continue a run

Most scenarios support `--path` + `--step-n` / `--loop-n` to continue from an existing log session directory.

Pattern:

```bash
rdagent data_science --competition us-fundamentals-iv --path ./log/<timestamp>/__session__/1/0_propose --step-n 1
```

If you’re unsure where the session folder is, list the newest run dir:

```bash
ls -lt ./log | head
find ./log -maxdepth 3 -type d -name '__session__' | tail
```
