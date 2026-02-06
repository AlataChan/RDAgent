# data_science with a custom dataset (not Kaggle)

Use `rdagent data_science` + `DataScienceScen` when you have your own dataset (US equities, options, fundamentals, etc.).

## Required env vars

```bash
export DS_SCEN="rdagent.scenarios.data_science.scen.DataScienceScen"
export DS_LOCAL_DATA_PATH="/abs/path/to/datasets-root"
```

`--competition` is effectively your task name (folder name under `DS_LOCAL_DATA_PATH`).

## Expected folder structure (minimum)

Inside `DS_LOCAL_DATA_PATH`:

```
source_data/
  <task-name>/
    prepare.py        # optional: turn raw data into train/test/submission + eval labels
<task-name>/
  description.md      # required
  train.csv           # typical tabular tasks
  test.csv
  sample_submission.csv
eval/                 # optional, but recommended (offline test scoring)
  <task-name>/
    submission_test.csv
    grade.py          # optional: compute score from submission + labels
    valid.py          # optional: validate submission format
```

You don’t have to use `train.csv`/`test.csv` if your task is non-tabular (e.g. time series). The pipeline mainly needs:

- `description.md` describing the objective + files + label column(s)
- a consistent way to evaluate model outputs (either built-in metric inference or `eval/<task>/grade.py`)

## Minimal run

```bash
rdagent data_science --competition <task-name> --loop-n 1
```

## Monitor logs

```bash
rdagent ui --port 19899 --data-science
```

## Helper scripts (this skill)

If you want a ready-made, real-data example for US equities:

- `scripts/collect_us_fundamentals_iv_features.py`: builds/updates a daily `features.csv` (Yahoo fundamentals + options IV via `yfinance`)
- `scripts/build_us_fundamentals_iv_competition.py`: turns `features.csv` into a `data_science` custom dataset folder under `DS_LOCAL_DATA_PATH`
