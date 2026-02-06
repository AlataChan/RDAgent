#!/usr/bin/env python3
"""
Build an RD-Agent `data_science` custom dataset ("competition") folder from a features CSV.

Input: a CSV produced by `collect_us_fundamentals_iv_features.py` with one row per (date, ticker).
Output: DS_LOCAL_DATA_PATH/<competition>/{train.csv,test.csv,sample_submission.csv,description.md}
        DS_LOCAL_DATA_PATH/eval/<competition>/submission_test.csv
        DS_LOCAL_DATA_PATH/source_data/<competition>/features.csv (copy)

Target: forward return over N trading days (default: 21) computed from Yahoo close prices.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def expand_path(value: str) -> Path:
    return Path(value).expanduser()


def resolve_out_root(raw: Optional[Path]) -> Path:
    if raw is not None:
        return raw.expanduser()
    env = (os.environ.get("DS_LOCAL_DATA_PATH") or "").strip()
    if env:
        return expand_path(env)
    return Path.home() / ".openclaw" / "datasets" / "rdagent"


def resolve_features_file(out_root: Path, competition: str) -> Path:
    env = (os.environ.get("RDAGENT_FEATURES_FILE") or "").strip()
    if env:
        return expand_path(env)
    return out_root / "source_data" / competition / "features.csv"


def build_description(competition: str, horizon: int) -> str:
    return "\n".join(
        [
            f"# {competition}",
            "",
            "US equities fundamentals + options IV (Yahoo Finance snapshot features).",
            "",
            "## Objective",
            f"Predict **{horizon}-trading-day forward return** for each (date, ticker) row.",
            "",
            "## Files",
            "- `train.csv`: features + `target`",
            "- `test.csv`: features only",
            "- `sample_submission.csv`: `id,target` template for `test.csv`",
            f"- `eval/{competition}/submission_test.csv`: labels for `test.csv` (offline scoring)",
            "",
            "## Target",
            "- Column: `target`",
            f"- Definition: `target = close[t+{horizon}] / close[t] - 1` (Yahoo close, trading-day shift)",
            "",
            "## Leakage rules (important)",
            "- Do not use any data after `date` when training.",
            "- Provided fundamentals/options are Yahoo snapshots; for strict point-in-time research you should replace them with filing-timestamped data.",
            "",
        ]
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    default_competition = (os.environ.get("RDAGENT_COMPETITION") or "us-fundamentals-iv").strip()
    if not default_competition:
        default_competition = "us-fundamentals-iv"

    parser = argparse.ArgumentParser(description="Build RD-Agent custom dataset from features CSV.")
    parser.add_argument(
        "--competition",
        default=default_competition,
        help="Competition/task name (folder name) (env: RDAGENT_COMPETITION).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        help="Datasets root (default: DS_LOCAL_DATA_PATH or ~/.openclaw/datasets/rdagent).",
    )
    parser.add_argument(
        "--features-file",
        type=Path,
        help="Input features.csv (default: RDAGENT_FEATURES_FILE or <out-root>/source_data/<competition>/features.csv).",
    )
    parser.add_argument("--horizon", type=int, default=21, help="Forward return horizon in trading days (default: 21)")
    parser.add_argument(
        "--test-dates",
        type=int,
        default=20,
        help="Hold out the last N labeled dates as test (default: 20)",
    )
    parser.add_argument(
        "--min-price-days",
        type=int,
        default=260,
        help="Days of price history to download for target calculation (default: 260)",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    competition = str(args.competition).strip() if args.competition else default_competition
    if not competition:
        competition = "us-fundamentals-iv"

    out_root = resolve_out_root(args.out_root)
    features_file = (args.features_file.expanduser() if args.features_file else resolve_features_file(out_root, competition)).expanduser()

    if not features_file.exists():
        eprint(f"Features file not found: {features_file}")
        return 2

    try:
        import pandas as pd  # type: ignore
        import yfinance as yf  # type: ignore
    except Exception as exc:
        eprint("Missing dependencies. Install with: python -m pip install -U yfinance pandas")
        eprint(str(exc))
        return 2

    features = pd.read_csv(features_file)
    if features.empty:
        eprint("Features CSV is empty.")
        return 2

    if "date" not in features.columns or "ticker" not in features.columns:
        eprint("Features CSV must include columns: date, ticker")
        return 2

    features["date"] = pd.to_datetime(features["date"], errors="coerce").dt.normalize()
    features = features.dropna(subset=["date", "ticker"]).copy()
    features["ticker"] = features["ticker"].astype(str)

    # Dedup (keep last row for each date,ticker).
    features = features.sort_values(by=["date", "ticker"], kind="stable")
    features = features.drop_duplicates(subset=["date", "ticker"], keep="last").copy()

    tickers = sorted(set(features["ticker"].tolist()))
    if not tickers:
        eprint("No tickers found in features.")
        return 2

    min_date = features["date"].min()
    max_date = features["date"].max()
    if min_date is None or max_date is None:
        eprint("Failed to determine date range from features.")
        return 2

    horizon = int(args.horizon)
    if horizon <= 0:
        eprint("--horizon must be > 0")
        return 2

    start = (min_date - pd.Timedelta(days=int(args.min_price_days))).date()
    end = (max_date + pd.Timedelta(days=int(horizon * 4 + 10))).date()

    eprint(f"Downloading price history from Yahoo: {start} -> {end} (tickers: {len(tickers)})")
    try:
        prices = yf.download(
            tickers=tickers,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as exc:
        eprint(f"yfinance download failed: {exc}")
        return 2

    if prices is None or prices.empty:
        eprint("No price data returned from Yahoo.")
        return 2

    targets: List[Tuple[Any, str, float]] = []

    def extract_close(df: Any, ticker: str) -> Optional[Any]:
        if df is None or getattr(df, "empty", True):
            return None

        # Single ticker: flat columns.
        if not isinstance(getattr(df, "columns", None), pd.MultiIndex):
            return df["Close"] if "Close" in df.columns else None

        cols = df.columns
        lvl0 = set(cols.get_level_values(0))
        lvl1 = set(cols.get_level_values(1))

        # group_by="ticker": (ticker, field)
        if ticker in lvl0 and "Close" in lvl1:
            try:
                sub = df[ticker]
                if getattr(sub, "empty", True):
                    return None
                return sub["Close"] if "Close" in sub.columns else None
            except Exception:
                pass

        # group_by="column": (field, ticker)
        if "Close" in lvl0 and ticker in lvl1:
            try:
                sub = df["Close"]
                if getattr(sub, "empty", True):
                    return None
                return sub[ticker] if ticker in sub.columns else None
            except Exception:
                pass

        return None

    # If single ticker, yfinance may return flat columns.
    if len(tickers) == 1 and isinstance(prices, pd.DataFrame) and "Close" in prices.columns:
        series = prices["Close"].copy()
        df_t = pd.DataFrame({"date": series.index.normalize(), "ticker": tickers[0], "close_yahoo": series.values})
        df_t["future_close"] = df_t["close_yahoo"].shift(-horizon)
        df_t["target"] = df_t["future_close"] / df_t["close_yahoo"] - 1.0
        df_t = df_t.dropna(subset=["target"])
        for row in df_t[["date", "ticker", "target"]].itertuples(index=False):
            targets.append((row.date, row.ticker, float(row.target)))
    else:
        for ticker in tickers:
            try:
                close_series = extract_close(prices, ticker)
                if close_series is None:
                    eprint(f"[{ticker}] Missing close series in downloaded data; skipping.")
                    continue
                close_series = close_series.dropna()
                if close_series.empty:
                    continue
                df_t = pd.DataFrame({"date": close_series.index.normalize(), "ticker": ticker, "close_yahoo": close_series.values})
                df_t["future_close"] = df_t["close_yahoo"].shift(-horizon)
                df_t["target"] = df_t["future_close"] / df_t["close_yahoo"] - 1.0
                df_t = df_t.dropna(subset=["target"])
                for row in df_t[["date", "ticker", "target"]].itertuples(index=False):
                    targets.append((row.date, row.ticker, float(row.target)))
            except Exception as exc:
                eprint(f"[{ticker}] Failed to compute targets: {exc}")

    if not targets:
        eprint("No targets computed. Need more price history and/or more rows in features.")
        return 2

    targets_df = pd.DataFrame(targets, columns=["date", "ticker", "target"])
    merged = features.merge(targets_df, on=["date", "ticker"], how="left")
    merged = merged.dropna(subset=["target"]).copy()

    if merged.empty:
        eprint("All rows missing target after merge.")
        return 2

    # Build stable id if missing.
    if "id" not in merged.columns:
        merged["id"] = merged["date"].dt.strftime("%Y-%m-%d") + "_" + merged["ticker"].astype(str)

    unique_dates = merged["date"].dropna().sort_values().unique().tolist()
    if len(unique_dates) <= int(args.test_dates):
        eprint(f"Not enough labeled dates ({len(unique_dates)}) to hold out test-dates={args.test_dates}.")
        return 2

    test_dates = set(unique_dates[-int(args.test_dates) :])
    is_test = merged["date"].isin(list(test_dates))

    train = merged[~is_test].copy()
    test_labeled = merged[is_test].copy()

    # Drop columns that are usually noise for modeling, but keep date/ticker/id.
    drop_cols = [c for c in ["snapshot_ts"] if c in train.columns]
    if drop_cols:
        train = train.drop(columns=drop_cols)
        test_labeled = test_labeled.drop(columns=drop_cols)

    # Output dirs
    # Output root resolved above (supports DS_LOCAL_DATA_PATH defaults).
    comp_dir = out_root / competition
    eval_dir = out_root / "eval" / competition
    source_dir = out_root / "source_data" / competition

    comp_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    # Train/test CSVs
    train_path = comp_dir / "train.csv"
    test_path = comp_dir / "test.csv"
    sample_path = comp_dir / "sample_submission.csv"
    desc_path = comp_dir / "description.md"
    labels_path = eval_dir / "submission_test.csv"
    source_features_path = source_dir / "features.csv"

    # Align test.csv columns with train.csv (minus target).
    feature_cols = [c for c in train.columns if c != "target"]
    test = test_labeled[feature_cols].copy()

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    sample = test[["id"]].copy()
    sample["target"] = 0.0
    sample.to_csv(sample_path, index=False)

    labels = test_labeled[["id", "target"]].copy()
    labels.to_csv(labels_path, index=False)

    desc_path.write_text(build_description(competition=competition, horizon=horizon), encoding="utf-8")

    # Copy raw features
    try:
        if source_features_path.resolve() != features_file.resolve():
            source_features_path.write_text(features_file.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception as exc:
        eprint(f"Warning: failed to copy features into source_data: {exc}")

    eprint(f"Wrote: {train_path} ({len(train)} rows)")
    eprint(f"Wrote: {test_path} ({len(test)} rows)")
    eprint(f"Wrote: {labels_path} ({len(labels)} rows)")
    eprint(f"Wrote: {desc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
