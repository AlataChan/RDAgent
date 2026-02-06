#!/usr/bin/env python3
"""
Collect a daily feature snapshot for US equities using Yahoo Finance via yfinance.

Features (starter set):
- Price/volume + realized vol (20D)
- Fundamental snapshot fields from Yahoo (Ticker info)
- Options chain implied vol features (ATM ~30D, skew, term structure)

This writes/updates a single CSV with one row per (date, ticker). The `date` is the
last available trading day close from Yahoo.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore


def eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def parse_iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}'. Expected YYYY-MM-DD.") from exc


def normalize_yahoo_ticker(value: str) -> str:
    return value.strip().replace(".", "-")


def read_tickers_file(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8")
    out: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def expand_path(value: str) -> Path:
    return Path(value).expanduser()


def resolve_datasets_root() -> Path:
    raw = os.environ.get("DS_LOCAL_DATA_PATH")
    if raw:
        return expand_path(raw)
    return Path.home() / ".openclaw" / "datasets" / "rdagent"


def resolve_default_features_file(competition: str) -> Path:
    raw = os.environ.get("RDAGENT_FEATURES_FILE")
    if raw:
        return expand_path(raw)
    root = resolve_datasets_root()
    return root / "source_data" / competition / "features.csv"


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if value != value:  # NaN
            return None
        return float(value)
    try:
        parsed = float(value)
        if parsed != parsed:
            return None
        return parsed
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value:
            return None
        return int(value)
    try:
        return int(str(value))
    except Exception:
        return None


def to_date_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return value
    return None


def pick_expiry(
    expiries: Sequence[date],
    *,
    asof: date,
    target_dte: int,
    min_dte: int,
    max_dte: int,
) -> Optional[Tuple[date, int]]:
    candidates: List[Tuple[int, int, date]] = []
    for exp in expiries:
        dte = (exp - asof).days
        if dte < min_dte or dte > max_dte:
            continue
        candidates.append((abs(dte - target_dte), dte, exp))
    if not candidates:
        return None
    candidates.sort()
    _, dte, exp = candidates[0]
    return exp, dte


@dataclass(frozen=True)
class OptionPick:
    strike: Optional[float]
    iv: Optional[float]
    volume: Optional[int]
    open_interest: Optional[int]
    bid: Optional[float]
    ask: Optional[float]


def pick_option_row(df: Any, target_strike: float) -> Optional[Dict[str, Any]]:
    # yfinance returns pandas DataFrames; keep it duck-typed.
    try:
        if df is None or getattr(df, "empty", True):
            return None
        if "strike" not in df.columns:
            return None
        strikes = df["strike"].astype(float)
        idx = (strikes - float(target_strike)).abs().idxmin()
        row = df.loc[idx]
        # If idx points to multiple rows, take the first.
        if hasattr(row, "to_dict"):
            if getattr(row, "shape", None) is not None and len(getattr(row, "shape")) == 2:
                row = row.iloc[0]
            return dict(row.to_dict())
        return None
    except Exception:
        return None


def parse_option_pick(row: Optional[Dict[str, Any]]) -> OptionPick:
    if not row:
        return OptionPick(
            strike=None,
            iv=None,
            volume=None,
            open_interest=None,
            bid=None,
            ask=None,
        )
    return OptionPick(
        strike=safe_float(row.get("strike")),
        iv=safe_float(row.get("impliedVolatility")),
        volume=safe_int(row.get("volume")),
        open_interest=safe_int(row.get("openInterest")),
        bid=safe_float(row.get("bid")),
        ask=safe_float(row.get("ask")),
    )


FUNDAMENTAL_FIELDS: Dict[str, str] = {
    "marketCap": "market_cap",
    "enterpriseValue": "enterprise_value",
    "sharesOutstanding": "shares_outstanding",
    "beta": "beta",
    "trailingPE": "trailing_pe",
    "forwardPE": "forward_pe",
    "pegRatio": "peg_ratio",
    "priceToBook": "price_to_book",
    "enterpriseToEbitda": "ev_to_ebitda",
    "trailingEps": "trailing_eps",
    "forwardEps": "forward_eps",
    "dividendYield": "dividend_yield",
    "payoutRatio": "payout_ratio",
    "profitMargins": "profit_margins",
    "grossMargins": "gross_margins",
    "operatingMargins": "operating_margins",
    "returnOnAssets": "return_on_assets",
    "returnOnEquity": "return_on_equity",
    "debtToEquity": "debt_to_equity",
    "currentRatio": "current_ratio",
    "quickRatio": "quick_ratio",
    "revenueGrowth": "revenue_growth",
    "earningsGrowth": "earnings_growth",
    "totalRevenue": "total_revenue",
    "ebitda": "ebitda",
    "freeCashflow": "free_cashflow",
    "totalCash": "total_cash",
    "totalDebt": "total_debt",
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    default_competition = (os.environ.get("RDAGENT_COMPETITION") or "us-fundamentals-iv").strip()
    if not default_competition:
        default_competition = "us-fundamentals-iv"
    default_tickers_file_raw = (os.environ.get("RDAGENT_TICKERS_FILE") or "").strip()
    default_tickers_file = expand_path(default_tickers_file_raw) if default_tickers_file_raw else None

    parser = argparse.ArgumentParser(
        description="Collect US equity fundamentals + options IV feature snapshots via Yahoo (yfinance).",
    )
    parser.add_argument(
        "--competition",
        default=default_competition,
        help="Task name used for default output paths (env: RDAGENT_COMPETITION).",
    )
    parser.add_argument("--tickers", help="Comma-separated tickers (e.g., AAPL,MSFT).")
    parser.add_argument(
        "--tickers-file",
        type=Path,
        default=default_tickers_file,
        help="Path to tickers file (one per line) (env: RDAGENT_TICKERS_FILE).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output CSV path (default: RDAGENT_FEATURES_FILE or $DS_LOCAL_DATA_PATH/source_data/<competition>/features.csv).",
    )
    parser.add_argument("--asof", type=parse_iso_date, default=date.today(), help="Snapshot date (YYYY-MM-DD).")
    parser.add_argument(
        "--no-normalize-tickers",
        action="store_true",
        help="Do not normalize tickers (default replaces '.' with '-' for Yahoo class shares).",
    )
    parser.add_argument(
        "--overwrite-date",
        action="store_true",
        help="If output exists, drop existing rows for collected trading date(s) before appending.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.25,
        help="Sleep between tickers to reduce Yahoo throttling (default: 0.25).",
    )
    parser.add_argument("--iv-target-dte", type=int, default=30, help="Target DTE for ATM IV (default: 30).")
    parser.add_argument("--iv-term-dte", type=int, default=90, help="Target DTE for term structure (default: 90).")
    parser.add_argument("--iv-min-dte", type=int, default=7, help="Minimum DTE to consider (default: 7).")
    parser.add_argument("--iv-max-dte", type=int, default=180, help="Maximum DTE to consider (default: 180).")

    args = parser.parse_args(list(argv) if argv is not None else None)

    competition = str(args.competition).strip() if args.competition else default_competition
    if not competition:
        competition = "us-fundamentals-iv"

    out_path: Path = (args.out.expanduser() if args.out else resolve_default_features_file(competition)).expanduser()

    tickers: List[str] = []
    if args.tickers:
        tickers.extend([t.strip() for t in args.tickers.split(",") if t.strip()])
    if args.tickers_file:
        tickers_file = args.tickers_file.expanduser()
        if not tickers_file.exists():
            eprint(f"Tickers file not found: {tickers_file}")
            return 2
        tickers.extend(read_tickers_file(tickers_file))

    if not tickers:
        eprint("No tickers provided. Use --tickers or --tickers-file.")
        return 2

    if not args.no_normalize_tickers:
        tickers = [normalize_yahoo_ticker(t) for t in tickers]

    # Import late so CLI help works without deps.
    try:
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        import yfinance as yf  # type: ignore
    except Exception as exc:
        eprint("Missing dependencies. Install with: python -m pip install -U yfinance pandas numpy")
        eprint(str(exc))
        return 2

    snapshot_ts = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    snapshot_date = args.asof

    rows: List[Dict[str, Any]] = []

    for i, raw_ticker in enumerate(tickers):
        ticker_symbol = raw_ticker.strip()
        if not ticker_symbol:
            continue
        if i > 0 and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

        try:
            t = yf.Ticker(ticker_symbol)
        except Exception as exc:
            eprint(f"[{ticker_symbol}] Failed to init yfinance ticker: {exc}")
            continue

        # Price history for close + RV.
        price_date: Optional[date] = None
        close: Optional[float] = None
        volume: Optional[int] = None
        rv_20d: Optional[float] = None
        ret_1d: Optional[float] = None

        try:
            hist = t.history(period="90d", interval="1d", auto_adjust=False, actions=False)
            if hist is not None and not hist.empty:
                hist = hist.dropna(subset=["Close"])
            if hist is not None and not hist.empty:
                last = hist.tail(1)
                idx = last.index[0]
                # idx is a Timestamp; normalize to date.
                try:
                    price_date = idx.to_pydatetime().date()
                except Exception:
                    price_date = None
                close = safe_float(last["Close"].iloc[0])
                volume = safe_int(last["Volume"].iloc[0]) if "Volume" in last.columns else None

                if len(hist) >= 2:
                    prev_close = safe_float(hist["Close"].iloc[-2])
                    if close is not None and prev_close is not None and prev_close != 0:
                        ret_1d = close / prev_close - 1.0

                tail = hist["Close"].dropna().tail(21)
                if len(tail) >= 3:
                    rets = np.log(tail / tail.shift(1)).dropna()
                    if len(rets) >= 2:
                        rv_20d = float(np.std(rets)) * float(np.sqrt(252.0))
        except Exception as exc:
            eprint(f"[{ticker_symbol}] Price history fetch failed: {exc}")

        if price_date is None or close is None:
            eprint(f"[{ticker_symbol}] Missing price_date/close; skipping.")
            continue

        # Fundamentals snapshot (Yahoo info). This is NOT point-in-time historical.
        info: Dict[str, Any] = {}
        try:
            if hasattr(t, "get_info"):
                info = t.get_info() or {}
            else:
                info = getattr(t, "info", {}) or {}
        except Exception as exc:
            eprint(f"[{ticker_symbol}] Fundamentals fetch failed (continuing): {exc}")
            info = {}

        spot = safe_float(info.get("regularMarketPrice")) or safe_float(info.get("currentPrice")) or close
        if spot is None:
            spot = close

        # Options chain IV features
        expiries: List[date] = []
        try:
            raw_exps = list(getattr(t, "options", []) or [])
            for exp in raw_exps:
                if not isinstance(exp, str):
                    continue
                try:
                    expiries.append(datetime.strptime(exp, "%Y-%m-%d").date())
                except Exception:
                    continue
            expiries.sort()
        except Exception as exc:
            eprint(f"[{ticker_symbol}] Options expiries fetch failed (continuing): {exc}")
            expiries = []

        expiry_30: Optional[date] = None
        dte_30: Optional[int] = None
        expiry_90: Optional[date] = None
        dte_90: Optional[int] = None

        iv_atm_30: Optional[float] = None
        iv_call_atm_30: Optional[float] = None
        iv_put_atm_30: Optional[float] = None
        iv_put_95_30: Optional[float] = None
        iv_call_105_30: Optional[float] = None
        iv_skew_30: Optional[float] = None

        iv_atm_90: Optional[float] = None
        iv_term_30_90: Optional[float] = None

        if expiries and spot is not None and spot > 0:
            pick30 = pick_expiry(
                expiries,
                asof=snapshot_date,
                target_dte=int(args.iv_target_dte),
                min_dte=int(args.iv_min_dte),
                max_dte=int(args.iv_max_dte),
            )
            pick90 = pick_expiry(
                expiries,
                asof=snapshot_date,
                target_dte=int(args.iv_term_dte),
                min_dte=int(args.iv_min_dte),
                max_dte=int(args.iv_max_dte),
            )

            if pick30:
                expiry_30, dte_30 = pick30
                try:
                    chain30 = t.option_chain(expiry_30.isoformat())
                    calls30 = getattr(chain30, "calls", None)
                    puts30 = getattr(chain30, "puts", None)

                    call_atm = parse_option_pick(pick_option_row(calls30, float(spot)))
                    put_atm = parse_option_pick(pick_option_row(puts30, float(spot)))
                    iv_call_atm_30 = call_atm.iv
                    iv_put_atm_30 = put_atm.iv
                    ivs = [v for v in [iv_call_atm_30, iv_put_atm_30] if isinstance(v, (int, float))]
                    if ivs:
                        iv_atm_30 = float(sum(ivs)) / float(len(ivs))

                    put_95 = parse_option_pick(pick_option_row(puts30, float(spot) * 0.95))
                    call_105 = parse_option_pick(pick_option_row(calls30, float(spot) * 1.05))
                    iv_put_95_30 = put_95.iv
                    iv_call_105_30 = call_105.iv
                    if iv_put_95_30 is not None and iv_call_105_30 is not None:
                        iv_skew_30 = float(iv_put_95_30) - float(iv_call_105_30)
                except Exception as exc:
                    eprint(f"[{ticker_symbol}] Options chain (30D) fetch failed (continuing): {exc}")

            if pick90:
                expiry_90, dte_90 = pick90
                try:
                    chain90 = t.option_chain(expiry_90.isoformat())
                    calls90 = getattr(chain90, "calls", None)
                    puts90 = getattr(chain90, "puts", None)

                    call_atm_90 = parse_option_pick(pick_option_row(calls90, float(spot)))
                    put_atm_90 = parse_option_pick(pick_option_row(puts90, float(spot)))
                    ivs90 = [v for v in [call_atm_90.iv, put_atm_90.iv] if isinstance(v, (int, float))]
                    if ivs90:
                        iv_atm_90 = float(sum(ivs90)) / float(len(ivs90))
                except Exception as exc:
                    eprint(f"[{ticker_symbol}] Options chain (term) fetch failed (continuing): {exc}")

        if iv_atm_30 is not None and iv_atm_90 is not None:
            iv_term_30_90 = float(iv_atm_30) - float(iv_atm_90)

        # Build row
        row: Dict[str, Any] = {
            "id": f"{price_date.isoformat()}_{ticker_symbol}",
            "date": price_date.isoformat(),
            "snapshot_date": snapshot_date.isoformat(),
            "snapshot_ts": snapshot_ts,
            "ticker": ticker_symbol,
            "currency": info.get("currency") if isinstance(info.get("currency"), str) else None,
            "close": close,
            "volume": volume,
            "spot": spot,
            "ret_1d": ret_1d,
            "rv_20d": rv_20d,
            "expiry_30d": to_date_str(expiry_30),
            "dte_30d": dte_30,
            "iv_atm_30d": iv_atm_30,
            "iv_call_atm_30d": iv_call_atm_30,
            "iv_put_atm_30d": iv_put_atm_30,
            "iv_put_95_30d": iv_put_95_30,
            "iv_call_105_30d": iv_call_105_30,
            "iv_skew_30d": iv_skew_30,
            "expiry_90d": to_date_str(expiry_90),
            "dte_90d": dte_90,
            "iv_atm_90d": iv_atm_90,
            "iv_term_30_90": iv_term_30_90,
            "sector": info.get("sector") if isinstance(info.get("sector"), str) else None,
            "industry": info.get("industry") if isinstance(info.get("industry"), str) else None,
        }

        for src, dst in FUNDAMENTAL_FIELDS.items():
            if dst in row:
                continue
            row[dst] = safe_float(info.get(src))

        rows.append(row)

    if not rows:
        eprint("No rows collected.")
        return 1

    new_df = pd.DataFrame(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = out_path.parent / f"{out_path.name}.lock"

    def merge_with_existing() -> Any:
        if out_path.exists():
            try:
                old_df = pd.read_csv(out_path)
            except Exception as exc:
                eprint(f"Failed to read existing CSV (will overwrite): {exc}")
                old_df = pd.DataFrame()
        else:
            old_df = pd.DataFrame()

        if not old_df.empty:
            combined_df = old_df
            if args.overwrite_date and "date" in old_df.columns:
                drop_dates = set(new_df["date"].dropna().astype(str).unique().tolist())
                combined_df = old_df[~old_df["date"].astype(str).isin(drop_dates)].copy()
            combined_df = pd.concat([combined_df, new_df], ignore_index=True, sort=False)
        else:
            combined_df = new_df

        if "date" in combined_df.columns and "ticker" in combined_df.columns:
            combined_df = combined_df.sort_values(by=["date", "ticker"], kind="stable")
        return combined_df

    def write_atomic_csv(df: Any) -> None:
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                delete=False,
                dir=str(out_path.parent),
                prefix=f"{out_path.name}.",
                suffix=".tmp",
                encoding="utf-8",
            ) as handle:
                tmp_path = Path(handle.name)
                df.to_csv(handle, index=False)
            tmp_path.replace(out_path)
        finally:
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    combined: Any
    if fcntl is None:
        combined = merge_with_existing()
        write_atomic_csv(combined)
    else:
        with open(lock_path, "w", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            combined = merge_with_existing()
            write_atomic_csv(combined)

    eprint(f"Wrote {len(new_df)} new rows. Output: {out_path} (total {len(combined)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
