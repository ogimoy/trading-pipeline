# scripts/01_download_yf.py
#
# Downloads daily OHLCV from Yahoo Finance for all tickers found in one or more
# semicolon-separated CSV files inside a folder (e.g. data/tickers/*.csv),
# and saves one Parquet per ticker to data/raw/yf_daily/.
#
# Usage (from repo root, in activated venv):
#   python scripts/01_download_yf.py --tickers-dir data/tickers --skip-existing --download-vix --end 2026-01-20
#
# Notes:
# - Expects each CSV to have tickers in the FIRST column (like your files).
# - Handles duplicates across files.
# - Retries downloads a few times with backoff.
# - Writes Parquet (fast, compact, great for DuckDB/Polars later).

import argparse
import os
import time
from glob import glob
from datetime import datetime
from typing import List, Set

import pandas as pd
import yfinance as yf

def load_tickers_from_dir(tickers_dir: str) -> List[str]:
    csv_files = sorted(glob(os.path.join(tickers_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {tickers_dir}")

    all_tickers: list[str] = []

    for csv_file in csv_files:
        # Auto-detect delimiter (comma vs semicolon, etc.)
        df = pd.read_csv(csv_file, sep=None, engine="python")
        df = df.dropna(axis=1, how="all")

        if df.shape[1] == 0:
            continue

        first_col = df.columns[0]  # e.g. "Symbol"
        tickers = df[first_col].dropna().astype(str).tolist()

        all_tickers.extend(tickers)

    # Clean + dedupe
    cleaned = []
    for t in all_tickers:
        t = t.strip()
        if not t:
            continue
        # optional: ignore weird rows (sometimes header repeats or bad lines)
        if t.lower() in {"symbol", "ticker"}:
            continue
        cleaned.append(t)

    return sorted(set(cleaned))

def yf_download_one(
    ticker: str,
    start_date: str,
    end_date: str,
    interval: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Download one ticker with retries. Returns empty DF on failure/empty response."""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=False,
                group_by="column",
            )

            if df is None or df.empty:
                return pd.DataFrame()

            # Rename Adj Close -> AdjClose
            if "Adj Close" in df.columns:
                df = df.rename(columns={"Adj Close": "AdjClose"})

            # Make index explicit and add ticker column
            df = df.reset_index()  # Date becomes a column (usually "Date")
            df.insert(0, "Ticker", ticker)

            return df

        except Exception as e:
            last_err = e
            # simple exponential-ish backoff
            time.sleep(0.5 * attempt)

    print(f"[FAIL] {ticker}: {last_err}")
    return pd.DataFrame()


def save_parquet(df: pd.DataFrame, out_dir: str, filename: str) -> str:
    """Save dataframe to parquet in out_dir with filename (no extension handling)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{filename}.parquet")
    df.to_parquet(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Yahoo Finance OHLCV and save as Parquet.")
    parser.add_argument("--tickers-dir", type=str, required=True, help="Directory containing ticker CSV files.")
    parser.add_argument("--out-dir", type=str, default=os.path.join("data", "raw", "yf_daily"))
    parser.add_argument("--start", type=str, default="2012-01-01")
    parser.add_argument("--end", type=str, default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--sleep", type=float, default=0.10, help="Sleep between tickers (seconds).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip tickers already saved to Parquet.")
    parser.add_argument("--download-vix", action="store_true", help="Also download VIX (^VIX) and save as VIX.parquet.")
    parser.add_argument("--sep", type=str, default=";", help="CSV separator for ticker lists (default ';').")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per ticker download.")
    args = parser.parse_args()

    tickers = load_tickers_from_dir(args.tickers_dir)

    print(f"Found {len(tickers)} unique tickers from: {args.tickers_dir}")
    print("First 10 tickers:", tickers[:10])

    os.makedirs(args.out_dir, exist_ok=True)

    failures: List[str] = []
    downloaded = 0
    skipped = 0
    empty = 0

    for ticker in tickers:
        out_path = os.path.join(args.out_dir, f"{ticker}.parquet")

        if args.skip_existing and os.path.exists(out_path):
            print(f"[SKIP] {ticker} (exists)")
            skipped += 1
            continue

        print(f"[DL] {ticker}")
        df = yf_download_one(
            ticker=ticker,
            start_date=args.start,
            end_date=args.end,
            interval=args.interval,
            max_retries=args.max_retries,
        )

        if df.empty:
            print(f"[EMPTY] {ticker}")
            failures.append(ticker)
            empty += 1
        else:
            save_parquet(df, args.out_dir, ticker)
            downloaded += 1

        time.sleep(args.sleep)

    if args.download_vix:
        vix_out = os.path.join(args.out_dir, "VIX.parquet")
        if args.skip_existing and os.path.exists(vix_out):
            print("[SKIP] VIX (exists)")
        else:
            print("[DL] VIX (^VIX)")
            vix = yf_download_one(
                ticker="^VIX",
                start_date=args.start,
                end_date=args.end,
                interval=args.interval,
                max_retries=args.max_retries,
            )
            if vix.empty:
                print("[EMPTY] VIX")
            else:
                # store under a stable name
                vix["Ticker"] = "VIX"
                vix.to_parquet(vix_out, index=False)
                print("[OK] VIX saved")

    print("\n=== Summary ===")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped:    {skipped}")
    print(f"Empty/Fail: {empty}")
    if failures:
        # Write failures list for easy reruns/debugging
        fail_path = os.path.join(args.out_dir, "_failures.txt")
        with open(fail_path, "w", encoding="utf-8") as f:
            for t in failures:
                f.write(f"{t}\n")
        print(f"Failures list saved to: {fail_path}")


if __name__ == "__main__":
    main()
