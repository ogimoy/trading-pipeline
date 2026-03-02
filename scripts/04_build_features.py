from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import polars as pl

from features.registry import build_plan, FEATURE_SETS


def _ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _atomic_replace_dir(tmp_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)


def _load_spy_series(core_dir: Path) -> pl.DataFrame:
    """
    Load SPY (ticker=spy) from core and return a compact date->spy_adjclose series.
    """
    spy_glob = str(core_dir / "ticker=spy" / "*.parquet")
    if not list((core_dir / "ticker=spy").glob("*.parquet")):
        raise FileNotFoundError(
            "SPY not found in core (data/derived/core/ticker=spy). "
            "Make sure SPY is downloaded and Script 02 has been run."
        )

    spy = (
        pl.scan_parquet(spy_glob)
        .select(["date", "adjclose"])
        .with_columns([
            pl.col("date").cast(pl.Date),
            pl.col("adjclose").cast(pl.Float64),
        ])
        .sort("date")
        .collect()
        .rename({"adjclose": "spy_adjclose"})
    )
    return spy


def _build_features_for_one_ticker(
    df: pl.DataFrame,
    spy: pl.DataFrame,
    plan,
    include_close_volume: bool,
    dtype: str,
) -> pl.DataFrame:
    """
    df: one ticker core data (already sorted by date)
    spy: date->spy_adjclose table
    returns: features dataframe with selected features + targets
    """

    # Join SPY series by date (left join; SPY should cover most dates)
    df = df.join(spy, on="date", how="left")

    # Base: logret_1d and ret_1d from adjclose
    df = df.with_columns([
        (pl.col("adjclose") / pl.col("adjclose").shift(1)).log().alias("logret_1d")
    ])

    df = df.with_columns([
        (pl.col("adjclose") / pl.col("adjclose").shift(1).over("ticker") - 1.0)
        .alias("ret_1d")
    ])

    # Rolling volatilities
    for w in plan.vol_windows:
        df = df.with_columns([
            pl.col("logret_1d")
              .rolling_std(window_size=w, min_samples=w)
              .alias(f"vol_{w}")
        ])

    # Normalization vol (ensure present)
    norm_vol_name = f"vol_{plan.norm_vol_window}"
    if norm_vol_name not in df.columns:
        df = df.with_columns([
            pl.col("logret_1d")
              .rolling_std(window_size=plan.norm_vol_window, min_samples=plan.norm_vol_window)
              .alias(norm_vol_name)
        ])

    # EMAs on adjclose
    for n in plan.ema_windows:
        df = df.with_columns([
            pl.col("adjclose")
              .ewm_mean(span=n, adjust=False, min_samples=n)
              .alias(f"ema_{n}")
        ])

    # Distances to EMAs (raw and vol-normalized)
    for n in plan.ema_windows:
        dist = (pl.col("adjclose") / pl.col(f"ema_{n}") - 1.0)
        df = df.with_columns([
            dist.alias(f"dist_close_ema_{n}"),
            (dist / pl.col(norm_vol_name)).alias(f"dist_close_ema_{n}_volnorm"),
        ])

    # Past returns and vol-normalized returns
    for h in plan.ret_windows:
        r = (pl.col("adjclose") / pl.col("adjclose").shift(h) - 1.0)
        df = df.with_columns([
            r.alias(f"ret_{h}"),
            (r / (pl.col(norm_vol_name) * (h ** 0.5))).alias(f"ret_{h}_volnorm"),
        ])

    # SPY returns and relative returns vs SPY
    # Use spy_adjclose series joined above
    for h in plan.rel_ret_windows:
        spy_r = (pl.col("spy_adjclose") / pl.col("spy_adjclose").shift(h) - 1.0)
        df = df.with_columns([
            spy_r.alias(f"spy_ret_{h}"),
            (pl.col(f"ret_{h}") - spy_r).alias(f"relret_spy_{h}"),
        ])

    # ATR (True Range + rolling mean)
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal([
        (pl.col("high") - pl.col("low")).abs(),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    ]).alias("true_range")

    df = df.with_columns([tr])

    df = df.with_columns([
        pl.col("true_range")
          .rolling_mean(window_size=plan.atr_window, min_samples=plan.atr_window)
          .alias(f"atr_{plan.atr_window}")
    ])

    # Volume MA
    df = df.with_columns([
        pl.col("volume")
          .rolling_mean(window_size=plan.volume_ma_window, min_samples=plan.volume_ma_window)
          .alias(f"volume_ma_{plan.volume_ma_window}")
    ])

    # Forward returns (targets) always
    for h in plan.fwd_ret_windows:
        df = df.with_columns([
            (pl.col("adjclose").shift(-h) / pl.col("adjclose") - 1.0).alias(f"fwd_ret_{h}")
        ])

    # --- Select final columns (keys + chosen features + targets) ---
    out_cols: List[str] = ["ticker", "date"]
    if include_close_volume:
        out_cols += ["adjclose", "volume"]

    out_cols += plan.selected
    out_cols += ["ret_1d"]
    out_cols += [f"fwd_ret_{h}" for h in plan.fwd_ret_windows]

    # De-duplicate while preserving order
    seen = set()
    out_cols = [c for c in out_cols if not (c in seen or seen.add(c))]

    out = df.select(out_cols)

    # Cast feature columns to float32/float64 (keep volume int, keys as-is)
    feature_dtype = pl.Float32 if dtype == "float32" else pl.Float64
    cast_exprs = []
    for c in out.columns:
        if c in ("ticker", "date", "volume"):
            continue
        cast_exprs.append(pl.col(c).cast(feature_dtype))
    out = out.with_columns(cast_exprs)

    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build precomputed features (Polars) from core dataset (per-ticker).")
    p.add_argument("--core-dir", default=os.path.join("data", "derived", "core"))
    p.add_argument("--quality-dir", default=os.path.join("data", "derived", "quality"))
    p.add_argument("--out-dir", default=os.path.join("data", "derived", "features", "current"))
    p.add_argument("--feature-set", default="default", choices=sorted(FEATURE_SETS.keys()))
    p.add_argument("--features", default="", help="Comma-separated feature names (overrides --feature-set).")
    p.add_argument("--overwrite", action="store_true")

    # Performance knobs
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--include-close-volume", action="store_true", help="Keep adjclose and volume columns in output.")
    p.add_argument("--limit-tickers", type=int, default=0, help="Test mode: only process first N tickers.")
    p.add_argument("--heartbeat-seconds", type=int, default=30, help="Print a heartbeat every N seconds.")
    args = p.parse_args()

    print("[INFO] Script 04 started")
    t0 = time.time()

    plan = build_plan(args.feature_set, args.features)

    core_dir = Path(args.core_dir)
    good_path = Path(args.quality_dir) / "good_tickers.parquet"

    if not good_path.exists():
        raise FileNotFoundError(f"Missing {good_path}. Run Script 03 first.")

    out_dir = Path(args.out_dir)
    tmp_dir = out_dir.parent / (out_dir.name + "_tmp_build")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{out_dir} exists. Use --overwrite to rebuild.")

    _ensure_clean_dir(tmp_dir)

    # Load universe
    good = pl.read_parquet(str(good_path))
    tickers = good.get_column("ticker").cast(pl.Utf8).unique().sort()

    # Ensure SPY exists for rel features
    if "spy" not in set(tickers.to_list()):
        tickers = pl.concat([tickers, pl.Series(["spy"])]).unique().sort()

    if args.limit_tickers and args.limit_tickers > 0:
        tickers = tickers.head(int(args.limit_tickers))

    ticker_list = tickers.to_list()
    n_total = len(ticker_list)
    print(f"[INFO] Tickers to process: {n_total}")

    # Load SPY series once
    print("[INFO] Loading SPY series...")
    spy = _load_spy_series(core_dir)
    print(f"[INFO] SPY rows: {spy.height}")

    # Per-ticker loop
    last_heartbeat = time.time()
    n_done = 0
    n_failed = 0

    for i, tkr in enumerate(ticker_list, 1):
        try:
            # Read this ticker's core parquet(s)
            ticker_path = core_dir / f"ticker={tkr}"
            files = list(ticker_path.glob("*.parquet"))
            if not files:
                # Skip silently if somehow missing (shouldn't happen for good tickers)
                n_failed += 1
                print(f"[WARN] Missing core parquet for ticker={tkr} (skipping)")
                continue

            df = (
                pl.scan_parquet(str(ticker_path / "*.parquet"))
                # core parquet doesn't store ticker column; it's encoded in the folder name ticker=<tkr>
                .select(["date", "open", "high", "low", "close", "adjclose", "volume"])
                .with_columns([
                    pl.lit(tkr).cast(pl.Utf8).alias("ticker"),                        pl.col("date").cast(pl.Date),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("adjclose").cast(pl.Float64),
                    pl.col("volume").cast(pl.Int64),
                ])
                .sort("date")
                .collect()
            )

            

            out = _build_features_for_one_ticker(
                df=df,
                spy=spy,
                plan=plan,
                include_close_volume=args.include_close_volume,
                dtype=args.dtype,
            )

            # Write this ticker immediately (partitioned layout expected)
            # We write a single file per ticker to keep file count manageable.
            out_ticker_dir = tmp_dir / f"ticker={tkr}"
            _ensure_clean_dir(out_ticker_dir)
            out.write_parquet(
                str(out_ticker_dir / "part-0.parquet"),
                compression="zstd",
                statistics=True,
            )

            n_done += 1

            # Progress print every 25 tickers
            if i % 25 == 0 or i == n_total:
                elapsed = time.time() - t0
                print(f"[PROGRESS] {i}/{n_total} tickers processed | ok={n_done} fail={n_failed} | elapsed={elapsed:.1f}s")

            # Heartbeat
            if time.time() - last_heartbeat >= args.heartbeat_seconds:
                elapsed = time.time() - t0
                print(f"[HEARTBEAT] still running... processed={i}/{n_total} | elapsed={elapsed:.1f}s")
                last_heartbeat = time.time()

        except Exception as e:
            n_failed += 1
            print(f"[FAIL] ticker={tkr}: {e}")

    # Metadata + success marker
    meta = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "feature_set": args.feature_set,
        "features": plan.selected,
        "dtype": args.dtype,
        "include_close_volume": bool(args.include_close_volume),
        "ema_windows": plan.ema_windows,
        "ret_windows": plan.ret_windows,
        "rel_ret_windows": plan.rel_ret_windows,
        "vol_windows": plan.vol_windows,
        "atr_window": plan.atr_window,
        "volume_ma_window": plan.volume_ma_window,
        "norm_vol_window": plan.norm_vol_window,
        "targets": [f"fwd_ret_{h}" for h in plan.fwd_ret_windows],
        "tickers_processed": n_total,
        "tickers_ok": n_done,
        "tickers_failed": n_failed,
    }
    (tmp_dir / "_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (tmp_dir / "_SUCCESS").write_text("", encoding="utf-8")

    _atomic_replace_dir(tmp_dir, out_dir)

    elapsed = time.time() - t0
    print(f"[OK] Features written to: {out_dir}")
    print(f"[OK] n_features_selected: {len(plan.selected)} | targets: {len(plan.fwd_ret_windows)}")
    print(f"[OK] tickers_ok={n_done} tickers_failed={n_failed} | elapsed={elapsed:.1f}s")
    print(f"[Tip] See: {out_dir / '_metadata.json'}")


if __name__ == "__main__":
    main()
