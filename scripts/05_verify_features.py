# scripts/05_verify_features.py
from __future__ import annotations

import re
from pathlib import Path

import polars as pl


FEATURES_DIR = Path("data/derived/features/current")
REPORTS_DIR = Path("reports/verify_v1")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

GLOB_PATH = str(FEATURES_DIR / "ticker=*/part-0.parquet")


def infer_warmup_rows(cols: list[str]) -> int:
    """
    Infer a warmup length from column name suffixes like ema_200, ret_400, vol_50, fwd_ret_50, etc.
    This is a cheap heuristic to separate "expected early NaNs" from real missingness.
    """
    pat = re.compile(r".*_(\d+)$")
    max_n = 0
    for c in cols:
        m = pat.match(c)
        if m:
            try:
                n = int(m.group(1))
                if n > max_n:
                    max_n = n
            except ValueError:
                pass
    # If nothing matches, fall back to a conservative default
    return max_n if max_n > 0 else 200


def missing_expr(col: str, dtype: pl.DataType) -> pl.Expr:
    """
    Treat NaN + null as missing for float columns; null only for non-floats.
    """
    if dtype in (pl.Float32, pl.Float64):
        return pl.col(col).is_null() | pl.col(col).is_nan()
    return pl.col(col).is_null()


def main() -> None:
    if not FEATURES_DIR.exists():
        raise FileNotFoundError(f"Features directory not found: {FEATURES_DIR}")

    scan = pl.scan_parquet(GLOB_PATH)
    schema = scan.collect_schema()
    cols = schema.names()

    # Hard expectations for your per-ticker outputs
    for required in ("ticker", "date"):
        if required not in cols:
            raise ValueError(f"Expected column '{required}' not found. Found columns: {cols}")

    key_cols = {"ticker", "date"}
    value_cols = [c for c in cols if c not in key_cols]

    warmup_n = infer_warmup_rows(cols)

    print(f"[INFO] Input: {GLOB_PATH}")
    print(f"[INFO] Columns: total={len(cols)} value_cols={len(value_cols)}")
    print(f"[INFO] Inferred warmup rows per ticker: {warmup_n}")

    # ---------------------------------------------------------------------
    # 1) Global counts (cheap)
    # ---------------------------------------------------------------------
    counts = scan.select(
        [
            pl.len().alias("n_rows"),
            pl.col("ticker").n_unique().alias("n_tickers"),
            pl.col("date").n_unique().alias("n_dates"),
        ]
    ).collect()

    counts.write_csv(REPORTS_DIR / "counts.csv")

    # ---------------------------------------------------------------------
    # 2) Missing rates overall (one lazy aggregate)
    # ---------------------------------------------------------------------
    miss_overall_wide = scan.select(
        [missing_expr(c, schema[c]).mean().alias(c) for c in value_cols]
    ).collect()

    miss_overall = (
        miss_overall_wide.transpose(
            include_header=True, header_name="column", column_names=["missing_rate"]
        )
        .sort("missing_rate", descending=True)
    )
    miss_overall.write_csv(REPORTS_DIR / "missing_rates_global.csv")

    # ---------------------------------------------------------------------
    # 3) Missing rates after warmup
    # Efficient approach: per-ticker row index, filter first warmup_n rows.
    # ---------------------------------------------------------------------
    warmed = (
        scan.sort(["ticker", "date"])
        .with_columns(pl.int_range(0, pl.len()).over("ticker").alias("_i"))
        .filter(pl.col("_i") >= warmup_n)
        .drop("_i")
    )

    miss_warm_wide = warmed.select(
        [missing_expr(c, schema[c]).mean().alias(c) for c in value_cols]
    ).collect()

    miss_warm = (
        miss_warm_wide.transpose(
            include_header=True,
            header_name="column",
            column_names=["missing_rate_after_warmup"],
        )
        .sort("missing_rate_after_warmup", descending=True)
    )
    miss_warm.write_csv(REPORTS_DIR / "missing_rates_after_warmup.csv")

    # ---------------------------------------------------------------------
    # 4) Constant or problematic columns after warmup (floats only)
    # - "is_constant": std == 0 (ignoring nulls/NaNs)
    # - "std_is_null": std is null (e.g., column entirely null/NaN after warmup)
    # ---------------------------------------------------------------------
    float_cols = [c for c in value_cols if schema[c] in (pl.Float32, pl.Float64)]

    if float_cols:
        std_wide = warmed.select(
            [pl.col(c).drop_nans().std().alias(c) for c in float_cols]
        ).collect()

        const_or_nullstd = (
            std_wide.transpose(include_header=True, header_name="column", column_names=["std"])
            .with_columns(
                [
                    pl.col("std").is_null().alias("std_is_null"),
                    (pl.col("std") == 0).alias("is_constant"),
                ]
            )
            .filter(pl.col("std_is_null") | pl.col("is_constant"))
            .sort("column")
        )
    else:
        const_or_nullstd = pl.DataFrame(
            {"column": [], "std": [], "std_is_null": [], "is_constant": []}
        )

    const_or_nullstd.write_csv(REPORTS_DIR / "constant_or_nullstd_after_warmup.csv")

    # ---------------------------------------------------------------------
    # 5) Forward-return alignment spot-check (tiny read)
    # Only runs if adjclose exists and fwd_ret_* columns exist.
    # Reads up to first 3 ticker files directly (no full-dataset compute).
    # ---------------------------------------------------------------------
    if "adjclose" in cols:
        ticker_dirs = sorted([p for p in FEATURES_DIR.glob("ticker=*") if p.is_dir()])[:3]
        tickers = [d.name.split("=", 1)[1] for d in ticker_dirs]

        checks: list[tuple[str, int, float | None]] = []
        for tkr in tickers:
            fpath = FEATURES_DIR / f"ticker={tkr}" / "part-0.parquet"
            if not fpath.exists():
                continue

            df = pl.read_parquet(fpath).sort("date")

            if "adjclose" not in df.columns:
                continue

            for h in (5, 10, 20, 50):
                col = f"fwd_ret_{h}"
                if col not in df.columns:
                    continue

                # expected: (adjclose[t+h]/adjclose[t]) - 1
                calc = (pl.col("adjclose").shift(-h) / pl.col("adjclose") - 1.0)

                # Compute max abs error ignoring tail nulls
                err = (
                    df.select((calc - pl.col(col)).abs().alias("err"))
                    .get_column("err")
                    .drop_nulls()
                )
                max_abs = err.max()
                checks.append((tkr, h, float(max_abs) if max_abs is not None else None))

        pl.DataFrame(checks, schema=["ticker", "horizon", "max_abs_error"], orient="row").write_csv(
            REPORTS_DIR / "fwd_ret_alignment_sample.csv"
        )
    else:
        # Still write an empty file so automation/pipeline consumers have stable outputs
        pl.DataFrame(
            {"ticker": [], "horizon": [], "max_abs_error": []}
        ).write_csv(REPORTS_DIR / "fwd_ret_alignment_sample.csv")

    # ---------------------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------------------
    n_rows = int(counts["n_rows"][0])
    n_tickers = int(counts["n_tickers"][0])
    n_dates = int(counts["n_dates"][0])

    print("[OK] Wrote reports to:", REPORTS_DIR)
    print(" - counts.csv")
    print(" - missing_rates_global.csv")
    print(" - missing_rates_after_warmup.csv")
    print(" - constant_or_nullstd_after_warmup.csv")
    print(" - fwd_ret_alignment_sample.csv")
    print(f"[INFO] rows={n_rows:,} tickers={n_tickers:,} dates={n_dates:,}")


if __name__ == "__main__":
    main()
