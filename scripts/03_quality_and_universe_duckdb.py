import argparse
import os
import duckdb


def main():
    p = argparse.ArgumentParser(description="Quality checks + universe selection from core dataset (DuckDB).")
    p.add_argument("--core-dir", default=os.path.join("data", "derived", "core"))
    p.add_argument("--out-dir", default=os.path.join("data", "derived", "quality"))
    p.add_argument("--overwrite", action="store_true")

    # Your chosen thresholds
    p.add_argument("--min-days", type=int, default=750)
    p.add_argument("--max-big-moves", type=int, default=10)
    p.add_argument("--big-move-threshold", type=float, default=0.5)  # abs(ret) > 0.5

    # "Currently trading" assumption: require last date close to global max date
    p.add_argument("--max-stale-days", type=int, default=30)

    args = p.parse_args()

    core_glob = os.path.join(args.core_dir, "**", "*.parquet")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.overwrite:
        for f in ["ticker_quality.parquet", "good_tickers.parquet", "bad_tickers.parquet"]:
            fp = os.path.join(args.out_dir, f)
            if os.path.exists(fp):
                try:
                    os.remove(fp)
                except OSError:
                    pass

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4;")

    # Read core
    con.execute(f"""
        CREATE OR REPLACE VIEW core AS
        SELECT * FROM read_parquet('{core_glob}');
    """)

    # Global max date (for "still trading" recency filter)
    max_date = con.execute("SELECT MAX(date) FROM core").fetchone()[0]
    if max_date is None:
        raise RuntimeError("Core dataset appears empty. Did Script 02 run successfully?")
    print(f"[Info] Global max date in core: {max_date}")

    # Per-row sanity: OHLC constraints
    # bad day if any are violated:
    # - high >= max(open, close, low)
    # - low  <= min(open, close, high)
    # Also check non-null (should already be true in core).
    con.execute(f"""
        CREATE OR REPLACE VIEW core_q AS
        SELECT
            *,
            CASE
              WHEN high < GREATEST(open, close, low) THEN 1
              WHEN low  > LEAST(open, close, high) THEN 1
              ELSE 0
            END AS is_bad_ohlc
        FROM core;
    """)

    # Returns per ticker (using close)
    con.execute("""
        CREATE OR REPLACE VIEW core_ret AS
        SELECT
            *,
            LAG(close) OVER (PARTITION BY ticker ORDER BY date) AS prev_close
        FROM core_q;
    """)

    # Daily return; mark "big moves" as abs(ret) > threshold
    con.execute(f"""
        CREATE OR REPLACE VIEW core_ret2 AS
        SELECT
            *,
            CASE
              WHEN prev_close IS NULL OR prev_close <= 0 THEN NULL
              ELSE (close / prev_close) - 1.0
            END AS ret,
            CASE
              WHEN prev_close IS NULL OR prev_close <= 0 THEN 0
              WHEN ABS((close / prev_close) - 1.0) > {float(args.big_move_threshold)} THEN 1
              ELSE 0
            END AS is_big_move
        FROM core_ret;
    """)

    # Aggregate metrics per ticker
    quality = con.execute(f"""
        SELECT
            ticker,
            COUNT(*) AS n_days,
            MIN(date) AS min_date,
            MAX(date) AS max_date,
            SUM(is_bad_ohlc) AS bad_ohlc_days,
            SUM(is_big_move) AS n_big_moves,
            -- optional extra diagnostics:
            AVG(CASE WHEN ret = 0 THEN 1 ELSE 0 END) AS pct_zero_returns
        FROM core_ret2
        GROUP BY 1
        ORDER BY 1;
    """).fetchdf()

    # Write full quality report
    out_quality = os.path.join(args.out_dir, "ticker_quality.parquet")
    quality.to_parquet(out_quality, index=False)
    print(f"[OK] Wrote quality report: {out_quality}")

    # Build good/bad universe inside DuckDB for clear logic and exact date arithmetic
    con.execute(f"""
        CREATE OR REPLACE VIEW ticker_quality AS
        SELECT
            ticker,
            n_days,
            min_date,
            max_date,
            bad_ohlc_days,
            n_big_moves,
            pct_zero_returns,
            -- flags
            (n_days >= {int(args.min_days)}) AS ok_days,
            (bad_ohlc_days = 0) AS ok_ohlc,
            (n_big_moves <= {int(args.max_big_moves)}) AS ok_big_moves,
            (DATE_DIFF('day', max_date, DATE '{max_date}') <= {int(args.max_stale_days)}) AS ok_recency
        FROM read_parquet('{out_quality}');
    """)

    # Good tickers
    good = con.execute("""
        SELECT ticker
        FROM ticker_quality
        WHERE ok_days AND ok_ohlc AND ok_big_moves AND ok_recency
        ORDER BY ticker;
    """).fetchdf()

    out_good = os.path.join(args.out_dir, "good_tickers.parquet")
    good.to_parquet(out_good, index=False)
    print(f"[OK] Wrote good tickers: {out_good} (n={len(good)})")

    # Bad tickers with reasons
    bad = con.execute("""
        SELECT
            ticker,
            n_days,
            min_date,
            max_date,
            bad_ohlc_days,
            n_big_moves,
            pct_zero_returns,
            ok_days,
            ok_ohlc,
            ok_big_moves,
            ok_recency
        FROM ticker_quality
        WHERE NOT (ok_days AND ok_ohlc AND ok_big_moves AND ok_recency)
        ORDER BY
            ok_recency ASC,
            ok_ohlc ASC,
            ok_days ASC,
            ok_big_moves ASC,
            ticker ASC;
    """).fetchdf()

    out_bad = os.path.join(args.out_dir, "bad_tickers.parquet")
    bad.to_parquet(out_bad, index=False)
    print(f"[OK] Wrote bad tickers: {out_bad} (n={len(bad)})")

    # Quick summary printed
    summary = con.execute("""
        SELECT
          SUM(CASE WHEN ok_days THEN 1 ELSE 0 END) AS ok_days,
          SUM(CASE WHEN ok_ohlc THEN 1 ELSE 0 END) AS ok_ohlc,
          SUM(CASE WHEN ok_big_moves THEN 1 ELSE 0 END) AS ok_big_moves,
          SUM(CASE WHEN ok_recency THEN 1 ELSE 0 END) AS ok_recency,
          SUM(CASE WHEN ok_days AND ok_ohlc AND ok_big_moves AND ok_recency THEN 1 ELSE 0 END) AS n_good,
          COUNT(*) AS n_total
        FROM ticker_quality;
    """).fetchdf()

    print("\n[Summary]")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
