import argparse
import os
import duckdb


def main():
    p = argparse.ArgumentParser(description="Build standardized core dataset from raw Parquet files.")
    p.add_argument("--in-dir", default=os.path.join("data", "raw", "yf_daily"))
    p.add_argument("--out-dir", default=os.path.join("data", "derived", "core"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--partition-by", choices=["ticker", "year_month", "none"], default="ticker")
    p.add_argument("--limit-tickers", type=int, default=0, help="If >0, process only first N tickers (test mode).")
    args = p.parse_args()

    if not os.path.isdir(args.in_dir):
        raise FileNotFoundError(f"Input directory not found: {args.in_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Optional: clear output directory
    if args.overwrite:
        for root, _, files in os.walk(args.out_dir):
            for f in files:
                if f.endswith(".parquet"):
                    try:
                        os.remove(os.path.join(root, f))
                    except OSError:
                        pass

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4;")

    glob_path = os.path.join(args.in_dir, "*.parquet")

    # Read all raw parquet files
    con.execute(f"""
        CREATE OR REPLACE VIEW raw AS
        SELECT * FROM read_parquet('{glob_path}');
    """)

    # Optional: limit tickers for testing
    if args.limit_tickers and args.limit_tickers > 0:
        n = int(args.limit_tickers)
        con.execute(f"""
            CREATE OR REPLACE VIEW raw_in AS
            SELECT *
            FROM raw
            WHERE Ticker IN (
                SELECT Ticker FROM raw GROUP BY 1 ORDER BY 1 LIMIT {n}
            );
        """)
        src = "raw_in"
    else:
        src = "raw"

    # Basic diagnostics
    diag = con.execute(f"""
        SELECT
          COUNT(*) AS n_rows,
          COUNT(DISTINCT Ticker) AS n_tickers,
          MIN(Date) AS min_date,
          MAX(Date) AS max_date
        FROM {src}
    """).fetchdf()

    print("\n[Raw stats]")
    print(diag.to_string(index=False))

    # Standardize + filter
    con.execute(f"""
        CREATE OR REPLACE VIEW core AS
        SELECT
          lower(Ticker) AS ticker,
          CAST(Date AS DATE) AS date,
          CAST(Open  AS DOUBLE) AS open,
          CAST(High  AS DOUBLE) AS high,
          CAST(Low   AS DOUBLE) AS low,
          CAST(Close AS DOUBLE) AS close,
          CAST(AdjClose AS DOUBLE) AS adjclose,
          CAST(Volume AS BIGINT) AS volume
        FROM {src}
        WHERE Date IS NOT NULL
          AND Close IS NOT NULL
          AND Volume IS NOT NULL
          AND Volume >= 0
          AND Open  > 0
          AND High  > 0
          AND Low   > 0
          AND Close > 0;
    """)

    # Deduplicate (ticker, date)
    con.execute("""
        CREATE OR REPLACE VIEW core_dedup AS
        SELECT ticker, date, open, high, low, close, adjclose, volume
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY ticker, date ORDER BY ticker) AS rn
            FROM core
        )
        WHERE rn = 1;
    """)

    # Choose final view
    final_view = "core_dedup"

    # Partition helpers
    if args.partition_by == "year_month":
        con.execute(f"""
            CREATE OR REPLACE VIEW out AS
            SELECT
              *,
              EXTRACT(year FROM date) AS year,
              LPAD(CAST(EXTRACT(month FROM date) AS VARCHAR), 2, '0') AS month
            FROM {final_view};
        """)
        out_view = "out"
        partition_cols = ["year", "month"]
    elif args.partition_by == "ticker":
        out_view = final_view
        partition_cols = ["ticker"]
    else:
        out_view = final_view
        partition_cols = []

    # Write output
    if partition_cols:
        part = ", ".join([f"'{c}'" for c in partition_cols])
        con.execute(f"""
            COPY (SELECT * FROM {out_view})
            TO '{args.out_dir}'
            (FORMAT PARQUET, PARTITION_BY ({part}), COMPRESSION ZSTD);
        """)
    else:
        out_file = os.path.join(args.out_dir, "core.parquet")
        con.execute(f"""
            COPY (SELECT * FROM {out_view})
            TO '{out_file}'
            (FORMAT PARQUET, COMPRESSION ZSTD);
        """)

    # Final sanity stats
    stats = con.execute(f"""
        SELECT
          COUNT(*) AS n_rows,
          COUNT(DISTINCT ticker) AS n_tickers,
          MIN(date) AS min_date,
          MAX(date) AS max_date
        FROM {out_view}
    """).fetchdf()

    print("\n[OK] Core dataset built:")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
