# scripts/04b_build_triggers.py
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import polars as pl


CORE_DIR = Path("data/derived/core")
OUT_DIR = Path("data/derived/triggers/current")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_H = [5, 10, 20, 50]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _atomic_write_parquet(df: pl.DataFrame, final_path: Path) -> None:
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    df.write_parquet(str(tmp_path), compression="zstd", statistics=True)
    if final_path.exists():
        final_path.unlink()
    tmp_path.rename(final_path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build trigger-conditioned trade sets from CORE (per-ticker loop; no hive partition required)."
    )
    p.add_argument("--core-dir", default=str(CORE_DIR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument(
        "--triggers",
        default=(
            "hammer,shooting_star,breakout_high,breakout_low,"
            "ema_cross_up,ema_cross_down,"
            "close_cross_above_ema,close_cross_below_ema,"
            "atr_expansion,range_expansion,volume_spike,vol_regime_shift"
        ),
    )
    p.add_argument("--horizons", default=",".join(map(str, DEFAULT_H)))

    # overwrite controls
    p.add_argument("--overwrite-all", action="store_true")
    p.add_argument("--overwrite-triggers", default="")

    # declustering + entry timing
    p.add_argument("--cooldown", type=int, default=0)
    p.add_argument("--entry-delay", type=int, default=0)

    # breakouts
    p.add_argument("--breakout-window", type=int, default=20)
    p.add_argument("--breakout-use-close", action="store_true")

    # ema cross fast/slow
    p.add_argument("--ema-fast", type=int, default=10)
    p.add_argument("--ema-slow", type=int, default=50)

    # close vs ema levels
    p.add_argument("--close-ema-levels", type=str, default="20,50,200")

    # atr expansion
    p.add_argument("--atr-window", type=int, default=14)
    p.add_argument("--atr-k", type=str, default="1.5,2.0,2.5")

    # range expansion
    p.add_argument("--range-window", type=int, default=20)
    p.add_argument("--range-k", type=str, default="1.5,2.0")

    # volume spike
    p.add_argument("--volume-ma-window", type=int, default=20)
    p.add_argument("--volume-k", type=str, default="1.5,2.0,3.0")

    # vol regime shift
    p.add_argument("--vol-fast", type=int, default=20)
    p.add_argument("--vol-slow", type=int, default=100)
    p.add_argument("--vol-k", type=str, default="1.25,1.5,2.0")

    # perf knobs
    p.add_argument("--limit-tickers", type=int, default=0)
    p.add_argument("--heartbeat-seconds", type=int, default=30)

    p.add_argument("--quality-dir", default=str(Path("data/derived/quality")))
    p.add_argument("--use-good-tickers", action="store_true", help="Restrict to quality-selected tickers (default ON).")
    p.add_argument("--no-good-tickers", dest="use_good_tickers", action="store_false", help="Do NOT restrict to good tickers.")
    p.set_defaults(use_good_tickers=True)

    p.add_argument("--include-spy", action="store_true", help="Ensure SPY is included (default ON).")
    p.add_argument("--no-spy", dest="include_spy", action="store_false")
    p.set_defaults(include_spy=True)


    return p.parse_args()


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return sorted(set(out))


def _parse_float_list(s: str) -> list[float]:
    out: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out


def _parse_str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def _load_good_tickers(quality_dir: Path, include_spy: bool) -> list[str]:
    good_path = quality_dir / "good_tickers.parquet"
    if not good_path.exists():
        raise FileNotFoundError(f"Missing {good_path}. Run Script 03 first.")
    good = pl.read_parquet(str(good_path)).select(pl.col("ticker").cast(pl.Utf8))
    tickers = good.get_column("ticker").unique().sort().to_list()
    if include_spy and "spy" not in set(tickers):
        tickers.append("spy")
        tickers = sorted(set(tickers))
    return tickers



def _ema_expr(x: pl.Expr, span: int) -> pl.Expr:
    alpha = 2.0 / (span + 1.0)
    return x.ewm_mean(alpha=alpha, adjust=False)


def _apply_cooldown(flag: pl.Expr, cooldown: int) -> pl.Expr:
    if cooldown <= 0:
        return flag
    win = cooldown + 1
    return pl.when(flag).then(
        flag.cast(pl.Int64).rolling_sum(window_size=win, min_samples=1) == 1
    ).otherwise(False)


def _fwd_ret_expr(h: int) -> pl.Expr:
    return (pl.col("adjclose").shift(-h) / pl.col("adjclose") - 1.0).alias(f"fwd_ret_{h}")


def _hammer_expr(o: pl.Expr, h: pl.Expr, l: pl.Expr, c: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    body = (c - o).abs()
    rng = (h - l)
    upper = h - pl.max_horizontal(o, c)
    lower = pl.min_horizontal(o, c) - l
    rng_ok = rng > 1e-12
    flag = rng_ok & (lower >= 2.0 * body) & (upper <= 1.0 * body) & (body <= 0.35 * rng)
    score = pl.when(rng_ok).then(lower / rng).otherwise(None)
    return flag, score


def _shooting_star_expr(o: pl.Expr, h: pl.Expr, l: pl.Expr, c: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    body = (c - o).abs()
    rng = (h - l)
    upper = h - pl.max_horizontal(o, c)
    lower = pl.min_horizontal(o, c) - l
    rng_ok = rng > 1e-12
    flag = rng_ok & (upper >= 2.0 * body) & (lower <= 1.0 * body) & (body <= 0.35 * rng)
    score = pl.when(rng_ok).then(upper / rng).otherwise(None)
    return flag, score


def _breakout_expr(x: pl.Expr, window: int, direction: str) -> tuple[pl.Expr, pl.Expr]:
    if direction == "high":
        ref = x.rolling_max(window_size=window, min_samples=window).shift(1)
        flag = (x > ref) & ref.is_not_null()
        score = pl.when(ref.is_not_null() & (ref.abs() > 1e-12)).then((x - ref) / ref.abs()).otherwise(None)
        return flag, score
    ref = x.rolling_min(window_size=window, min_samples=window).shift(1)
    flag = (x < ref) & ref.is_not_null()
    score = pl.when(ref.is_not_null() & (ref.abs() > 1e-12)).then((ref - x) / ref.abs()).otherwise(None)
    return flag, score


def _ema_cross_expr(close: pl.Expr, fast: int, slow: int) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    ef = _ema_expr(close, fast)
    es = _ema_expr(close, slow)
    ef1 = ef.shift(1)
    es1 = es.shift(1)
    up = (ef > es) & (ef1 <= es1) & ef1.is_not_null() & es1.is_not_null()
    dn = (ef < es) & (ef1 >= es1) & ef1.is_not_null() & es1.is_not_null()
    score = pl.when(close.abs() > 1e-12).then((ef - es) / close.abs()).otherwise(None)
    return up, dn, score


def _close_cross_ema_expr(close: pl.Expr, n: int, direction: str) -> tuple[pl.Expr, pl.Expr]:
    ema = _ema_expr(close, n)
    c1 = close.shift(1)
    e1 = ema.shift(1)
    if direction == "above":
        flag = (close > ema) & (c1 <= e1) & c1.is_not_null() & e1.is_not_null()
        score = pl.when(ema.abs() > 1e-12).then((close - ema) / ema.abs()).otherwise(None)
        return flag, score
    flag = (close < ema) & (c1 >= e1) & c1.is_not_null() & e1.is_not_null()
    score = pl.when(ema.abs() > 1e-12).then((ema - close) / ema.abs()).otherwise(None)
    return flag, score


def _true_range_expr() -> pl.Expr:
    prev_close = pl.col("close").shift(1)
    return pl.max_horizontal([
        (pl.col("high") - pl.col("low")).abs(),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    ])


def _logret_expr() -> pl.Expr:
    return (pl.col("adjclose") / pl.col("adjclose").shift(1)).log()


def _trigger_exists(out_dir: Path, trigger_name: str) -> bool:
    return (out_dir / f"trigger={trigger_name}" / "part-0.parquet").exists()


def _should_write_trigger(out_dir: Path, trigger_name: str, overwrite_all: bool, overwrite_set: set[str]) -> bool:
    if overwrite_all or trigger_name in overwrite_set:
        return True
    return not _trigger_exists(out_dir, trigger_name)


def _list_tickers(core_dir: Path) -> list[str]:
    return sorted([p.name.split("ticker=")[-1] for p in core_dir.glob("ticker=*") if p.is_dir()])


def main() -> None:
    args = _parse_args()
    t0 = time.time()

    core_dir = Path(args.core_dir)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    horizons = _parse_int_list(args.horizons)
    cooldown = int(args.cooldown)
    entry_delay = int(args.entry_delay)

    trigger_families = set(_parse_str_list(args.triggers))
    overwrite_set = set(_parse_str_list(args.overwrite_triggers))
    overwrite_all = bool(args.overwrite_all)

    close_ema_levels = _parse_int_list(args.close_ema_levels)
    atr_k_list = _parse_float_list(args.atr_k)
    range_k_list = _parse_float_list(args.range_k)
    volume_k_list = _parse_float_list(args.volume_k)
    vol_k_list = _parse_float_list(args.vol_k)

    atr_n = int(args.atr_window)
    range_n = int(args.range_window)
    vol_ma_n = int(args.volume_ma_window)
    vol_fast = int(args.vol_fast)
    vol_slow = int(args.vol_slow)

    quality_dir = Path(args.quality_dir)

    if args.use_good_tickers:
        tickers = _load_good_tickers(quality_dir, include_spy=bool(args.include_spy))
    else:
        tickers = _list_tickers(core_dir)
        if args.include_spy and "spy" not in set(tickers):
            tickers.append("spy")
            tickers = sorted(set(tickers))

        if args.limit_tickers and args.limit_tickers > 0:
            tickers = tickers[: int(args.limit_tickers)]

        print(f"[INFO] tickers={len(tickers)} core_dir={core_dir}")
        print(f"[INFO] cooldown={cooldown} entry_delay={entry_delay} horizons={horizons}")

    # Determine concrete trigger names we might write
    # (We store outputs in dict: trigger_name -> list of per-ticker DataFrames)
    outputs: Dict[str, List[pl.DataFrame]] = {}

    def add_df(trigger_name: str, df: pl.DataFrame) -> None:
        if df.height == 0:
            return
        outputs.setdefault(trigger_name, []).append(df)

    # Metadata
    meta_path = out_dir / "_metadata.json"
    meta = _read_json(meta_path)
    meta.setdefault("built_history", [])
    meta.setdefault("triggers", {})
    meta.setdefault("row_counts", {})

    # Precompute which triggers are needed (skip existing unless overwrite)
    planned: Dict[str, dict] = {}

    def plan(trigger_name: str, cfg: dict) -> bool:
        if not _should_write_trigger(out_dir, trigger_name, overwrite_all, overwrite_set):
            print(f"[SKIP] trigger={trigger_name} already exists")
            return False
        planned[trigger_name] = cfg
        return True

    # Plan triggers
    if "hammer" in trigger_families:
        plan("hammer", {"family": "hammer"})
    if "shooting_star" in trigger_families:
        plan("shooting_star", {"family": "shooting_star"})

    if "breakout_high" in trigger_families:
        name = f"breakout_high_{int(args.breakout_window)}{'_close' if args.breakout_use_close else '_high'}"
        plan(name, {"family": "breakout_high", "window": int(args.breakout_window), "use_close": bool(args.breakout_use_close)})
    if "breakout_low" in trigger_families:
        name = f"breakout_low_{int(args.breakout_window)}{'_close' if args.breakout_use_close else '_low'}"
        plan(name, {"family": "breakout_low", "window": int(args.breakout_window), "use_close": bool(args.breakout_use_close)})

    if "ema_cross_up" in trigger_families:
        plan(f"ema_cross_up_{int(args.ema_fast)}_{int(args.ema_slow)}", {"family": "ema_cross_up", "fast": int(args.ema_fast), "slow": int(args.ema_slow)})
    if "ema_cross_down" in trigger_families:
        plan(f"ema_cross_down_{int(args.ema_fast)}_{int(args.ema_slow)}", {"family": "ema_cross_down", "fast": int(args.ema_fast), "slow": int(args.ema_slow)})

    if "close_cross_above_ema" in trigger_families:
        for n in close_ema_levels:
            plan(f"close_cross_above_ema_{n}", {"family": "close_cross_above_ema", "ema": n})
    if "close_cross_below_ema" in trigger_families:
        for n in close_ema_levels:
            plan(f"close_cross_below_ema_{n}", {"family": "close_cross_below_ema", "ema": n})

    if "atr_expansion" in trigger_families:
        for k in atr_k_list:
            plan(f"atr_expansion_{k:.2f}x_atr_{atr_n}", {"family": "atr_expansion", "atr_window": atr_n, "k": float(k)})

    if "range_expansion" in trigger_families:
        for k in range_k_list:
            plan(f"range_expansion_{k:.2f}x_mean_{range_n}", {"family": "range_expansion", "range_window": range_n, "k": float(k)})

    if "volume_spike" in trigger_families:
        for k in volume_k_list:
            plan(f"volume_spike_{k:.2f}x_ma_{vol_ma_n}", {"family": "volume_spike", "volume_ma_window": vol_ma_n, "k": float(k)})

    if "vol_regime_shift" in trigger_families:
        for k in vol_k_list:
            plan(f"vol_regime_{vol_fast}_over_{vol_slow}_x{k:.2f}", {"family": "vol_regime_shift", "vol_fast": vol_fast, "vol_slow": vol_slow, "k": float(k)})

    if not planned:
        print("[OK] Nothing to do (all requested triggers already exist).")
        return

    # Per-ticker loop
    last_heartbeat = time.time()

    for i, tkr in enumerate(tickers, 1):
        ticker_path = core_dir / f"ticker={tkr}"
        files = list(ticker_path.glob("*.parquet"))
        if not files:
            continue

        df = (
            pl.scan_parquet(str(ticker_path / "*.parquet"))
            .select(["date", "open", "high", "low", "close", "adjclose", "volume"])
            .with_columns([
                pl.lit(tkr).cast(pl.Utf8).alias("ticker"),
                pl.col("date").cast(pl.Date),
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

        # outcomes
        for hh in horizons:
            df = df.with_columns([(pl.col("adjclose").shift(-hh) / pl.col("adjclose") - 1.0).alias(f"fwd_ret_{hh}")])

        # entry delay: shift outcomes backward
        if entry_delay != 0:
            for hh in horizons:
                df = df.with_columns([pl.col(f"fwd_ret_{hh}").shift(-entry_delay).alias(f"fwd_ret_{hh}")])

        # shared series
        df = df.with_columns([
            _true_range_expr().alias("true_range"),
            _logret_expr().alias("logret_1d"),
            (pl.col("high") - pl.col("low")).alias("range_hl"),
        ])

        df = df.with_columns([
            pl.col("true_range").rolling_mean(window_size=atr_n, min_samples=atr_n).alias(f"atr_{atr_n}"),
            pl.col("range_hl").rolling_mean(window_size=range_n, min_samples=range_n).alias(f"range_mean_{range_n}"),
            pl.col("volume").rolling_mean(window_size=vol_ma_n, min_samples=vol_ma_n).alias(f"volume_ma_{vol_ma_n}"),
            pl.col("logret_1d").rolling_std(window_size=vol_fast, min_samples=vol_fast).alias(f"vol_{vol_fast}"),
            pl.col("logret_1d").rolling_std(window_size=vol_slow, min_samples=vol_slow).alias(f"vol_{vol_slow}"),
        ])

        o = pl.col("open")
        h = pl.col("high")
        l = pl.col("low")
        c = pl.col("close")

        # helper to extract rows for a trigger
        def emit(trigger_name: str, side: int, flag: pl.Expr, score: pl.Expr) -> None:
            if trigger_name not in planned:
                return
            sub = (
                df.lazy()
                .with_columns([
                    _apply_cooldown(flag, cooldown).alias("flag"),
                    score.alias("score"),
                ])
                .filter(pl.col("flag") == True)
                .select([
                    "ticker", "date",
                    pl.lit(trigger_name).alias("trigger"),
                    pl.lit(int(side)).alias("side"),
                    pl.col("score").cast(pl.Float64).alias("score"),
                    pl.col("adjclose").cast(pl.Float64).alias("adjclose"),
                    pl.col("volume").cast(pl.Int64).alias("volume"),
                    *[pl.col(f"fwd_ret_{hh}").cast(pl.Float64) for hh in horizons],
                ])
                .filter(pl.all_horizontal([pl.col(f"fwd_ret_{hh}").is_not_null() for hh in horizons]))
                .collect()
            )
            add_df(trigger_name, sub)

        # hammer / star
        if "hammer" in planned:
            flag, score = _hammer_expr(o, h, l, c)
            emit("hammer", +1, flag, score)
        if "shooting_star" in planned:
            flag, score = _shooting_star_expr(o, h, l, c)
            emit("shooting_star", -1, flag, score)

        # breakouts
        if any(k.startswith("breakout_high_") for k in planned):
            x = pl.col("close") if args.breakout_use_close else pl.col("high")
            flag, score = _breakout_expr(x, int(args.breakout_window), "high")
            emit(f"breakout_high_{int(args.breakout_window)}{'_close' if args.breakout_use_close else '_high'}", +1, flag, score)

        if any(k.startswith("breakout_low_") for k in planned):
            x = pl.col("close") if args.breakout_use_close else pl.col("low")
            flag, score = _breakout_expr(x, int(args.breakout_window), "low")
            emit(f"breakout_low_{int(args.breakout_window)}{'_close' if args.breakout_use_close else '_low'}", -1, flag, score)

        # ema fast/slow cross
        if any(k.startswith("ema_cross_") for k in planned):
            up, dn, score = _ema_cross_expr(c, int(args.ema_fast), int(args.ema_slow))
            emit(f"ema_cross_up_{int(args.ema_fast)}_{int(args.ema_slow)}", +1, up, score)
            emit(f"ema_cross_down_{int(args.ema_fast)}_{int(args.ema_slow)}", -1, dn, score)

        # close vs ema level crosses
        for n in close_ema_levels:
            emit(f"close_cross_above_ema_{n}", +1, _close_cross_ema_expr(c, n, "above")[0], _close_cross_ema_expr(c, n, "above")[1])
            emit(f"close_cross_below_ema_{n}", -1, _close_cross_ema_expr(c, n, "below")[0], _close_cross_ema_expr(c, n, "below")[1])

        # atr expansion
        tr_col = pl.col("true_range")
        atr_col = pl.col(f"atr_{atr_n}")
        for k in atr_k_list:
            name = f"atr_expansion_{k:.2f}x_atr_{atr_n}"
            flag = atr_col.is_not_null() & (atr_col > 1e-12) & (tr_col > (pl.lit(float(k)) * atr_col))
            score = pl.when(atr_col.is_not_null() & (atr_col > 1e-12)).then(tr_col / atr_col).otherwise(None)
            emit(name, +1, flag, score)

        # range expansion
        r = pl.col("range_hl")
        rm = pl.col(f"range_mean_{range_n}")
        for k in range_k_list:
            name = f"range_expansion_{k:.2f}x_mean_{range_n}"
            flag = rm.is_not_null() & (rm > 1e-12) & (r > (pl.lit(float(k)) * rm))
            score = pl.when(rm.is_not_null() & (rm > 1e-12)).then(r / rm).otherwise(None)
            emit(name, +1, flag, score)

        # volume spike
        v = pl.col("volume").cast(pl.Float64)
        vm = pl.col(f"volume_ma_{vol_ma_n}").cast(pl.Float64)
        for k in volume_k_list:
            name = f"volume_spike_{k:.2f}x_ma_{vol_ma_n}"
            flag = vm.is_not_null() & (vm > 1e-12) & (v > (pl.lit(float(k)) * vm))
            score = pl.when(vm.is_not_null() & (vm > 1e-12)).then(v / vm).otherwise(None)
            emit(name, +1, flag, score)

        # vol regime shift
        vf = pl.col(f"vol_{vol_fast}")
        vs = pl.col(f"vol_{vol_slow}")
        for k in vol_k_list:
            name = f"vol_regime_{vol_fast}_over_{vol_slow}_x{k:.2f}"
            flag = vs.is_not_null() & (vs > 1e-12) & vf.is_not_null() & (vf > (pl.lit(float(k)) * vs))
            score = pl.when(vs.is_not_null() & (vs > 1e-12) & vf.is_not_null()).then(vf / vs).otherwise(None)
            emit(name, +1, flag, score)

        # heartbeat
        if time.time() - last_heartbeat >= int(args.heartbeat_seconds):
            elapsed = time.time() - t0
            print(f"[HEARTBEAT] processed={i}/{len(tickers)} elapsed={elapsed:.1f}s")
            last_heartbeat = time.time()

    # Write triggers
    for trigger_name, dfs in outputs.items():
        out_path = out_dir / f"trigger={trigger_name}"
        _ensure_dir(out_path)

        full = pl.concat(dfs, how="vertical") if len(dfs) > 1 else dfs[0]
        _atomic_write_parquet(full, out_path / "part-0.parquet")

        meta["triggers"][trigger_name] = {
            "config": planned[trigger_name],
            "cooldown": int(args.cooldown),
            "entry_delay": int(args.entry_delay),
            "horizons": horizons,
            "built_utc": datetime.now(timezone.utc).isoformat(),
        }
        meta["row_counts"][trigger_name] = int(full.height)

        print(f"[OK] wrote trigger={trigger_name} rows={full.height}")

    meta["built_history"].append({
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "core_dir": str(core_dir),
        "out_dir": str(out_dir),
        "trigger_families_requested": sorted(trigger_families),
        "overwrite_all": bool(args.overwrite_all),
        "overwrite_triggers": sorted(overwrite_set),
        "cooldown": int(args.cooldown),
        "entry_delay": int(args.entry_delay),
        "horizons": horizons,
    })
    _write_json(out_dir / "_metadata.json", meta)

    elapsed = time.time() - t0
    print(f"[OK] done. elapsed={elapsed:.1f}s")
    print(f"[Tip] See: {out_dir / '_metadata.json'}")


if __name__ == "__main__":
    main()
