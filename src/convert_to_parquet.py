"""Convert raw hackathon CSVs to typed Parquet (streaming for the large tables) and print a profile."""
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Hackathon Data Set _ Gradient"
OUT = ROOT / "data" / "parquet"
OUT.mkdir(parents=True, exist_ok=True)

SWAP_SCHEMA = {
    "event_id": pl.Utf8, "rider_id": pl.Utf8, "station_id": pl.Utf8, "event_ts": pl.Utf8,
    "event_type": pl.Utf8, "attempt_seq": pl.Int16, "queue_wait_sec": pl.Int32,
    "battery_in_id": pl.Utf8, "battery_out_id": pl.Utf8,
    "soc_in_pct": pl.Float32, "soh_in_pct": pl.Float32, "soc_out_pct": pl.Float32, "soh_out_pct": pl.Float32,
    "km_since_last_swap": pl.Float32, "tariff_code": pl.Utf8, "list_price_inr": pl.Float32,
    "discount_inr": pl.Float32, "amount_charged_inr": pl.Float32, "payment_mode": pl.Utf8,
    "energy_to_recharge_kwh": pl.Float32, "station_firmware": pl.Utf8, "sync_mode": pl.Utf8,
}

HOURLY_SCHEMA = {
    "station_id": pl.Utf8, "hour_start": pl.Utf8, "charged_2w_avg": pl.Float32, "charged_2w_min": pl.Float32,
    "charged_3w_min": pl.Float32, "packs_charging": pl.Float32, "packs_quarantined": pl.Float32,
    "chargers_online": pl.Float32, "ambient_temp_c": pl.Float32, "cabinet_temp_c": pl.Float32,
    "avg_charge_minutes": pl.Float32, "outage_minutes": pl.Float32, "grid_kwh": pl.Float32,
    "telemetry_status": pl.Utf8,
}


def convert_big(name, schema, ts_col):
    t = time.time()
    lf = pl.scan_csv(RAW / f"{name}.csv", schema=schema)
    lf = lf.with_columns(pl.col(ts_col).str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S", strict=False))
    lf.sink_parquet(OUT / f"{name}.parquet", compression="zstd")
    print(f"[{name}] written in {time.time()-t:.1f}s")


def convert_small(name, date_cols=(), ts_cols=()):
    df = pl.read_csv(RAW / f"{name}.csv", infer_schema_length=None, try_parse_dates=False)
    for c in date_cols:
        df = df.with_columns(pl.col(c).cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False))
    for c in ts_cols:
        df = df.with_columns(pl.col(c).cast(pl.Utf8).str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S", strict=False))
    df.write_parquet(OUT / f"{name}.parquet", compression="zstd")
    print(f"[{name}] {df.shape} written")
    return df


def profile(name):
    df = pl.read_parquet(OUT / f"{name}.parquet")
    print("\n" + "=" * 100)
    print(f"TABLE {name}: shape={df.shape}")
    rows = []
    for c in df.columns:
        s = df[c]
        nulls = s.null_count()
        info = {"col": c, "dtype": str(s.dtype), "nulls": nulls, "null_pct": round(100 * nulls / len(s), 2)}
        if s.dtype == pl.Utf8:
            nun = s.n_unique()
            info["n_unique"] = nun
            if nun <= 30:
                vc = s.value_counts(sort=True)
                info["values"] = "; ".join(f"{r[0]}={r[1]}" for r in vc.iter_rows())
            else:
                info["sample"] = ", ".join(map(str, s.drop_nulls().head(3).to_list()))
        elif s.dtype in (pl.Boolean,):
            vc = s.value_counts(sort=True)
            info["values"] = "; ".join(f"{r[0]}={r[1]}" for r in vc.iter_rows())
        else:
            nn = s.drop_nulls()
            if len(nn):
                if s.dtype.is_numeric():
                    info["min"], info["p50"], info["max"], info["mean"] = (
                        nn.min(), round(float(nn.median()), 3), nn.max(), round(float(nn.mean()), 3))
                else:
                    info["min"], info["max"] = nn.min(), nn.max()
        rows.append(info)
    for r in rows:
        print("  " + " | ".join(f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    convert_small("stations", date_cols=["commissioned_date", "decommissioned_date", "firmware_updated_date",
                                          "competitor_within_1_5km_since"])
    convert_small("fleet_partners", date_cols=["contract_start_date", "amendment_date"])
    convert_small("riders", date_cols=["signup_date"])
    convert_small("batteries", date_cols=["manufacture_date", "commission_date", "retired_date"])
    convert_small("support_tickets", ts_cols=["created_ts"])
    convert_small("city_daily_context", date_cols=["date"])
    convert_big("station_hourly_status", HOURLY_SCHEMA, "hour_start")
    convert_big("swap_events", SWAP_SCHEMA, "event_ts")
    for n in ["stations", "fleet_partners", "riders", "batteries", "support_tickets", "city_daily_context",
              "station_hourly_status", "swap_events"]:
        profile(n)
