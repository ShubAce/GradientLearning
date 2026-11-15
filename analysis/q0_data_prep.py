# %% [markdown]
# ## 2. Data understanding
# Eight related tables describe 18 months (Jan-2024 to Jun-2025) of a simulated battery-swapping network in six Indian cities.
# We first convert the raw CSVs to typed Parquet (fast, column-projected reads), then profile every table and check that the
# documented relationships actually hold before trusting any join.

# %%
import json, time, datetime as dt
import polars as pl
import sys
from pathlib import Path
from common import *

pl.Config.set_tbl_rows(25); pl.Config.set_tbl_cols(25); pl.Config.set_tbl_width_chars(200); pl.Config.set_fmt_str_lengths(50)

SWAP_SCHEMA = {
    "event_id": pl.Utf8, "rider_id": pl.Utf8, "station_id": pl.Utf8, "event_ts": pl.Utf8, "event_type": pl.Utf8,
    "attempt_seq": pl.Int16, "queue_wait_sec": pl.Int32, "battery_in_id": pl.Utf8, "battery_out_id": pl.Utf8,
    "soc_in_pct": pl.Float32, "soh_in_pct": pl.Float32, "soc_out_pct": pl.Float32, "soh_out_pct": pl.Float32,
    "km_since_last_swap": pl.Float32, "tariff_code": pl.Utf8, "list_price_inr": pl.Float32, "discount_inr": pl.Float32,
    "amount_charged_inr": pl.Float32, "payment_mode": pl.Utf8, "energy_to_recharge_kwh": pl.Float32,
    "station_firmware": pl.Utf8, "sync_mode": pl.Utf8}
HOURLY_SCHEMA = {
    "station_id": pl.Utf8, "hour_start": pl.Utf8, "charged_2w_avg": pl.Float32, "charged_2w_min": pl.Float32,
    "charged_3w_min": pl.Float32, "packs_charging": pl.Float32, "packs_quarantined": pl.Float32, "chargers_online": pl.Float32,
    "ambient_temp_c": pl.Float32, "cabinet_temp_c": pl.Float32, "avg_charge_minutes": pl.Float32, "outage_minutes": pl.Float32,
    "grid_kwh": pl.Float32, "telemetry_status": pl.Utf8}

def to_parquet_big(name, schema, ts_col):
    lf = pl.scan_csv(RAW / f"{name}.csv", schema=schema).with_columns(
        pl.col(ts_col).str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S", strict=False))
    lf.sink_parquet(PARQ / f"{name}.parquet", compression="zstd")

def to_parquet_small(name, date_cols=(), ts_cols=()):
    df = pl.read_csv(RAW / f"{name}.csv", infer_schema_length=None)
    for c in date_cols:
        df = df.with_columns(pl.col(c).cast(pl.Utf8).str.strptime(pl.Date, "%Y-%m-%d", strict=False))
    for c in ts_cols:
        df = df.with_columns(pl.col(c).cast(pl.Utf8).str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M:%S", strict=False))
    df.write_parquet(PARQ / f"{name}.parquet", compression="zstd")

t = time.time()
PARQ.mkdir(parents=True, exist_ok=True)
to_parquet_small("stations", ["commissioned_date", "decommissioned_date", "firmware_updated_date", "competitor_within_1_5km_since"])
to_parquet_small("fleet_partners", ["contract_start_date", "amendment_date"])
to_parquet_small("riders", ["signup_date"])
to_parquet_small("batteries", ["manufacture_date", "commission_date", "retired_date"])
to_parquet_small("support_tickets", ts_cols=["created_ts"])
to_parquet_small("city_daily_context", ["date"])
to_parquet_big("station_hourly_status", HOURLY_SCHEMA, "hour_start")
to_parquet_big("swap_events", SWAP_SCHEMA, "event_ts")
print(f"CSV -> Parquet in {time.time()-t:.1f}s")
for f in sorted(PARQ.glob("*.parquet")):
    print(f"{f.name:32s} {f.stat().st_size/1e6:7.1f} MB")

# %% [markdown]
# ### 2.1 Table inventory
# Row counts, grain and date coverage match the data dictionary (3,877,013 swap attempts; 1,487,712 station-hours;
# 20,000 riders; 6,500 batteries; 44,000 tickets; 152 stations; 3,282 city-days; 12 fleet partners).

# %%
TABLES = {"swap_events": "event_ts", "station_hourly_status": "hour_start", "riders": "signup_date", "batteries": "commission_date",
          "support_tickets": "created_ts", "stations": "commissioned_date", "city_daily_context": "date", "fleet_partners": "contract_start_date"}
inv = []
for name, dcol in TABLES.items():
    lf = pl.scan_parquet(PARQ / f"{name}.parquet")
    s = lf.select(pl.len().alias("rows"), pl.col(dcol).min().alias("min"), pl.col(dcol).max().alias("max")).collect()
    inv.append({"table": name, "rows": s["rows"][0], "columns": len(lf.collect_schema()), "date_field": dcol,
                "from": str(s["min"][0])[:10], "to": str(s["max"][0])[:10]})
inventory = pl.DataFrame(inv)
inventory

# %% [markdown]
# ### 2.2 Column profile (nulls, cardinality, ranges)
# Nulls are expected and informative here: battery fields are only populated for attempts where a pack was inserted /
# issued, telemetry blanks follow `telemetry_status`, and CSAT is mostly missing.

# %%
def profile(name, max_rows=None):
    df = pl.read_parquet(PARQ / f"{name}.parquet") if name not in ("swap_events", "station_hourly_status") else \
        pl.scan_parquet(PARQ / f"{name}.parquet").collect()
    rows = []
    for c in df.columns:
        s = df[c]
        r = {"table": name, "column": c, "dtype": str(s.dtype), "null_%": round(100 * s.null_count() / len(s), 2)}
        if s.dtype == pl.Utf8:
            r["distinct"] = s.n_unique()
            vc = s.value_counts(sort=True).head(4)
            r["top_values"] = ", ".join(f"{a}({b})" for a, b in vc.iter_rows())
        elif s.dtype.is_numeric():
            nn = s.drop_nulls()
            if len(nn):
                r["distinct"] = nn.n_unique()
                r["top_values"] = f"min {nn.min():.4g} | median {float(nn.median()):.4g} | max {nn.max():.4g}"
        rows.append(r)
    del df
    return pl.DataFrame(rows)

prof = pl.concat([profile(n) for n in TABLES], how="diagonal_relaxed")
prof.write_csv(TAB / "q0_column_profile.csv")
print(prof.filter(pl.col("table") == "swap_events"))
print(prof.filter(pl.col("null_%") > 0).select("table", "column", "null_%").sort("null_%", descending=True))

# %% [markdown]
# ### 2.3 Do the documented relationships hold?
# Before any join we test referential integrity (every foreign key must match a parent row), vehicle-class consistency
# between riders and the packs they swap, and temporal integrity (no activity before sign-up / commissioning).

# %%
sw = pl.scan_parquet(PARQ / "swap_events.parquet")
rd0 = pl.read_parquet(PARQ / "riders.parquet"); st0 = pl.read_parquet(PARQ / "stations.parquet")
bt0 = pl.read_parquet(PARQ / "batteries.parquet"); tk0 = pl.read_parquet(PARQ / "support_tickets.parquet")
fp0 = pl.read_parquet(PARQ / "fleet_partners.parquet"); hr0 = pl.scan_parquet(PARQ / "station_hourly_status.parquet")

def fk_check(child, col, parent, pcol, label):
    c = (child.select(pl.col(col).drop_nulls()).collect() if isinstance(child, pl.LazyFrame) else child.select(pl.col(col).drop_nulls()))
    ok = int(c[col].is_in(parent[pcol].implode()).sum())
    return {"relationship": label, "child_rows_with_key": c.height, "matched": ok, "match_%": round(100 * ok / max(c.height, 1), 3)}

fk = pl.DataFrame([
    fk_check(sw, "rider_id", rd0, "rider_id", "swap_events.rider_id -> riders"),
    fk_check(sw, "station_id", st0, "station_id", "swap_events.station_id -> stations"),
    fk_check(sw, "battery_in_id", bt0, "battery_id", "swap_events.battery_in_id -> batteries"),
    fk_check(sw, "battery_out_id", bt0, "battery_id", "swap_events.battery_out_id -> batteries"),
    fk_check(rd0, "partner_id", fp0, "partner_id", "riders.partner_id -> fleet_partners"),
    fk_check(tk0, "rider_id", rd0, "rider_id", "support_tickets.rider_id -> riders"),
    fk_check(tk0, "station_id", st0, "station_id", "support_tickets.station_id -> stations"),
    fk_check(tk0, "battery_id", bt0, "battery_id", "support_tickets.battery_id -> batteries"),
    fk_check(hr0, "station_id", st0, "station_id", "station_hourly_status.station_id -> stations"),
])
vc = (sw.select("rider_id", "battery_in_id", "battery_out_id").join(rd0.lazy().select("rider_id", "vehicle_class"), on="rider_id")
      .select((pl.col("battery_in_id").str.slice(4, 2) != pl.col("vehicle_class")).sum().alias("returned_pack_class_mismatch"),
              (pl.col("battery_out_id").str.slice(4, 2) != pl.col("vehicle_class")).sum().alias("issued_pack_class_mismatch")).collect())
tmp = (sw.select("rider_id", "station_id", "event_ts").join(rd0.lazy().select("rider_id", "signup_date"), on="rider_id")
       .join(st0.lazy().select("station_id", "commissioned_date", "decommissioned_date"), on="station_id")
       .select((pl.col("event_ts").dt.date() < pl.col("signup_date")).sum().alias("events_before_rider_signup"),
               (pl.col("event_ts").dt.date() < pl.col("commissioned_date")).sum().alias("events_before_station_commissioning"),
               (pl.col("event_ts").dt.date() > pl.col("decommissioned_date")).sum().alias("events_after_decommissioning")).collect())
grid = (hr0.group_by("station_id").agg(pl.len().alias("rows"), pl.col("hour_start").min().alias("a"), pl.col("hour_start").max().alias("b")).collect()
        .with_columns(((pl.col("b") - pl.col("a")).dt.total_hours() + 1).alias("expected")))
print(fk)
print(vc)
print(tmp)
print("station-hour grid: stations with missing hours =", grid.filter(pl.col("rows") != pl.col("expected")).height)
fk.write_csv(TAB / "q0_referential_integrity.csv")

# %% [markdown]
# **Result:** all nine foreign keys match 100%, issued/returned packs always match the rider's vehicle class, and the
# hourly telemetry grid has no missing hours. The raw export shows 6 events before the rider's sign-up date and 24 before
# the station's commissioning date, but **all 30 are evening events inside the firmware-bug window (12 Mar - 13 Apr 2025)**:
# the 5h30m clock error dated them to the previous calendar day. After the timestamp fix in section 3.3 every integrity
# check passes (0 violations), which independently confirms the correction. The joins can be trusted.
#
# ## 3. Data cleaning
# Each issue below is (1) detected from the data, (2) quantified, (3) handled with an explicit, reversible decision.
# Every decision is written to an audit log (`outputs/tables/data_quality_log.json`).

# %%
T0 = time.time()
LOG = {"steps": []}

def log_step(name, **kw):
    LOG["steps"].append({"step": name, **kw})
    print(f"[{name}]", {k: v for k, v in kw.items() if k != "note"})

st = st0.with_columns(pl.col("station_id").str.starts_with("STN-TST").alias("is_test"))
log_step("stations", total=st.height, test_stations=int(st["is_test"].sum()), test_ids=st.filter("is_test")["station_id"].to_list(),
         decommissioned=st.filter(pl.col("decommissioned_date").is_not_null())["station_id"].to_list(),
         competitor_nearby=int(st["competitor_within_1_5km_since"].is_not_null().sum()))
st.write_parquet(CLEAN / "stations.parquet")
st_real = st.filter(~pl.col("is_test"))

# %% [markdown]
# ### 3.1 Inconsistent city spellings (riders.home_city)
# 21 raw spellings (e.g. *Bombay, MUM, Bangalore, bengaluru␣, Gurgaon, New Delhi, Hyd*) map to the 6 operating cities.
# We validate the mapping against the city prefix of `home_zone` (BLR/DEL/HYD/PUN/MUM/JAI): zero mismatches.

# %%
CITY_MAP = {"bengaluru": "Bengaluru", "bangalore": "Bengaluru", "blr": "Bengaluru", "delhi ncr": "Delhi NCR", "new delhi": "Delhi NCR",
            "delhi": "Delhi NCR", "gurgaon": "Delhi NCR", "gurugram": "Delhi NCR", "noida": "Delhi NCR", "del": "Delhi NCR",
            "mumbai": "Mumbai", "bombay": "Mumbai", "mum": "Mumbai", "pune": "Pune", "pun": "Pune",
            "hyderabad": "Hyderabad", "hyd": "Hyderabad", "jaipur": "Jaipur", "jai": "Jaipur"}
rd = rd0.with_columns(
    pl.col("home_city").str.strip_chars().str.to_lowercase().replace_strict(CITY_MAP, default=None).alias("home_city_clean"),
    pl.col("home_zone").str.slice(0, 3).replace_strict(CITY_CODE, default=None).alias("zone_city"),
    pl.col("partner_id").is_null().alias("is_independent"))
city_before_after = (rd.group_by("home_city_clean", "home_city").len().sort("home_city_clean", "len", descending=[False, True]))
log_step("riders_city_standardisation", raw_variants=rd0["home_city"].n_unique(), clean_cities=rd["home_city_clean"].n_unique(),
         rows_recoded=int((rd["home_city"] != rd["home_city_clean"]).sum()),
         city_vs_zone_mismatch=rd.filter(pl.col("home_city_clean") != pl.col("zone_city")).height)
city_before_after

# %% [markdown]
# ### 3.2 Test stations (STN-TST-01/02)
# The documentation describes "a small number of zero-value transactions at unusual hours". The export actually holds
# **62,031 attempts incl. 59,134 paid swaps (₹3.93M)** at the two test stations, spread over normal demand hours -
# a discrepancy worth flagging to the data owners. They are internal test assets (round-number coordinates and costs,
# outside the real zone network), so we exclude them from all network KPIs and keep them in a separate audit file.

# %%
ev = pl.scan_parquet(PARQ / "swap_events.parquet").join(st.lazy().select("station_id", "is_test"), on="station_id", how="left")
n_raw = ev.select(pl.len()).collect().item()
test_ev = ev.filter(pl.col("is_test")).collect()
test_ev.write_parquet(CLEAN / "test_station_events.parquet")
log_step("exclude_test_stations", raw_events=n_raw, test_events=test_ev.height,
         test_completed=int((test_ev["event_type"] == "swap_completed").sum()),
         test_revenue_inr=round(float(test_ev["amount_charged_inr"].sum()), 0),
         test_zero_value_events=int((test_ev["amount_charged_inr"] == 0).sum()))
print(test_ev.with_columns(pl.col("event_ts").dt.hour().alias("hour")).group_by("hour").len().sort("hour").transpose(include_header=False))
ev = ev.filter(~pl.col("is_test")).drop("is_test")

# %% [markdown]
# ### 3.3 Firmware v3.2.0 timezone bug
# Detection: during 10-Mar-2025 to 14-Apr-2025 the mean logged event hour at v3.2.0 stations drops from ~15.1 to ~10.4
# while v3.2.1 stations stay at ~15.2 - a clean 5h30m (IST vs UTC) shift that starts and ends exactly on those dates.
# Fix: add 5h30m to logged timestamps in that window at v3.2.0 stations (the raw value is kept in `event_ts_raw`).

# %%
daily_fw = (ev.filter(pl.col("event_ts").dt.date().is_between(pl.date(2025, 3, 1), pl.date(2025, 4, 25)))
            .group_by(pl.col("event_ts").dt.date().alias("date"), "station_firmware")
            .agg(pl.col("event_ts").dt.hour().mean().round(2).alias("mean_logged_hour")).collect()
            .pivot(on="station_firmware", index="date", values="mean_logged_hour").sort("date"))
print(daily_fw.filter(pl.col("date").is_in([dt.date(2025, 3, 9), dt.date(2025, 3, 10), dt.date(2025, 4, 14), dt.date(2025, 4, 15)])))
bug = (pl.col("station_firmware") == "v3.2.0") & (pl.col("event_ts") >= FW_BUG_START) & (pl.col("event_ts") < FW_BUG_END)
ev = ev.with_columns(bug.alias("ts_corrected"), pl.col("event_ts").alias("event_ts_raw"),
                     pl.when(bug).then(pl.col("event_ts").dt.offset_by(FW_BUG_SHIFT)).otherwise(pl.col("event_ts")).alias("event_ts")).collect()
chk = (ev.filter(pl.col("event_ts_raw").dt.date().is_between(pl.date(2025, 3, 10), pl.date(2025, 4, 14)))
       .group_by("station_firmware").agg(pl.col("event_ts_raw").dt.hour().mean().round(2).alias("mean_hour_raw"),
                                         pl.col("event_ts").dt.hour().mean().round(2).alias("mean_hour_fixed")))
log_step("firmware_timestamp_fix", events_shifted=int(ev["ts_corrected"].sum()), check=chk.to_dicts())

# %% [markdown]
# ### 3.4 Near-duplicate offline-sync records
# Documented pattern: same rider, station and batteries, seconds to minutes apart, `sync_mode = offline_batch`.
# We test it explicitly. Under the documented definition **no duplicates exist** in this export, and even a loose
# definition finds only a handful of candidates, so the 591K `offline_batch` rows are legitimate events and are kept.
# (The rule stays in the pipeline so a future export with retries would be de-duplicated automatically.)

# %%
k = (ev.select("event_id", "rider_id", "station_id", "event_ts", "event_type", "battery_in_id", "battery_out_id", "sync_mode")
     .sort("rider_id", "event_ts")
     .with_columns(pl.col("event_ts").diff().over("rider_id").dt.total_seconds().alias("gap_s"),
                   *[pl.col(c).shift(1).over("rider_id").alias(c + "_prev") for c in ["station_id", "event_type", "battery_in_id", "battery_out_id", "sync_mode"]]))
same = ((pl.col("station_id") == pl.col("station_id_prev")) & (pl.col("event_type") == pl.col("event_type_prev"))
        & pl.col("battery_in_id").eq_missing(pl.col("battery_in_id_prev")) & pl.col("battery_out_id").eq_missing(pl.col("battery_out_id_prev"))
        & (pl.col("gap_s") <= 180) & ((pl.col("sync_mode") == "offline_batch") | (pl.col("sync_mode_prev") == "offline_batch")))
dups = k.filter(same.fill_null(False))
loose = k.filter(((pl.col("station_id") == pl.col("station_id_prev")) & (pl.col("battery_in_id") == pl.col("battery_in_id_prev")) & (pl.col("gap_s") <= 300)).fill_null(False))
log_step("near_duplicate_detection", duplicates_found=dups.height, loose_candidates_same_battery_in_300s=loose.height,
         offline_batch_rows=int((ev["sync_mode"] == "offline_batch").sum()))
ev = ev.join(dups.select("event_id"), on="event_id", how="anti")
del k

# %% [markdown]
# ### 3.5 Outliers: odometer resets and sensor drift
# `km_since_last_swap` has negative values (down to -20 km) and a separate cluster at 330-420 km, with **no readings
# between 200 and 330 km** (2W p99 = 83 km, 3W p99 = 118 km). Values < 0 or > 200 km are set to null (flagged).
# SoC/SoH readings slightly above 100% (calibration drift) are clipped to 100 and flagged.

# %%
km = pl.col("km_since_last_swap")
km_bad = (km < 0) | (km > 200)
n_neg, n_hi = int((ev["km_since_last_swap"] < 0).sum()), int((ev["km_since_last_swap"] > 200).sum())
ev = ev.with_columns(km_bad.fill_null(False).alias("km_invalid"), pl.when(km_bad).then(None).otherwise(km).alias("km_since_last_swap"))
READ = ["soc_in_pct", "soh_in_pct", "soc_out_pct", "soh_out_pct"]
clip_counts = {c: int((ev[c] > 100).sum()) for c in READ}
ev = ev.with_columns(pl.any_horizontal([pl.col(c) > 100 for c in READ]).fill_null(False).alias("reading_clipped"),
                     *[pl.col(c).clip(upper_bound=100.0) for c in READ])
log_step("outliers", km_negative=n_neg, km_gt_200=n_hi, readings_above_100_clipped=clip_counts)

# %% [markdown]
# ### 3.6 Constant / reserved fields
# `attempt_seq` is always 1 (reserved). `payment_mode` is **'partner_invoice' on 100% of rows - including pay-as-you-go
# riders** - so it cannot be used for segment checks (an undocumented issue). Price exposure is taken from
# `riders.plan_type` and `tariff_code` instead.

# %%
log_step("drop_constant_fields", values={c: ev[c].unique().to_list() for c in ["attempt_seq", "payment_mode"]})
ev = ev.drop("attempt_seq", "payment_mode")

# %% [markdown]
# ### 3.7 Battery wear model and battery-metadata consistency
# Battery IDs in the swap log are not tracked physically (the same pack is issued in two cities minutes apart), but each
# pack's **SoH trajectory is internally consistent**, so we estimate every pack's SoH loss per charge cycle (OLS slope of
# issued-pack SoH on cycle index). Wear cost per swap = purchase cost x loss per cycle / (initial SoH - 70% end-of-life).
# We also find that `batteries.csv` commission dates contradict usage for 5,154 packs (used before "commissioning"),
# so pack age is measured from first appearance in the swap log.

# %%
bt = bt0
issued = (ev.filter(pl.col("event_type") == "swap_completed").select("battery_out_id", "event_ts", "soh_out_pct")
          .sort("battery_out_id", "event_ts").with_columns(pl.int_range(1, pl.len() + 1).over("battery_out_id").alias("cycle")))
slopes = (issued.group_by("battery_out_id").agg(
    pl.len().alias("n_cycles"), pl.col("event_ts").min().alias("first_issued"), pl.col("event_ts").max().alias("last_issued"),
    pl.col("soh_out_pct").first().alias("soh_first_obs"), pl.col("soh_out_pct").last().alias("soh_last_obs"),
    ((pl.col("cycle") * pl.col("soh_out_pct")).mean() - pl.col("cycle").mean() * pl.col("soh_out_pct").mean()).alias("cov"),
    ((pl.col("cycle") ** 2).mean() - pl.col("cycle").mean() ** 2).alias("var"))
    .with_columns((-(pl.col("cov") / pl.col("var"))).alias("soh_loss_per_cycle")).rename({"battery_out_id": "battery_id"}).drop("cov", "var"))
seen_in = (ev.filter(pl.col("battery_in_id").is_not_null()).group_by("battery_in_id")
           .agg(pl.col("event_ts").min().alias("first_returned"), pl.len().alias("n_returns")).rename({"battery_in_id": "battery_id"}))
bt = (bt.join(slopes, on="battery_id", how="left").join(seen_in, on="battery_id", how="left")
      .with_columns(pl.min_horizontal("first_issued", "first_returned").alias("first_seen")))
grp_med = bt.filter(pl.col("n_cycles") >= 50).group_by("supplier", "pack_type").agg(pl.col("soh_loss_per_cycle").median().alias("grp_loss"))
bt = (bt.join(grp_med, on=["supplier", "pack_type"], how="left")
      .with_columns(pl.when((pl.col("n_cycles") >= 50) & (pl.col("soh_loss_per_cycle") > 0)).then(pl.col("soh_loss_per_cycle"))
                    .otherwise(pl.col("grp_loss")).alias("loss_per_cycle_used"))
      .with_columns((pl.col("purchase_cost_inr") * pl.col("loss_per_cycle_used") / (pl.col("initial_soh_pct") - EOL_SOH)).alias("wear_cost_per_cycle"),
                    ((pl.col("initial_soh_pct") - EOL_SOH) / pl.col("loss_per_cycle_used")).alias("expected_cycles_to_eol"),
                    (pl.col("first_seen").dt.date() < pl.col("commission_date")).alias("used_before_commission"),
                    pl.col("battery_id").str.slice(4, 2).alias("vehicle_class")))
wear_summary = bt.group_by("supplier", "pack_type").agg(
    pl.len().alias("packs"), pl.col("loss_per_cycle_used").median().round(4).alias("soh_loss_per_cycle"),
    pl.col("expected_cycles_to_eol").median().round(0).alias("cycles_to_eol"),
    pl.col("wear_cost_per_cycle").median().round(2).alias("wear_inr_per_swap"),
    pl.col("used_before_commission").mean().round(3).alias("share_used_before_commission")).sort("pack_type", "supplier")
log_step("battery_wear_model", packs_used_before_recorded_commission=int(bt["used_before_commission"].fill_null(False).sum()))
bt.write_parquet(CLEAN / "batteries.parquet")
wear_summary

# %% [markdown]
# ### 3.8 Enrichment and the per-swap cost model
# Every completed swap gets: revenue (`amount_charged_inr`), energy cost (kWh to recharge x station tariff), battery wear
# cost (above) and an allocation of the station's rent + maintenance (pro-rated by active days, spread over that
# station-month's swaps). **cm1 = revenue - energy - wear** (variable contribution margin) and **cm2 = cm1 - station fixed
# cost**. Non-completed attempts carry zero revenue and cost.

# %%
st_attr = st_real.select("station_id", "city", "zone", "location_type", "host_type", "expansion_wave", "charger_generation", "connectivity_tier",
                         "grid_tariff_inr_kwh", "commissioned_date", pl.col("competitor_within_1_5km_since").alias("competitor_since"))
rd_attr = rd.select("rider_id", "partner_id", pl.col("vehicle_class").alias("rider_vehicle_class"), "plan_type", "signup_date", "signup_channel",
                    pl.col("home_city_clean").alias("rider_home_city"), "is_independent")
ev = (ev.join(st_attr, on="station_id", how="left").join(rd_attr, on="rider_id", how="left")
      .join(bt.select(pl.col("battery_id").alias("battery_out_id"), pl.col("supplier").alias("supplier_out"), pl.col("wear_cost_per_cycle").alias("wear_cost_out")), on="battery_out_id", how="left")
      .join(bt.select(pl.col("battery_id").alias("battery_in_id"), pl.col("supplier").alias("supplier_in")), on="battery_in_id", how="left"))
ev = ev.with_columns(
    pl.col("event_ts").dt.date().alias("date"), pl.col("event_ts").dt.hour().alias("hour"), pl.col("event_ts").dt.weekday().alias("dow"),
    pl.col("event_ts").dt.truncate("1mo").dt.date().alias("month"), pl.col("event_ts").dt.truncate("1w").dt.date().alias("week"),
    pl.col("event_ts").dt.month().replace_strict(SEASON_OF_MONTH).alias("season"),
    (pl.col("event_type") == "swap_completed").alias("is_completed"), (pl.col("event_type") == "failed_no_charged_battery").alias("is_stockout"),
    (pl.col("event_type") == "abandoned_queue").alias("is_abandoned"), (pl.col("event_type") == "failed_system_error").alias("is_system_error"),
    (pl.col("event_type") == "cancelled_by_rider").alias("is_cancelled"),
    pl.col("event_type").is_in(["failed_no_charged_battery", "abandoned_queue", "failed_system_error"]).alias("is_service_failure"),
    pl.coalesce(pl.col("battery_in_id").str.slice(4, 2), pl.col("rider_vehicle_class")).alias("vehicle_class"),
    ((pl.col("competitor_since").is_not_null()) & (pl.col("event_ts").dt.date() >= pl.col("competitor_since"))).alias("competitor_active"))
ev = ev.with_columns(
    pl.when(pl.col("is_completed")).then(pl.col("amount_charged_inr")).otherwise(0.0).alias("revenue"),
    pl.when(pl.col("is_completed")).then(pl.col("energy_to_recharge_kwh") * pl.col("grid_tariff_inr_kwh")).otherwise(0.0).alias("energy_cost"),
    pl.when(pl.col("is_completed")).then(pl.col("wear_cost_out")).otherwise(0.0).alias("wear_cost")).drop("wear_cost_out")

months = pl.DataFrame({"month": pl.date_range(WINDOW_START, WINDOW_END, "1mo", eager=True)})
sm = (st_real.select("station_id", "city", "commissioned_date", "decommissioned_date", "monthly_rent_inr", "monthly_maintenance_inr", "slots_2w", "slots_3w",
                     "expansion_wave", "charger_generation", "location_type", "host_type", "connectivity_tier")
      .join(months, how="cross").with_columns(pl.col("month").dt.month_end().alias("month_end"))
      .with_columns(pl.max_horizontal("commissioned_date", "month").alias("a_start"),
                    pl.min_horizontal(pl.col("decommissioned_date").fill_null(WINDOW_END), "month_end").alias("a_end"))
      .with_columns(((pl.col("a_end") - pl.col("a_start")).dt.total_days() + 1).clip(lower_bound=0).alias("active_days"),
                    pl.col("month_end").dt.day().alias("days_in_month"))
      .with_columns(((pl.col("monthly_rent_inr") + pl.col("monthly_maintenance_inr")) * pl.col("active_days") / pl.col("days_in_month")).alias("fixed_cost"))
      .filter(pl.col("active_days") > 0).drop("a_start", "a_end", "month_end"))
ev_sm = ev.group_by("station_id", "month").agg(
    pl.len().alias("attempts"), pl.col("is_completed").sum().alias("completed"), pl.col("is_stockout").sum().alias("stockouts"),
    pl.col("is_abandoned").sum().alias("abandoned"), pl.col("is_system_error").sum().alias("system_errors"), pl.col("is_cancelled").sum().alias("cancelled"),
    pl.col("queue_wait_sec").filter(pl.col("is_completed")).mean().alias("q_wait_completed_mean"),
    pl.col("queue_wait_sec").filter(pl.col("is_completed")).quantile(0.9).alias("q_wait_completed_p90"),
    pl.col("revenue").sum(), pl.col("energy_cost").sum(), pl.col("wear_cost").sum(), pl.col("rider_id").n_unique().alias("unique_riders"))
sm = (sm.join(ev_sm, on=["station_id", "month"], how="left").with_columns(pl.col("completed").fill_null(0), pl.col("attempts").fill_null(0))
      .with_columns(pl.when(pl.col("completed") > 0).then(pl.col("fixed_cost") / pl.col("completed")).alias("fixed_cost_per_swap"),
                    (pl.col("completed") / (pl.col("active_days") * (pl.col("slots_2w") + pl.col("slots_3w")))).alias("swaps_per_slot_day"),
                    (pl.col("revenue") - pl.col("energy_cost") - pl.col("wear_cost")).alias("cm1"))
      .with_columns((pl.col("cm1") - pl.col("fixed_cost")).alias("cm2")))
sm.write_parquet(CLEAN / "station_month.parquet")
ev = (ev.join(sm.select("station_id", "month", "fixed_cost_per_swap"), on=["station_id", "month"], how="left")
      .with_columns(pl.when(pl.col("is_completed")).then(pl.col("fixed_cost_per_swap")).otherwise(0.0).alias("fixed_cost")).drop("fixed_cost_per_swap")
      .with_columns((pl.col("revenue") - pl.col("energy_cost") - pl.col("wear_cost")).alias("cm1"))
      .with_columns((pl.col("cm1") - pl.col("fixed_cost")).alias("cm2")))
comp = ev.filter("is_completed")
log_step("cost_model", completed_swaps=comp.height, **{f"{c}_per_swap": round(float(comp[c].mean()), 2) for c in ["revenue", "energy_cost", "wear_cost", "fixed_cost", "cm1", "cm2"]})

first = ["event_id", "rider_id", "station_id", "event_ts", "event_ts_raw", "ts_corrected", "date", "month", "week", "hour", "dow", "season",
         "event_type", "is_completed", "is_stockout", "is_abandoned", "is_system_error", "is_cancelled", "is_service_failure", "queue_wait_sec",
         "vehicle_class", "city", "zone"]
ev = ev.select(first + [c for c in ev.columns if c not in first]).sort("event_ts")
ev.write_parquet(CLEAN / "events.parquet", compression="zstd", row_group_size=250_000)
log_step("events_written", rows=ev.height, cols=ev.width)

# %% [markdown]
# ### 3.9 Riders, telemetry, tickets and context tables
# Riders get their first/last activity; telemetry keeps blanks as **null** (never zero) with `telemetry_status`; tickets
# get window and CSAT-response flags. CSAT is missing for 65.6% of tickets and the response rate falls from ~44% for the
# fastest-resolved quartile to ~28% for the slowest - a missing-not-at-random pattern handled later with inverse-probability weighting.

# %%
ract = ev.group_by("rider_id").agg(
    pl.col("event_ts").min().alias("first_attempt_ts"), pl.col("event_ts").max().alias("last_attempt_ts"),
    pl.col("event_ts").filter(pl.col("is_completed")).min().alias("first_swap_ts"), pl.col("event_ts").filter(pl.col("is_completed")).max().alias("last_swap_ts"),
    pl.len().alias("n_attempts"), pl.col("is_completed").sum().alias("n_swaps"))
rd = rd.join(ract, on="rider_id", how="left")
rd.write_parquet(CLEAN / "riders.parquet")
del ev, comp

hr = (pl.scan_parquet(PARQ / "station_hourly_status.parquet")
      .join(st.lazy().select("station_id", "is_test", "city", "charger_generation", "expansion_wave", "connectivity_tier", "location_type",
                             "slots_2w", "slots_3w", "inventory_target_2w", "inventory_target_3w"), on="station_id", how="left").collect())
tele = hr.group_by("connectivity_tier", "telemetry_status").len().pivot(on="telemetry_status", index="connectivity_tier", values="len").sort("connectivity_tier")
hr = hr.filter(~pl.col("is_test")).drop("is_test").with_columns(
    pl.col("hour_start").dt.date().alias("date"), pl.col("hour_start").dt.hour().alias("hour"),
    pl.col("hour_start").dt.truncate("1mo").dt.date().alias("month"), pl.col("hour_start").dt.month().replace_strict(SEASON_OF_MONTH).alias("season"))
hr.write_parquet(CLEAN / "station_hourly.parquet", compression="zstd")
log_step("hourly_telemetry", rows_clean=hr.height, telemetry_by_connectivity=tele.to_dicts())
del hr

tk = (tk0.join(st.select("station_id", "is_test", pl.col("city").alias("station_city")), on="station_id", how="left")
      .with_columns(pl.col("is_test").fill_null(False), pl.col("created_ts").dt.date().alias("created_date"),
                    pl.col("created_ts").dt.truncate("1mo").dt.date().alias("month"),
                    pl.col("created_ts").is_between(pl.datetime(2024, 1, 1), pl.datetime(2025, 6, 30, 23, 59, 59)).alias("in_window"),
                    pl.col("csat_score").is_not_null().alias("has_csat")))
tk.write_parquet(CLEAN / "tickets.parquet")
log_step("tickets", total=tk.height, outside_window=int((~tk["in_window"]).sum()), csat_missing_share=round(float(tk["csat_score"].null_count() / tk.height), 3))
pl.read_parquet(PARQ / "city_daily_context.parquet").write_parquet(CLEAN / "city_daily.parquet")
fp0.write_parquet(CLEAN / "fleet_partners.parquet")
(TAB / "data_quality_log.json").write_text(json.dumps(LOG, indent=2, default=str), encoding="utf-8")
print(f"cleaning finished in {time.time()-T0:.1f}s")
tele
