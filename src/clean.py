"""Cleaning pipeline: raw Parquet -> data/clean/*.parquet + outputs/tables/data_quality_log.json

Decisions (each is logged with before/after counts):
  1. Test stations STN-TST-* are excluded from network analysis (kept in a separate audit file).
  2. Firmware v3.2.0 timezone bug: logged event_ts in [2025-03-10, 2025-04-15) at v3.2.0 stations shifted +5h30m.
  3. Near-duplicate offline-sync retries: same rider+station+event_type+battery_in+battery_out within 180 s where
     at least one record is offline_batch -> later record dropped.
  4. km_since_last_swap < 0 or > 200 km -> set to null (odometer resets; no valid readings exist in 120-330 km).
  5. SoC/SoH readings > 100% -> clipped to 100 (sensor calibration drift), flagged.
  6. riders.home_city standardised to 6 cities (validated against the home_zone prefix).
  7. Telemetry blanks are kept as null (never zero-filled); telemetry_status retained for filtering.
  8. Constant / reserved fields dropped from the analytical table: attempt_seq (always 1), payment_mode
     (always 'partner_invoice' in this export - unusable; plan_type/tariff_code used for price exposure).
Cost model attached to every completed swap (see docs in the log):
  revenue      = amount_charged_inr (0 for non-completed attempts)
  energy_cost  = energy_to_recharge_kwh x station grid_tariff_inr_kwh
  wear_cost    = battery purchase_cost x (SoH points lost per cycle) / (initial SoH - 70% end-of-life)
  fixed_cost   = station (rent + maintenance), pro-rated by active days, spread over that station-month's swaps
  cm1 = revenue - energy_cost - wear_cost        (variable contribution margin)
  cm2 = cm1 - fixed_cost                          (contribution after station operating costs)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CITY_CODE, CLEAN, EOL_SOH, FW_BUG_END, FW_BUG_SHIFT, FW_BUG_START, PARQ, SEASON_OF_MONTH, TAB,
                    WINDOW_END, WINDOW_START)

T0 = time.time()
LOG: dict = {"steps": []}


def log_step(name: str, **kw):
    kw = {k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in kw.items()}
    LOG["steps"].append({"step": name, **kw})
    print(f"[{time.time()-T0:6.1f}s] {name}: {kw}")


# =============================================================== stations
st = pl.read_parquet(PARQ / "stations.parquet").with_columns(
    pl.col("station_id").str.starts_with("STN-TST").alias("is_test"))
log_step("stations", total=st.height, test_stations=int(st["is_test"].sum()),
         test_ids=st.filter("is_test")["station_id"].to_list(),
         decommissioned=st.filter(pl.col("decommissioned_date").is_not_null())["station_id"].to_list(),
         competitor_nearby=int(st["competitor_within_1_5km_since"].is_not_null().sum()))
st.write_parquet(CLEAN / "stations.parquet")
st_real = st.filter(~pl.col("is_test"))

# =============================================================== riders: city standardisation
CITY_MAP = {
    "bengaluru": "Bengaluru", "bangalore": "Bengaluru", "blr": "Bengaluru",
    "delhi ncr": "Delhi NCR", "new delhi": "Delhi NCR", "delhi": "Delhi NCR", "gurgaon": "Delhi NCR",
    "gurugram": "Delhi NCR", "noida": "Delhi NCR", "del": "Delhi NCR",
    "mumbai": "Mumbai", "bombay": "Mumbai", "mum": "Mumbai",
    "pune": "Pune", "pun": "Pune",
    "hyderabad": "Hyderabad", "hyd": "Hyderabad",
    "jaipur": "Jaipur", "jai": "Jaipur",
}
rd = pl.read_parquet(PARQ / "riders.parquet")
n_variants = rd["home_city"].n_unique()
rd = rd.with_columns(
    pl.col("home_city").str.strip_chars().str.to_lowercase().replace_strict(CITY_MAP, default=None).alias("home_city_clean"),
    pl.col("home_zone").str.slice(0, 3).replace_strict(CITY_CODE, default=None).alias("zone_city"),
    pl.col("partner_id").is_null().alias("is_independent"),
)
unmapped = rd.filter(pl.col("home_city_clean").is_null())["home_city"].unique().to_list()
mismatch = rd.filter(pl.col("home_city_clean") != pl.col("zone_city")).height
log_step("riders_city_standardisation", raw_variants=n_variants, clean_cities=rd["home_city_clean"].n_unique(),
         unmapped_values=unmapped, rows_recoded=int((rd["home_city"] != rd["home_city_clean"]).sum()),
         city_vs_zone_mismatch=mismatch,
         declared_shift_missing=int(rd["declared_shift"].null_count()), age_band_missing=int(rd["age_band"].null_count()))

# =============================================================== events
ev = pl.scan_parquet(PARQ / "swap_events.parquet")
n_raw = ev.select(pl.len()).collect().item()
ev = ev.join(st.lazy().select("station_id", "is_test"), on="station_id", how="left")
test_ev = ev.filter(pl.col("is_test")).collect()
test_ev.write_parquet(CLEAN / "test_station_events.parquet")
tst_summary = test_ev.group_by("event_type").agg(pl.len().alias("n"), pl.col("amount_charged_inr").sum().alias("revenue")).sort("n", descending=True)
log_step("exclude_test_stations", raw_events=n_raw, test_events=test_ev.height,
         test_completed=int((test_ev["event_type"] == "swap_completed").sum()),
         test_revenue_inr=round(float(test_ev["amount_charged_inr"].sum()), 0),
         test_zero_value_events=int((test_ev["amount_charged_inr"] == 0).sum()),
         note="Documentation describes a small number of zero-value test transactions; the export actually holds "
              "full paid activity at both STN-TST stations. Excluded from network KPIs (internal test assets).")
ev = ev.filter(~pl.col("is_test")).drop("is_test")

# ---- firmware timestamp bug
bug = ((pl.col("station_firmware") == "v3.2.0") & (pl.col("event_ts") >= FW_BUG_START) & (pl.col("event_ts") < FW_BUG_END))
ev = ev.with_columns(
    bug.alias("ts_corrected"),
    pl.col("event_ts").alias("event_ts_raw"),
    pl.when(bug).then(pl.col("event_ts").dt.offset_by(FW_BUG_SHIFT)).otherwise(pl.col("event_ts")).alias("event_ts"),
)
ev = ev.collect()
n_shift = int(ev["ts_corrected"].sum())
chk = (ev.filter(pl.col("event_ts_raw").dt.date().is_between(pl.date(2025, 3, 10), pl.date(2025, 4, 14)))
       .group_by("station_firmware").agg(pl.col("event_ts_raw").dt.hour().mean().round(2).alias("mean_hour_raw"),
                                         pl.col("event_ts").dt.hour().mean().round(2).alias("mean_hour_fixed")))
log_step("firmware_timestamp_fix", events_shifted=n_shift, share_of_all=round(n_shift / ev.height, 4),
         check=chk.to_dicts(), crossed_midnight=int((ev.filter("ts_corrected")["event_ts"].dt.date() != ev.filter("ts_corrected")["event_ts_raw"].dt.date()).sum()))

# ---- near-duplicate offline-sync retries
k = (ev.select("event_id", "rider_id", "station_id", "event_ts", "event_type", "battery_in_id", "battery_out_id", "sync_mode")
     .sort("rider_id", "event_ts")
     .with_columns(
         pl.col("event_ts").diff().over("rider_id").dt.total_seconds().alias("gap_s"),
         *[pl.col(c).shift(1).over("rider_id").alias(c + "_prev") for c in ["station_id", "event_type", "battery_in_id", "battery_out_id", "sync_mode"]])
     )
same = ((pl.col("station_id") == pl.col("station_id_prev")) & (pl.col("event_type") == pl.col("event_type_prev"))
        & (pl.col("battery_in_id").eq_missing(pl.col("battery_in_id_prev"))) & (pl.col("battery_out_id").eq_missing(pl.col("battery_out_id_prev")))
        & (pl.col("gap_s") <= 180) & ((pl.col("sync_mode") == "offline_batch") | (pl.col("sync_mode_prev") == "offline_batch")))
dups = k.filter(same.fill_null(False))
# looser candidate count (any sync mode, same rider+station+battery_in within 300 s) for transparency
loose = k.filter(((pl.col("station_id") == pl.col("station_id_prev")) & (pl.col("battery_in_id") == pl.col("battery_in_id_prev")) & (pl.col("gap_s") <= 300)).fill_null(False))
log_step("near_duplicate_detection", rule="same rider+station+event_type+batteries, <=180s apart, >=1 offline_batch",
         duplicates_found=dups.height, loose_candidates_same_battery_in_300s=loose.height,
         offline_batch_rows=int((ev["sync_mode"] == "offline_batch").sum()),
         note="The documented offline-sync duplicate pattern is essentially absent in this export; offline_batch rows "
              "are legitimate events and are kept. Rule applied anyway for reproducibility.")
ev = ev.join(dups.select("event_id"), on="event_id", how="anti")
del k

# ---- outliers
km = pl.col("km_since_last_swap")
km_bad = (km < 0) | (km > 200)
n_km_neg = int((ev["km_since_last_swap"] < 0).sum()); n_km_hi = int((ev["km_since_last_swap"] > 200).sum())
ev = ev.with_columns(km_bad.fill_null(False).alias("km_invalid"), pl.when(km_bad).then(None).otherwise(km).alias("km_since_last_swap"))
clip_counts = {}
for c in ["soc_in_pct", "soh_in_pct", "soc_out_pct", "soh_out_pct"]:
    clip_counts[c] = int((ev[c] > 100).sum())
ev = ev.with_columns(
    pl.any_horizontal([pl.col(c) > 100 for c in ["soc_in_pct", "soh_in_pct", "soc_out_pct", "soh_out_pct"]]).fill_null(False).alias("reading_clipped"),
    *[pl.col(c).clip(upper_bound=100.0) for c in ["soc_in_pct", "soh_in_pct", "soc_out_pct", "soh_out_pct"]])
log_step("outliers", km_negative=n_km_neg, km_gt_200=n_km_hi, km_valid_range="0-200 km (2W p99=83 km, 3W p99=118 km)",
         readings_above_100_clipped=clip_counts)

# ---- reserved / constant fields
const = {c: ev[c].unique().to_list() for c in ["attempt_seq", "payment_mode"]}
log_step("drop_constant_fields", values=const,
         note="payment_mode is 'partner_invoice' on 100% of rows (incl. pay-as-you-go riders) - field unusable")
ev = ev.drop("attempt_seq", "payment_mode")

# =============================================================== battery wear model (per battery, from SoH trajectories)
bt = pl.read_parquet(PARQ / "batteries.parquet")
issued = (ev.filter(pl.col("event_type") == "swap_completed").select("battery_out_id", "event_ts", "soh_out_pct")
          .sort("battery_out_id", "event_ts")
          .with_columns(pl.int_range(1, pl.len() + 1).over("battery_out_id").alias("cycle")))
slopes = (issued.group_by("battery_out_id").agg(
    pl.len().alias("n_cycles"),
    pl.col("event_ts").min().alias("first_issued"), pl.col("event_ts").max().alias("last_issued"),
    pl.col("soh_out_pct").first().alias("soh_first_obs"), pl.col("soh_out_pct").last().alias("soh_last_obs"),
    ((pl.col("cycle") * pl.col("soh_out_pct")).mean() - pl.col("cycle").mean() * pl.col("soh_out_pct").mean()).alias("cov"),
    ((pl.col("cycle") ** 2).mean() - pl.col("cycle").mean() ** 2).alias("var"),
).with_columns((-(pl.col("cov") / pl.col("var"))).alias("soh_loss_per_cycle")).rename({"battery_out_id": "battery_id"}).drop("cov", "var"))
seen_in = ev.filter(pl.col("battery_in_id").is_not_null()).group_by("battery_in_id").agg(
    pl.col("event_ts").min().alias("first_returned"), pl.len().alias("n_returns")).rename({"battery_in_id": "battery_id"})
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
log_step("battery_wear_model", method="per-battery OLS slope of issued-pack SoH vs cycle index; group median if <50 cycles",
         summary=wear_summary.to_dicts(),
         packs_used_before_recorded_commission=int(bt["used_before_commission"].fill_null(False).sum()),
         note="batteries.csv commission_date/lot metadata contradicts usage for many packs: age is measured from "
              "first appearance in swap events instead")
bt.write_parquet(CLEAN / "batteries.parquet")

# =============================================================== enrich events
st_attr = st_real.select("station_id", "city", "zone", "location_type", "host_type", "expansion_wave", "charger_generation",
                         "connectivity_tier", "grid_tariff_inr_kwh", "commissioned_date",
                         pl.col("competitor_within_1_5km_since").alias("competitor_since"))
rd_attr = rd.select("rider_id", "partner_id", pl.col("vehicle_class").alias("rider_vehicle_class"), "plan_type", "signup_date",
                    "signup_channel", pl.col("home_city_clean").alias("rider_home_city"), "is_independent")
b_out = bt.select(pl.col("battery_id").alias("battery_out_id"), pl.col("supplier").alias("supplier_out"),
                  pl.col("wear_cost_per_cycle").alias("wear_cost_out"))
b_in = bt.select(pl.col("battery_id").alias("battery_in_id"), pl.col("supplier").alias("supplier_in"))
ev = (ev.join(st_attr, on="station_id", how="left").join(rd_attr, on="rider_id", how="left")
      .join(b_out, on="battery_out_id", how="left").join(b_in, on="battery_in_id", how="left"))
ev = ev.with_columns(
    pl.col("event_ts").dt.date().alias("date"),
    pl.col("event_ts").dt.hour().alias("hour"),
    pl.col("event_ts").dt.weekday().alias("dow"),            # 1=Mon .. 7=Sun
    pl.col("event_ts").dt.truncate("1mo").dt.date().alias("month"),
    pl.col("event_ts").dt.truncate("1w").dt.date().alias("week"),
    pl.col("event_ts").dt.month().replace_strict(SEASON_OF_MONTH).alias("season"),
    (pl.col("event_type") == "swap_completed").alias("is_completed"),
    (pl.col("event_type") == "failed_no_charged_battery").alias("is_stockout"),
    (pl.col("event_type") == "abandoned_queue").alias("is_abandoned"),
    (pl.col("event_type") == "failed_system_error").alias("is_system_error"),
    (pl.col("event_type") == "cancelled_by_rider").alias("is_cancelled"),
    pl.col("event_type").is_in(["failed_no_charged_battery", "abandoned_queue", "failed_system_error"]).alias("is_service_failure"),
    pl.coalesce(pl.col("battery_in_id").str.slice(4, 2), pl.col("rider_vehicle_class")).alias("vehicle_class"),
    ((pl.col("competitor_since").is_not_null()) & (pl.col("event_ts").dt.date() >= pl.col("competitor_since"))).alias("competitor_active"),
)
ev = ev.with_columns(
    pl.when(pl.col("is_completed")).then(pl.col("amount_charged_inr")).otherwise(0.0).alias("revenue"),
    pl.when(pl.col("is_completed")).then(pl.col("energy_to_recharge_kwh") * pl.col("grid_tariff_inr_kwh")).otherwise(0.0).alias("energy_cost"),
    pl.when(pl.col("is_completed")).then(pl.col("wear_cost_out")).otherwise(0.0).alias("wear_cost"),
).drop("wear_cost_out")

# =============================================================== station-month fixed costs and KPIs
months = pl.DataFrame({"month": pl.date_range(WINDOW_START, WINDOW_END, "1mo", eager=True)})
sm = (st_real.select("station_id", "city", "commissioned_date", "decommissioned_date", "monthly_rent_inr", "monthly_maintenance_inr",
                     "slots_2w", "slots_3w", "expansion_wave", "charger_generation", "location_type", "host_type", "connectivity_tier")
      .join(months, how="cross")
      .with_columns(pl.col("month").dt.month_end().alias("month_end"))
      .with_columns(pl.max_horizontal("commissioned_date", "month").alias("a_start"),
                    pl.min_horizontal(pl.col("decommissioned_date").fill_null(WINDOW_END), "month_end").alias("a_end"))
      .with_columns(((pl.col("a_end") - pl.col("a_start")).dt.total_days() + 1).clip(lower_bound=0).alias("active_days"),
                    pl.col("month_end").dt.day().alias("days_in_month"))
      .with_columns(((pl.col("monthly_rent_inr") + pl.col("monthly_maintenance_inr")) * pl.col("active_days") / pl.col("days_in_month")).alias("fixed_cost"))
      .filter(pl.col("active_days") > 0)
      .drop("a_start", "a_end", "month_end"))
ev_sm = ev.group_by("station_id", "month").agg(
    pl.len().alias("attempts"), pl.col("is_completed").sum().alias("completed"),
    pl.col("is_stockout").sum().alias("stockouts"), pl.col("is_abandoned").sum().alias("abandoned"),
    pl.col("is_system_error").sum().alias("system_errors"), pl.col("is_cancelled").sum().alias("cancelled"),
    pl.col("queue_wait_sec").filter(pl.col("is_completed")).mean().alias("q_wait_completed_mean"),
    pl.col("queue_wait_sec").filter(pl.col("is_completed")).quantile(0.9).alias("q_wait_completed_p90"),
    pl.col("revenue").sum(), pl.col("energy_cost").sum(), pl.col("wear_cost").sum(),
    pl.col("rider_id").n_unique().alias("unique_riders"))
sm = sm.join(ev_sm, on=["station_id", "month"], how="left").with_columns(
    pl.col("completed").fill_null(0), pl.col("attempts").fill_null(0))
sm = sm.with_columns(
    pl.when(pl.col("completed") > 0).then(pl.col("fixed_cost") / pl.col("completed")).alias("fixed_cost_per_swap"),
    (pl.col("completed") / (pl.col("active_days") * (pl.col("slots_2w") + pl.col("slots_3w")))).alias("swaps_per_slot_day"),
    (pl.col("revenue") - pl.col("energy_cost") - pl.col("wear_cost")).alias("cm1"),
).with_columns((pl.col("cm1") - pl.col("fixed_cost")).alias("cm2"))
orphan_cost = sm.filter((pl.col("completed") == 0))["fixed_cost"].sum()
sm.write_parquet(CLEAN / "station_month.parquet")
ev = ev.join(sm.select("station_id", "month", "fixed_cost_per_swap"), on=["station_id", "month"], how="left").with_columns(
    pl.when(pl.col("is_completed")).then(pl.col("fixed_cost_per_swap")).otherwise(0.0).alias("fixed_cost")).drop("fixed_cost_per_swap")
ev = ev.with_columns((pl.col("revenue") - pl.col("energy_cost") - pl.col("wear_cost")).alias("cm1")).with_columns(
    (pl.col("cm1") - pl.col("fixed_cost")).alias("cm2"))
comp = ev.filter("is_completed")
log_step("cost_model", completed_swaps=comp.height,
         revenue_per_swap=round(float(comp["revenue"].mean()), 2), energy_cost_per_swap=round(float(comp["energy_cost"].mean()), 2),
         wear_cost_per_swap=round(float(comp["wear_cost"].mean()), 2), fixed_cost_per_swap=round(float(comp["fixed_cost"].mean()), 2),
         cm1_per_swap=round(float(comp["cm1"].mean()), 2), cm2_per_swap=round(float(comp["cm2"].mean()), 2),
         station_months_with_cost_but_no_swaps=int(sm.filter(pl.col("completed") == 0).height), unallocated_fixed_cost=round(float(orphan_cost), 0),
         missing_wear_cost=int(comp["wear_cost"].null_count()), missing_energy=int(comp["energy_cost"].null_count()))

# column order and write
first = ["event_id", "rider_id", "station_id", "event_ts", "event_ts_raw", "ts_corrected", "date", "month", "week", "hour", "dow", "season",
         "event_type", "is_completed", "is_stockout", "is_abandoned", "is_system_error", "is_cancelled", "is_service_failure",
         "queue_wait_sec", "vehicle_class", "city", "zone"]
ev = ev.select(first + [c for c in ev.columns if c not in first]).sort("event_ts")
ev.write_parquet(CLEAN / "events.parquet", compression="zstd", row_group_size=250_000)
log_step("events_written", rows=ev.height, cols=ev.width, columns=ev.columns)

# =============================================================== riders enriched with activity
ract = ev.group_by("rider_id").agg(
    pl.col("event_ts").min().alias("first_attempt_ts"), pl.col("event_ts").max().alias("last_attempt_ts"),
    pl.col("event_ts").filter(pl.col("is_completed")).min().alias("first_swap_ts"),
    pl.col("event_ts").filter(pl.col("is_completed")).max().alias("last_swap_ts"),
    pl.len().alias("n_attempts"), pl.col("is_completed").sum().alias("n_swaps"))
rd = rd.join(ract, on="rider_id", how="left")
log_step("riders_written", riders=rd.height, never_attempted=int(rd["n_attempts"].null_count()),
         signed_up_before_window=int((rd["signup_date"] < WINDOW_START).sum()), signed_up_in_window=int((rd["signup_date"] >= WINDOW_START).sum()))
rd.write_parquet(CLEAN / "riders.parquet")

# =============================================================== hourly telemetry
hr = (pl.scan_parquet(PARQ / "station_hourly_status.parquet")
      .join(st.lazy().select("station_id", "is_test", "city", "charger_generation", "expansion_wave", "connectivity_tier", "location_type",
                             "slots_2w", "slots_3w", "inventory_target_2w", "inventory_target_3w"), on="station_id", how="left")
      .collect())
n_hr = hr.height
tele = hr.group_by("connectivity_tier", "telemetry_status").len().sort("connectivity_tier", "telemetry_status")
hr = hr.filter(~pl.col("is_test")).drop("is_test").with_columns(
    pl.col("hour_start").dt.date().alias("date"), pl.col("hour_start").dt.hour().alias("hour"),
    pl.col("hour_start").dt.truncate("1mo").dt.date().alias("month"),
    pl.col("hour_start").dt.month().replace_strict(SEASON_OF_MONTH).alias("season"))
log_step("hourly_telemetry", rows_raw=n_hr, rows_clean=hr.height, telemetry_by_connectivity=tele.to_dicts(),
         note="blank fields kept as null; aggregate with null-aware means and report coverage")
hr.write_parquet(CLEAN / "station_hourly.parquet", compression="zstd")

# =============================================================== tickets
tk = (pl.read_parquet(PARQ / "support_tickets.parquet")
      .join(st.select("station_id", "is_test", pl.col("city").alias("station_city")), on="station_id", how="left")
      .with_columns(pl.col("is_test").fill_null(False),
                    pl.col("created_ts").dt.date().alias("created_date"),
                    pl.col("created_ts").dt.truncate("1mo").dt.date().alias("month"),
                    pl.col("created_ts").is_between(pl.datetime(2024, 1, 1), pl.datetime(2025, 6, 30, 23, 59, 59)).alias("in_window"),
                    pl.col("csat_score").is_not_null().alias("has_csat")))
log_step("tickets", total=tk.height, outside_window=int((~tk["in_window"]).sum()), test_station_tickets=int(tk["is_test"].sum()),
         csat_missing_share=round(float(tk["csat_score"].null_count() / tk.height), 3),
         csat_response_by_resolution_speed=tk.filter(pl.col("resolution_status") == "resolved").with_columns(
             pl.col("resolution_hours").qcut(4, labels=["Q1 fastest", "Q2", "Q3", "Q4 slowest"]).alias("speed_q"))
         .group_by("speed_q").agg(pl.col("has_csat").mean().round(3).alias("csat_response_rate"), pl.col("csat_score").mean().round(2).alias("csat_mean")).sort("speed_q").to_dicts(),
         distinct_comments=tk["rider_comment"].n_unique())
tk.write_parquet(CLEAN / "tickets.parquet")

# =============================================================== small tables
pl.read_parquet(PARQ / "city_daily_context.parquet").write_parquet(CLEAN / "city_daily.parquet")
pl.read_parquet(PARQ / "fleet_partners.parquet").write_parquet(CLEAN / "fleet_partners.parquet")

LOG["runtime_sec"] = round(time.time() - T0, 1)
(TAB / "data_quality_log.json").write_text(json.dumps(LOG, indent=2, default=str), encoding="utf-8")
print("done in", LOG["runtime_sec"], "s")
