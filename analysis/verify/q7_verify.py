# %% [markdown]
# # Q7 verification (independent re-derivation)
# Adversarial re-computation of the Q7 data-quality / anomaly claims. Written from scratch: it does NOT import or copy
# analysis/q7_data_quality_anomalies.py. Clean data via src/common loaders; the raw parquet is read only where a
# before/after comparison needs the uncleaned value (km, SoC/SoH, home_city).
# Run: PYTHONIOENCODING=utf-8 python analysis/verify/q7_verify.py [section ...]   (sections 1-15; default all)

# %%
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import *  # noqa: E402,F401,F403

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(24)
pl.Config.set_tbl_width_chars(230)
T0 = time.time()
RUN = set(int(a) for a in sys.argv[1:]) or set(range(1, 16))
N_EV = events(["event_id"]).select(pl.len()).collect().item()
FT = ["failed_no_charged_battery", "abandoned_queue", "failed_system_error"]


def hdr(t):
    print(f"\n{'#' * 100}\n# {t}   [{time.time() - T0:.0f}s]\n{'#' * 100}", flush=True)


def pct(a, b, d=2):
    return f"{100 * a / b:.{d}f}%"


print(f"clean attempts N_EV = {N_EV:,}")

# %% [markdown]
# ## 1 · Firmware fix: hour profile, tariff clock, and window edges

# %%
if 1 in RUN:
    hdr("1. firmware fix")
    nshift = events(["ts_corrected"]).select(pl.col("ts_corrected").sum()).collect().item()
    print(f"ts_corrected rows {nshift:,} = {pct(nshift, N_EV)} of clean attempts")
    W = (events(["station_firmware", "event_ts", "event_ts_raw", "city", "tariff_code", "is_independent", "event_type"])
         .filter(pl.col("event_ts_raw").is_between(dt.datetime(2025, 3, 1), dt.datetime(2025, 4, 22)))
         .with_columns(pl.col("event_ts_raw").dt.hour().alias("h_raw"), pl.col("event_ts").dt.hour().alias("h_fix"),
                       pl.col("event_ts_raw").dt.date().alias("d_raw"))
         .collect())
    inw = W.filter(pl.col("d_raw").is_between(dt.date(2025, 3, 10), dt.date(2025, 4, 14)))
    a0 = inw.filter(pl.col("station_firmware") == "v3.2.0")
    a1 = inw.filter(pl.col("station_firmware") == "v3.2.1")

    def prof(s):
        c = np.bincount(s.to_numpy().astype(int), minlength=24).astype(float)
        return c / c.sum()

    pr, pf, pref = prof(a0["h_raw"]), prof(a0["h_fix"]), prof(a1["h_fix"])
    H = np.arange(24)
    for lab, p_ in [("v3.2.0 raw", pr), ("v3.2.0 fixed", pf), ("v3.2.1 ref", pref)]:
        print(f"{lab:13s}: 00-05h {100 * p_[:6].sum():5.2f}% | 12-22h {100 * p_[12:23].sum():5.2f}% | mean hour {np.dot(H, p_):.2f} | busiest {p_.argmax()}h")
    print(f"TVD to reference: raw {50 * np.abs(pr - pref).sum():.2f} pts, fixed {50 * np.abs(pf - pref).sum():.2f} pts; n v3.2.0 = {a0.height:,}, v3.2.1 = {a1.height:,}")
    pn_raw, pn_fix = prof(inw["h_raw"]), prof(inw["h_fix"])
    print(f"NETWORK in window: 12-22h share raw {100 * pn_raw[12:23].sum():.2f}% vs fixed {100 * pn_fix[12:23].sum():.2f}%; 00-05h raw {100 * pn_raw[:6].sum():.2f}% vs fixed {100 * pn_fix[:6].sum():.2f}%")
    pk = a0.filter((pl.col("tariff_code") == "PEAK") & pl.col("city").is_in(PILOT_CITIES))
    pk1 = a1.filter((pl.col("tariff_code") == "PEAK") & pl.col("city").is_in(PILOT_CITIES))
    print(f"TARIFF CLOCK v3.2.0 pilot-city PEAK rows n={pk.height:,}: inside 12-22h raw {pct(pk['h_raw'].is_between(12, 22).sum(), pk.height, 1)} "
          f"vs fixed {pct(pk['h_fix'].is_between(12, 22).sum(), pk.height, 1)}; v3.2.1 reference n={pk1.height:,}: {pct(pk1['h_fix'].is_between(12, 22).sum(), pk1.height, 1)}")
    # tariff-code by corrected hour for the whole post-pilot period (pilot cities, independent riders): any residual mis-timing?
    tc = (events(["city", "date", "hour", "tariff_code", "is_independent", "station_firmware", "event_ts_raw", "ts_corrected"])
          .filter(pl.col("city").is_in(PILOT_CITIES) & (pl.col("date") >= PILOT_START) & pl.col("is_independent"))
          .group_by("hour", "tariff_code").len().collect().pivot(on="tariff_code", index="hour", values="len").sort("hour").fill_null(0))
    print("pilot cities, independent riders, post-pilot: tariff_code by CORRECTED hour"); print(tc)
    bad = (events(["city", "date", "hour", "tariff_code", "is_independent", "station_firmware", "event_ts_raw", "event_ts", "ts_corrected"])
           .filter(pl.col("city").is_in(PILOT_CITIES) & (pl.col("date") >= PILOT_START) & pl.col("is_independent"))
           .filter(((pl.col("tariff_code") == "PEAK") & ~pl.col("hour").is_between(12, 22))
                   | ((pl.col("tariff_code") != "PEAK") & pl.col("hour").is_between(12, 22)))
           .collect())
    print(f"tariff/hour contradictions after the fix (whole post-pilot period): {bad.height:,}")
    if bad.height:
        print(bad.group_by("station_firmware", "date", "tariff_code").agg(pl.len(), pl.col("event_ts_raw").min().alias("raw_min"),
                                                                        pl.col("event_ts_raw").max().alias("raw_max")).sort("len", descending=True).head(12))
    # window edges
    v0 = W.filter(pl.col("station_firmware") == "v3.2.0")
    v1 = W.filter(pl.col("station_firmware") == "v3.2.1")
    gap0 = v0.filter(pl.col("event_ts").is_between(dt.datetime(2025, 3, 10, 0, 0), dt.datetime(2025, 3, 10, 5, 29, 59))).height
    gap1 = v1.filter(pl.col("event_ts").is_between(dt.datetime(2025, 3, 10, 0, 0), dt.datetime(2025, 3, 10, 5, 29, 59))).height
    end0 = v0.filter(pl.col("event_ts_raw").is_between(dt.datetime(2025, 4, 14, 18, 30), dt.datetime(2025, 4, 14, 23, 59, 59))).height
    end1 = v1.filter(pl.col("event_ts_raw").is_between(dt.datetime(2025, 4, 14, 18, 30), dt.datetime(2025, 4, 14, 23, 59, 59))).height
    # pre-bug ratio v3.2.0 / v3.2.1 for the same clock windows on 2-8 Mar (true time == logged time then)
    pre = W.filter(pl.col("d_raw").is_between(dt.date(2025, 3, 2), dt.date(2025, 3, 8)))
    mins = pl.col("event_ts_raw").dt.hour().cast(pl.Int32) * 60 + pl.col("event_ts_raw").dt.minute().cast(pl.Int32)
    ev_ = pre.filter(mins >= 18 * 60 + 30)
    ng_ = pre.filter(mins < 5 * 60 + 30)
    r_eve = ev_.filter(pl.col("station_firmware") == "v3.2.0").height / ev_.filter(pl.col("station_firmware") == "v3.2.1").height
    r_ngt = ng_.filter(pl.col("station_firmware") == "v3.2.0").height / ng_.filter(pl.col("station_firmware") == "v3.2.1").height
    e9 = W.filter((pl.col("d_raw") == dt.date(2025, 3, 9)) & (mins >= 18 * 60 + 30))
    o9, e9x = e9.filter(pl.col("station_firmware") == "v3.2.0").height, r_eve * e9.filter(pl.col("station_firmware") == "v3.2.1").height
    print(f"EDGE start: v3.2.0 at corrected 10-Mar 00:00-05:29 = {gap0} (expected {r_ngt * gap1:.0f} from v3.2.1 n={gap1} x pre ratio {r_ngt:.3f})")
    print(f"EDGE start: v3.2.0 logged 9-Mar 18:30-23:59 = {o9} vs expected {e9x:.0f} -> excess {o9 - e9x:+.0f} (z={(o9 - e9x) / np.sqrt(e9x):.1f})")
    print(f"EDGE end: v3.2.0 logged 14-Apr 18:30-23:59 = {end0} (v3.2.1 same window {end1}; expected if bugged-on-logged-time {r_eve * end1:.0f})")
    # row-level evidence on 9-Mar evening: pilot-city independent rows logged in peak hours but billed OFFPEAK/STD
    e9p = (e9.filter((pl.col("station_firmware") == "v3.2.0") & pl.col("city").is_in(PILOT_CITIES) & pl.col("is_independent")
                     & (pl.col("h_raw").is_between(18, 22)))
           .group_by("tariff_code").len())
    print("9-Mar 18:30-22:59 logged, v3.2.0 pilot-city independent riders, tariff codes:", e9p.to_dicts())
    del W, inw, a0, a1, v0, v1, bad

# %% [markdown]
# ## 2 · Test stations: what would keeping them do?

# %%
if 2 in RUN:
    hdr("2. test stations")
    ts_ = pl.read_parquet(CLEAN / "test_station_events.parquet",
                          columns=["station_id", "rider_id", "event_ts", "event_type", "amount_charged_inr", "list_price_inr"])
    t_att = ts_.height
    t_comp = int((ts_["event_type"] == "swap_completed").sum())
    t_fail = int(ts_["event_type"].is_in(FT).sum())
    t_rev = float(ts_.filter(pl.col("event_type") == "swap_completed")["amount_charged_inr"].sum())
    t_rev_all = float(ts_["amount_charged_inr"].sum())
    zero = ts_.filter(pl.col("amount_charged_inr") == 0)
    print(f"test attempts {t_att:,}; completed {t_comp:,}; failures {t_fail:,}; revenue (completed) {t_rev:,.0f} / all rows {t_rev_all:,.0f}")
    print(f"zero-value rows {zero.height:,}; of which completed {int((zero['event_type'] == 'swap_completed').sum())}; share of raw {pct(t_att, 3877013)}")
    tm = ts_.group_by("station_id").agg(pl.col("event_ts").min().alias("first"), pl.col("event_ts").max().alias("last"),
                                        pl.col("event_ts").dt.truncate("1mo").n_unique().alias("months"),
                                        (pl.col("event_type") == "swap_completed").sum().alias("comp"))
    print(tm)
    t_night = float((ts_["event_ts"].dt.hour() < 6).mean())
    net = events(["hour", "city", "is_completed", "event_type", "revenue", "month"])
    agg = net.select(pl.len().alias("att"), (pl.col("hour") < 6).mean().alias("night")).collect().row(0, named=True)
    print(f"night share (00-05h): test {100 * t_night:.2f}% vs clean network {100 * agg['night']:.2f}%")
    blr = (net.filter(pl.col("city") == "Bengaluru")
           .select(pl.len().alias("att"), pl.col("is_completed").sum().alias("comp"), pl.col("event_type").is_in(FT).sum().alias("fail"),
                   pl.col("revenue").sum().alias("rev")).collect().row(0, named=True))
    netrev = net.select(pl.col("revenue").sum()).collect().item()
    print(f"Bengaluru clean: att {blr['att']:,}, comp {blr['comp']:,}, fail rate {100 * blr['fail'] / blr['att']:.3f}%, revenue {blr['rev']:,.0f}")
    print(f"keep test: BLR swaps +{100 * t_comp / blr['comp']:.2f}%, BLR revenue +{100 * t_rev / blr['rev']:.2f}%, network revenue +{100 * t_rev / netrev:.2f}%, "
          f"BLR fail rate -> {100 * (blr['fail'] + t_fail) / (blr['att'] + t_att):.3f}%")
    bm = net.filter(pl.col("city") == "Bengaluru").group_by("month").agg(pl.col("is_completed").sum().alias("c")).collect()
    tmm = ts_.filter(pl.col("event_type") == "swap_completed").group_by(pl.col("event_ts").dt.truncate("1mo").dt.date().alias("month")).agg(pl.len().alias("t"))
    bm = bm.join(tmm, on="month", how="left").fill_null(0).sort("month")
    j24, j25 = bm.filter(pl.col("month") == dt.date(2024, 1, 1)).row(0, named=True), bm.filter(pl.col("month") == dt.date(2025, 6, 1)).row(0, named=True)
    print(f"BLR completed growth Jun-25/Jan-24: without {j25['c'] / j24['c']:.3f}x, with test {(j25['c'] + j25['t']) / (j24['c'] + j24['t']):.3f}x; "
          f"test swaps Jan-24 {j24['t']:,} Jun-25 {j25['t']:,}")
    smb = station_month().filter(pl.col("city") == "Bengaluru")
    full = smb.filter(pl.col("active_days") == pl.col("month").dt.month_end().dt.day())
    print(f"BLR completed per station-month: all rows {smb['completed'].sum() / smb.height:,.0f} (n={smb.height}); full-month rows only "
          f"{full['completed'].sum() / full.height:,.0f}; Launch-wave BLR full months {full.filter(pl.col('expansion_wave') == 'Launch')['completed'].mean():,.0f}; "
          f"test per station-month {t_comp / tm['months'].sum():,.0f}")
    rd = riders().select("rider_id", "n_attempts")
    tr = ts_.group_by("rider_id").agg(pl.len().alias("nt")).join(rd, on="rider_id", how="left")
    print(f"riders at test stations {tr.height:,}; test-only (no clean attempts) {int(tr['n_attempts'].is_null().sum())}; "
          f"share of their attempts at test stations {pct(tr['nt'].sum(), tr['nt'].sum() + tr['n_attempts'].fill_null(0).sum(), 1)}")
    del ts_, tr

# %% [markdown]
# ## 3 · Telemetry blanks: null profile, zero-fill bias, informativeness

# %%
if 3 in RUN:
    hdr("3. telemetry")
    TF = ["charged_2w_avg", "charged_2w_min", "charged_3w_min", "packs_charging", "packs_quarantined", "chargers_online",
          "ambient_temp_c", "cabinet_temp_c", "avg_charge_minutes", "outage_minutes", "grid_kwh"]
    hr = hourly(["station_id", "date", "hour", "telemetry_status", "connectivity_tier", "slots_3w"] + TF).collect()
    print(f"station-hours {hr.height:,}")
    print(hr.group_by("connectivity_tier", "telemetry_status").len().sort("connectivity_tier", "telemetry_status"))
    nulls = hr.group_by("telemetry_status").agg(pl.len().alias("n"), *[pl.col(c).null_count().alias(c) for c in TF]).sort("telemetry_status")
    print("null counts by status:"); print(nulls)
    print("partial rows with charged_3w_min null at stations WITH 3W slots:",
          hr.filter((pl.col("telemetry_status") == "partial") & (pl.col("slots_3w") > 0) & pl.col("charged_3w_min").is_null()).height)
    so = (hr.group_by("connectivity_tier").agg(pl.len().alias("hours"), (pl.col("telemetry_status") == "missing").sum().alias("miss"),
                                               (pl.col("charged_2w_min") == 0).sum().alias("so_null"),
                                               (pl.col("charged_2w_min").fill_null(0) == 0).sum().alias("so_zero"),
                                               pl.col("charged_2w_avg").mean().alias("avg_null"), pl.col("charged_2w_avg").fill_null(0).mean().alias("avg_zero"))
          .sort("connectivity_tier").with_columns((100 * pl.col("miss") / pl.col("hours")).alias("miss_pct"),
                                                  (100 * (pl.col("avg_zero") / pl.col("avg_null") - 1)).alias("stock_bias_pct")))
    print(so)
    print(f"stockout-hours network: null-aware {so['so_null'].sum():,} -> zero-filled {so['so_zero'].sum():,} (x{so['so_zero'].sum() / so['so_null'].sum():.2f}); "
          f"missing {so['miss'].sum():,} = {pct(so['miss'].sum(), so['hours'].sum())}")
    eh = (events(["station_id", "date", "hour", "is_stockout", "connectivity_tier"]).filter(pl.col("connectivity_tier") == "poor")
          .group_by("station_id", "date", "hour").agg(pl.len().alias("att"), pl.col("is_stockout").sum().alias("so")).collect())
    j = (hr.filter(pl.col("connectivity_tier") == "poor").select("station_id", "date", "hour", "telemetry_status")
         .join(eh, on=["station_id", "date", "hour"], how="left").fill_null(0)
         .group_by(pl.col("telemetry_status") == "missing").agg(pl.col("att").sum(), pl.col("so").sum(), pl.len().alias("hours")).sort("telemetry_status"))
    print(j.with_columns((100 * pl.col("so") / pl.col("att")).alias("stockout_rate"), (pl.col("att") / pl.col("hours")).alias("att_per_hour")))
    tab = np.array([[r["so"], r["att"] - r["so"]] for r in j.iter_rows(named=True)])
    print(f"chi2 p (stockout rate, blank vs reported hours) = {stats.chi2_contingency(tab)[1]:.3f}")
    del hr, eh

# %% [markdown]
# ## 4 · Near-duplicates

# %%
if 4 in RUN:
    hdr("4. near-duplicates")
    k = (events(["rider_id", "station_id", "event_ts", "event_type", "battery_in_id", "battery_out_id", "sync_mode"])
         .select(pl.col("rider_id").hash().alias("r"), pl.col("station_id").cast(pl.Categorical).to_physical().alias("s"), "event_ts",
                 pl.col("event_type").cast(pl.Categorical).to_physical().alias("t"),
                 pl.col("battery_in_id").hash().alias("bi"), pl.col("battery_out_id").hash().alias("bo"),
                 (pl.col("sync_mode") == "offline_batch").alias("off"), pl.col("station_id").alias("sid"))
         .sort("r", "event_ts")
         .with_columns([pl.col(c).shift(1).alias(c + "0") for c in ["r", "s", "event_ts", "t", "bi", "bo", "off"]])
         .filter(pl.col("r") == pl.col("r0"))
         .with_columns((pl.col("event_ts") - pl.col("event_ts0")).dt.total_seconds().alias("gap"))
         .filter(pl.col("gap") <= 600).collect())
    S, T, B = pl.col("s") == pl.col("s0"), pl.col("t") == pl.col("t0"), (pl.col("bi") == pl.col("bi0")) & (pl.col("bo") == pl.col("bo0"))
    OFF, g = pl.col("off") | pl.col("off0"), pl.col("gap")
    rules = {"D0 documented": S & T & B & (g <= 180) & OFF, "D1 no battery IDs, offline": S & T & (g <= 180) & OFF,
             "D2 any sync": S & T & (g <= 180), "D3 any type": S & (g <= 180), "D6 600 s": S & T,
             "same battery_in <=300s any station": (pl.col("bi") == pl.col("bi0")) & (g <= 300), "gap == 0": g == 0}
    for nm, ex in rules.items():
        n_ = k.filter(ex).height
        print(f"{nm:38s}: {n_:7,} pairs = {pct(n_, N_EV, 3)} of attempts")
    ss = k.filter(S & T)
    hist, _ = np.histogram(ss["gap"].to_numpy(), bins=np.arange(0, 601, 30))
    print(f"same station+type gap histogram (30-s bins): {hist.tolist()}; mean 0-120 s {hist[:4].mean():.0f}/bin vs 120-600 s {hist[4:].mean():.0f}/bin; "
          f"chi2 uniformity p = {stats.chisquare(hist).pvalue:.3f}")
    stoff = events(["station_id", "sync_mode"]).group_by("station_id").agg((pl.col("sync_mode") == "offline_batch").mean().alias("p")).collect()
    d2 = k.filter(S & T & (g <= 180)).join(stoff, left_on="sid", right_on="station_id", how="left")
    print(f"D2 offline involved {100 * d2.select(OFF.mean()).item():.1f}% vs expected {100 * (1 - (1 - d2['p']) ** 2).mean():.1f}%")
    base = events(["event_type"]).select(pl.col("event_type").is_in(FT).sum()).collect().item()
    tmap = dict(zip(range(10), []))
    d2_types = events(["event_type"]).select(pl.col("event_type").cast(pl.Categorical).to_physical().alias("code"), "event_type").unique().collect()
    fail_codes = d2_types.filter(pl.col("event_type").is_in(FT))["code"].to_list()
    comp_code = d2_types.filter(pl.col("event_type") == "swap_completed")["code"].item()
    d2f = int(d2["t"].is_in(fail_codes).sum()); d2c = int((d2["t"] == comp_code).sum())
    print(f"failure rate {100 * base / N_EV:.3f}% -> {100 * (base - d2f) / (N_EV - d2.height):.3f}% if later D2 record dropped; completed -{d2c:,} ({pct(d2c, 3586675)})")
    del k, ss, d2

# %% [markdown]
# ## 5 · Revenue logic and hidden PARTNER pilot pricing

# %%
if 5 in RUN:
    hdr("5. tariff arithmetic")
    rv = (events(["is_completed", "tariff_code", "list_price_inr", "discount_inr", "amount_charged_inr", "partner_id", "plan_type", "city", "date", "hour"])
          .with_columns((pl.col("amount_charged_inr") - (pl.col("list_price_inr") - pl.col("discount_inr"))).round(1).alias("res")))
    nc = rv.filter(~pl.col("is_completed")).select(pl.len().alias("n"), pl.col("list_price_inr").is_not_null().sum().alias("lp"),
                                                   pl.col("tariff_code").is_not_null().sum().alias("tc"),
                                                   pl.col("amount_charged_inr").sum().alias("amt"),
                                                   (pl.col("list_price_inr") - pl.col("discount_inr")).sum().alias("lmd")).collect().row(0, named=True)
    rev = rv.filter(pl.col("is_completed")).select(pl.col("amount_charged_inr").sum()).collect().item()
    print(f"non-completed {nc['n']:,}: list price populated {nc['lp']:,}, tariff populated {nc['tc']:,}, amount sum {nc['amt']:.0f}; "
          f"list-discount sum {nc['lmd']:,.0f} = +{100 * nc['lmd'] / rev:.2f}% of revenue {rev:,.0f}")
    fp = partners().select("partner_id", "peak_surcharge_billable")
    c = rv.filter(pl.col("is_completed")).join(fp.lazy(), on="partner_id", how="left").with_columns(
        pl.col("city").is_in(PILOT_CITIES).alias("pc"), (pl.col("date") >= PILOT_START).alias("post"))
    rt = c.group_by("tariff_code", "res").len().collect().sort("tariff_code", "len", descending=[False, True])
    print("residual (amount - (list - discount)) by tariff code:"); print(rt.filter(pl.col("len") > 50))
    hp = (c.filter(pl.col("tariff_code") == "PARTNER").group_by("peak_surcharge_billable", "pc", "post")
          .agg(pl.len().alias("n"), (pl.col("res") == 15).sum().alias("p15"), (pl.col("res") == -8).sum().alias("m8"),
               pl.col("res").is_in([5.0, -5.0, 10.0, -10.0]).sum().alias("pm5_10")).collect().sort("peak_surcharge_billable", "pc", "post"))
    print(hp)
    p15 = int(hp["p15"].sum()); m8 = int(hp["m8"].sum())
    denom = int(hp.filter(pl.col("pc") & pl.col("post"))["n"].sum())
    denomY = int(hp.filter(pl.col("pc") & pl.col("post") & (pl.col("peak_surcharge_billable") == "Y"))["n"].sum())
    print(f"PARTNER +15: {p15:,}; -8: {m8:,}; total {p15 + m8:,} = {pct(p15 + m8, denom, 1)} of {denom:,} post-pilot partner swaps in pilot cities "
          f"({pct(p15 + m8, denomY, 1)} of the {denomY:,} by Y-flag partners); net INR {15 * p15 - 8 * m8:,}")
    hrs = c.filter((pl.col("tariff_code") == "PARTNER") & pl.col("res").is_in([15.0, -8.0])).group_by("res").agg(
        pl.len(), pl.col("hour").is_between(12, 22).mean().alias("share_12_22"), pl.col("hour").min().alias("hmin"), pl.col("hour").max().alias("hmax")).collect()
    print("hours of hidden partner adjustments:"); print(hrs)
    pk_coded = c.filter(pl.col("tariff_code").is_in(["PEAK", "OFFPEAK"])).group_by("tariff_code", pl.col("partner_id").is_null().alias("indep")).len().collect()
    print("PEAK/OFFPEAK coded completed swaps by independent flag:"); print(pk_coded)
    n_coded = int(pk_coded["len"].sum())
    print(f"all pilot-priced completed swaps = {n_coded + p15 + m8:,}; share hidden in PARTNER = {pct(p15 + m8, n_coded + p15 + m8, 1)}")
    # unexplained deviations
    expl = (pl.when(pl.col("tariff_code") == "PEAK").then(pl.col("res") == 15)
            .when(pl.col("tariff_code") == "OFFPEAK").then(pl.col("res") == -8)
            .when(pl.col("tariff_code") == "PARTNER").then(pl.col("res").is_in([0.0, 15.0, -8.0]))
            .otherwise(pl.col("res") == 0))
    ux = c.filter(~expl).select(pl.len().alias("n"), pl.col("res").sum().alias("net"), pl.col("res").abs().sum().alias("gross"),
                                pl.col("res").unique().sort().alias("vals")).collect().row(0, named=True)
    print(f"unexplained: {ux['n']:,} swaps ({pct(ux['n'], 3586675, 3)}), net {ux['net']:,.0f}, gross {ux['gross']:,.0f}, values {ux['vals']}")
    pp = c.group_by("plan_type", "tariff_code").agg(pl.len(), pl.col("discount_inr").mean().alias("disc"), pl.col("res").mean().alias("res")).collect().sort("plan_type", "tariff_code")
    print(pp)
    print(f"prepaid_pack completed swaps: {int(pp.filter(pl.col('plan_type') == 'prepaid_pack')['len'].sum()):,}")

# %% [markdown]
# ## 6 · Sub-70% SoH packs issued; the 15-Jun-2025 'retirement'

# %%
if 6 in RUN:
    hdr("6. sub-EOL packs / retirement")
    bt = batteries().select("battery_id", "supplier", "pack_type", "retired_date", "current_soh_pct", "manufacturing_lot")
    iss = (events(["date", "battery_out_id", "soh_out_pct", "is_completed"]).filter(pl.col("is_completed"))
           .join(bt.lazy().rename({"battery_id": "battery_out_id"}), on="battery_out_id", how="left").collect())
    sub = iss.filter(pl.col("soh_out_pct") < EOL_SOH)
    print(f"completed swaps issuing SoH<70 packs: {sub.height:,} ({pct(sub.height, iss.height)} of completed); first {sub['date'].min()}; "
          f"suppliers {sub.group_by('supplier', 'pack_type').len().to_dicts()}; distinct packs {sub['battery_out_id'].n_unique():,}; "
          f"of them from 'retired' packs {int(sub['retired_date'].is_not_null().sum()):,}")
    aft = iss.filter(pl.col("date") > pl.col("retired_date"))
    on_ = iss.filter(pl.col("date") >= pl.col("retired_date"))
    print(f"issued strictly after retired_date: {aft.height:,} swaps, {aft['battery_out_id'].n_unique():,} packs, median SoH {aft['soh_out_pct'].median():.1f}; "
          f"incl. 15-Jun itself {on_.height:,}")
    print(f"current_soh of retired packs: max {bt.filter(pl.col('retired_date').is_not_null())['current_soh_pct'].max()}, "
          f"active packs min {bt.filter(pl.col('retired_date').is_null())['current_soh_pct'].min()}")
    RET = dt.date(2025, 6, 15)
    w = (events(["date", "is_stockout", "is_completed", "supplier_out", "battery_out_id"])
         .filter(pl.col("date").is_between(RET - dt.timedelta(days=7), RET + dt.timedelta(days=7)) & (pl.col("date") != RET))
         .group_by(pl.col("date") > RET).agg(pl.len().alias("att"), (100 * pl.col("is_stockout").mean()).alias("stockout_pct"),
                                             (100 * (pl.col("supplier_out") == "Kyron").filter(pl.col("is_completed")).mean()).alias("kyron_issued_pct"),
                                             pl.col("battery_out_id").filter(pl.col("supplier_out") == "Kyron").n_unique().alias("kyron_packs"))
         .collect().sort("date"))
    print("7 days before vs after 15-Jun-2025:"); print(w)
    # same-season control for the stockout move: same window one year earlier
    w24 = (events(["date", "is_stockout"]).filter(pl.col("date").is_between(dt.date(2024, 6, 8), dt.date(2024, 6, 22)) & (pl.col("date") != dt.date(2024, 6, 15)))
           .group_by(pl.col("date") > dt.date(2024, 6, 15)).agg((100 * pl.col("is_stockout").mean()).alias("stockout_pct")).collect().sort("date"))
    print("same window 2024 (control):", w24.to_dicts())
    del iss, sub, aft

# %% [markdown]
# ## 7 · Battery IDs teleport

# %%
if 7 in RUN:
    hdr("7. battery ID chains")
    bp = (events(["rider_id", "event_ts", "battery_in_id", "battery_out_id", "city", "is_completed"]).filter(pl.col("is_completed"))
          .select(pl.col("rider_id").hash().alias("r"), "event_ts", pl.col("battery_in_id").hash().alias("bi"), pl.col("battery_out_id").hash().alias("bo"),
                  pl.col("city").cast(pl.Categorical).to_physical().alias("c")))
    rc = (bp.sort("r", "event_ts").with_columns(pl.col("r").shift(1).alias("r0"), pl.col("bo").shift(1).alias("bo0"))
          .filter(pl.col("r") == pl.col("r0")).select(pl.len().alias("n"), (pl.col("bi") == pl.col("bo0")).sum().alias("m")).collect().row(0, named=True))
    print(f"rider chain: returned pack == last issued pack in {rc['m']:,} of {rc['n']:,} = {pct(rc['m'], rc['n'], 3)}")
    pc = (bp.sort("bo", "event_ts").with_columns(pl.col("bo").shift(1).alias("bo0"), pl.col("c").shift(1).alias("c0"), pl.col("event_ts").shift(1).alias("t0"))
          .filter(pl.col("bo") == pl.col("bo0")).with_columns(((pl.col("event_ts") - pl.col("t0")).dt.total_seconds() / 60).alias("gm"))
          .select(pl.len().alias("n"), (pl.col("c") != pl.col("c0")).sum().alias("dc"),
                  ((pl.col("c") != pl.col("c0")) & (pl.col("gm") <= 60)).sum().alias("dc60"),
                  ((pl.col("c") != pl.col("c0")) & (pl.col("gm") <= 10)).sum().alias("dc10"), pl.col("gm").median().alias("med")).collect().row(0, named=True))
    print(f"pack chain: {pc['n']:,} consecutive re-issues; other city {pc['dc']:,} ({pct(pc['dc'], pc['n'], 1)}); other city <=60 min {pc['dc60']:,} "
          f"({pct(pc['dc60'], pc['n'], 2)}); <=10 min {pc['dc10']:,}; median gap {pc['med']:.0f} min")

# %% [markdown]
# ## 8 · Metadata contradictions (3W slots, commission/manufacture dates, lot codes, firmware field)

# %%
if 8 in RUN:
    hdr("8. metadata")
    st = stations().filter(~pl.col("is_test")).select("station_id", "slots_3w")
    e3 = (events(["station_id", "vehicle_class", "is_stockout"]).join(st.lazy(), on="station_id")
          .group_by((pl.col("slots_3w") == 0).alias("zero3"), "vehicle_class").agg(pl.len().alias("att"), (100 * pl.col("is_stockout").mean()).alias("so_pct"),
                                                                                    pl.col("station_id").n_unique().alias("st"))
          .collect().sort("zero3", "vehicle_class"))
    print(e3)
    z3 = e3.filter(pl.col("zero3") & (pl.col("vehicle_class") == "3W"))
    print(f"3W attempts at slots_3w=0 stations: {z3['att'].item():,} ({pct(z3['att'].item(), N_EV)}) at {z3['st'].item()} stations "
          f"(of {st.filter(pl.col('slots_3w') == 0).height} zero-slot stations)")
    for zf in (True, False):
        sub_ = e3.filter(pl.col("zero3") == zf)
        print(f"  zero3={zf}: 3W share {pct(sub_.filter(pl.col('vehicle_class') == '3W')['att'].sum(), sub_['att'].sum())}")
    b = batteries().with_columns((pl.col("commission_date") - pl.col("first_seen").dt.date()).dt.total_days().alias("early"),
                                 (pl.col("first_seen").dt.date() < pl.col("manufacture_date")).alias("pre_mfg"))
    ubc = b.filter(pl.col("first_seen").dt.date() < pl.col("commission_date"))
    print(f"packs used before commission: {ubc.height:,} of {b.height:,} ({pct(ubc.height, b.height, 1)}), median {ubc['early'].median():.0f} d early; "
          f"before manufacture {int(b['pre_mfg'].fill_null(False).sum()):,}")
    lot = b.with_columns(pl.col("manufacturing_lot").str.slice(3, 2).cast(pl.Int32).alias("yy"), pl.col("manufacturing_lot").str.slice(5, 2).cast(pl.Int32).alias("mm"))
    valid = lot.filter(pl.col("mm").is_between(1, 12))
    match = valid.filter((pl.col("manufacture_date").dt.year() == 2000 + pl.col("yy")) & (pl.col("manufacture_date").dt.month() == pl.col("mm")))
    print(f"lot code month == manufacture month: {match.height:,} of {valid.height:,} valid-code packs ({pct(match.height, valid.height, 1)}); "
          f"invalid codes: {sorted(lot.filter(~pl.col('mm').is_between(1, 12))['manufacturing_lot'].unique().to_list())}")
    fwa = events(["station_id", "station_firmware", "date"]).group_by("station_id").agg(pl.col("station_firmware").n_unique().alias("k")).collect()
    pre = events(["station_firmware", "date"]).filter((pl.col("station_firmware") == "v3.2.0") & (pl.col("date") < dt.date(2025, 3, 10))).select(pl.len()).collect().item()
    pre1 = events(["station_firmware", "date"]).filter((pl.col("station_firmware") == "v3.2.1") & (pl.col("date") < dt.date(2025, 4, 14))).select(pl.len()).collect().item()
    print(f"stations with >1 firmware value: {int((fwa['k'] > 1).sum())}; v3.2.0 attempts before 10-Mar-25 install: {pre:,}; "
          f"v3.2.1 attempts before its 14-Apr-25 update date: {pre1:,}")

# %% [markdown]
# ## 9 · CSAT MNAR

# %%
if 9 in RUN:
    hdr("9. CSAT")
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from sklearn.metrics import roc_auc_score
    tk = tickets().filter(pl.col("in_window"))
    print(tk.group_by("resolution_status").agg(pl.len(), pl.col("has_csat").mean()).sort("resolution_status"))
    r = tk.filter(pl.col("resolution_status") == "resolved").select("resolution_hours", "category", "channel", "priority", "has_csat", "csat_score").to_pandas()
    r["dec"] = pd.qcut(r["resolution_hours"], 10, labels=False) + 1
    print(r.groupby("dec").agg(n=("has_csat", "size"), resp=("has_csat", "mean"), csat=("csat_score", "mean")).round(3).T)
    y = r["has_csat"].astype(int)
    m = smf.glm("y ~ C(dec) + C(category) + C(channel) + C(priority)", data=r.assign(y=y), family=sm.families.Binomial()).fit()
    ph = m.predict(r)
    resp = r[r["has_csat"]]
    wts = 1 / ph[r["has_csat"]]
    ipw = float(np.sum(wts * resp["csat_score"]) / wts.sum())
    naive = float(resp["csat_score"].mean())
    se = float(resp["csat_score"].std() / np.sqrt(len(resp)))
    print(f"naive {naive:.3f} (normal CI {naive - 1.96 * se:.3f}-{naive + 1.96 * se:.3f}); IPW (GLM logit) {ipw:.3f}; AUC {roc_auc_score(y, ph):.3f}; n resolved {len(r):,}")
    n_un = tk.filter(pl.col("resolution_status") == "unresolved").height
    for s_ in (1, 3, 5):
        print(f"  unresolved scored {s_}: population CSAT {(len(r) * ipw + n_un * s_) / (len(r) + n_un):.3f} (unresolved share {pct(n_un, len(r) + n_un, 1)})")
    # does CSAT itself depend on resolution speed? (the lever that would make IPW matter)
    print("CSAT by category (responders):", resp.groupby("category")["csat_score"].mean().round(2).to_dict())

# %% [markdown]
# ## 10 · Station failure funnel and stage-2 GLM

# %%
if 10 in RUN:
    hdr("10. station funnel")
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    sv = (events(["station_id", "city", "month", "vehicle_class", "event_type"])
          .group_by("station_id", "city", "month", "vehicle_class").agg(pl.len().alias("a"), pl.col("event_type").is_in(FT).sum().alias("f")).collect())
    cell = sv.group_by("city", "month", "vehicle_class").agg(pl.col("a").sum().alias("A"), pl.col("f").sum().alias("F"))
    sv = sv.join(cell, on=["city", "month", "vehicle_class"]).with_columns(((pl.col("F") - pl.col("f")) / (pl.col("A") - pl.col("a"))).alias("peer"))
    print(f"cells with no peers (peer rate undefined): {sv.filter(pl.col('peer').is_null() | pl.col('peer').is_nan()).height}")
    fu = (sv.filter(pl.col("peer").is_not_null() & pl.col("peer").is_not_nan()).group_by("station_id")
          .agg(pl.col("a").sum().alias("att"), pl.col("f").sum().alias("O"), (pl.col("a") * pl.col("peer")).sum().alias("E"))
          .with_columns((pl.col("O") / pl.col("E")).alias("smr"), ((pl.col("O") - pl.col("E")) / pl.col("E").sqrt()).alias("z")))
    z = fu["z"].to_numpy()
    lo, hi = np.percentile(z, [10, 90])
    phi_w = float(np.mean(np.clip(z, lo, hi) ** 2))
    phi_p = float(np.sum(z ** 2) / (len(z) - 1))
    print(f"stations {fu.height}; Poisson |z|>3.09: high {(z > 3.09).sum()} low {(z < -3.09).sum()}; phi winsorised {phi_w:.1f}, plain Pearson {phi_p:.1f}")
    print(f"after phi_w: high {(z / np.sqrt(phi_w) > 3.09).sum()} low {(z / np.sqrt(phi_w) < -3.09).sum()}")
    st = stations().filter(~pl.col("is_test")).select("station_id", "charger_generation", "expansion_wave", "location_type", "connectivity_tier", "commissioned_date")
    smu = station_month().group_by("station_id").agg(pl.col("swaps_per_slot_day").mean().alias("util"))
    fd = fu.join(st, on="station_id").join(smu, on="station_id", how="left").to_pandas()
    for col in ["charger_generation", "expansion_wave"]:
        print(fd.groupby(col).apply(lambda d: pd.Series({"st": len(d), "pooled_OE": d["O"].sum() / d["E"].sum()}), include_groups=False).round(3).T)
    print(pd.crosstab(fd["expansion_wave"], fd["charger_generation"]))
    fd["lu"] = np.log(fd["util"]); fd["lE"] = np.log(fd["E"])
    g2 = smf.glm("O ~ C(charger_generation) + C(expansion_wave) + C(location_type) + C(connectivity_tier) + lu", data=fd,
                 family=sm.families.Poisson(), offset=fd["lE"]).fit(scale="X2")
    print(f"stage-2 Pearson scale {g2.scale:.2f}; deviance explained {100 * (1 - g2.deviance / g2.null_deviance):.1f}%")
    out = pd.DataFrame({"RR": np.exp(g2.params), "lo": np.exp(g2.conf_int()[0]), "hi": np.exp(g2.conf_int()[1]), "p": g2.pvalues}).round(3)
    print(out)
    mu = g2.fittedvalues
    z2 = (fd["O"] - mu) / np.sqrt(mu * g2.scale)
    z2p = (fd["O"] - mu) / np.sqrt(mu)
    print(f"stage-2 anomalies |z|>3.09 with quasi scale: {(np.abs(z2) > 3.09).sum()}; with pure Poisson limits: {(np.abs(z2p) > 3.09).sum()}; "
          f"max |O/mu - 1| = {np.abs(fd['O'] / mu - 1).max():.3f}; 99.8% limit at median mu would be +/-{3.09 * np.sqrt(g2.scale / np.median(mu)):.3f}")
    fd["z2"] = z2
    print(fd.reindex(fd["z2"].abs().sort_values(ascending=False).index).head(4)[["station_id", "charger_generation", "expansion_wave", "O", "E", "z2"]].round(2))
    # simpler, more interpretable alternative: what does generation alone explain?
    g1 = smf.glm("O ~ C(charger_generation)", data=fd, family=sm.families.Poisson(), offset=fd["lE"]).fit(scale="X2")
    print(f"generation-only model: scale {g1.scale:.1f}, deviance explained {100 * (1 - g1.deviance / g1.null_deviance):.1f}%")

# %% [markdown]
# ## 11 · Month-boundary failure regime vs temperature and price

# %%
if 11 in RUN:
    hdr("11. month regime")
    d = (events(["date", "event_type"]).group_by("date").agg(pl.len().alias("a"), pl.col("event_type").is_in(FT).mean().alias("r")).collect().sort("date"))
    cd = city_daily().group_by("date").agg(pl.col("max_temp_c").mean().alias("tmax"), pl.col("heat_alert").sum().alias("heat"))
    d = d.join(cd, on="date", how="left").to_pandas()
    d["date"] = pd.to_datetime(d["date"])
    rows = []
    for m0 in pd.date_range("2024-02-01", "2025-06-01", freq="MS"):
        for kind, piv in [("1st", m0), ("16th", m0 + pd.Timedelta(days=15))]:
            b_ = d[(d["date"] >= piv - pd.Timedelta(days=7)) & (d["date"] < piv)]
            a_ = d[(d["date"] >= piv) & (d["date"] < piv + pd.Timedelta(days=7))]
            rows.append(dict(pivot=piv.date(), kind=kind, before=100 * b_["r"].mean(), after=100 * a_["r"].mean(),
                             step=100 * (a_["r"].mean() - b_["r"].mean()), t_step=a_["tmax"].mean() - b_["tmax"].mean(),
                             att_step_pct=100 * (a_["a"].mean() / b_["a"].mean() - 1)))
    sp = pd.DataFrame(rows)
    print(sp[sp["kind"] == "1st"].round(2).to_string(index=False))
    print(f"max |step| around the 16th: {sp.loc[sp.kind == '16th', 'step'].abs().max():.2f} pts; max |tmax step| on the 16th {sp.loc[sp.kind == '16th', 't_step'].abs().max():.2f} C, "
          f"on the 1st {sp.loc[sp.kind == '1st', 't_step'].abs().max():.2f} C")
    mm = (events(["month", "event_type"]).group_by("month").agg(pl.len().alias("a"), (100 * pl.col("event_type").is_in(FT).mean()).alias("fail_pct")).collect().sort("month"))
    print(mm.to_pandas().round(2).T.to_string())
    print(f"attempts in Mar+Jun: {int(mm.filter(pl.col('month').dt.month().is_in([3, 6]))['a'].sum()):,}")
    # daily temperature around the boundaries -- smooth or stepped?
    for piv in ["2024-04-01", "2024-05-01", "2024-07-01", "2025-05-01"]:
        p0 = pd.Timestamp(piv)
        seg = d[(d["date"] >= p0 - pd.Timedelta(days=3)) & (d["date"] < p0 + pd.Timedelta(days=3))]
        print(piv, "daily fail%:", (100 * seg["r"]).round(2).tolist(), "| mean city max temp:", seg["tmax"].round(1).tolist())
    # station-hourly ambient temperature and charged-pack stock around 1-Jul-2024 (mechanism check)
    hh = (hourly(["date", "ambient_temp_c", "cabinet_temp_c", "charged_2w_avg", "packs_quarantined", "chargers_online"])
          .filter(pl.col("date").is_between(dt.date(2024, 6, 24), dt.date(2024, 7, 7)))
          .group_by("date").agg(pl.all().exclude("date").mean()).collect().sort("date"))
    print(hh)
    # heat-alert days within city-month
    cdy = (events(["city", "date", "event_type"]).group_by("city", "date").agg(pl.col("event_type").is_in(FT).mean().alias("r")).collect()
           .join(city_daily().select("city", "date", "heat_alert"), on=["city", "date"], how="left").with_columns(pl.col("date").dt.truncate("1mo").alias("m")))
    hh2 = (cdy.group_by("city", "m").agg(pl.col("r").filter(pl.col("heat_alert")).mean().alias("rh"), pl.col("r").filter(~pl.col("heat_alert")).mean().alias("rn"),
                                          pl.col("heat_alert").sum().alias("nh"), (~pl.col("heat_alert")).sum().alias("nn"))
           .filter((pl.col("nh") > 0) & (pl.col("nn") > 0)).with_columns((100 * (pl.col("rh") - pl.col("rn"))).alias("diff")).to_pandas())
    ci = stats.t.interval(0.95, len(hh2) - 1, loc=hh2["diff"].mean(), scale=stats.sem(hh2["diff"]))
    print(f"heat-alert vs non-alert days in same city-month: {len(hh2)} city-months; unweighted mean {hh2['diff'].mean():+.2f} pts (CI {ci[0]:+.2f}..{ci[1]:+.2f}), "
          f"weighted by heat days {np.average(hh2['diff'], weights=hh2['nh']):+.2f}; p = {stats.ttest_1samp(hh2['diff'], 0).pvalue:.2f}; heat days total {int(hh2['nh'].sum())}")
    print(hh2.round(2).to_string(index=False))

# %% [markdown]
# ## 12 · Battery lots: is the KY-2407/08/09 3W anomaly real or an exposure artefact?

# %%
if 12 in RUN:
    hdr("12. lots")
    b = batteries()
    bl = b.filter((pl.col("n_cycles") >= 50) & pl.col("soh_loss_per_cycle").is_not_null())
    k2 = bl.filter((pl.col("supplier") == "Kyron") & (pl.col("pack_type") == "2W_2.1kWh"))["soh_loss_per_cycle"]
    o2 = bl.filter((pl.col("supplier") != "Kyron") & (pl.col("pack_type") == "2W_2.1kWh"))["soh_loss_per_cycle"]
    print(f"Kyron 2W median {k2.median():.4f} vs Cellora+Amptek 2W {o2.median():.4f} (x{k2.median() / o2.median():.2f})")
    k3 = (b.filter((pl.col("supplier") == "Kyron") & (pl.col("pack_type") == "3W_4.8kWh"))
          .group_by("manufacturing_lot").agg(pl.len().alias("packs"), (pl.col("n_cycles") >= 50).sum().alias("n50"),
                                             pl.col("soh_loss_per_cycle").filter(pl.col("n_cycles") >= 50).median().alias("slope_med"),
                                             pl.col("n_cycles").median().alias("cyc_med"), pl.col("first_seen").min().alias("first_seen"),
                                             pl.col("retired_date").is_not_null().sum().alias("retired"), pl.col("current_soh_pct").median().alias("soh_now"),
                                             pl.col("initial_soh_pct").median().alias("soh0"))
          .sort("manufacturing_lot"))
    print(k3)
    fl = ["KY-2407", "KY-2408", "KY-2409"]
    peers = bl.filter((pl.col("supplier") == "Kyron") & (pl.col("pack_type") == "3W_4.8kWh") & ~pl.col("manufacturing_lot").is_in(fl))["soh_loss_per_cycle"]
    for lt in fl:
        x = bl.filter((pl.col("supplier") == "Kyron") & (pl.col("pack_type") == "3W_4.8kWh") & (pl.col("manufacturing_lot") == lt))["soh_loss_per_cycle"]
        print(f"{lt} 3W: n={x.len()} median {x.median():.4f} vs other Kyron 3W lots {peers.median():.4f} ({100 * (x.median() / peers.median() - 1):+.0f}%), "
              f"MWU p={stats.mannwhitneyu(x.to_numpy(), peers.to_numpy()).pvalue:.2g}")
    still = b.filter((pl.col("supplier") == "Kyron") & (pl.col("pack_type") == "3W_4.8kWh") & pl.col("manufacturing_lot").is_in(fl) & pl.col("retired_date").is_null())
    print(f"still in service from flagged lots: {still.height}, current SoH {still['current_soh_pct'].min():.1f}-{still['current_soh_pct'].max():.1f}, "
          f"median slope {still['soh_loss_per_cycle'].median():.4f}, cycles median {still['n_cycles'].median():.0f}")
    # exposure check: slope over the FIRST 150 cycles only (same exposure for every pack), from the issued-pack trajectories
    ids = b.filter((pl.col("supplier") == "Kyron") & (pl.col("pack_type") == "3W_4.8kWh")).select("battery_id", "manufacturing_lot")
    tr = (events(["battery_out_id", "event_ts", "soh_out_pct", "is_completed"]).filter(pl.col("is_completed"))
          .join(ids.lazy().rename({"battery_id": "battery_out_id"}), on="battery_out_id").collect()
          .sort("battery_out_id", "event_ts").with_columns(pl.int_range(1, pl.len() + 1).over("battery_out_id").alias("cyc")))
    early = (tr.filter(pl.col("cyc") <= 150).group_by("battery_out_id", "manufacturing_lot")
             .agg(pl.len().alias("n"), (-pl.cov("cyc", "soh_out_pct") / pl.col("cyc").var()).alias("slope150"),
                  pl.col("event_ts").min().alias("t0"), pl.col("event_ts").filter(pl.col("cyc") == 150).first().alias("t150"))
             .filter(pl.col("n") == 150))
    print(early.group_by(pl.col("manufacturing_lot").is_in(fl).alias("flagged")).agg(pl.len(), pl.col("slope150").median(),
          pl.col("t0").median(), ((pl.col("t150") - pl.col("t0")).dt.total_days()).median().alias("days_for_150")))
    # same calendar window comparison (exposure-matched): SoH loss per cycle using only swaps in Jan-Jun 2025
    cal = (tr.filter(pl.col("event_ts") >= dt.datetime(2025, 1, 1)).with_columns(pl.int_range(1, pl.len() + 1).over("battery_out_id").alias("c2"))
           .group_by("battery_out_id", "manufacturing_lot").agg(pl.len().alias("n"), (-pl.cov("c2", "soh_out_pct") / pl.col("c2").var()).alias("s"))
           .filter(pl.col("n") >= 50))
    print(cal.group_by(pl.col("manufacturing_lot").is_in(fl).alias("flagged")).agg(pl.len(), pl.col("s").median()))
    bm = bl.group_by("supplier", "pack_type", "bms_firmware").agg(pl.len(), pl.col("soh_loss_per_cycle").median()).sort("supplier", "pack_type", "bms_firmware")
    print(bm)
    for (sp_, pt_), gdf in bl.group_by(["supplier", "pack_type"]):
        arrs = [g_["soh_loss_per_cycle"].to_numpy() for _, g_ in gdf.group_by("bms_firmware")]
        print(sp_, pt_, f"Kruskal p = {stats.kruskal(*arrs).pvalue:.3f}")

# %% [markdown]
# ## 13 · Connectivity, firmware, energy

# %%
if 13 in RUN:
    hdr("13. connectivity / energy")
    ct = (events(["connectivity_tier", "sync_mode", "is_system_error"]).group_by("connectivity_tier")
          .agg(pl.len().alias("n"), (100 * (pl.col("sync_mode") == "offline_batch").mean()).alias("off_pct"), pl.col("is_system_error").sum().alias("se"))
          .collect().sort("connectivity_tier").with_columns((100 * pl.col("se") / pl.col("n")).alias("se_pct")))
    print(ct)
    print(f"chi2 p system error by tier: {stats.chi2_contingency(np.column_stack([ct['se'], ct['n'] - ct['se']]))[1]:.3f}")
    print(events(["station_firmware", "is_system_error"]).group_by("station_firmware").agg(pl.len(), (100 * pl.col("is_system_error").mean()).alias("se_pct")).collect())
    en = (events(["energy_to_recharge_kwh", "soc_in_pct", "soh_in_pct", "vehicle_class", "is_completed", "charger_generation"]).filter(pl.col("is_completed"))
          .with_columns((pl.when(pl.col("vehicle_class") == "3W").then(4.8).otherwise(2.1) * (100 - pl.col("soc_in_pct")) / 100 * pl.col("soh_in_pct") / 100).alias("need"))
          .with_columns((pl.col("energy_to_recharge_kwh") / pl.col("need")).alias("k")))
    print(f"corr(energy, need) = {en.select(pl.corr('energy_to_recharge_kwh', 'need')).collect().item():.3f}")
    eg = en.group_by("charger_generation").agg(pl.col("k").median()).collect().sort("charger_generation")
    print(eg, f"Gen1/Gen3 - 1 = {100 * (eg['k'][0] / eg['k'][2] - 1):.1f}%")

# %% [markdown]
# ## 14 · Small documented issues (km, SoC/SoH > 100, city spellings) and scorecard counts

# %%
if 14 in RUN:
    hdr("14. small issues")
    km = (pl.scan_parquet(PARQ / "swap_events.parquet").filter(~pl.col("station_id").str.starts_with("STN-TST") & pl.col("km_since_last_swap").is_not_null())
          .select(pl.col("km_since_last_swap").cast(pl.Float64).alias("km"), pl.col("battery_in_id").str.slice(4, 2).alias("vc")).collect())
    bad = (km["km"] < 0) | (km["km"] > 200)
    print(f"km readings {km.height:,}; invalid {int(bad.sum()):,} ({pct(int(bad.sum()), km.height)}); neg {int((km['km'] < 0).sum()):,}; >200 {int((km['km'] > 200).sum()):,}")
    k2 = km.filter(pl.col("vc") == "2W")
    print(f"2W mean raw {k2['km'].mean():.2f} sd {k2['km'].std():.2f}; clean mean {k2.filter(pl.col('km').is_between(0, 200))['km'].mean():.2f} "
          f"sd {k2.filter(pl.col('km').is_between(0, 200))['km'].std():.2f}; valid range {km.filter(pl.col('km').is_between(0, 200))['km'].min():.1f}-"
          f"{km.filter(pl.col('km').is_between(0, 200))['km'].max():.1f}; outlier ranges {km.filter(pl.col('km') < 0)['km'].min():.0f}..{km.filter(pl.col('km') < 0)['km'].max():.0f} / "
          f"{km.filter(pl.col('km') > 200)['km'].min():.0f}..{km.filter(pl.col('km') > 200)['km'].max():.0f}")
    del km
    sc = pd.read_csv(TAB / "q7_data_quality_scorecard.csv")
    print(sc["kind"].value_counts().to_dict(), "| severity:", sc["severity"].value_counts().to_dict(), "| rows", len(sc))
    print(sc.loc[sc["kind"] == "undocumented", ["id", "issue", "severity"]].to_string(index=False))

# %% [markdown]
# ## 15 · Extra adversarial checks (not in the analyst's script)
# a) the TRUE pilot tariff schedule (derived from the data) as a sharper firmware clock, and whether hidden PARTNER
#    +15/-8 rows follow it; b) wear cost the cost model charges on packs already below 70% EOL; c) temperature steps at
#    month boundaries and the cross-city contrast on 1-Jul-2024 (price vs heat); d) funnel with generation x hot-city;
#    e) integer-minute truncation in the teleport count; f) billing disputes, firmware and competitor DiDs.

# %%
if 15 in RUN:
    hdr("15. extra checks")
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    PKH, OPH = [12, 13, 19, 20, 21, 22], [23, 0, 1, 2, 3, 4, 5, 14, 15, 16]

    def sched(h):
        return pl.when(h.is_in(PKH)).then(pl.lit("PEAK")).when(h.is_in(OPH)).then(pl.lit("OFFPEAK")).otherwise(pl.lit("STD"))

    # a) tariff schedule
    tz = (events(["city", "date", "hour", "event_ts_raw", "tariff_code", "is_independent", "station_firmware", "is_completed",
                  "list_price_inr", "discount_inr", "amount_charged_inr", "partner_id"])
          .filter(pl.col("city").is_in(PILOT_CITIES) & (pl.col("date") >= PILOT_START))
          .with_columns(sched(pl.col("hour")).alias("exp"), sched(pl.col("event_ts_raw").dt.hour()).alias("exp_raw"),
                        (pl.col("amount_charged_inr") - (pl.col("list_price_inr") - pl.col("discount_inr"))).round(1).alias("res")))
    ind = tz.filter(pl.col("is_independent"))
    print("hour bands by tariff code (independents, pilot cities, post-pilot):",
          ind.group_by("tariff_code").agg(pl.col("hour").unique().sort().alias("hours")).collect().sort("tariff_code").to_dicts())
    bad = ind.filter(pl.col("exp") != pl.col("tariff_code")).collect()
    print(f"contradictions with the derived schedule (corrected clock), whole post-pilot period: {bad.height}; "
          f"all v3.2.0 on 2025-03-09: {bad.filter((pl.col('station_firmware') == 'v3.2.0') & (pl.col('date') == dt.date(2025, 3, 9))).height}; "
          f"logged {bad['event_ts_raw'].min()} .. {bad['event_ts_raw'].max()}; codes {bad['tariff_code'].unique().to_list()}")
    win = ind.filter(pl.col("event_ts_raw").dt.date().is_between(dt.date(2025, 3, 10), dt.date(2025, 4, 14)) & (pl.col("station_firmware") == "v3.2.0")).collect()
    pk = win.filter(pl.col("tariff_code") == "PEAK")
    print(f"v3.2.0 in window: code agrees with schedule raw {pct((win['exp_raw'] == win['tariff_code']).sum(), win.height, 1)} vs corrected "
          f"{pct((win['exp'] == win['tariff_code']).sum(), win.height, 1)} (n={win.height:,}); PEAK rows inside TRUE peak hours raw "
          f"{pct(pk['event_ts_raw'].dt.hour().is_in(PKH).sum(), pk.height, 1)} vs fixed {pct(pk['hour'].is_in(PKH).sum(), pk.height, 1)} (n={pk.height:,})")
    hid = (tz.filter(pl.col("is_completed") & (pl.col("tariff_code") == "PARTNER") & pl.col("res").is_in([15.0, -8.0]))
           .group_by("res").agg(pl.len(), (pl.col("exp") == pl.when(pl.col("res") == 15).then(pl.lit("PEAK")).otherwise(pl.lit("OFFPEAK"))).mean().alias("on_schedule"))
           .collect())
    print("hidden PARTNER adjustments on the PEAK/OFFPEAK schedule:", hid.to_dicts())
    # b) wear past EOL
    w = (events(["month", "is_completed", "soh_out_pct", "wear_cost", "cm1"]).filter(pl.col("is_completed"))
         .group_by("month").agg(pl.len().alias("swaps"), (pl.col("soh_out_pct") < EOL_SOH).sum().alias("sub70"),
                                pl.col("wear_cost").filter(pl.col("soh_out_pct") < EOL_SOH).sum().alias("wear_sub70"), pl.col("cm1").sum().alias("cm1"))
         .with_columns((pl.col("cm1") / pl.col("swaps")).alias("cm1_ps"), ((pl.col("cm1") + pl.col("wear_sub70")) / pl.col("swaps")).alias("cm1_ps_no_wear_past_eol"),
                       (pl.col("wear_sub70") / pl.col("swaps")).alias("wear_past_eol_ps"))
         .collect().sort("month").filter(pl.col("month") >= dt.date(2025, 4, 1)))
    print(w)
    tw = events(["is_completed", "soh_out_pct", "wear_cost"]).filter(pl.col("is_completed") & (pl.col("soh_out_pct") < EOL_SOH)).select(
        pl.col("wear_cost").sum().alias("tot"), pl.col("wear_cost").mean().alias("mean"), pl.len()).collect().row(0, named=True)
    print(f"wear charged on {tw['len']:,} swaps that issued a pack already below 70% SoH: INR {tw['tot']:,.0f} (INR {tw['mean']:.1f}/swap)")
    # c) temperature at month boundaries; cross-city contrast on 1-Jul-2024
    cdx = (events(["city", "date", "event_type"]).group_by("city", "date").agg(pl.len().alias("a"), pl.col("event_type").is_in(FT).mean().alias("r")).collect()
           .join(city_daily().select("city", "date", "max_temp_c"), on=["city", "date"]))
    jj = (cdx.filter(pl.col("date").is_between(dt.date(2024, 6, 24), dt.date(2024, 7, 7)))
          .group_by("city", (pl.col("date") >= BASE_PRICE_CHANGE).alias("post")).agg(pl.col("max_temp_c").mean().round(1).alias("tmax"),
                                                                                     (100 * pl.col("r").mean()).round(2).alias("fail_pct"), pl.col("a").mean().round(0).alias("att_day"))
          .sort("city", "post"))
    print("1-Jul-2024, 7 days before vs after, by city:"); print(jj.pivot(on="post", index="city", values=["tmax", "fail_pct", "att_day"]))
    cdx = cdx.with_columns(pl.col("date").dt.truncate("1mo").alias("m"))
    cm = cdx.group_by("city", "m").agg(pl.col("max_temp_c").mean().alias("t"), pl.col("r").mean().alias("r"))
    dd = cdx.with_columns((pl.col("max_temp_c") - pl.col("max_temp_c").mean().over("city", "m")).alias("dt_"))
    print(f"within city-month SD of daily max temp {dd['dt_'].std():.2f} C; corr(city-month mean max temp, failure rate) = {np.corrcoef(cm['t'], cm['r'])[0, 1]:.3f}")
    print(cm.with_columns(pl.col("t").cut([34, 36, 38]).alias("band")).group_by("band").agg(pl.len().alias("city_months"), (100 * pl.col("r").mean()).round(2).alias("fail_pct")).sort("band"))
    hhx = (hourly(["date", "ambient_temp_c"]).filter(pl.col("date").is_between(dt.date(2024, 6, 24), dt.date(2024, 7, 7)))
           .group_by(pl.col("date") >= BASE_PRICE_CHANGE).agg(pl.col("ambient_temp_c").mean()).collect().sort("date"))
    print(f"station ambient temp 7 d before/after 1-Jul-2024: {hhx['ambient_temp_c'][0]:.1f} -> {hhx['ambient_temp_c'][1]:.1f} C")
    # d) funnel with a generation x hot-city interaction
    HOT = ["Delhi NCR", "Jaipur", "Hyderabad"]
    sv = (events(["station_id", "city", "month", "vehicle_class", "event_type"]).group_by("station_id", "city", "month", "vehicle_class")
          .agg(pl.len().alias("a"), pl.col("event_type").is_in(FT).sum().alias("f")).collect())
    cell = sv.group_by("city", "month", "vehicle_class").agg(pl.col("a").sum().alias("A"), pl.col("f").sum().alias("F"))
    sv = (sv.join(cell, on=["city", "month", "vehicle_class"]).with_columns(((pl.col("F") - pl.col("f")) / (pl.col("A") - pl.col("a"))).alias("peer"),
                                                                           pl.col("month").dt.month().is_in([4, 5, 6]).alias("summer"), pl.col("city").is_in(HOT).alias("hot"))
          .join(stations().select("station_id", "charger_generation", "expansion_wave", "location_type", "connectivity_tier"), on="station_id"))
    print("stage-1 pooled O/E by hot city x Apr-Jun x generation:")
    print(sv.group_by("hot", "summer", "charger_generation").agg((pl.col("f").sum() / (pl.col("a") * pl.col("peer")).sum()).round(2).alias("oe"))
          .pivot(on="charger_generation", index=["hot", "summer"], values="oe").sort("hot", "summer"))
    fu = (sv.group_by("station_id", "charger_generation", "expansion_wave", "location_type", "connectivity_tier", "hot")
          .agg(pl.col("f").sum().alias("O"), (pl.col("a") * pl.col("peer")).sum().alias("E"))
          .join(station_month().group_by("station_id").agg(pl.col("swaps_per_slot_day").mean().alias("util")), on="station_id").to_pandas())
    fu["lu"], fu["lE"] = np.log(fu["util"]), np.log(fu["E"])
    for lab, f_ in [("analyst spec", "O ~ C(charger_generation) + C(expansion_wave) + C(location_type) + C(connectivity_tier) + lu"),
                    ("+ generation x hot city", "O ~ C(charger_generation)*hot + C(expansion_wave) + C(location_type) + C(connectivity_tier) + lu")]:
        g_ = smf.glm(f_, data=fu, family=sm.families.Poisson(), offset=fu["lE"]).fit(scale="X2")
        mu_ = g_.fittedvalues
        z_ = (fu["O"] - mu_) / np.sqrt(mu_ * g_.scale)
        small = mu_ < 300
        print(f"{lab:24s}: scale {g_.scale:5.2f}; deviance explained {100 * (1 - g_.deviance / g_.null_deviance):.1f}%; |z|>3.09: {(np.abs(z_) > 3.09).sum()} "
              f"{fu.loc[np.abs(z_) > 3.09, 'station_id'].tolist()}; 99.8% band at median mu +/-{100 * 3.09 * np.sqrt(g_.scale / np.median(mu_)):.0f}%; "
              f"{small.sum()} stations with mu<300 (Wave2: {(fu.loc[small, 'expansion_wave'] == 'Wave2_2025H1').sum()}; hot-city: {int(fu.loc[small, 'hot'].sum())}) "
              f"pooled O/mu {fu.loc[small, 'O'].sum() / mu_[small].sum():.2f}")
    # e) teleport minute truncation
    pc = (events(["event_ts", "battery_out_id", "city", "is_completed"]).filter(pl.col("is_completed"))
          .select("event_ts", pl.col("battery_out_id").hash().alias("bo"), pl.col("city").cast(pl.Categorical).to_physical().alias("c"))
          .sort("bo", "event_ts").with_columns(pl.col("bo").shift(1).alias("b0"), pl.col("c").shift(1).alias("c0"), pl.col("event_ts").shift(1).alias("t0"))
          .filter((pl.col("bo") == pl.col("b0")) & (pl.col("c") != pl.col("c0")))
          .with_columns((pl.col("event_ts") - pl.col("t0")).alias("gap"))
          .select((pl.col("gap").dt.total_minutes() <= 10).sum().alias("int_min_le10"), (pl.col("gap").dt.total_seconds() <= 600).sum().alias("strict_le10"),
                  (pl.col("gap").dt.total_minutes() <= 60).sum().alias("int_min_le60"), (pl.col("gap").dt.total_seconds() <= 3600).sum().alias("strict_le60"))
          .collect())
    print("cross-city re-issues of the same pack:", pc.to_dicts())
    # f) billing disputes after unexplained deviations; firmware and competitor DiDs
    expl = (pl.when(pl.col("tariff_code") == "PEAK").then(pl.col("res") == 15).when(pl.col("tariff_code") == "OFFPEAK").then(pl.col("res") == -8)
            .when(pl.col("tariff_code") == "PARTNER").then(pl.col("res").is_in([0.0, 15.0, -8.0])).otherwise(pl.col("res") == 0))
    cc = (events(["event_id", "rider_id", "event_ts", "is_completed", "tariff_code", "list_price_inr", "discount_inr", "amount_charged_inr"])
          .filter(pl.col("is_completed")).with_columns((pl.col("amount_charged_inr") - (pl.col("list_price_inr") - pl.col("discount_inr"))).round(1).alias("res")))
    bdt = tickets().filter(pl.col("category") == "billing_dispute").select("rider_id", "created_ts")

    def drate(df):
        hit = (df.join(bdt, on="rider_id").filter((pl.col("created_ts") >= pl.col("event_ts")) & (pl.col("created_ts") <= pl.col("event_ts") + pl.duration(days=7)))
               ["event_id"].n_unique())
        return hit, df.height

    k1, n1 = drate(cc.filter(~expl).select("event_id", "rider_id", "event_ts").collect())
    k0, n0 = drate(cc.filter(expl & (pl.col("event_id").str.slice(3).cast(pl.Int64) % 40 == 7)).select("event_id", "rider_id", "event_ts").collect())
    print(f"billing-dispute ticket within 7 d: deviations {pct(k1, n1)} (n={n1:,}) vs 1-in-40 normal sample {pct(k0, n0)} (n={n0:,}); "
          f"chi2 p = {stats.chi2_contingency(np.array([[k1, n1 - k1], [k0, n0 - k0]]))[1]:.2f}")
    fe = (events(["date", "station_firmware", "is_system_error"]).filter(pl.col("date").is_between(dt.date(2025, 2, 1), dt.date(2025, 4, 14)))
          .group_by("station_firmware", (pl.col("date") >= dt.date(2025, 3, 10)).alias("bug")).agg((100 * pl.col("is_system_error").mean()).alias("se")).collect())
    fd_ = {(r["station_firmware"], r["bug"]): r["se"] for r in fe.iter_rows(named=True)}
    print(f"firmware system-error DiD (bug window - 1 Feb..9 Mar, v3.2.0 - v3.2.1): {(fd_[('v3.2.0', True)] - fd_[('v3.2.0', False)]) - (fd_[('v3.2.1', True)] - fd_[('v3.2.1', False)]):+.3f} pts")
    cp = (events(["date", "station_id", "competitor_since"]).filter(pl.col("date").is_between(COMPETITOR_ENTRY - dt.timedelta(days=56), COMPETITOR_ENTRY + dt.timedelta(days=55)))
          .group_by(pl.col("competitor_since").is_not_null().alias("tr"), (pl.col("date") >= COMPETITOR_ENTRY).alias("post"))
          .agg((pl.len() / pl.struct("station_id", "date").n_unique()).alias("apd")).collect())
    cpd = {(r["tr"], r["post"]): r["apd"] for r in cp.iter_rows(named=True)}
    print(f"competitor entry DiD (attempts per station-day, ratio of ratios): {(cpd[(True, True)] / cpd[(True, False)]) / (cpd[(False, True)] / cpd[(False, False)]):.3f}")

print(f"\nruntime {time.time() - T0:.0f}s")
