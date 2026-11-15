# %% [markdown]
# # Q2 - Service failures and customer experience
# **Question.** How do queue wait, failed attempts and abandoned swaps relate to station, hour of day, season and
# vehicle class? Is service quality spread evenly across the network, or concentrated?
#
# **Approach.**
# 1. Failure taxonomy and rates (2W vs 3W).
# 2. Queue-wait distributions (completed vs abandoned), a Kaplan-Meier "patience" curve for abandonment, and queue
#    wait vs station utilisation.
# 3. Hour of day (firmware-corrected), day of week, month/season, plus an hour x month heatmap.
# 4. City x charger generation x season, and city-day weather (heat alerts, max temperature, grid outages, rain).
# 5. Concentration: Lorenz / Gini across stations and station-hours (against a chance-only null), excess failures on a
#    fair (Apr-Jun) denominator, and an observed/expected test of whether the largest sites are unusual for their city.
# 6. Heat mechanism from hourly telemetry (ambient -> cabinet temperature -> charge time -> charged inventory), a
#    station-day binomial GLM (city, month and weekday controls, station-clustered errors), and a trap check: does the
#    hourly stock level predict WHICH hours fail once the day and the hour of day are held fixed? (It does not: the
#    heat effect is a day-level association.)
# 7. 3W specifics; 8. like-for-like trend at a fixed set of stations; 9. support tickets vs failure rates;
# 10. sizing for recommendations, and what a rider does after a failure (retry or not).
#
# **Definitions** (docs/analysis_brief.md): service failure = stockout (`failed_no_charged_battery`) + `abandoned_queue`
# + `failed_system_error`; `cancelled_by_rider` is reported but not counted as a failure. Hours are the corrected local
# hour (firmware v3.2.0 bug fixed in `src/clean.py`). Telemetry blanks stay null; we report coverage.

# %%
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from scipy import stats

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # running inside a notebook
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from common import *  # noqa: E402,F401,F403

warnings.filterwarnings("ignore")
set_style()
T0 = time.time()
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(24)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_float_precision(4)

HOT_MONTHS = [4, 5, 6]                       # Apr-Jun: the only months where failures depart from baseline
HOT_CITIES = ["Delhi NCR", "Jaipur", "Hyderabad"]
GEN_COLORS = {"Gen1": PALETTE["accent"], "Gen2": PALETTE["warn"], "Gen3": PALETTE["good"]}
FAIL = pl.col("is_service_failure")


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (vectorised)."""
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, centre - half, centre + half


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    cum = np.cumsum(x)
    return (n + 1 - 2 * cum.sum() / cum[-1]) / n


def top_share(x, frac):
    x = np.sort(np.asarray(x, float))[::-1]
    k = int(np.ceil(frac * len(x)))
    return x[:k].sum() / x.sum()


def conc_curve(x):
    """Concentration curve: units sorted from worst to best -> (cum share of units, cum share of x)."""
    x = np.sort(np.asarray(x, float))[::-1]
    return np.r_[0, np.arange(1, len(x) + 1) / len(x)], np.r_[0, np.cumsum(x) / x.sum()]


def mh_rr(df: pd.DataFrame, a, n1, c, n0):
    """Mantel-Haenszel risk ratio across strata with Greenland-Robins 95% CI.
    a/n1 = events/total in the exposed group, c/n0 = events/total in the reference group (one row per stratum)."""
    d = df.loc[(df[n1] > 0) & (df[n0] > 0), [a, n1, c, n0]].astype("float64")  # float: avoid uint overflow
    N = d[n1] + d[n0]
    R = (d[a] * d[n0] / N).sum()
    S = (d[c] * d[n1] / N).sum()
    P = ((d[n1] * d[n0] * (d[a] + d[c]) - d[a] * d[c] * N) / N ** 2).sum()
    rr = R / S
    se = np.sqrt(P / (R * S))
    return rr, rr * np.exp(-1.96 * se), rr * np.exp(1.96 * se), len(d)


def headline(fig, title, subtitle=None, top=0.84):
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.text(0.01, 0.995, title, ha="left", va="top", fontsize=15, fontweight="bold", color="#1a1a1a")
    if subtitle:
        fig.text(0.01, 0.995 - (1 - top) * 0.55, subtitle, ha="left", va="top", fontsize=11, color="#555555")


def month_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))


def ptable(name, df, n=None):
    print(f"\n--- {name} ---")
    print(df if n is None else df.head(n))


# %% [markdown]
# ## 1. Failure taxonomy
# Every attempt ends as one of five outcomes. Stockouts and abandonments dominate the failures; system errors are a
# small, flat noise floor.

# %%
EV_COLS = ["station_id", "event_ts", "event_ts_raw", "ts_corrected", "date", "month", "hour", "dow", "event_type",
           "is_completed", "is_stockout", "is_abandoned", "is_system_error", "is_cancelled", "is_service_failure",
           "queue_wait_sec", "vehicle_class", "city", "charger_generation", "expansion_wave", "connectivity_tier",
           "station_firmware", "sync_mode"]
ev = events(EV_COLS)
st_all = stations().filter(~pl.col("is_test"))

tax = ev.group_by("event_type", "vehicle_class").agg(pl.len().alias("n")).collect()
tax_w = (tax.pivot(on="vehicle_class", index="event_type", values="n")
         .with_columns((pl.col("2W") + pl.col("3W")).alias("all")))
tot = {c: int(tax_w[c].sum()) for c in ["2W", "3W", "all"]}
tax_w = (tax_w.with_columns(
    (pl.col("all") / tot["all"]).alias("rate_all"), (pl.col("2W") / tot["2W"]).alias("rate_2W"),
    (pl.col("3W") / tot["3W"]).alias("rate_3W"), pl.col("event_type").is_in(FAIL_TYPES).alias("service_failure"))
    .sort("all", descending=True))
fails = tax_w.filter("service_failure")
N_ATT, N_FAIL = tot["all"], int(fails["all"].sum())
FR_ALL = N_FAIL / N_ATT
FR_2W = fails["2W"].sum() / tot["2W"]
FR_3W = fails["3W"].sum() / tot["3W"]
tax_w.write_csv(TAB / "q2_taxonomy.csv")
ptable("Outcome taxonomy (counts and rates by vehicle class)", tax_w)
print(f"[TAX] attempts={N_ATT:,}  service failures={N_FAIL:,}  failure rate={pct(FR_ALL, 2)}  "
      f"(2W {pct(FR_2W, 2)} of {tot['2W']:,}; 3W {pct(FR_3W, 2)} of {tot['3W']:,})")
for r in fails.iter_rows(named=True):
    print(f"[TAX] {r['event_type']}: {r['all']:,} = {pct(r['rate_all'], 2)} of attempts, "
          f"{pct(r['all'] / N_FAIL, 1)} of service failures")
for dim in ["connectivity_tier", "station_firmware", "sync_mode"]:
    se_d = ev.group_by(dim).agg(pl.len().alias("n"), pl.col("is_system_error").mean().alias("r")).sort(dim).collect()
    print(f"[TAX] system-error rate by {dim}: " + ", ".join(f"{r[dim]} {pct(r['r'], 3)} (n={r['n']:,})" for r in se_d.iter_rows(named=True)))
canc = tax_w.filter(pl.col("event_type") == "cancelled_by_rider").row(0, named=True)
print(f"[TAX] cancelled_by_rider (not a service failure): {canc['all']:,} = {pct(canc['rate_all'], 2)}")

# %% [markdown]
# ## 2. Queue wait: completed vs abandoned, rider patience, and load
# `queue_wait_sec` = arrival to completion (completed) or to exit (abandoned). Stockouts / system errors exit within
# 90 s. We estimate rider **patience** with Kaplan-Meier: abandonment is the event, completion censors the wait
# (assumes service time is independent of patience).

# %%
qq = (ev.group_by("event_type").agg(
    pl.len().alias("n"), pl.col("queue_wait_sec").mean().alias("mean_s"),
    *[pl.col("queue_wait_sec").quantile(q).alias(f"p{int(q * 100)}_s") for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)],
    pl.col("queue_wait_sec").min().alias("min_s"), pl.col("queue_wait_sec").max().alias("max_s"))
    .sort("n", descending=True).collect())
qq.write_csv(TAB / "q2_queue_wait_quantiles.csv")
ptable("Queue wait (seconds) by outcome", qq)
qc = qq.filter(pl.col("event_type") == "swap_completed").row(0, named=True)
qa = qq.filter(pl.col("event_type") == "abandoned_queue").row(0, named=True)
print(f"[QW] completed: median {qc['p50_s'] / 60:.1f} min, mean {qc['mean_s'] / 60:.2f} min, p90 {qc['p90_s'] / 60:.1f} min,"
      f" max {qc['max_s'] / 60:.0f} min")
print(f"[QW] abandoned: median {qa['p50_s'] / 60:.1f} min, p10 {qa['p10_s'] / 60:.1f} min, min {qa['min_s'] / 60:.2f} min,"
      f" max {qa['max_s'] / 60:.0f} min; median ratio abandoned/completed = {qa['p50_s'] / qc['p50_s']:.1f}x")

# Kaplan-Meier patience curve
qd = (ev.filter(pl.col("event_type").is_in(["swap_completed", "abandoned_queue"]))
      .group_by("queue_wait_sec").agg(pl.col("is_abandoned").sum().alias("ab"), pl.col("is_completed").sum().alias("comp"))
      .sort("queue_wait_sec").collect())
t_s = qd["queue_wait_sec"].to_numpy().astype(float)
d_ab = qd["ab"].to_numpy().astype(float)
d_cp = qd["comp"].to_numpy().astype(float)
at_risk = (d_ab + d_cp).sum() - np.r_[0, np.cumsum(d_ab + d_cp)[:-1]]
S_km = np.cumprod(1 - d_ab / at_risk)
km_med = t_s[np.argmax(S_km <= 0.5)] / 60 if (S_km <= 0.5).any() else np.nan
share_ab_beyond_p90 = d_ab[t_s > qc["p90_s"]].sum() / d_ab.sum()


def S_at(minutes):
    idx = np.searchsorted(t_s, minutes * 60, side="right") - 1
    return S_km[max(idx, 0)]


minute = (t_s // 60).astype(int)
pc = (pd.DataFrame({"minute": minute, "ab": d_ab, "comp": d_cp}).groupby("minute", as_index=False).sum())
pc["at_risk_start"] = (pc["ab"] + pc["comp"]).sum() - np.r_[0, np.cumsum(pc["ab"] + pc["comp"])[:-1]]
pc["hazard_per_min"] = pc["ab"] / pc["at_risk_start"]
pc["share_abandoned_among_ended"] = pc["ab"] / (pc["ab"] + pc["comp"])
pc["km_still_waiting"] = [S_at(m + 1) for m in pc["minute"]]
pc.to_csv(TAB / "q2_patience_curve.csv", index=False)
cross50 = pc.loc[(pc["share_abandoned_among_ended"] >= 0.5) & (pc["ab"] + pc["comp"] >= 500), "minute"].min()
print(f"[QW] Kaplan-Meier patience: still willing to wait at 5 min {pct(S_at(5))}, 10 min {pct(S_at(10))}, "
      f"15 min {pct(S_at(15))}, 20 min {pct(S_at(20))}; median patience = {km_med:.1f} min")
for m in (5, 8, 10, 12, 15):
    r = pc.loc[pc["minute"] == m].iloc[0]
    print(f"[QW] hazard of abandoning during minute {m}-{m + 1}: {pct(r['hazard_per_min'])} "
          f"(at risk {int(r['at_risk_start']):,})")
print(f"[QW] first minute where abandonments outnumber completions among attempts ending then: {cross50} min")
print(f"[QW] {pct(share_ab_beyond_p90)} of abandoners had waited longer than the completed-swap p90 "
      f"({qc['p90_s'] / 60:.1f} min)")

# %%
# Station-hour aggregate (used for load, telemetry and concentration analyses)
SH = (ev.with_columns(pl.col("event_ts").dt.truncate("1h").alias("hour_start"))
      .group_by("station_id", "hour_start").agg(
          pl.len().cast(pl.Int32).alias("att"), pl.col("is_completed").sum().cast(pl.Int32).alias("comp"),
          FAIL.sum().cast(pl.Int32).alias("fail"), pl.col("is_stockout").sum().cast(pl.Int32).alias("so"),
          pl.col("is_abandoned").sum().cast(pl.Int32).alias("ab"), pl.col("is_system_error").sum().cast(pl.Int32).alias("se"),
          (pl.col("vehicle_class") == "3W").sum().cast(pl.Int32).alias("att3"),
          ((pl.col("vehicle_class") == "3W") & pl.col("is_stockout")).sum().cast(pl.Int32).alias("so3"),
          ((pl.col("vehicle_class") == "3W") & pl.col("is_abandoned")).sum().cast(pl.Int32).alias("ab3"),
          pl.col("queue_wait_sec").filter(pl.col("is_completed")).cast(pl.Int64).sum().alias("qws"),
          pl.col("queue_wait_sec").filter(pl.col("is_completed")).cast(pl.Float64).pow(2).sum().alias("qws2"))
      .collect(engine="streaming").sort("station_id", "hour_start"))  # sorted: makes the seeded null simulation reproducible
slots = st_all.select("station_id", "slots_2w", "slots_3w", (pl.col("slots_2w") + pl.col("slots_3w")).alias("slots"))
SH = SH.join(slots.select("station_id", "slots"), on="station_id", how="left").with_columns(
    (pl.col("comp") / pl.col("slots")).alias("util"), (pl.col("att") / pl.col("slots")).alias("demand_util"))
print(f"[LOAD] station-hours with >=1 attempt: {SH.height:,}")

util_edges = [0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 99]
util_lbl = ["<0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-0.75", "0.75-1.0", "1.0-1.5", ">=1.5"]
qu = (SH.filter(pl.col("comp") > 0)
      .with_columns(pl.col("util").cut(util_edges[1:-1], labels=util_lbl, left_closed=True).alias("util_bin"))
      .group_by("util_bin").agg(pl.len().alias("station_hours"), pl.col("comp").sum(), pl.col("att").sum(),
                                (pl.col("qws").sum() / pl.col("comp").sum() / 60).alias("mean_wait_min"),
                                pl.col("qws2").sum().alias("ss"),
                                (pl.col("ab").sum() / pl.col("att").sum()).alias("abandon_rate"))
      .with_columns(((pl.col("ss") / pl.col("comp") - (pl.col("mean_wait_min") * 60) ** 2) / pl.col("comp")).sqrt().alias("se_s"))
      .with_columns((pl.col("mean_wait_min") - 1.96 * pl.col("se_s") / 60).alias("ci_lo_min"),
                    (pl.col("mean_wait_min") + 1.96 * pl.col("se_s") / 60).alias("ci_hi_min"))
      .drop("ss", "se_s").sort("util_bin"))
du = (SH.with_columns(pl.col("demand_util").cut(util_edges[1:-1], labels=util_lbl, left_closed=True).alias("demand_bin"))
      .group_by("demand_bin").agg(pl.len().alias("station_hours"), pl.col("att").sum(),
                                  (pl.col("ab").sum() / pl.col("att").sum()).alias("abandon_rate"),
                                  (pl.col("so").sum() / pl.col("att").sum()).alias("stockout_rate"))
      .sort("demand_bin"))
qu.write_csv(TAB / "q2_queue_vs_utilisation.csv")
du.write_csv(TAB / "q2_abandon_vs_demand_per_slot.csv")
ptable("Queue wait vs utilisation (completed swaps per slot-hour)", qu)
ptable("Abandonment / stockouts vs demand (attempts per slot-hour)", du)
q_lo, q_hi = qu.row(0, named=True), qu.row(qu.height - 1, named=True)
st_wait = (ev.filter("is_completed").group_by("station_id").agg(pl.col("queue_wait_sec").mean().alias("m")).collect())["m"]
mo_wait = (ev.filter("is_completed").group_by("month").agg(pl.col("queue_wait_sec").mean().alias("m")).collect())["m"]
print(f"[LOAD] mean completed wait by utilisation bin ranges {qu['mean_wait_min'].min():.2f}-{qu['mean_wait_min'].max():.2f} min; "
      f"lightest bin ({q_lo['util_bin']}) {q_lo['mean_wait_min'] * 60:.1f} s (95% CI {q_lo['ci_lo_min'] * 60:.1f}-{q_lo['ci_hi_min'] * 60:.1f}) vs "
      f"busiest bin ({q_hi['util_bin']}) {q_hi['mean_wait_min'] * 60:.1f} s (95% CI {q_hi['ci_lo_min'] * 60:.1f}-{q_hi['ci_hi_min'] * 60:.1f})")
print(f"[LOAD] station mean completed wait ranges {st_wait.min():.0f}-{st_wait.max():.0f} s across {len(st_wait)} stations;"
      f" monthly network mean {mo_wait.min():.0f}-{mo_wait.max():.0f} s")
d_lo, d_hi = du.row(0, named=True), du.row(du.height - 1, named=True)
print(f"[LOAD] demand per slot-hour {d_lo['demand_bin']} -> {d_hi['demand_bin']}: abandonment {pct(d_lo['abandon_rate'], 2)} -> "
      f"{pct(d_hi['abandon_rate'], 2)}, stockouts {pct(d_lo['stockout_rate'], 2)} -> {pct(d_hi['stockout_rate'], 2)}")

# %% [markdown]
# ## 3. When do failures happen? Hour, weekday, month, season
# Hours use the corrected timestamp. We also show what the uncorrected firmware clock would have done to the hour
# profile of the affected events.

# %%
def rates(df_lazy, keys):
    return (df_lazy.group_by(keys).agg(
        pl.len().alias("attempts"), FAIL.sum().alias("failures"), FAIL.mean().alias("failure_rate"),
        pl.col("is_stockout").mean().alias("stockout_rate"), pl.col("is_abandoned").mean().alias("abandon_rate"),
        pl.col("is_system_error").mean().alias("system_error_rate"), pl.col("is_cancelled").mean().alias("cancel_rate"))
        .sort(keys).collect())


by_hour = rates(ev, ["hour"]).with_columns((pl.col("attempts") / N_ATT).alias("share_attempts"),
                                           (pl.col("failures") / N_FAIL).alias("share_failures"))
by_dow = rates(ev, ["dow"])
by_month = rates(ev, ["month"])
by_hour.write_csv(TAB / "q2_by_hour.csv")
by_dow.write_csv(TAB / "q2_by_dow.csv")
by_month.write_csv(TAB / "q2_by_month.csv")
ptable("By hour (corrected)", by_hour)
ptable("By day of week (1=Mon)", by_dow)
ptable("By month", by_month)
EVE = [19, 20, 21, 22]
eve = by_hour.filter(pl.col("hour").is_in(EVE))
oth = by_hour.filter(~pl.col("hour").is_in(EVE))
print(f"[HOUR] evening peak 19:00-22:59: {pct(eve['share_attempts'].sum())} of attempts, {pct(eve['share_failures'].sum())} of"
      f" failures; failure rate {pct(eve['failures'].sum() / eve['attempts'].sum(), 2)} vs {pct(oth['failures'].sum() / oth['attempts'].sum(), 2)}"
      f" in other hours")
pk = by_hour.filter(pl.col("hour").is_between(12, 22))
print(f"[HOUR] pilot PEAK window 12:00-22:59: {pct(pk['share_attempts'].sum())} of attempts, {pct(pk['share_failures'].sum())} of failures")
by_dow_jm = rates(ev.filter(~pl.col("month").dt.month().is_in(HOT_MONTHS)), ["dow"])
by_dow.join(by_dow_jm.select("dow", pl.col("failure_rate").alias("failure_rate_jul_mar")), on="dow").write_csv(TAB / "q2_by_dow.csv")
print(f"[DOW] failure rate range across weekdays: full period (Jan-24 to Jun-25) {pct(by_dow['failure_rate'].min(), 2)}-"
      f"{pct(by_dow['failure_rate'].max(), 2)}; Jul-Mar only {pct(by_dow_jm['failure_rate'].min(), 2)}-{pct(by_dow_jm['failure_rate'].max(), 2)}")
bm = by_month.with_columns(pl.col("month").dt.month().alias("m"), pl.col("month").dt.year().alias("y"))
base_mo = bm.filter(~pl.col("m").is_in(HOT_MONTHS))
BASE_FR = base_mo["failures"].sum() / base_mo["attempts"].sum()
print(f"[MONTH] Jul-Mar baseline failure rate {pct(BASE_FR, 2)} (monthly range {pct(base_mo['failure_rate'].min(), 2)}-"
      f"{pct(base_mo['failure_rate'].max(), 2)}); Apr-Jun months: " +
      ", ".join(f"{r['month']:%b-%y} {pct(r['failure_rate'], 1)}" for r in bm.filter(pl.col("m").is_in(HOT_MONTHS)).iter_rows(named=True)))
hot_ev = bm.filter(pl.col("m").is_in(HOT_MONTHS))
print(f"[MONTH] Apr-Jun = {pct(hot_ev['attempts'].sum() / N_ATT)} of attempts but {pct(hot_ev['failures'].sum() / N_FAIL)} of failures")

# firmware-clock check on the shifted events
fw = (ev.filter("ts_corrected").select(pl.col("hour"), pl.col("event_ts_raw").dt.hour().alias("hour_raw"), FAIL).collect())
print(f"[FW] {fw.height:,} events had the +5h30m correction; share in 19:00-22:59 corrected = "
      f"{pct(fw['hour'].is_in(EVE).mean())} vs raw clock = {pct(fw['hour_raw'].is_in(EVE).mean())}; "
      f"their failures placed in 19-22h: corrected {pct(fw.filter('is_service_failure')['hour'].is_in(EVE).mean())}"
      f" vs raw {pct(fw.filter('is_service_failure')['hour_raw'].is_in(EVE).mean())}")
del fw

# hour x month matrices
hm = rates(ev, ["month", "hour"])
months = sorted(hm["month"].unique().to_list())
M_fail = hm.pivot(on="month", index="hour", values="failure_rate").sort("hour")
M_so = hm.pivot(on="month", index="hour", values="stockout_rate").sort("hour")
M_fail.write_csv(TAB / "q2_hour_month_failure_rate.csv")
M_so.write_csv(TAB / "q2_hour_month_stockout_rate.csv")
A_fail = M_fail.select([str(m) for m in months]).to_numpy()
A_so = M_so.select([str(m) for m in months]).to_numpy()
imax = np.unravel_index(np.nanargmax(A_fail), A_fail.shape)
print(f"[HEAT] worst hour x month cell: {months[imax[1]]:%b-%y} {imax[0]:02d}:00 failure rate {pct(A_fail[imax])}; "
      f"best cell {pct(np.nanmin(A_fail))}")
may_eve = hm.filter(pl.col("month").dt.month() == 5, pl.col("hour").is_in(EVE))
may_oth = hm.filter(pl.col("month").dt.month() == 5, ~pl.col("hour").is_in(EVE))
print(f"[HEAT] May (both years) failure rate 19-22h {pct(may_eve['failures'].sum() / may_eve['attempts'].sum())} vs other hours "
      f"{pct(may_oth['failures'].sum() / may_oth['attempts'].sum())}")
cool_eve = hm.filter(~pl.col("month").dt.month().is_in(HOT_MONTHS), pl.col("hour").is_in(EVE))
cool_oth = hm.filter(~pl.col("month").dt.month().is_in(HOT_MONTHS), ~pl.col("hour").is_in(EVE))
print(f"[HEAT] Jul-Mar failure rate 19-22h {pct(cool_eve['failures'].sum() / cool_eve['attempts'].sum(), 2)} vs other hours "
      f"{pct(cool_oth['failures'].sum() / cool_oth['attempts'].sum(), 2)}")
ecomp = (ev.filter(~pl.col("month").dt.month().is_in(HOT_MONTHS))
         .with_columns(pl.col("hour").is_in(EVE).alias("eve"))
         .group_by("eve").agg((pl.col("vehicle_class") == "3W").mean().alias("share_3w"),
                              FAIL.filter(pl.col("vehicle_class") == "2W").mean().alias("fr_2w"),
                              FAIL.filter(pl.col("vehicle_class") == "3W").mean().alias("fr_3w")).sort("eve").collect())
e_t, e_f = ecomp.filter(pl.col("eve")).row(0, named=True), ecomp.filter(~pl.col("eve")).row(0, named=True)
print(f"[HEAT] Jul-Mar composition check: 3W share of attempts 19-22h {pct(e_t['share_3w'])} vs {pct(e_f['share_3w'])} other hours; "
      f"within class: 2W {pct(e_t['fr_2w'], 2)} vs {pct(e_f['fr_2w'], 2)}, 3W {pct(e_t['fr_3w'], 2)} vs {pct(e_f['fr_3w'], 2)}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), sharey=True)
mlabels = [f"{m:%b}\n{m:%y}" if m.month in (1, 4, 7, 10) else f"{m:%b}" for m in months]
for ax, A, ttl, vmax in [(axes[0], A_fail, "Service-failure rate (stockout + abandoned + system error)", 0.20),
                         (axes[1], A_so, "Stockout rate (no charged battery)", 0.14)]:
    im = ax.imshow(A, aspect="auto", cmap="YlOrRd", vmin=0.02, vmax=vmax, origin="upper")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(mlabels, fontsize=9.5)
    ax.set_yticks(range(0, 24, 2))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.set_title(ttl, fontsize=11.5)
    ax.grid(False)
    for x in (3, 15):
        ax.add_patch(plt.Rectangle((x - 0.5, -0.5), 3, 24, fill=False, ec="#333333", lw=1.4, ls="--"))
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, format=mtick.PercentFormatter(1.0, decimals=0))
    cb.set_ticks(np.arange(0.02, vmax + 1e-9, 0.02))
    cb.ax.tick_params(labelsize=9.5)
axes[0].set_ylabel("Hour of day (corrected local time)")
axes[0].annotate(f"Peak {pct(A_fail[imax])}\n({months[imax[1]]:%b-%y}, {imax[0]:02d}:00)", xy=(imax[1], imax[0]),
                 xytext=(imax[1] + 3.2, imax[0] - 6), fontsize=10, arrowprops=dict(arrowstyle="->", color="#222222"),
                 bbox=dict(boxstyle="round", fc="white", ec="#999999"))
axes[0].text(9.5, 3.5, f"Jul-Mar: flat ~{pct(BASE_FR, 1)}\nat every hour\n(within vehicle class)", ha="center", fontsize=10.5,
             bbox=dict(boxstyle="round", fc="white", ec="#999999"))
headline(fig, f"Failures are a summer problem: Apr-Jun cells reach {pct(A_fail[imax], 0)}, the rest of the year is a flat ~{pct(BASE_FR, 0)}",
         "Share of swap attempts failing, by corrected hour of day x month, Jan 2024 - Jun 2025 (dashed boxes = Apr-Jun). "
         "Evening hours (19-22h) are worst inside the summer window.", top=0.86)
save_fig(fig, "q2_02_hour_month_heatmap")

# %% [markdown]
# ## 4. Where: city x charger generation x season, and city-day weather
# Outside Apr-Jun every city and every charger generation sits at the same ~3.9%. The summer excess appears only in
# hot cities and scales with charger generation. Gen comparisons are made inside the same city and month (Mantel-
# Haenszel), because Gen3 stations exist mainly in 2025.

# %%
cm = rates(ev, ["city", "month"])
cm.pivot(on="city", index="month", values="failure_rate").sort("month").write_csv(TAB / "q2_city_month_failure_rate.csv")
ev_s = ev.with_columns(pl.when(pl.col("month").dt.month().is_in(HOT_MONTHS)).then(pl.lit("Apr-Jun"))
                       .otherwise(pl.lit("Jul-Mar")).alias("season_grp"))
cgs = rates(ev_s, ["season_grp", "city", "charger_generation"])
p_, lo_, hi_ = wilson(cgs["failures"].to_numpy(), cgs["attempts"].to_numpy())
cgs = cgs.with_columns(pl.Series("ci_lo", lo_), pl.Series("ci_hi", hi_))
cgs.write_csv(TAB / "q2_city_generation_season.csv")
ptable("Failure rate by season group x city x generation", cgs.select("season_grp", "city", "charger_generation", "attempts",
                                                                    "failure_rate", "ci_lo", "ci_hi"))
cs = rates(ev_s, ["season_grp", "city"]).pivot(on="season_grp", index="city", values="failure_rate").sort("Apr-Jun", descending=True)
ptable("Failure rate by city: Apr-Jun vs Jul-Mar", cs)
cool_cg = cgs.filter(pl.col("season_grp") == "Jul-Mar")
print(f"[CITY] Jul-Mar failure rate across all 18 city x generation cells: {pct(cool_cg['failure_rate'].min(), 2)}-"
      f"{pct(cool_cg['failure_rate'].max(), 2)}")

# Gen comparison within city x month strata (Apr-Jun 2025: all generations present)
gm = (ev.filter(pl.col("month") >= pl.date(2025, 4, 1))
      .group_by("city", "month", "charger_generation").agg(pl.len().alias("n"), FAIL.sum().alias("f")).collect().to_pandas())
gw = gm.pivot_table(index=["city", "month"], columns="charger_generation", values=["n", "f"], aggfunc="sum").fillna(0)
gw.columns = [f"{a}_{b}" for a, b in gw.columns]
gen_rr = {}
for g in ["Gen1", "Gen2"]:
    gen_rr[g] = mh_rr(gw.reset_index(), f"f_{g}", f"n_{g}", "f_Gen3", "n_Gen3")
    print(f"[GEN] Apr-Jun 2025, same city & month: {g} vs Gen3 failure risk ratio (MH) = {gen_rr[g][0]:.2f} "
          f"(95% CI {gen_rr[g][1]:.2f}-{gen_rr[g][2]:.2f}; {gen_rr[g][3]} city-month strata)")
# Is it the wave (site choice) or the charger? Compare waves inside the same generation, city and month
wv = (ev.group_by("city", "month", "expansion_wave", "charger_generation").agg(pl.len().alias("n"), FAIL.sum().alias("f"))
      .collect().to_pandas())
wave_rr = {}
for g, exp_w, ref_w in [("Gen2", "Wave1_2024H2", "Launch"), ("Gen3", "Wave2_2025H1", "Wave1_2024H2")]:
    s = wv[wv["charger_generation"] == g].pivot_table(index=["city", "month"], columns="expansion_wave", values=["n", "f"],
                                                      aggfunc="sum").fillna(0)
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    wave_rr[(g, exp_w)] = mh_rr(s.reset_index(), f"f_{exp_w}", f"n_{exp_w}", f"f_{ref_w}", f"n_{ref_w}")
    r = wave_rr[(g, exp_w)]
    print(f"[WAVE] {g} only, same city & month: {exp_w} vs {ref_w} failure risk ratio (MH) = {r[0]:.2f} (95% CI {r[1]:.2f}-{r[2]:.2f};"
          f" {r[3]} strata)")
# Effect modification: the pooled Gen1 vs Gen3 ratio mixes hot and cool cities (heat x generation interaction)
gw_r = gw.reset_index()
COOL_CITIES = [c for c in CITIES if c not in HOT_CITIES]
gen_city_rows = []
for lab, members in [(c, [c]) for c in HOT_CITIES + COOL_CITIES] + [("Hot cities pooled", HOT_CITIES),
                                                                     ("Other cities pooled", COOL_CITIES)]:
    r = mh_rr(gw_r[gw_r["city"].isin(members)], "f_Gen1", "n_Gen1", "f_Gen3", "n_Gen3")
    gen_city_rows.append({"city": lab, "gen1_vs_gen3_rr": r[0], "ci_lo": r[1], "ci_hi": r[2], "month_strata": r[3],
                          "gen1_attempts": int(gw_r.loc[gw_r["city"].isin(members), "n_Gen1"].sum()),
                          "gen1_failures": int(gw_r.loc[gw_r["city"].isin(members), "f_Gen1"].sum())})
gen_city = pd.DataFrame(gen_city_rows)
gen_city.to_csv(TAB / "q2_gen1_vs_gen3_rr_by_city.csv", index=False)
GEN_RR_CITY = dict(zip(gen_city["city"], gen_city["gen1_vs_gen3_rr"]))
print("[GEN] Apr-Jun 2025, Gen1 vs Gen3 MH risk ratio by city (month strata): " + ", ".join(
    f"{r.city} {r.gen1_vs_gen3_rr:.2f} ({r.ci_lo:.2f}-{r.ci_hi:.2f})" for r in gen_city.itertuples()))
hc25 = (ev.filter(pl.col("month") >= pl.date(2025, 4, 1), pl.col("city").is_in(HOT_CITIES))
        .group_by("charger_generation").agg(pl.len().alias("n"), FAIL.mean().alias("fr")).sort("charger_generation").collect())
print("[GEN] Apr-Jun 2025 only, hot cities: " + ", ".join(f"{r['charger_generation']} {pct(r['fr'], 2)} (n={r['n']:,})"
                                                        for r in hc25.iter_rows(named=True)))
hcall = (ev.filter(pl.col("month").dt.month().is_in(HOT_MONTHS), pl.col("city").is_in(HOT_CITIES))
         .group_by("charger_generation").agg(pl.len().alias("n"), FAIL.mean().alias("fr")).sort("charger_generation").collect())
print("[GEN] Apr-Jun 2024+2025 (both summers), hot cities: " + ", ".join(
    f"{r['charger_generation']} {pct(r['fr'], 2)} (n={r['n']:,})" for r in hcall.iter_rows(named=True))
      + "  [Gen3 exists only in 2025]")

# %%
# City-day weather context (join on city + date)
cdd = (ev.group_by("city", "date").agg(pl.len().alias("att"), FAIL.sum().alias("fail"), pl.col("is_stockout").sum().alias("so"),
                                        pl.col("is_abandoned").sum().alias("ab"), pl.col("is_system_error").sum().alias("se"))
       .collect().join(city_daily(), on=["city", "date"], how="left")
       .with_columns((pl.col("fail") / pl.col("att")).alias("fr"), pl.col("date").dt.truncate("1mo").alias("month")))
print(f"[WX] city-days {cdd.height:,}; unmatched to city_daily: {cdd.filter(pl.col('max_temp_c').is_null()).height}")
tb = (cdd.with_columns((pl.col("max_temp_c") // 2 * 2).alias("tmax_bin"))
      .group_by("tmax_bin").agg(pl.len().alias("city_days"), pl.col("att").sum(), (pl.col("fail").sum() / pl.col("att").sum()).alias("failure_rate"),
                                (pl.col("so").sum() / pl.col("att").sum()).alias("stockout_rate"))
      .sort("tmax_bin"))
tb.write_csv(TAB / "q2_failure_by_daily_max_temp.csv")
ptable("Failure rate by city-day max temperature (2C bins)", tb)
for lab, cond in [("<34C", pl.col("max_temp_c") < 34), ("34-38C", pl.col("max_temp_c").is_between(34, 38, closed="left")),
                  (">=38C", pl.col("max_temp_c") >= 38)]:
    s = cdd.filter(cond)
    print(f"[WX] city-days with max temp {lab}: {s.height} days, failure rate {pct(s['fail'].sum() / s['att'].sum(), 2)}")


def mh_binary(df: pl.DataFrame, flag: pl.Expr, event: str, label: str):
    """MH risk ratio of `event` for days where flag is true vs false, within city-month strata."""
    g = (df.with_columns(flag.alias("x")).group_by("city", "month", "x").agg(pl.col(event).sum().alias("e"), pl.col("att").sum().alias("n"))
         .to_pandas().pivot_table(index=["city", "month"], columns="x", values=["e", "n"], aggfunc="sum").fillna(0))
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    rr = mh_rr(g.reset_index(), "e_True", "n_True", "e_False", "n_False")
    print(f"[WX] {label}: MH risk ratio {rr[0]:.2f} (95% CI {rr[1]:.2f}-{rr[2]:.2f}) within {rr[3]} city-month strata")
    return rr


wx = {}
ha = cdd.filter(pl.col("heat_alert"))
print(f"[WX] heat-alert city-days: {ha.height} (cities: {sorted(ha['city'].unique().to_list())}; months: "
      f"{sorted(ha['date'].dt.month().unique().to_list())}); raw failure rate {pct(ha['fail'].sum() / ha['att'].sum())} vs "
      f"{pct(cdd.filter(~pl.col('heat_alert'))['fail'].sum() / cdd.filter(~pl.col('heat_alert'))['att'].sum())} on other days")
wx["heat_alert"] = mh_binary(cdd, pl.col("heat_alert"), "fail", "heat alert vs no alert, service failures")
wx["tmax_ge36"] = mh_binary(cdd, pl.col("max_temp_c") >= 36, "fail", "daily max >=36C vs <36C (same city-month), service failures")
wx["grid_outage"] = mh_binary(cdd, pl.col("grid_outage_hours") > 0, "fail", "grid outage >0h vs none, service failures")
wx["grid_outage_se"] = mh_binary(cdd, pl.col("grid_outage_hours") > 0, "se", "grid outage >0h vs none, system errors")
wx["grid_outage_so"] = mh_binary(cdd, pl.col("grid_outage_hours") > 0, "so", "grid outage >0h vs none, stockouts")
wx["rain"] = mh_binary(cdd, pl.col("rainfall_mm") >= 20, "fail", "rain >=20mm vs less, service failures")
wx["rain_ab"] = mh_binary(cdd, pl.col("rainfall_mm") >= 20, "ab", "rain >=20mm vs less, abandonments")
print(f"[WX] flood-disruption city-days: {cdd.filter(pl.col('flood_disruption')).height} (too few to test)")
pd.DataFrame([{"comparison": k, "mh_rr": v[0], "ci_lo": v[1], "ci_hi": v[2], "strata": v[3]} for k, v in wx.items()]).to_csv(
    TAB / "q2_weather_mh_risk_ratios.csv", index=False)
# Where exactly is the threshold? (1C bins) And what rule sets the official heat alert?
t1 = (cdd.with_columns(pl.col("max_temp_c").floor().alias("t")).group_by("t")
      .agg(pl.len().alias("city_days"), (pl.col("fail").sum() / pl.col("att").sum()).alias("fr")).sort("t"))
print("[WX] failure rate by city-day max temperature (1C bins, >=33C): " + ", ".join(
    f"{int(r['t'])}C {pct(r['fr'])} ({r['city_days']}d)" for r in t1.filter(pl.col("t") >= 33).iter_rows(named=True)))
ha_rule = (cdd.with_columns(pl.col("max_temp_c").cut([40, 42], labels=["<=40C", "40-42C", ">42C"]).alias("band"))
           .group_by("band").agg(pl.len().alias("city_days"), pl.col("heat_alert").sum().alias("alert_days")).sort("band"))
print("[WX] heat_alert vs daily max: " + ", ".join(f"{r['band']}: {r['alert_days']} of {r['city_days']} city-days"
                                                    for r in ha_rule.iter_rows(named=True))
      + " -> the official alert is a fixed rule at about 42C, ~6C above where failures start to climb")

# %%
cool_ref = cgs.filter(pl.col("season_grp") == "Jul-Mar")
cool_ref_rate = cool_ref["failures"].sum() / cool_ref["attempts"].sum()
fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
order = cs["city"].to_list()
gens = ["Gen1", "Gen2", "Gen3"]
w = 0.26
hot_cg = cgs.filter(pl.col("season_grp") == "Apr-Jun")
for i, g in enumerate(gens):
    sub = hot_cg.filter(pl.col("charger_generation") == g)
    vals, los, his = [], [], []
    for c in order:
        r = sub.filter(pl.col("city") == c)
        vals.append(r["failure_rate"][0] if r.height else np.nan)
        los.append(r["failure_rate"][0] - r["ci_lo"][0] if r.height else 0)
        his.append(r["ci_hi"][0] - r["failure_rate"][0] if r.height else 0)
    x = np.arange(len(order)) + (i - 1) * w
    ax.bar(x, vals, w, color=GEN_COLORS[g], label=f"{g} (Apr-Jun)", yerr=[los, his], capsize=2, error_kw=dict(lw=0.8))
    for xi, v in zip(x, vals):
        if np.isfinite(v) and v > 0.07:
            ax.text(xi, v + 0.004, f"{100 * v:.0f}%", ha="center", fontsize=9.5)
ax.axhline(cool_ref_rate, color=PALETTE["primary"], ls="--", lw=1.5, label=f"Jul-Mar, every city & generation: {pct(cool_ref_rate)}")
ax.set_xticks(range(len(order)))
ax.set_xticklabels([f"{c}\nRR {GEN_RR_CITY[c]:.2f}" for c in order], fontsize=10)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax.set_ylabel("Service-failure rate, Apr-Jun")
ax.set_title("Summer failure rate by city and charger generation (95% CI)", fontsize=11.5)
ax.set_ylim(0, 0.20)
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.04))
ax.legend(loc="upper right")
ax = axes[1]
pdx = cdd.select("city", "max_temp_c", "fr", "att").to_pandas()
for c in CITIES:
    s = pdx[pdx["city"] == c]
    ax.scatter(s["max_temp_c"], s["fr"], s=7, alpha=0.35, color=CITY_COLORS[c], label=c)
tbp = tb.filter(pl.col("city_days") >= 5).to_pandas()
ax.plot(tbp["tmax_bin"] + 1, tbp["failure_rate"], color="#222222", lw=2.2, label="Mean (2C bins)")
ax.axvline(36, color="#555555", ls=":", lw=1.2)
ax.text(35.4, 0.282, f"36C: playbook trigger\n(>=36C days fail x{wx['tmax_ge36'][0]:.2f},\nsame city-month)", fontsize=9.5,
        ha="right", va="center", bbox=dict(boxstyle="round", fc="white", ec="#999999"))
ax.axvline(42, color=PALETTE["accent"], ls="--", lw=1.0)
ax.text(42.4, 0.285, "official heat alert\n(fixed rule, >42C)", fontsize=9.5, va="center", color=PALETTE["accent"])
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax.set_ylim(0, 0.31)
ax.set_xlabel("City daily maximum temperature (C)")
ax.set_ylabel("Service-failure rate (city-day)")
ax.set_title("Each dot = one city-day (n=3,282)", fontsize=11.5)
ax.legend(ncol=2, fontsize=9, loc="center left", markerscale=2)
headline(fig, "The summer spike is concentrated in older (Gen1/Gen2) cabinets in Delhi NCR, Jaipur and Hyderabad",
         "Left: Apr-Jun failure rate by city x charger generation (Gen1/Gen2 bars pool Apr-Jun 2024+2025; Gen3 bars are Apr-Jun 2025 "
         "only - no Gen3 site existed in 2024).\nRR under each city = Gen1 vs Gen3 MH risk ratio, Apr-Jun 2025, same city-month. "
         "Right: city-day failure rate vs daily max temperature - a threshold, not a slope.", top=0.84)
save_fig(fig, "q2_03_city_generation_heat")

# %% [markdown]
# ## 5. Is service quality even? Concentration across stations and station-hours
# Failures scale with volume, so we compare the failure concentration curve with the attempts curve and also look
# at **excess failures** = observed minus what each station's attempts would produce at the Jul-Mar baseline rate of
# its vehicle class. At station-hour grain, sparse counts alone create extreme concentration, so we compare with
# simulated chance-only nulls.

# %%
SMC = (ev.group_by("station_id", "month", "vehicle_class").agg(
    pl.len().alias("att"), FAIL.sum().alias("fail"), pl.col("is_stockout").sum().alias("so"),
    pl.col("is_abandoned").sum().alias("ab"), pl.col("is_system_error").sum().alias("se"),
    pl.col("is_completed").sum().alias("comp")).collect())
base_cls = (SMC.filter(~pl.col("month").dt.month().is_in(HOT_MONTHS)).group_by("vehicle_class")
            .agg((pl.col("fail").sum() / pl.col("att").sum()).alias("b")))
BASE = dict(zip(base_cls["vehicle_class"], base_cls["b"]))
print(f"[CONC] Jul-Mar baseline failure rate: 2W {pct(BASE['2W'], 2)}, 3W {pct(BASE['3W'], 2)} (3W/2W = {BASE['3W'] / BASE['2W']:.2f}x)")
SMC = SMC.join(base_cls, on="vehicle_class").with_columns((pl.col("fail") - pl.col("att") * pl.col("b")).alias("excess"))
attrs = st_all.select("station_id", "city", "zone", "expansion_wave", "charger_generation", "location_type", "host_type",
                      "connectivity_tier", "slots_2w", "slots_3w", "commissioned_date")
smo = station_month().select("station_id", "month", "active_days", "completed", "slots_2w", "slots_3w")
util_st = smo.group_by("station_id").agg(
    (pl.col("completed").sum() / (pl.col("active_days") * (pl.col("slots_2w") + pl.col("slots_3w"))).sum()).alias("swaps_per_slot_day"),
    pl.col("active_days").sum().alias("active_days"))
hotm = pl.col("month").dt.month().is_in(HOT_MONTHS)
yr = pl.col("month").dt.year()
SC = (SMC.group_by("station_id").agg(
    pl.col("att").sum(), pl.col("fail").sum(), pl.col("so").sum(), pl.col("ab").sum(), pl.col("se").sum(),
    pl.col("excess").sum(),
    pl.col("fail").filter(hotm & (yr == 2024)).sum().alias("fail_AJ24"), pl.col("att").filter(hotm & (yr == 2024)).sum().alias("att_AJ24"),
    pl.col("fail").filter(hotm & (yr == 2025)).sum().alias("fail_AJ25"), pl.col("att").filter(hotm & (yr == 2025)).sum().alias("att_AJ25"),
    pl.col("fail").filter(~hotm).sum().alias("fail_JM"), pl.col("att").filter(~hotm).sum().alias("att_JM"),
    pl.col("att").filter(pl.col("vehicle_class") == "3W").sum().alias("att_3w"),
    pl.col("fail").filter(pl.col("vehicle_class") == "3W").sum().alias("fail_3w"),
    pl.col("so").filter(pl.col("vehicle_class") == "3W").sum().alias("so_3w"))
    .join(attrs, on="station_id", how="left").join(util_st, on="station_id", how="left")
    .with_columns((pl.col("fail") / pl.col("att")).alias("failure_rate"), (pl.col("so") / pl.col("att")).alias("stockout_rate"),
                  (pl.col("ab") / pl.col("att")).alias("abandon_rate"),
                  (pl.col("fail_AJ24") / pl.col("att_AJ24")).alias("fr_AprJun24"), (pl.col("fail_AJ25") / pl.col("att_AJ25")).alias("fr_AprJun25"),
                  (pl.col("fail_JM") / pl.col("att_JM")).alias("fr_JulMar"), (pl.col("att") / pl.col("active_days")).alias("attempts_per_day"))
    .sort("excess", descending=True))
n_st = SC.height
k10 = int(np.ceil(0.10 * n_st))
G_att, G_fail, G_exc = gini(SC["att"]), gini(SC["fail"]), gini(SC["excess"].clip(lower_bound=0))
ts_att = top_share(SC["att"], 0.10)
ts_fail = top_share(SC["fail"], 0.10)
ts_exc = top_share(SC["excess"].clip(lower_bound=0), 0.10)
G_rate = gini(SC["failure_rate"])
print(f"[CONC] {n_st} stations. Gini: attempts {G_att:.3f}, failures {G_fail:.3f}, positive excess failures {G_exc:.3f}; "
      f"Gini of station failure RATE {G_rate:.3f}")
print(f"[CONC] top 10% of stations ({k10}): {pct(ts_att)} of attempts, {pct(ts_fail)} of failures, {pct(ts_exc)} of excess failures")
tot_exc = SMC["excess"].sum()
tot_pos = SC["excess"].clip(lower_bound=0).sum()
print(f"[CONC] network excess failures over baseline = {tot_exc:,.0f} of {N_FAIL:,} failures ({pct(tot_exc / N_FAIL)}); "
      f"stations with positive excess: {(SC['excess'] > 0).sum()} of {n_st}")
print(f"[CONC] station failure-rate range {pct(SC['failure_rate'].min(), 2)}-{pct(SC['failure_rate'].max(), 2)}; Jul-Mar range "
      f"{pct(SC['fr_JulMar'].min(), 2)}-{pct(SC['fr_JulMar'].max(), 2)}")
# Is the Jul-Mar baseline really the same at every station? Raw rates mix 2W and 3W riders (3W fail ~2.6x as often),
# so test station heterogeneity raw and after standardising for each station's vehicle-class mix.
jm_st = (SMC.filter(~hotm).group_by("station_id", "vehicle_class").agg(pl.col("att").sum(), pl.col("fail").sum())
         .join(base_cls, on="vehicle_class")
         .with_columns((pl.col("att") * pl.col("b")).alias("e"), (pl.col("att") * pl.col("b") * (1 - pl.col("b"))).alias("v"))
         .group_by("station_id").agg(pl.col("att").sum(), pl.col("fail").sum(), pl.col("e").sum(), pl.col("v").sum()))
p_jm = jm_st["fail"].sum() / jm_st["att"].sum()
chi_raw = float((((jm_st["fail"] - jm_st["att"] * p_jm) ** 2) / (jm_st["att"] * p_jm * (1 - p_jm))).sum())
chi_cls = float((((jm_st["fail"] - jm_st["e"]) ** 2) / jm_st["v"]).sum())
df_jm = jm_st.height - 1
oe_jm = jm_st["fail"] / jm_st["e"]
print(f"[BASE] Jul-Mar, {jm_st.height} stations: raw station rates {pct((jm_st['fail'] / jm_st['att']).min(), 2)}-"
      f"{pct((jm_st['fail'] / jm_st['att']).max(), 2)} (heterogeneity chi2 {chi_raw:.0f} on {df_jm} df, p={stats.chi2.sf(chi_raw, df_jm):.1g}); "
      f"after vehicle-class mix: observed/expected {oe_jm.min():.2f}-{oe_jm.max():.2f}, chi2 {chi_cls:.0f} on {df_jm} df "
      f"(p={stats.chi2.sf(chi_cls, df_jm):.3f})")
top15_ids = SC.sort("excess", descending=True).head(15)["station_id"].to_list()

# segment view of excess failures
SMCs = SMC.join(attrs.select("station_id", "city", "charger_generation"), on="station_id").with_columns(
    pl.when(~hotm).then(pl.lit("Jul-Mar, all stations"))
    .when(pl.col("city").is_in(HOT_CITIES) & (pl.col("charger_generation") == "Gen1")).then(pl.lit("Apr-Jun, hot-city Gen1"))
    .when(pl.col("city").is_in(HOT_CITIES) & (pl.col("charger_generation") == "Gen2")).then(pl.lit("Apr-Jun, hot-city Gen2"))
    .when(pl.col("city").is_in(HOT_CITIES)).then(pl.lit("Apr-Jun, hot-city Gen3"))
    .otherwise(pl.lit("Apr-Jun, other cities")).alias("segment"))
seg = (SMCs.group_by("segment").agg(pl.col("att").sum(), pl.col("fail").sum(), pl.col("excess").sum(),
                                     pl.col("station_id").n_unique().alias("stations"))
       .with_columns((pl.col("att") / pl.col("att").sum()).alias("share_attempts"), (pl.col("fail") / pl.col("fail").sum()).alias("share_failures"),
                     (pl.col("excess") / pl.col("excess").sum()).alias("share_excess"), (pl.col("fail") / pl.col("att")).alias("failure_rate"))
       .sort("share_excess", descending=True))
# Fair denominator: Jul-Mar has zero excess by construction, so compare excess with the share of Apr-Jun attempts
aj_tot = seg.filter(pl.col("segment").str.starts_with("Apr-Jun"))["att"].sum()
seg = seg.with_columns(pl.when(pl.col("segment").str.starts_with("Apr-Jun")).then(pl.col("att") / aj_tot)
                       .alias("share_aprjun_attempts"))
seg.write_csv(TAB / "q2_excess_failure_segments.csv")
ptable("Excess failures by segment", seg)
g1 = seg.filter(pl.col("segment") == "Apr-Jun, hot-city Gen1").row(0, named=True)
print(f"[CONC] Apr-Jun at hot-city Gen1 stations: {g1['stations']} stations, {pct(g1['share_aprjun_attempts'])} of Apr-Jun attempts "
      f"({pct(g1['share_attempts'])} of all 18-month attempts), {pct(g1['share_failures'])} of all failures, "
      f"{pct(g1['share_excess'])} of excess failures (failure rate {pct(g1['failure_rate'])})")
hot_all = seg.filter(pl.col("segment").str.starts_with("Apr-Jun, hot"))
print(f"[CONC] Apr-Jun at all hot-city stations: {pct(hot_all['share_aprjun_attempts'].sum())} of Apr-Jun attempts "
      f"({pct(hot_all['share_attempts'].sum())} of all attempts), {pct(hot_all['share_excess'].sum())} of excess failures")
st36 = SC.filter(pl.col("city").is_in(HOT_CITIES), pl.col("charger_generation") == "Gen1")
print(f"[CONC] the same {st36.height} hot-city Gen1 stations over all 18 months: {pct(st36['att'].sum() / N_ATT)} of attempts, "
      f"{pct(st36['fail'].sum() / N_FAIL)} of failures (the network's busiest third, not a small corner)")
for y_ in (2024, 2025):
    sy = SMCs.filter(hotm, yr == y_)
    g1y = sy.filter(pl.col("segment") == "Apr-Jun, hot-city Gen1")
    print(f"[CONC] Apr-Jun {y_}: excess {sy['excess'].sum():,.0f}; hot-city Gen1 = {pct(g1y['att'].sum() / sy['att'].sum())} of that "
          f"summer's attempts and {pct(g1y['excess'].sum() / sy['excess'].sum())} of its excess")
# robustness: each city's own Jul-Mar rate (per vehicle class) as the baseline instead of the network rate
base_cc = SMCs.filter(~hotm).group_by("city", "vehicle_class").agg((pl.col("fail").sum() / pl.col("att").sum()).alias("bc"))
smc2 = SMCs.join(base_cc, on=["city", "vehicle_class"]).with_columns((pl.col("fail") - pl.col("att") * pl.col("bc")).alias("exc2"))
print(f"[CONC] robustness, city x class baseline: total excess {smc2['exc2'].sum():,.0f}; Apr-Jun hot-city Gen1 share "
      f"{pct(smc2.filter(pl.col('segment') == 'Apr-Jun, hot-city Gen1')['exc2'].sum() / smc2['exc2'].sum())}")

# Are the stations with the most excess special, or just the biggest hot-city Gen1 sites? Compare each of the 36
# hot-city Gen1 stations with its own city x month x vehicle-class rate in Apr-Jun (observed / expected).
hg1 = SMCs.filter(pl.col("segment") == "Apr-Jun, hot-city Gen1")
ref_cm = hg1.group_by("city", "month", "vehicle_class").agg((pl.col("fail").sum() / pl.col("att").sum()).alias("r"))
OE36 = (hg1.join(ref_cm, on=["city", "month", "vehicle_class"])
        .with_columns((pl.col("att") * pl.col("r")).alias("e"), (pl.col("att") * pl.col("r") * (1 - pl.col("r"))).alias("v"))
        .group_by("station_id", "city").agg(pl.col("att").sum().alias("att_AJ"), pl.col("fail").sum().alias("fail_AJ"),
                                            pl.col("e").sum().alias("expected_AJ"), pl.col("v").sum().alias("var_AJ"),
                                            pl.col("excess").sum().alias("excess_AJ"))
        .with_columns((pl.col("fail_AJ") / pl.col("expected_AJ")).alias("oe_vs_city_month_class"),
                      (pl.col("fail_AJ") / pl.col("att_AJ")).alias("failure_rate_AJ"),
                      pl.col("station_id").is_in(top15_ids).alias("top15_by_excess"))
        .sort("excess_AJ", descending=True))
chi_36 = float((((OE36["fail_AJ"] - OE36["expected_AJ"]) ** 2) / OE36["var_AJ"]).sum())
df_36 = OE36.height - len(HOT_CITIES)
OE36.write_csv(TAB / "q2_hot_city_gen1_station_oe.csv")
t15o, r21o = OE36.filter("top15_by_excess"), OE36.filter(~pl.col("top15_by_excess"))
print(f"[OE36] {OE36.height} hot-city Gen1 stations in Apr-Jun vs their own city x month x class rate: O/E "
      f"{OE36['oe_vs_city_month_class'].min():.2f}-{OE36['oe_vs_city_month_class'].max():.2f}; extra between-station variation chi2 "
      f"{chi_36:.1f} on {df_36} df (p={stats.chi2.sf(chi_36, df_36):.2f})")
print(f"[OE36] top 15 by excess: median O/E {t15o['oe_vs_city_month_class'].median():.2f} vs {r21o['oe_vs_city_month_class'].median():.2f} "
      f"for the other {r21o.height}; median Apr-Jun attempts {t15o['att_AJ'].median():,.0f} vs {r21o['att_AJ'].median():,.0f}; by city "
      + ", ".join(f"{c} {t15o.filter(pl.col('city') == c).height} of {OE36.filter(pl.col('city') == c).height}" for c in HOT_CITIES))
s36 = {c: st36[c].drop_nans().drop_nulls() for c in ["fr_AprJun24", "fr_AprJun25", "fr_JulMar"]}
print(f"[OE36] all {st36.height} hot-city Gen1 stations fail in both summers: Apr-Jun 2024 station rates {pct(s36['fr_AprJun24'].min())}-"
      f"{pct(s36['fr_AprJun24'].max())} ({s36['fr_AprJun24'].len()} stations), Apr-Jun 2025 {pct(s36['fr_AprJun25'].min())}-"
      f"{pct(s36['fr_AprJun25'].max())} ({s36['fr_AprJun25'].len()}), vs Jul-Mar {pct(s36['fr_JulMar'].min())}-{pct(s36['fr_JulMar'].max())}")

# station-hour concentration vs chance-only nulls
rng = np.random.default_rng(59500)
sh_att = SH["att"].to_numpy()
sh_fail = SH["fail"].to_numpy()
sm_rate = (SMC.group_by("station_id", "month").agg((pl.col("fail").sum() / pl.col("att").sum()).alias("p")))
p_sm = (SH.select("station_id", pl.col("hour_start").dt.truncate("1mo").dt.date().alias("month"))
        .join(sm_rate, on=["station_id", "month"], how="left")["p"].to_numpy())
null_flat = rng.binomial(sh_att, FR_ALL)
null_sm = rng.binomial(sh_att, p_sm)
G_sh = {"observed": gini(sh_fail), "null_flat_rate": gini(null_flat), "null_station_month_rate": gini(null_sm)}
T_sh = {"observed": top_share(sh_fail, 0.10), "null_flat_rate": top_share(null_flat, 0.10),
        "null_station_month_rate": top_share(null_sm, 0.10)}
zero_sh = {"observed": (sh_fail == 0).mean(), "null_flat_rate": (null_flat == 0).mean(), "null_station_month_rate": (null_sm == 0).mean()}
conc_tbl = pd.DataFrame({"gini": G_sh, "top10pct_share": T_sh, "share_station_hours_zero_failures": zero_sh})
print("\n--- Station-hour concentration of failures vs chance-only nulls ---")
print(conc_tbl.round(4))
print(f"[CONC] station-hours (>=1 attempt): Gini observed {G_sh['observed']:.3f} vs {G_sh['null_flat_rate']:.3f} if every attempt "
      f"failed at the network rate, vs {G_sh['null_station_month_rate']:.3f} with each station-month's own rate; top 10% of "
      f"station-hours hold {pct(T_sh['observed'])} of failures (null {pct(T_sh['null_flat_rate'])} / {pct(T_sh['null_station_month_rate'])})")
summary_rows = [
    {"level": "station", "metric": "gini_attempts", "value": G_att}, {"level": "station", "metric": "gini_failures", "value": G_fail},
    {"level": "station", "metric": "gini_positive_excess", "value": G_exc}, {"level": "station", "metric": "gini_failure_rate", "value": G_rate},
    {"level": "station", "metric": "top10pct_share_attempts", "value": ts_att}, {"level": "station", "metric": "top10pct_share_failures", "value": ts_fail},
    {"level": "station", "metric": "top10pct_share_excess", "value": ts_exc}]
summary_rows += [{"level": "station_hour", "metric": f"gini_{k}", "value": v} for k, v in G_sh.items()]
summary_rows += [{"level": "station_hour", "metric": f"top10pct_share_{k}", "value": v} for k, v in T_sh.items()]
pd.DataFrame(summary_rows).to_csv(TAB / "q2_concentration_summary.csv", index=False)
del null_flat, null_sm, p_sm

# %%
# Tickets per station (used in the top-15 site table and section 9)
tk = (tickets().filter(pl.col("in_window") & ~pl.col("is_test") & pl.col("station_id").is_not_null()))
tk_st = tk.group_by("station_id").agg(
    pl.len().alias("tickets_all"), (pl.col("category") == "no_battery_available").sum().alias("tk_no_battery"),
    (pl.col("category") == "long_queue").sum().alias("tk_long_queue"))
SC = SC.join(tk_st, on="station_id", how="left").with_columns(
    [pl.col(c).fill_null(0) for c in ["tickets_all", "tk_no_battery", "tk_long_queue"]]).with_columns(
    ((pl.col("tk_no_battery") + pl.col("tk_long_queue")) / pl.col("att") * 1000).alias("nb_lq_tickets_per_1k"),
    (pl.col("tickets_all") / pl.col("att") * 1000).alias("tickets_per_1k"))
SC.write_csv(TAB / "q2_station_scorecard.csv")
top15_raw = SC.filter(pl.col("station_id").is_in(top15_ids))
TOP15_EXC_SHARE = top15_raw["excess"].sum() / tot_exc
TOP15_ATT_SHARE = top15_raw["att"].sum() / N_ATT
TOP15_FAIL_SHARE = top15_raw["fail"].sum() / N_FAIL
hot15 = SC.sort("excess", descending=True).head(15).join(
    OE36.select("station_id", "att_AJ", pl.col("oe_vs_city_month_class").round(3)), on="station_id", how="left").select(
    "station_id", "city", "zone", "expansion_wave", "charger_generation", "location_type", "host_type", "connectivity_tier",
    "slots_2w", "slots_3w", pl.col("swaps_per_slot_day").round(2), pl.col("attempts_per_day").round(1), "att", "att_AJ", "fail",
    pl.col("excess").round(0), "oe_vs_city_month_class", pl.col("failure_rate").round(4), pl.col("fr_AprJun24").round(4),
    pl.col("fr_AprJun25").round(4), pl.col("fr_JulMar").round(4), pl.col("stockout_rate").round(4), pl.col("abandon_rate").round(4),
    pl.col("nb_lq_tickets_per_1k").round(2))
hot15.write_csv(TAB / "q2_top15_hot_city_gen1_sites.csv")
ptable("15 largest hot-city Gen1 sites by absolute excess failures (not unusual: O/E vs own city-month rate shown)", hot15)
print(f"[TOP15] cities: {dict(hot15.group_by('city').len().iter_rows())}; generation: {dict(hot15.group_by('charger_generation').len().iter_rows())};"
      f" wave: {dict(hot15.group_by('expansion_wave').len().iter_rows())}; location: {dict(hot15.group_by('location_type').len().iter_rows())}")
print(f"[TOP15] the 15 stations with the most excess hold {pct(TOP15_EXC_SHARE)} of net excess failures, {pct(TOP15_FAIL_SHARE)} of all "
      f"failures and {pct(TOP15_ATT_SHARE)} of attempts; Apr-Jun failure rate 2024 median {pct(hot15['fr_AprJun24'].median())}, 2025 median "
      f"{pct(hot15['fr_AprJun25'].median())}; Jul-Mar median {pct(hot15['fr_JulMar'].median(), 2)}")
rest = SC.filter(~pl.col("station_id").is_in(hot15["station_id"].to_list()))
print(f"[TOP15] utilisation (swaps per slot-day): these 15 median {hot15['swaps_per_slot_day'].median():.2f} vs other stations "
      f"{rest['swaps_per_slot_day'].median():.2f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 6.3), gridspec_kw={"width_ratios": [1, 1.1]})
ax = axes[0]
for vals, lab, col, lw in [(SC["att"], f"Attempts (Gini {G_att:.2f})", PALETTE["muted"], 2),
                           (SC["fail"], f"All failures (Gini {G_fail:.2f})", PALETTE["primary"], 2.2),
                           (SC["excess"].clip(lower_bound=0), f"Excess failures over baseline (Gini {G_exc:.2f})", PALETTE["accent"], 2.6)]:
    xx, yy = conc_curve(vals)
    ax.plot(xx, yy, color=col, lw=lw, label=lab)
ax.plot([0, 1], [0, 1], color="#BBBBBB", ls=":", lw=1)
ax.axvline(0.10, color="#444444", ls="--", lw=1)
ax.annotate(f"The {len(top15_ids)} stations with the most excess\n(all hot-city Gen1) hold "
            f"{pct(TOP15_EXC_SHARE, 0)} of excess\nfailures on {pct(TOP15_ATT_SHARE, 0)} of attempts",
            xy=(0.10, ts_exc), xytext=(0.27, 0.27), fontsize=10.5, arrowprops=dict(arrowstyle="->", color="#222222"),
            bbox=dict(boxstyle="round", fc="white", ec="#999999"))
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Share of stations (worst first)")
ax.set_ylabel("Cumulative share")
ax.set_title(f"Concentration across {n_st} stations", fontsize=11.5)
ax.legend(loc="lower right", fontsize=9.5, title="Each curve ranks stations by its own measure", title_fontsize=9.5)
ax = axes[1]
segp = seg.filter(pl.col("segment").str.starts_with("Apr-Jun")).sort("share_excess").to_pandas()
seg_lbl = {"Apr-Jun, hot-city Gen1": "Hot-city Gen1", "Apr-Jun, hot-city Gen2": "Hot-city Gen2",
           "Apr-Jun, hot-city Gen3": "Hot-city Gen3", "Apr-Jun, other cities": "Other cities (all gens)"}
y = np.arange(len(segp))
ax.barh(y + 0.2, segp["share_aprjun_attempts"], 0.38, color=PALETTE["muted"], label="Share of Apr-Jun attempts")
ax.barh(y - 0.2, segp["share_excess"], 0.38, color=PALETTE["accent"], label="Share of excess failures")
for yi, (a, e, fr, ns) in enumerate(zip(segp["share_aprjun_attempts"], segp["share_excess"], segp["failure_rate"], segp["stations"])):
    ax.text(a + 0.01, yi + 0.2, f"{pct(a, 0)} of attempts", va="center", fontsize=10, color="#555555")
    inside = e > 0.6  # long bar: write the label inside it so it stays within the panel
    ax.text(e - 0.01 if inside else max(e, 0) + 0.01, yi - 0.2, f"{pct(e, 0)} of excess  (failure rate {pct(fr, 1)})", va="center",
            ha="right" if inside else "left", fontsize=10, color="white" if inside else PALETTE["accent"],
            fontweight="bold" if inside else "normal")
ax.set_yticks(y)
ax.set_yticklabels([f"{seg_lbl[s]}\n({n} stations)" for s, n in zip(segp["segment"], segp["stations"])])
ax.axvline(0, color="#888888", lw=0.8)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax.set_xlim(0, 1.0)
ax.set_title("Apr-Jun only: share of attempts vs share of excess failures", fontsize=11.5)
ax.legend(loc="lower right")
headline(fig, f"In Apr-Jun, the {g1['stations']} hot-city Gen1 stations handle {pct(g1['share_aprjun_attempts'], 0)} of attempts but "
         f"produce {pct(g1['share_excess'], 0)} of excess failures",
         f"Excess = observed - attempts x Jul-Mar baseline of the vehicle class (2W {pct(BASE['2W'], 1)}, 3W {pct(BASE['3W'], 1)}): "
         f"{tot_exc:,.0f} of {N_FAIL:,} failures. Jul-Mar ({pct(1 - aj_tot / N_ATT, 0)} of attempts) has zero excess by construction.\n"
         f"Over all 18 months the same {st36.height} stations carry {pct(st36['att'].sum() / N_ATT, 0)} of attempts and "
         f"{pct(st36['fail'].sum() / N_FAIL, 0)} of failures. Hot cities = Delhi NCR, Jaipur, Hyderabad.", top=0.84)
save_fig(fig, "q2_04_concentration_lorenz")

# %% [markdown]
# ## 6. Mechanism from hourly telemetry: heat -> hot cabinet -> slower Gen1 charging -> empty cabinets on hot days
# Telemetry rows with `telemetry_status == 'missing'` have every sensor field null; `partial` rows carry values.
# We exclude missing rows (null-aware) and report coverage. Event station-hours are joined on station + hour_start.
# The telemetry links (heat -> cabinet -> charge time -> afternoon empty cabinets) are measured facts. Whether empty
# cabinets explain WHICH hours fail is tested separately in 6b, holding the day and the hour of day fixed.

# %%
HR_COLS = ["station_id", "hour_start", "hour", "month", "telemetry_status", "connectivity_tier", "charger_generation", "city",
           "charged_2w_avg", "charged_2w_min", "charged_3w_min", "packs_quarantined", "chargers_online", "ambient_temp_c",
           "cabinet_temp_c", "avg_charge_minutes", "outage_minutes", "slots_2w", "slots_3w", "inventory_target_2w"]
HR = hourly(HR_COLS).collect()
J = (HR.join(SH.drop("slots", "util", "demand_util"), on=["station_id", "hour_start"], how="left")
     .with_columns([pl.col(c).fill_null(0) for c in ["att", "comp", "fail", "so", "ab", "se", "att3", "so3", "ab3", "qws"]])
     .with_columns((pl.col("att") - pl.col("att3")).alias("att2"), (pl.col("so") - pl.col("so3")).alias("so2"),
                   (pl.col("ab") - pl.col("ab3")).alias("ab2")))
unmatched = SH.join(HR.select("station_id", "hour_start"), on=["station_id", "hour_start"], how="anti").height
cov = (J.group_by("connectivity_tier", "telemetry_status").agg(pl.len().alias("station_hours"), pl.col("att").sum().alias("attempts"))
       .with_columns((pl.col("station_hours") / pl.col("station_hours").sum().over("connectivity_tier")).alias("share_of_tier_hours"))
       .sort("connectivity_tier", "telemetry_status"))
cov.write_csv(TAB / "q2_telemetry_coverage.csv")
ptable("Telemetry coverage by connectivity tier", cov)
cov_s = J.group_by("telemetry_status").agg(pl.len().alias("h"), pl.col("att").sum().alias("a"))
cov_d = {r["telemetry_status"]: r for r in cov_s.iter_rows(named=True)}
print(f"[TEL] station-hours {J.height:,}: ok {pct(cov_d['ok']['h'] / J.height)}, partial {pct(cov_d['partial']['h'] / J.height)}, "
      f"missing {pct(cov_d['missing']['h'] / J.height, 2)}; attempts in hours with usable telemetry "
      f"{pct(1 - cov_d['missing']['a'] / N_ATT, 2)}; event station-hours without a telemetry row: {unmatched}")
JT = J.filter(pl.col("telemetry_status") != "missing")
amb = JT.select(pl.corr("ambient_temp_c", "cabinet_temp_c").alias("r"),
                (pl.col("cabinet_temp_c") - pl.col("ambient_temp_c")).mean().alias("d"))
print(f"[TEL] ambient vs cabinet temperature: r = {amb['r'][0]:.3f}; cabinet runs {amb['d'][0]:.1f}C above ambient on average")
print("[TEL] cabinet minus ambient by generation: " + ", ".join(
    f"{r['charger_generation']} {r['d']:.2f}C" for r in JT.group_by("charger_generation").agg(
        (pl.col("cabinet_temp_c") - pl.col("ambient_temp_c")).mean().alias("d")).sort("charger_generation").iter_rows(named=True)))

CB = 2.5
mech = (JT.with_columns(((pl.col("cabinet_temp_c") / CB).floor() * CB).alias("cab_bin"))
        .group_by("charger_generation", "cab_bin").agg(
            pl.len().alias("station_hours"), pl.col("att").sum().alias("attempts"),
            pl.col("ambient_temp_c").mean().alias("ambient_c"), pl.col("avg_charge_minutes").mean().alias("charge_min"),
            (pl.col("charged_2w_min") == 0).mean().alias("zero_2w_share"),
            (pl.col("charged_2w_avg") / pl.col("inventory_target_2w")).mean().alias("charged_vs_target"),
            (pl.col("so").sum() / pl.col("att").sum()).alias("stockout_rate"),
            (pl.col("ab").sum() / pl.col("att").sum()).alias("abandon_rate"),
            (pl.col("fail").sum() / pl.col("att").sum()).alias("failure_rate"))
        .sort("charger_generation", "cab_bin"))
mech.write_csv(TAB / "q2_mechanism_cabinet_temp.csv")
ptable("Mechanism by cabinet temperature (2.5C bins) x generation", mech.filter(pl.col("station_hours") >= 200))


def band(g, lo, hi):
    s = JT.filter(pl.col("charger_generation") == g, pl.col("cabinet_temp_c") >= lo, pl.col("cabinet_temp_c") < hi)
    return dict(h=s.height, chg=s["avg_charge_minutes"].mean(), zero=(s["charged_2w_min"] == 0).mean(),
                so=s["so"].sum() / max(s["att"].sum(), 1), fr=s["fail"].sum() / max(s["att"].sum(), 1), att=s["att"].sum())


MB = {}
for g in ["Gen1", "Gen2", "Gen3"]:
    MB[g] = {"<40": band(g, -50, 40), ">=45": band(g, 45, 99), ">=50": band(g, 50, 99)}
    a, b, c = MB[g]["<40"], MB[g][">=45"], MB[g][">=50"]
    print(f"[MECH] {g}: cabinet <40C -> charge {a['chg']:.0f} min, zero-2W-stock hours {pct(a['zero'], 2)}, stockout rate {pct(a['so'], 2)} "
          f"(n={a['h']:,} h) | >=45C -> {b['chg']:.0f} min, {pct(b['zero'])}, {pct(b['so'])} (n={b['h']:,} h) | "
          f">=50C -> {c['chg']:.0f} min, {pct(c['zero'])}, {pct(c['so'])} (n={c['h']:,} h)")
hot_share = (JT.group_by("charger_generation").agg((pl.col("cabinet_temp_c") >= 45).mean().alias("s"))
             .sort("charger_generation"))
print("[MECH] share of station-hours with cabinet >=45C: " + ", ".join(f"{r['charger_generation']} {pct(r['s'], 2)}" for r in hot_share.iter_rows(named=True)))
cab_city = (JT.filter(pl.col("month").dt.month().is_in(HOT_MONTHS)).group_by("city").agg(
    (pl.col("cabinet_temp_c") >= 45).mean().alias("share_ge45")).sort("share_ge45", descending=True))
print("[MECH] Apr-Jun share of station-hours with cabinet >=45C by city: " + ", ".join(
    f"{r['city']} {pct(r['share_ge45'])}" for r in cab_city.iter_rows(named=True)))

inv = (JT.with_columns(pl.col("charged_2w_min").clip(upper_bound=10).alias("min_bin"))
       .group_by("min_bin").agg(pl.len().alias("station_hours"), pl.col("att2").sum().alias("attempts_2w"),
                                (pl.col("so2").sum() / pl.col("att2").sum()).alias("stockout_rate_2w"),
                                (pl.col("ab2").sum() / pl.col("att2").sum()).alias("abandon_rate_2w"),
                                (pl.col("qws").sum() / pl.col("comp").sum() / 60).alias("mean_wait_min"))
       .sort("min_bin"))
inv.write_csv(TAB / "q2_inventory_vs_stockout.csv")
ptable("2W stockout / abandonment by minimum charged-2W packs in the hour (10 = 10+)", inv)
i0 = inv.filter(pl.col("min_bin") == 0).row(0, named=True)
i5 = inv.filter(pl.col("min_bin") >= 5)
r5 = i5["stockout_rate_2w"].dot(i5["attempts_2w"]) / i5["attempts_2w"].sum()
a5 = i5["abandon_rate_2w"].dot(i5["attempts_2w"]) / i5["attempts_2w"].sum()
print(f"[INV] POOLED (no strata; confounded by day and hour, see 6b): hours with zero charged 2W packs: 2W stockout "
      f"{pct(i0['stockout_rate_2w'], 2)}, abandonment {pct(i0['abandon_rate_2w'], 2)} vs {pct(r5, 2)} / {pct(a5, 2)} when >=5 packs; "
      f"completed wait {i0['mean_wait_min']:.2f} min (unchanged)")

# chargers offline, quarantined packs, outages
J2 = J.with_columns((pl.col("slots_2w") + pl.col("slots_3w") - pl.col("chargers_online")).alias("chargers_offline"))
oth_rows = []
for lab, cond in [("chargers_offline>=1", pl.col("chargers_offline") >= 1), ("all chargers online", pl.col("chargers_offline") <= 0),
                  ("packs_quarantined>=1", pl.col("packs_quarantined") >= 1), ("outage_minutes>0", pl.col("outage_minutes") > 0),
                  ("no outage", pl.col("outage_minutes") == 0)]:
    s = J2.filter(cond)
    p, lo, hi = wilson(s["fail"].sum(), s["att"].sum())
    oth_rows.append({"condition": lab, "station_hours": s.height, "attempts": int(s["att"].sum()), "failure_rate": float(p),
                     "ci_lo": float(lo), "ci_hi": float(hi), "system_error_rate": s["se"].sum() / s["att"].sum(),
                     "stockout_rate": s["so"].sum() / s["att"].sum()})
oth_df = pd.DataFrame(oth_rows)
oth_df.to_csv(TAB / "q2_other_telemetry_conditions.csv", index=False)
print("\n--- Other telemetry conditions ---")
print(oth_df.round(4).to_string(index=False))
del J2

# %%
# Station-day binomial GLM, separately per generation: failure ~ daily max cabinet temp band + city + month + weekday
sd_e = (ev.group_by("station_id", "date").agg(pl.len().alias("att"), FAIL.sum().alias("fail"),
                                              pl.col("is_stockout").sum().alias("so")).collect())
sd_t = (JT.with_columns(pl.col("hour_start").dt.date().alias("date")).group_by("station_id", "date")
        .agg(pl.col("cabinet_temp_c").max().alias("cab_max"), pl.len().alias("tele_hours")))
SD = (sd_e.join(sd_t, on=["station_id", "date"], how="inner").join(attrs.select("station_id", "city", "charger_generation"), on="station_id")
      .with_columns(pl.col("date").dt.strftime("%Y-%m").alias("ym"), pl.col("date").dt.weekday().cast(pl.Utf8).alias("wd"),
                    pl.col("cab_max").cut([40, 45, 50, 55], labels=["<40", "40-45", "45-50", "50-55", ">=55"], left_closed=True).alias("cab_band")))
print(f"[GLM] station-days with events and telemetry: {SD.height:,} of {sd_e.height:,} station-days with events "
      f"({pct(SD.height / sd_e.height, 2)} coverage)")
glm_rows = []
for g in ["Gen1", "Gen2", "Gen3"]:
    d = SD.filter(pl.col("charger_generation") == g).to_pandas()
    d["cab_band"] = pd.Categorical(d["cab_band"], categories=["<40", "40-45", "45-50", "50-55", ">=55"])
    X = pd.get_dummies(d[["cab_band", "city", "ym", "wd"]], drop_first=True, dtype=float)
    X = X.loc[:, X.sum() > 0]
    X = sm.add_constant(X)
    yv = np.column_stack([d["fail"], d["att"] - d["fail"]])
    res = sm.GLM(yv, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": d["station_id"].astype("category").cat.codes})
    for b in ["40-45", "45-50", "50-55", ">=55"]:
        col = f"cab_band_{b}"
        if col in res.params:
            ci = res.conf_int().loc[col]
            glm_rows.append({"generation": g, "cabinet_max_band": b, "odds_ratio_vs_lt40": np.exp(res.params[col]),
                             "ci_lo": np.exp(ci[0]), "ci_hi": np.exp(ci[1]), "p_value": res.pvalues[col],
                             "station_days_in_band": int((d["cab_band"] == b).sum()), "station_days_total": len(d),
                             "stations": d["station_id"].nunique()})
glm_df = pd.DataFrame(glm_rows)
glm_df.to_csv(TAB / "q2_glm_heat_by_generation.csv", index=False)
print("\n--- Station-day GLM: odds ratio of a failed attempt vs days with max cabinet <40C (controls: city, month, weekday; "
      "station-clustered 95% CI) ---")
print(glm_df.round(3).to_string(index=False))
# station-day curve used in the mechanism chart (day's peak cabinet temperature -> that day's stockout rate)
sdc = (SD.with_columns(((pl.col("cab_max") / CB).floor() * CB).alias("cab_max_bin"))
       .group_by("charger_generation", "cab_max_bin").agg(pl.len().alias("station_days"), pl.col("att").sum(),
                                                          (pl.col("so").sum() / pl.col("att").sum()).alias("stockout_rate"),
                                                          (pl.col("fail").sum() / pl.col("att").sum()).alias("failure_rate"))
       .sort("charger_generation", "cab_max_bin"))
sdc.write_csv(TAB / "q2_station_day_cabinet_max_vs_stockout.csv")
for g in ["Gen1", "Gen2", "Gen3"]:
    s = SD.filter(pl.col("charger_generation") == g)
    lo_, hi_ = s.filter(pl.col("cab_max") < 40), s.filter(pl.col("cab_max") >= 45)
    print(f"[MECH-DAY] {g}: days with peak cabinet <40C stockout rate {pct(lo_['so'].sum() / lo_['att'].sum(), 2)} "
          f"({lo_.height:,} station-days) vs >=45C {pct(hi_['so'].sum() / hi_['att'].sum(), 2)} ({hi_.height:,} station-days)")
# load parity behind the Gen3 counterfactual: do hot-city Gen3 sites serve a comparable load in the same summer?
lpar = (SD.filter(pl.col("city").is_in(HOT_CITIES), pl.col("date") >= pl.date(2025, 4, 1), pl.col("charger_generation").is_in(["Gen1", "Gen3"]))
        .group_by("charger_generation").agg(pl.col("att").mean().alias("apd"), pl.len().alias("station_days")).sort("charger_generation"))
print("[LOAD-GEN] hot cities, Apr-Jun 2025, attempts per active station-day: " + ", ".join(
    f"{r['charger_generation']} {r['apd']:.1f} ({r['station_days']:,} station-days)" for r in lpar.iter_rows(named=True)))

# %% [markdown]
# ### 6b. Trap check: does the hourly stock level explain WHICH hours fail? (Simpson's paradox test)
# Pooled, hours with zero charged 2W packs show more stockouts (section 6). But empty hours cluster on hot days and
# in the afternoon, which are risky for other reasons. We compare zero-stock with stocked hours inside strata that
# hold the station-day fixed (and the 19-22h window), or the hour of day fixed (city x generation x month x hour),
# using Mantel-Haenszel risk ratios. Lagged stock uses a time-based self-join on `hour_start`; a row shift would
# misalign hours around dropped 'missing' rows. Unit: station-hours with >=1 2W attempt and a non-null hourly
# minimum of charged 2W packs.

# %%
H2 = (JT.filter(pl.col("att2") > 0, pl.col("charged_2w_min").is_not_null())
      .select("station_id", "hour_start", "city", "charger_generation", "att2", "so2", "ab2", "charged_2w_min")
      .with_columns(pl.col("hour_start").dt.date().alias("date"), pl.col("hour_start").dt.hour().alias("h"),
                    pl.col("hour_start").dt.truncate("1mo").alias("mo"), (pl.col("charged_2w_min") == 0).alias("zero"))
      .with_columns(pl.col("h").is_in(EVE).alias("eve")))


def mh_flag(df: pl.DataFrame, keys, flag, event):
    """MH risk ratio of a 2W outcome per 2W attempt, flag true vs false, within strata `keys`."""
    g = df.group_by(keys + [flag]).agg(pl.col("att2").sum().alias("n"), pl.col(event).sum().alias("e")).to_pandas()
    w = g.pivot_table(index=keys, columns=flag, values=["n", "e"], aggfunc="sum", fill_value=0)
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    return mh_rr(w.reset_index(), "e_True", "n_True", "e_False", "n_False")


OUT_LBL = {"so2": "2W stockout", "ab2": "2W abandonment"}
zc = H2.group_by("zero").agg(pl.col("att2").sum(), pl.col("so2").sum(), pl.col("ab2").sum())
zt, zf = zc.filter(pl.col("zero")).row(0, named=True), zc.filter(~pl.col("zero")).row(0, named=True)
hour_rows = [{"comparison": "pooled (no strata), crude RR", "outcome": OUT_LBL[o], "rr": (zt[o] / zt["att2"]) / (zf[o] / zf["att2"]),
              "ci_lo": np.nan, "ci_hi": np.nan, "strata": 1, "rate_flag": zt[o] / zt["att2"], "rate_other": zf[o] / zf["att2"]}
             for o in ["so2", "ab2"]]
for lab, keys in [("same station-day", ["station_id", "date"]),
                  ("same station-day and same window (19-22h vs other)", ["station_id", "date", "eve"]),
                  ("same city x generation x month x hour of day", ["city", "charger_generation", "mo", "h"])]:
    for o in ["so2", "ab2"]:
        r = mh_flag(H2, keys, "zero", o)
        hour_rows.append({"comparison": lab, "outcome": OUT_LBL[o], "rr": r[0], "ci_lo": r[1], "ci_hi": r[2], "strata": r[3]})
TL = JT.filter(pl.col("charged_2w_min").is_not_null()).select("station_id", "hour_start", "charged_2w_min")
for L in [0, 1, 2, 3, 4, 6]:
    lg = TL.select("station_id", (pl.col("hour_start") + pl.duration(hours=L)).alias("hour_start"),
                   (pl.col("charged_2w_min") == 0).alias("zero_lag"))
    zj = H2.join(lg, on=["station_id", "hour_start"], how="inner")
    r0, r1 = zj.filter(pl.col("zero_lag")), zj.filter(~pl.col("zero_lag"))
    r = mh_flag(zj, ["city", "charger_generation", "mo", "h"], "zero_lag", "so2")
    hour_rows.append({"comparison": f"zero stock {L}h earlier (time join); same city x generation x month x hour", "outcome": OUT_LBL["so2"],
                      "rr": r[0], "ci_lo": r[1], "ci_hi": r[2], "strata": r[3], "lag_hours": L,
                      "rate_flag": r0["so2"].sum() / r0["att2"].sum(), "rate_other": r1["so2"].sum() / r1["att2"].sum()})
del TL, lg, zj, r0, r1
hr_df = pd.DataFrame(hour_rows)
hr_df.to_csv(TAB / "q2_hourly_stock_vs_failure_mh.csv", index=False)
print("\n--- Zero-stock hour vs stocked hour: 2W outcomes (rate_flag / rate_other = pooled rates) ---")
print(hr_df.round(4).to_string(index=False))


def hrr(comp, outcome="2W stockout"):
    return hr_df[(hr_df["comparison"] == comp) & (hr_df["outcome"] == outcome)].iloc[0]


HR_SD, HR_SDE, HR_CGMH = (hrr("same station-day"), hrr("same station-day and same window (19-22h vs other)"),
                          hrr("same city x generation x month x hour of day"))
for o in ["2W stockout", "2W abandonment"]:
    a_, b_, c_, d_ = (hrr("pooled (no strata), crude RR", o), hrr("same station-day", o),
                      hrr("same station-day and same window (19-22h vs other)", o), hrr("same city x generation x month x hour of day", o))
    print(f"[HOURLY] {o}, zero-stock vs stocked hour: pooled crude RR {a_['rr']:.2f} ({pct(a_['rate_flag'], 2)} vs {pct(a_['rate_other'], 2)}); "
          f"same station-day MH RR {b_['rr']:.2f} ({b_['ci_lo']:.2f}-{b_['ci_hi']:.2f}); + same window {c_['rr']:.2f} ({c_['ci_lo']:.2f}-"
          f"{c_['ci_hi']:.2f}); same city x gen x month x hour {d_['rr']:.2f} ({d_['ci_lo']:.2f}-{d_['ci_hi']:.2f})")
lags = hr_df[hr_df["lag_hours"].notna()]
LAG_RR_RANGE = (lags.loc[lags["lag_hours"] >= 1, "rr"].min(), lags.loc[lags["lag_hours"] >= 1, "rr"].max())
print("[LAG] 2W stockout when charged_2w_min was 0 L hours earlier (time-based self-join): pooled " + ", ".join(
    f"L={int(r.lag_hours)} {pct(r.rate_flag, 2)} vs {pct(r.rate_other, 2)}" for r in lags.itertuples())
      + f" | MH RR within city x gen x month x hour: " + ", ".join(f"L={int(r.lag_hours)} {r.rr:.2f}" for r in lags.itertuples())
      + f" -> 1-6 h lags {LAG_RR_RANGE[0]:.2f}-{LAG_RR_RANGE[1]:.2f}: no carry-over once hour of day is held fixed")

# Hot Gen1 days: the hour profile of stockouts vs the hour profile of empty cabinets
hd = (H2.filter(pl.col("city").is_in(HOT_CITIES), pl.col("charger_generation") == "Gen1", pl.col("hour_start").dt.month().is_in(HOT_MONTHS))
      .join(sd_t.select("station_id", "date", "cab_max"), on=["station_id", "date"], how="left")
      .filter(pl.col("cab_max").is_not_null()).with_columns((pl.col("cab_max") >= 45).alias("hot_day")))
prof = (hd.group_by("hot_day", "h").agg(pl.len().alias("station_hours"), pl.col("att2").sum(), pl.col("so2").sum(),
                                        pl.col("zero").mean().alias("zero_stock_share"),
                                        pl.col("charged_2w_min").mean().alias("mean_min_charged_2w"))
        .with_columns((pl.col("so2") / pl.col("att2")).alias("stockout_rate_2w")).sort("hot_day", "h"))
prof.write_csv(TAB / "q2_hot_gen1_day_hour_profile.csv")
hot_h = hd.filter(pl.col("hot_day"))


def win(df, cond):
    s = df.filter(cond)
    return s["so2"].sum() / s["att2"].sum(), s["zero"].mean(), int(s["att2"].sum())


W018, W1217, W1922 = win(hot_h, pl.col("h") <= 18), win(hot_h, pl.col("h").is_between(12, 17)), win(hot_h, pl.col("h").is_in(EVE))
W1922_OTH = win(hd.filter(~pl.col("hot_day")), pl.col("h").is_in(EVE))
ph018 = prof.filter(pl.col("hot_day"), pl.col("h") <= 18)
n_hot_days = hot_h.select(pl.struct("station_id", "date").n_unique()).item()
print(f"[HOTDAY] hot-city Gen1, Apr-Jun, station-days with peak cabinet >=45C ({n_hot_days:,} station-days): 2W stockout 0-18h "
      f"{pct(W018[0])} (hourly range {pct(ph018['stockout_rate_2w'].min())}-{pct(ph018['stockout_rate_2w'].max())}; hourly empty share "
      f"{pct(ph018['zero_stock_share'].min())}-{pct(ph018['zero_stock_share'].max())}); 12-17h {pct(W1217[0])} with {pct(W1217[1])} of hours "
      f"empty; 19-22h {pct(W1922[0])} with only {pct(W1922[1])} of hours empty (other Apr-Jun days at 19-22h: {pct(W1922_OTH[0])})")
eg = (hot_h.filter(pl.col("h").is_in(EVE))
      .with_columns(pl.col("charged_2w_min").cut([0.5, 2.5, 5.5], labels=["0", "1-2", "3-5", "6+"]).alias("stock"))
      .group_by("stock").agg(pl.col("att2").sum(), (pl.col("so2").sum() / pl.col("att2").sum()).alias("so")).sort("stock"))
print("[HOTDAY] same hot days, 19-22h, 2W stockout by hourly minimum of charged 2W packs: " + ", ".join(
    f"{r['stock']} packs {pct(r['so'])} (n={r['att2']:,})" for r in eg.iter_rows(named=True)))
nt = (hd.filter(pl.col("h") <= 8).group_by("hot_day")
      .agg(pl.col("att2").sum(), (pl.col("so2").sum() / pl.col("att2").sum()).alias("so"), pl.col("charged_2w_min").mean().alias("ms"))
      .sort("hot_day", descending=True))
NT = {r["hot_day"]: r for r in nt.iter_rows(named=True)}
print("[HOTDAY] same stations, 0-8h when cabinets are full: " + ", ".join(
    f"{'hot days' if k else 'other days'} 2W stockout {pct(r['so'])} (mean hourly-min charged 2W packs {r['ms']:.1f}; n={r['att2']:,})"
    for k, r in NT.items()))

# %%
fig = plt.figure(figsize=(17, 11))
gsp = fig.add_gridspec(2, 3, height_ratios=[1, 1.1])
axes = [fig.add_subplot(gsp[0, 0]), fig.add_subplot(gsp[0, 1]), fig.add_subplot(gsp[0, 2]), fig.add_subplot(gsp[1, 0]),
        fig.add_subplot(gsp[1, 1:])]
ax = axes[0]
ab = (JT.with_columns((pl.col("ambient_temp_c") // 2 * 2).alias("ab")).group_by("ab")
      .agg(pl.col("cabinet_temp_c").mean().alias("cab"), pl.len().alias("n")).filter(pl.col("n") >= 200).sort("ab"))
ax.plot(ab["ab"] + 1, ab["cab"], color=PALETTE["primary"], lw=2.4, marker="o", ms=3.5)
ax.plot([0, 50], [0, 50], color="#BBBBBB", ls=":", lw=1)
ax.text(3, 47, f"r = {amb['r'][0]:.2f}\ncabinet = ambient + {amb['d'][0]:.1f}C", fontsize=10.5, va="top")
ax.set_xlabel("Ambient temperature (C)")
ax.set_ylabel("Cabinet temperature (C)")
ax.set_title("1. Cabinets run ~6.6C above the air", fontsize=11.5)
mp = mech.filter(pl.col("station_hours") >= 300).to_pandas()
for i, (colname, ttl, ylab) in enumerate([
        ("charge_min", "2. Above ~40C, Gen1 charging slows sharply", "Avg charge time per pack (min)"),
        ("zero_2w_share", "3. In hot hours, Gen1 cabinets run out of packs", "Share of hours with 0 charged 2W packs")]):
    ax = axes[i + 1]
    for g in ["Gen1", "Gen2", "Gen3"]:
        s = mp[mp["charger_generation"] == g]
        ax.plot(s["cab_bin"] + CB / 2, s[colname], color=GEN_COLORS[g], lw=2.4, marker="o", ms=3.5, label=g)
    ax.axvspan(40, 60, color=PALETTE["accent"], alpha=0.06)
    ax.set_xlabel("Cabinet temperature in the hour (C)")
    ax.set_ylabel(ylab)
    ax.set_title(ttl, fontsize=11.5)
    ax.set_xlim(5, 60)
axes[2].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
axes[1].legend(title="Charger", loc="upper left")
ax = axes[3]
sdp = sdc.filter(pl.col("station_days") >= 60).to_pandas()
for g in ["Gen1", "Gen2", "Gen3"]:
    s = sdp[sdp["charger_generation"] == g]
    ax.plot(s["cab_max_bin"] + CB / 2, s["stockout_rate"], color=GEN_COLORS[g], lw=2.4, marker="o", ms=3.5, label=g)
ax.axvspan(40, 62, color=PALETTE["accent"], alpha=0.06)
ax.set_xlabel("Peak cabinet temperature of the day (C)")
ax.set_ylabel("Stockout rate that day (share of attempts)")
ax.set_title("4. Hot days: Gen1 stockouts 5x (day-level association)", fontsize=11.5)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
g1b = MB["Gen1"]
axes[1].annotate(f"Gen1: {g1b['<40']['chg']:.0f} -> {g1b['>=50']['chg']:.0f} min\nabove 50C", xy=(52.5, g1b['>=50']['chg']),
                 xytext=(12, 160), fontsize=10, arrowprops=dict(arrowstyle="->", color="#222222"))
axes[2].annotate(f"Gen1 >=50C:\n{pct(g1b['>=50']['zero'], 0)} of hours empty", xy=(52.5, g1b['>=50']['zero']),
                 xytext=(12, 0.62), fontsize=10, arrowprops=dict(arrowstyle="->", color="#222222"))
g1d = SD.filter(pl.col("charger_generation") == "Gen1")
g1_hi = g1d.filter(pl.col("cab_max") >= 45)
g1_lo = g1d.filter(pl.col("cab_max") < 40)
SD_G1_HI, SD_G1_LO = g1_hi["so"].sum() / g1_hi["att"].sum(), g1_lo["so"].sum() / g1_lo["att"].sum()
glm_g1 = glm_df[(glm_df["generation"] == "Gen1") & (glm_df["cabinet_max_band"] == "45-50")].iloc[0]
axes[3].text(0.03, 0.97, f"Gen1 station-days, peak >=45C:\n{pct(SD_G1_HI, 1)} vs {pct(SD_G1_LO, 1)} when peak <40C\n"
             f"GLM odds ratio (45-50C) {glm_g1['odds_ratio_vs_lt40']:.2f}", transform=axes[3].transAxes, fontsize=10, va="top",
             bbox=dict(boxstyle="round", fc="white", ec="#999999"))
# Panel 5: within hot days, the hour profile of stockouts vs empty cabinets
ax = axes[4]
ph_hot = prof.filter(pl.col("hot_day")).sort("h").to_pandas()
ph_oth = prof.filter(~pl.col("hot_day")).sort("h").to_pandas()
axb = ax.twinx()
axb.bar(ph_hot["h"], ph_hot["zero_stock_share"], 0.72, color=PALETTE["muted"], alpha=0.35,
        label="Hot days: share of hours with 0 charged 2W packs (right axis)")
axb.set_ylim(0, 1.0)
axb.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
axb.set_ylabel("Share of station-hours with 0 charged 2W packs")
axb.grid(False)
axb.spines["right"].set_visible(True)
ax.set_zorder(axb.get_zorder() + 1)
ax.patch.set_visible(False)
ax.plot(ph_hot["h"], ph_hot["stockout_rate_2w"], color=PALETTE["accent"], lw=2.6, marker="o", ms=4.5,
        label="Hot days (peak cabinet >=45C): 2W stockout rate")
ax.plot(ph_oth["h"], ph_oth["stockout_rate_2w"], color=PALETTE["primary"], lw=2.0, ls="--", marker="s", ms=3.5,
        label="Other Apr-Jun days: 2W stockout rate")
ax.axvspan(18.5, 22.5, color=PALETTE["accent"], alpha=0.07)
ax.set_xticks(range(0, 24, 2))
ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
ax.set_xlim(-0.6, 23.6)
ax.set_ylim(0, 0.26)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Hour of day (corrected local time)")
ax.set_ylabel("2W stockout rate (share of 2W attempts)")
ax.set_title(f"5. Within a hot day, stockouts do not follow the empty hours (hot-city Gen1, Apr-Jun; {n_hot_days:,} hot station-days)",
             fontsize=11.5)
ax.text(0.01, 0.97, "Zero-stock vs stocked hour, 2W stockout (MH RR):\n"
        f"  same station-day: {HR_SD['rr']:.2f} ({HR_SD['ci_lo']:.2f}-{HR_SD['ci_hi']:.2f})\n"
        f"  + same 19-22h window: {HR_SDE['rr']:.2f} ({HR_SDE['ci_lo']:.2f}-{HR_SDE['ci_hi']:.2f})\n"
        f"  same city x gen x month x hour: {HR_CGMH['rr']:.2f} ({HR_CGMH['ci_lo']:.2f}-{HR_CGMH['ci_hi']:.2f})\n"
        f"  zero stock 1-6 h earlier: {LAG_RR_RANGE[0]:.2f}-{LAG_RR_RANGE[1]:.2f}",
        transform=ax.transAxes, fontsize=9.5, va="top", ha="left", bbox=dict(boxstyle="round", fc="white", ec="#999999"))
ax.text(10.5, 0.135, f"0-18h: {pct(W018[0])} whether {pct(ph018['zero_stock_share'].min(), 0)} or "
        f"{pct(ph018['zero_stock_share'].max(), 0)} of hours are empty", ha="center", fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="#999999"))
ax.text(20.5, 0.215, f"19-22h: {pct(W1922[0])}\nonly {pct(W1922[1])} of hours empty", ha="center", fontsize=10,
        color=PALETTE["accent"], bbox=dict(boxstyle="round", fc="white", ec=PALETTE["accent"]))
h1_, l1_ = ax.get_legend_handles_labels()
h2_, l2_ = axb.get_legend_handles_labels()
ax.legend(h1_ + h2_, l1_ + l2_, loc="center", bbox_to_anchor=(0.56, 0.675), fontsize=9.5)
headline(fig, "On hot days Gen1 cabinets overheat, charge 2x slower and run empty; the same station-days see 5x the stockouts",
         f"Panels 1-3: hourly telemetry ({J.height / 1e6:.2f}M station-hours; 'missing' rows excluded = {pct(cov_d['missing']['h'] / J.height, 1)}); "
         f"shaded = above 40C. Panel 4: {SD.height:,} station-days. Panel 5: hot-city Gen1 station-hours with a 2W attempt.\n"
         "The heat effect is a day-level association: within a day, the hourly stock level does not predict which hours fail. "
         "Gen3 chargers barely react to heat.", top=0.9)
save_fig(fig, "q2_05_heat_mechanism")

# %% [markdown]
# ## 7. Vehicle class: 3W riders fail about twice as often
# 3W vs 2W compared inside the same station and month (Mantel-Haenszel), so station mix cannot explain the gap.

# %%
vc = rates(ev, ["month", "vehicle_class"])
vc.write_csv(TAB / "q2_vehicle_class_monthly.csv")
vcs = rates(ev_s, ["season_grp", "vehicle_class"])
ptable("Rates by season group x vehicle class", vcs)
smv = SMC.to_pandas().pivot_table(index=["station_id", "month"], columns="vehicle_class", values=["att", "so", "ab", "fail"], aggfunc="sum").fillna(0)
smv.columns = [f"{a}_{b}" for a, b in smv.columns]
smv = smv.reset_index()
rr3 = {k: mh_rr(smv, f"{k}_3W", "att_3W", f"{k}_2W", "att_2W") for k in ["so", "ab", "fail"]}
for k, lab in [("so", "stockout"), ("ab", "abandonment"), ("fail", "any service failure")]:
    print(f"[3W] 3W vs 2W {lab} risk ratio within station-month (MH) = {rr3[k][0]:.2f} (95% CI {rr3[k][1]:.2f}-{rr3[k][2]:.2f}; "
          f"{rr3[k][3]:,} station-month strata)")
smv_hot = smv[pd.to_datetime(smv["month"]).dt.month.isin(HOT_MONTHS)]
smv_cool = smv[~pd.to_datetime(smv["month"]).dt.month.isin(HOT_MONTHS)]
rr3h = mh_rr(smv_hot, "so_3W", "att_3W", "so_2W", "att_2W")
rr3c = mh_rr(smv_cool, "so_3W", "att_3W", "so_2W", "att_2W")
print(f"[3W] stockout RR 3W vs 2W: Jul-Mar {rr3c[0]:.2f} (CI {rr3c[1]:.2f}-{rr3c[2]:.2f}), Apr-Jun {rr3h[0]:.2f} (CI {rr3h[1]:.2f}-{rr3h[2]:.2f})")
v_cool = vcs.filter(pl.col("season_grp") == "Jul-Mar")
v_hot = vcs.filter(pl.col("season_grp") == "Apr-Jun")
for r in v_cool.iter_rows(named=True):
    h = v_hot.filter(pl.col("vehicle_class") == r["vehicle_class"]).row(0, named=True)
    print(f"[3W] {r['vehicle_class']}: stockout {pct(r['stockout_rate'], 2)} (Jul-Mar) -> {pct(h['stockout_rate'], 2)} (Apr-Jun); "
          f"abandonment {pct(r['abandon_rate'], 2)} -> {pct(h['abandon_rate'], 2)}; failure {pct(r['failure_rate'], 2)} -> {pct(h['failure_rate'], 2)}")

# 3W slots vs 3W demand
s3 = (SC.with_columns(pl.col("slots_3w").cast(pl.Utf8).alias("slots_3w_grp"))
      .group_by("slots_3w_grp").agg(pl.len().alias("stations"), pl.col("att_3w").sum(), pl.col("fail_3w").sum(), pl.col("so_3w").sum(),
                                     (pl.col("att_3w").sum() / pl.col("active_days").sum()).alias("att_3w_per_station_day"))
      .with_columns((pl.col("fail_3w") / pl.col("att_3w")).alias("failure_rate_3w"), (pl.col("so_3w") / pl.col("att_3w")).alias("stockout_rate_3w"),
                    (pl.col("att_3w") / pl.col("att_3w").sum()).alias("share_of_3w_attempts"))
      .sort("slots_3w_grp"))
s3.write_csv(TAB / "q2_3w_slots_vs_demand.csv")
ptable("3W demand and failures by station 3W slot count", s3)
z3 = s3.filter(pl.col("slots_3w_grp") == "0").row(0, named=True)
comp3_zero = (ev.filter(pl.col("vehicle_class") == "3W", pl.col("is_completed"))
              .join(st_all.lazy().select("station_id", "slots_3w"), on="station_id").filter(pl.col("slots_3w") == 0)
              .select(pl.len()).collect().item())
print(f"[3W] stations with ZERO 3W slots: {z3['stations']} stations take {pct(z3['share_of_3w_attempts'])} of 3W attempts "
      f"({z3['att_3w']:,}); {comp3_zero:,} 3W swaps completed there; 3W stockout rate {pct(z3['stockout_rate_3w'], 2)} vs "
      f"{pct(s3.filter(pl.col('slots_3w_grp') != '0')['so_3w'].sum() / s3.filter(pl.col('slots_3w_grp') != '0')['att_3w'].sum(), 2)} at stations with 3W slots")
with3 = SC.filter(pl.col("slots_3w") > 0).with_columns((pl.col("att_3w") / pl.col("active_days") / pl.col("slots_3w")).alias("d3"),
                                                       (pl.col("so_3w") / pl.col("att_3w")).alias("r3"))
rho3 = stats.spearmanr(with3["d3"].to_numpy(), with3["r3"].to_numpy())
print(f"[3W] stations with 3W slots: Spearman(3W attempts per 3W slot-day, 3W stockout rate) = {rho3[0]:.3f} (p={rho3[1]:.3f}, n={with3.height})")
J3 = JT.filter(pl.col("slots_3w") > 0, pl.col("charged_3w_min").is_not_null())
z3h = J3.filter(pl.col("charged_3w_min") == 0)
nz3h = J3.filter(pl.col("charged_3w_min") > 0)
print(f"[3W] 3W-equipped station-hours with telemetry: {J3.height:,}; share with 0 charged 3W packs {pct(z3h.height / J3.height, 2)}; "
      f"3W stockout rate in those hours {pct(z3h['so3'].sum() / z3h['att3'].sum(), 1)} vs {pct(nz3h['so3'].sum() / nz3h['att3'].sum(), 1)} otherwise; "
      f"mean charged_3w_min {J3['charged_3w_min'].mean():.2f}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
for cls, col in [("2W", PALETTE["primary"]), ("3W", PALETTE["accent"])]:
    s = vc.filter(pl.col("vehicle_class") == cls).sort("month")
    ax.plot(s["month"], s["stockout_rate"], color=col, lw=2.4, marker="o", ms=4, label=f"{cls} stockout")
    ax.plot(s["month"], s["abandon_rate"], color=col, lw=1.8, ls="--", marker="s", ms=3, label=f"{cls} abandoned")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax.set_ylabel("Share of attempts")
ax.set_title(f"Same station & month: 3W stockout risk = {rr3['so'][0]:.1f}x 2W, abandonment = {rr3['ab'][0]:.1f}x", fontsize=11.5)
ax.legend(ncol=2, loc="upper right")
month_axis(ax)
ax = axes[1]
s3p = s3.to_pandas()
xx = np.arange(len(s3p))
ax.bar(xx - 0.2, s3p["att_3w_per_station_day"], 0.4, color=PALETTE["muted"], label="3W attempts per station-day")
ax2 = ax.twinx()
ax2.bar(xx + 0.2, s3p["failure_rate_3w"], 0.4, color=PALETTE["accent"], label="3W failure rate")
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax2.set_ylim(0, 0.20)
ax2.yaxis.set_major_locator(mtick.MultipleLocator(0.04))
ax2.grid(False)
ax2.spines["right"].set_visible(True)
for xi, (a, f_, n) in enumerate(zip(s3p["att_3w_per_station_day"], s3p["failure_rate_3w"], s3p["stations"])):
    ax.text(xi - 0.2, a + 0.2, f"{a:.1f}", ha="center", fontsize=10)
    ax2.text(xi + 0.2, f_ + 0.003, pct(f_, 1), ha="center", fontsize=10, color=PALETTE["accent"])
ax.set_xticks(xx)
ax.set_xticklabels([f"{s} 3W slots\n({n} stations)" for s, n in zip(s3p["slots_3w_grp"], s3p["stations"])])
ax.set_ylim(0, 10)
ax.set_ylabel("3W attempts per station-day")
ax2.set_ylabel("3W service-failure rate")
ax.set_title("3W demand and failure rate do not depend on 3W slots", fontsize=11.5)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, fontsize=9.5)
headline(fig, f"3W riders fail about {rr3['fail'][0]:.1f}x as often as 2W riders at the same station, all year round",
         f"Left: monthly stockout and abandonment rates by vehicle class. Right: {pct(z3['share_of_3w_attempts'], 0)} of 3W attempts "
         "happen at stations recorded with zero 3W slots - slot metadata does not describe how 3W packs are served.", top=0.84)
save_fig(fig, "q2_06_vehicle_class_3w")

# %% [markdown]
# ## 8. Did service get worse? Like-for-like at a fixed set of stations
# The Launch stations open for all of Jan-Jun 2024 and Jan-Jun 2025 give a same-station, same-month comparison
# (the decommissioned station is excluded if it is a Launch site). Busiest = top 10% of Launch stations by attempts per
# active day in H1-2024, with membership held fixed so the line is not distorted by new stations joining the group.

# %%
smf = SMC.group_by("station_id", "month").agg(pl.col("att").sum(), pl.col("fail").sum(), pl.col("comp").sum()).join(
    attrs.select("station_id", "expansion_wave", "city", "charger_generation"), on="station_id")
fixed_ids = st_all.filter((pl.col("expansion_wave") == "Launch") & pl.col("decommissioned_date").is_null())["station_id"].to_list()
print("[YOY] decommissioned stations: " + ", ".join(f"{r['station_id']} ({r['expansion_wave']}, {r['decommissioned_date']})"
                                                     for r in st_all.filter(pl.col("decommissioned_date").is_not_null()).iter_rows(named=True)))
fx = (smf.filter(pl.col("station_id").is_in(fixed_ids), pl.col("month").dt.month() <= 6)
      .with_columns(pl.col("month").dt.year().alias("y"), pl.col("month").dt.month().alias("m")))
yoy = (fx.group_by("y", "m").agg(pl.col("att").sum(), pl.col("fail").sum(), pl.col("station_id").n_unique().alias("stations"))
       .with_columns((pl.col("fail") / pl.col("att")).alias("fr")).sort("m", "y"))
days = {m: [31, 29 if y == 2024 else 28, 31, 30, 31, 30][m - 1] for y in (2024, 2025) for m in range(1, 7)}
yoy_w = yoy.pivot(on="y", index="m", values=["fr", "att"]).sort("m")
yoy_w = yoy_w.with_columns((pl.col("att_2025") / pl.col("att_2024")).alias("attempt_growth"), (pl.col("fr_2025") - pl.col("fr_2024")).alias("fr_change_pts"))
yoy_w.write_csv(TAB / "q2_yoy_fixed_stations.csv")
ptable(f"Fixed set of {len(fixed_ids)} Launch stations: failure rate by calendar month, 2024 vs 2025", yoy_w)
fxp = fx.to_pandas()
fxp["key"] = fxp["station_id"] + "_" + fxp["m"].astype(str)
fw_ = fxp.pivot_table(index="key", columns="y", values=["fail", "att"], aggfunc="sum").fillna(0)
fw_.columns = [f"{a}_{b}" for a, b in fw_.columns]
rr_yoy = mh_rr(fw_.reset_index(), "fail_2025", "att_2025", "fail_2024", "att_2024")
fw_hot = fw_[fw_.index.str.split("_").str[-1].astype(int).isin(HOT_MONTHS)]
rr_yoy_hot = mh_rr(fw_hot.reset_index(), "fail_2025", "att_2025", "fail_2024", "att_2024")
h1 = fx.group_by("y").agg(pl.col("att").sum(), pl.col("fail").sum())
h1d = {r["y"]: r for r in h1.iter_rows(named=True)}
print(f"[YOY] fixed {len(fixed_ids)} Launch stations, H1-2025 vs H1-2024: attempts x{h1d[2025]['att'] / h1d[2024]['att']:.2f}, failures "
      f"x{h1d[2025]['fail'] / h1d[2024]['fail']:.2f}; same station & month failure risk ratio (MH) {rr_yoy[0]:.3f} "
      f"(95% CI {rr_yoy[1]:.3f}-{rr_yoy[2]:.3f}); Apr-Jun only {rr_yoy_hot[0]:.3f} (CI {rr_yoy_hot[1]:.3f}-{rr_yoy_hot[2]:.3f})")
for r in yoy_w.filter(pl.col("m").is_in([1, 5, 6])).iter_rows(named=True):
    print(f"[YOY] month {r['m']}: fixed-set failure rate {pct(r['fr_2024'], 1)} (2024) vs {pct(r['fr_2025'], 1)} (2025); attempts x{r['attempt_growth']:.2f}")
hg1 = (fx.filter(pl.col("city").is_in(HOT_CITIES), pl.col("charger_generation") == "Gen1", pl.col("m") == 5)
       .group_by("y").agg(pl.col("att").sum(), pl.col("fail").sum(), pl.col("station_id").n_unique().alias("n")).sort("y"))
for r in hg1.iter_rows(named=True):
    print(f"[YOY] May-{r['y']} hot-city Gen1 Launch stations: {r['att'] / r['n'] / 31:.1f} attempts per station-day, failure rate {pct(r['fail'] / r['att'])}")

net = smf.group_by("month").agg((pl.col("fail").sum() / pl.col("att").sum()).alias("Network"))
grp_lines = {"Network": net}
for wv, lab in [("Launch", "Launch stations"), ("Wave1_2024H2", "Wave 1 (2024H2)"), ("Wave2_2025H1", "Wave 2 (2025H1)")]:
    grp_lines[lab] = smf.filter(pl.col("expansion_wave") == wv).group_by("month").agg((pl.col("fail").sum() / pl.col("att").sum()).alias(lab))
H1_24 = pl.col("month") < pl.date(2024, 7, 1)
apd24 = (smf.filter(H1_24).group_by("station_id").agg(pl.col("att").sum())
         .join(smo.filter(H1_24).group_by("station_id").agg(pl.col("active_days").sum().alias("d24")), on="station_id")
         .with_columns((pl.col("att") / pl.col("d24")).alias("apd")).sort("apd", descending=True))
busiest = apd24.head(k10)["station_id"].to_list()
BUSY = f"Busiest {k10} in H1-24 (fixed set)"
grp_lines[BUSY] = smf.filter(pl.col("station_id").is_in(busiest)).group_by("month").agg(
    (pl.col("fail").sum() / pl.col("att").sum()).alias(BUSY))
bfx = fw_[fw_.index.str.rsplit("_", n=1).str[0].isin(busiest)]
rr_busy = mh_rr(bfx.reset_index(), "fail_2025", "att_2025", "fail_2024", "att_2024")
print(f"[YOY] busiest {k10} stations by H1-24 attempts/day: same station & month failure risk ratio 2025 vs 2024 (MH) = {rr_busy[0]:.3f} "
      f"(95% CI {rr_busy[1]:.3f}-{rr_busy[2]:.3f}); their H1-24 attempts/day median {apd24.head(k10)['apd'].median():.1f} vs other Launch "
      f"{apd24.tail(apd24.height - k10)['apd'].median():.1f}")
lines = net
for k, v in grp_lines.items():
    if k != "Network":
        lines = lines.join(v, on="month", how="left")
lines = lines.sort("month")
lines.write_csv(TAB / "q2_failure_rate_by_station_group_monthly.csv")
ptable("Monthly failure rate: network vs station groups", lines)
bz = SC.filter(pl.col("station_id").is_in(busiest))
print(f"[YOY] busiest {k10} stations (H1-2024 load): generations {dict(bz.group_by('charger_generation').len().iter_rows())}, cities "
      f"{dict(bz.group_by('city').len().iter_rows())}")
may25 = lines.filter(pl.col("month") == pl.date(2025, 5, 1)).row(0, named=True)
may24 = lines.filter(pl.col("month") == pl.date(2024, 5, 1)).row(0, named=True)
print(f"[YOY] May-24 network {pct(may24['Network'])} -> May-25 network {pct(may25['Network'])}; May-25 Launch {pct(may25['Launch stations'])}, "
      f"Wave1 {pct(may25['Wave 1 (2024H2)'])}, Wave2 {pct(may25['Wave 2 (2025H1)'])}; busiest {k10}: May-24 {pct(may24[BUSY])}, May-25 {pct(may25[BUSY])}")
mix25 = (smf.filter(pl.col("month") == pl.date(2025, 5, 1)).group_by("expansion_wave").agg(pl.col("att").sum())
         .with_columns((pl.col("att") / pl.col("att").sum()).alias("s")).sort("expansion_wave"))
print("[YOY] May-25 attempt mix by wave: " + ", ".join(f"{r['expansion_wave']} {pct(r['s'])}" for r in mix25.iter_rows(named=True)))

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.45, 1]})
ax = axes[0]
style = {"Network": (PALETTE["primary"], 3.0, "-"), "Launch stations": (PALETTE["accent"], 2.0, "-"),
         "Wave 1 (2024H2)": (PALETTE["warn"], 2.0, "-"), "Wave 2 (2025H1)": (PALETTE["good"], 2.0, "-"),
         BUSY: ("#6C4F9E", 2.0, "--")}
lp = lines.to_pandas()
for col, (c, lw, ls) in style.items():
    ax.plot(lp["month"], lp[col], color=c, lw=lw, ls=ls, marker="o", ms=3.5, label=col)
ax.annotate(f"May-25 network {pct(may25['Network'])}\n(Launch {pct(may25['Launch stations'])}, Wave 2 {pct(may25['Wave 2 (2025H1)'])})",
            xy=(pd.Timestamp("2025-05-01"), may25["Network"]), xytext=(pd.Timestamp("2024-08-01"), 0.115), fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#222222"))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax.set_ylim(0.02, 0.19)
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.02))
ax.set_ylabel("Service-failure rate")
ax.set_title("Monthly failure rate by station group", fontsize=11.5)
ax.legend(fontsize=9.5, loc="upper center", ncol=3)
month_axis(ax)
ax = axes[1]
yp = yoy_w.to_pandas()
xx = np.arange(len(yp))
ax.bar(xx - 0.2, yp["fr_2024"], 0.4, color=PALETTE["muted"], label="2024")
ax.bar(xx + 0.2, yp["fr_2025"], 0.4, color=PALETTE["primary"], label="2025")
for xi, (a, b, gr) in enumerate(zip(yp["fr_2024"], yp["fr_2025"], yp["attempt_growth"])):
    ax.text(xi, max(a, b) + 0.004, f"x{gr:.2f}\nattempts", ha="center", fontsize=9)
ax.set_xticks(xx)
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun"])
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax.set_ylim(0, 0.15)
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.02))
ax.set_title(f"Same {len(fixed_ids)} Launch stations, same month: risk ratio {rr_yoy[0]:.3f} (CI {rr_yoy[1]:.3f}-{rr_yoy[2]:.3f})", fontsize=11.5)
ax.legend(loc="upper left")
headline(fig, "Like-for-like, failure rates did not change year on year; the network average improved only because of the new stations",
         f"Fixed Launch stations served {h1d[2025]['att'] / h1d[2024]['att']:.2f}x the attempts in H1-2025 at the same failure rate. "
         "The lower network rate in May-25 comes from the mix of new Gen3-heavy Wave stations, not from fixing the old ones.", top=0.84)
save_fig(fig, "q2_07_same_station_yoy")

# %% [markdown]
# ## 9. Customer-experience link: support tickets vs failure rates
# Tickets are agent-categorised (not verified); we use `no_battery_available` and `long_queue`, the two categories
# that should track stockouts and abandonment.

# %%
rho_nb = stats.spearmanr((SC["tk_no_battery"] / SC["att"] * 1000).to_numpy(), SC["stockout_rate"].to_numpy())
rho_lq = stats.spearmanr((SC["tk_long_queue"] / SC["att"] * 1000).to_numpy(), SC["abandon_rate"].to_numpy())
rho_all = stats.spearmanr(SC["nb_lq_tickets_per_1k"].to_numpy(), SC["failure_rate"].to_numpy())
print(f"[TK] in-window station-linked tickets: {tk.height:,} (no_battery_available {int(SC['tk_no_battery'].sum()):,}, "
      f"long_queue {int(SC['tk_long_queue'].sum()):,})")
print(f"[TK] station level (n={n_st}): Spearman no_battery tickets/1K attempts vs stockout rate = {rho_nb[0]:.3f} (p={rho_nb[1]:.2g}); "
      f"long_queue/1K vs abandonment rate = {rho_lq[0]:.3f} (p={rho_lq[1]:.2g}); both/1K vs failure rate = {rho_all[0]:.3f} (p={rho_all[1]:.2g})")
tk_sm = (tk.group_by("station_id", "month").agg((pl.col("category") == "no_battery_available").sum().alias("nb"),
                                                (pl.col("category") == "long_queue").sum().alias("lq"), pl.len().alias("all_tk")))
smt = (SMC.group_by("station_id", "month").agg(pl.col("att").sum(), pl.col("fail").sum(), pl.col("so").sum(), pl.col("ab").sum())
       .join(tk_sm, on=["station_id", "month"], how="left").with_columns([pl.col(c).fill_null(0) for c in ["nb", "lq", "all_tk"]]))
rho_sm = stats.spearmanr((smt["nb"] / smt["att"]).to_numpy(), (smt["so"] / smt["att"]).to_numpy())
print(f"[TK] station-month level (n={smt.height:,}): Spearman no_battery tickets per attempt vs stockout rate = {rho_sm[0]:.3f} (p={rho_sm[1]:.2g})")
tkm = (smt.group_by("month").agg(pl.col("att").sum(), pl.col("fail").sum(), pl.col("so").sum(), pl.col("ab").sum(), pl.col("nb").sum(),
                                 pl.col("lq").sum(), pl.col("all_tk").sum())
       .with_columns((pl.col("nb") / pl.col("att") * 1000).alias("nb_per_1k"), (pl.col("lq") / pl.col("att") * 1000).alias("lq_per_1k"),
                     (pl.col("all_tk") / pl.col("att") * 1000).alias("tickets_per_1k"), (pl.col("fail") / pl.col("att")).alias("failure_rate"),
                     (pl.col("nb") / pl.col("so") * 100).alias("nb_per_100_stockouts"), (pl.col("lq") / pl.col("ab") * 100).alias("lq_per_100_abandons"))
       .sort("month"))
tkm.write_csv(TAB / "q2_tickets_monthly.csv")
ptable("Monthly tickets per 1K attempts vs failure rate", tkm.select("month", "att", "failure_rate", "nb_per_1k", "lq_per_1k", "tickets_per_1k",
                                                                   "nb_per_100_stockouts", "lq_per_100_abandons"))
rho_m = stats.spearmanr(tkm["nb_per_1k"].to_numpy(), tkm["failure_rate"].to_numpy())
print(f"[TK] monthly (n=18): Spearman no_battery tickets/1K vs failure rate = {rho_m[0]:.3f} (p={rho_m[1]:.2g})")
# denominator check: the rates above use station-linked tickets; include tickets without a station_id as well
tk_any = tickets().filter(pl.col("in_window") & ~pl.col("is_test"))
tka_m = tk_any.group_by("month").agg(pl.len().alias("all_tk_incl_unlinked"),
                                     (pl.col("category") == "no_battery_available").sum().alias("nb_incl_unlinked"))
tkm = tkm.join(tka_m, on="month", how="left").with_columns(
    (pl.col("all_tk_incl_unlinked") / pl.col("att") * 1000).alias("tickets_per_1k_incl_unlinked"),
    (pl.col("nb_incl_unlinked") / pl.col("so") * 100).alias("nb_per_100_stockouts_incl_unlinked"))
tkm.write_csv(TAB / "q2_tickets_monthly.csv")
print(f"[TK] in-window tickets: {tk_any.height:,}, of which {tk_any.filter(pl.col('station_id').is_null()).height:,} have no station_id; "
      f"station-level and monthly rates below use the {tk.height:,} station-linked tickets")
for mo in [pl.date(2024, 1, 1), pl.date(2024, 5, 1), pl.date(2025, 1, 1), pl.date(2025, 5, 1)]:
    r = tkm.filter(pl.col("month") == mo).row(0, named=True)
    print(f"[TK] {r['month']:%b-%y}: station-linked tickets/1K attempts {r['tickets_per_1k']:.2f} (no_battery {r['nb_per_1k']:.2f}, long_queue "
          f"{r['lq_per_1k']:.2f}); no_battery tickets per 100 stockouts {r['nb_per_100_stockouts']:.1f}; failure rate {pct(r['failure_rate'])} | "
          f"all tickets: {r['tickets_per_1k_incl_unlinked']:.2f} per 1K, no_battery {r['nb_per_100_stockouts_incl_unlinked']:.1f} per 100 stockouts")
SC.select("station_id", "city", "charger_generation", "expansion_wave", "att", "failure_rate", "stockout_rate", "abandon_rate",
          "tk_no_battery", "tk_long_queue", "nb_lq_tickets_per_1k", "tickets_per_1k").write_csv(TAB / "q2_tickets_vs_failures_station.csv")
# which categories drive the rise in tickets per attempt? same cool months (Jan-Mar) in both years
q1 = pl.col("month").dt.month() <= 3
att_q1 = {y: SMC.filter(q1, pl.col("month").dt.year() == y)["att"].sum() for y in (2024, 2025)}
cat_q1 = (tk.filter(q1).group_by(pl.col("month").dt.year().alias("y"), "category").len()
          .pivot(on="y", index="category", values="len").sort("category"))
cat_q1 = cat_q1.select("category", "2024", "2025").with_columns((pl.col("2024") / att_q1[2024] * 1000).alias("per_1k_Q1_24"),
                             (pl.col("2025") / att_q1[2025] * 1000).alias("per_1k_Q1_25")).sort("per_1k_Q1_25", descending=True)
cat_q1.write_csv(TAB / "q2_ticket_categories_q1_yoy.csv")
ptable("Tickets per 1K attempts by category, Jan-Mar 2024 vs Jan-Mar 2025 (failure rate flat at ~3.9% in both)", cat_q1)

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
ax = axes[0]
scp = SC.select("failure_rate", "nb_lq_tickets_per_1k", "charger_generation", "att").to_pandas()
for g in ["Gen1", "Gen2", "Gen3"]:
    s = scp[scp["charger_generation"] == g]
    ax.scatter(s["failure_rate"], s["nb_lq_tickets_per_1k"], s=s["att"] / 900, alpha=0.65, color=GEN_COLORS[g], label=g, edgecolor="white", lw=0.5)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Station service-failure rate (Jan 2024 - Jun 2025)")
ax.set_ylabel("'No battery' + 'long queue' tickets per 1K attempts")
ax.set_title(f"Stations: Spearman rho = {rho_all[0]:.2f} (n={n_st}; bubble = attempts)", fontsize=11.5)
ax.legend(title="Charger", loc="upper left")
ax = axes[1]
tp = tkm.to_pandas()
ax.plot(tp["month"], tp["nb_per_1k"], color=PALETTE["accent"], lw=2.4, marker="o", ms=4, label="'No battery' tickets per 1K attempts")
ax.plot(tp["month"], tp["lq_per_1k"], color=PALETTE["warn"], lw=2.2, marker="s", ms=3.5, label="'Long queue' tickets per 1K attempts")
ax.set_ylabel("Tickets per 1K attempts")
ax2 = ax.twinx()
ax2.plot(tp["month"], tp["failure_rate"], color=PALETTE["primary"], lw=1.8, ls="--", label="Service-failure rate (right)")
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax2.grid(False)
ax2.spines["right"].set_visible(True)
ax2.set_ylabel("Service-failure rate")
h1_, l1_ = ax.get_legend_handles_labels()
h2_, l2_ = ax2.get_legend_handles_labels()
ax.legend(h1_ + h2_, l1_ + l2_, loc="upper left", fontsize=9.5)
ax.set_title("Network by month", fontsize=11.5)
month_axis(ax)
headline(fig, "Complaints follow failures: stations and months with more failures generate more 'no battery' and 'long queue' tickets",
         "Support tickets linked to a station, Jan 2024 - Jun 2025 (test stations and out-of-window tickets excluded). Ticket categories are agent-assigned.",
         top=0.84)
save_fig(fig, "q2_08_tickets_vs_failures")

# %% [markdown]
# ## 2b. Figure: queue wait, patience and load

# %%
fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
ax = axes[0]
bins = np.arange(0, 40.5, 0.5)
qh = (ev.filter(pl.col("event_type").is_in(["swap_completed", "abandoned_queue"]))
      .with_columns((pl.col("queue_wait_sec") / 30).floor().alias("b")).group_by("event_type", "b").agg(pl.len().alias("n")).collect().to_pandas())
for et, col, lab in [("swap_completed", PALETTE["primary"], "Completed"), ("abandoned_queue", PALETTE["accent"], "Abandoned")]:
    s = qh[qh["event_type"] == et].sort_values("b")
    ax.fill_between(s["b"] * 0.5 + 0.25, s["n"] / s["n"].sum() / 0.5, step="mid", alpha=0.35, color=col)
    ax.plot(s["b"] * 0.5 + 0.25, s["n"] / s["n"].sum() / 0.5, color=col, lw=1.8, label=lab)
ax.axvline(qc["p50_s"] / 60, color=PALETTE["primary"], ls="--", lw=1.2)
ax.axvline(qa["p50_s"] / 60, color=PALETTE["accent"], ls="--", lw=1.2)
ax.text(qc["p50_s"] / 60 + 0.5, 0.215, f"median {qc['p50_s'] / 60:.1f} min (p90 {qc['p90_s'] / 60:.1f})", color=PALETTE["primary"], fontsize=10)
ax.text(qa["p50_s"] / 60 + 0.5, 0.12, f"median {qa['p50_s'] / 60:.1f} min", color=PALETTE["accent"], fontsize=10)
ax.set_ylim(0, 0.26)
ax.set_xlim(0, 30)
ax.set_xlabel("Queue wait (minutes)")
ax.set_ylabel("Density (share per minute)")
ax.set_title("Wait before completion vs before giving up", fontsize=11.5)
ax.legend(loc="upper right")
ax = axes[1]
ax.plot(t_s / 60, S_km, color=PALETTE["accent"], lw=2.6)
for m in (5, 10, 15, 20):
    ax.plot(m, S_at(m), "o", color="#222222", ms=5)
    ax.text(m + 0.5, S_at(m) + 0.02, pct(S_at(m), 0), fontsize=10)
if np.isfinite(km_med):
    ax.axhline(0.5, color="#888888", ls=":", lw=1)
    ax.axvline(km_med, color="#888888", ls=":", lw=1)
    ax.text(km_med + 0.4, 0.52, f"median patience\n{km_med:.1f} min", fontsize=10)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
ax.set_xlim(0, 30)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Minutes waited")
ax.set_ylabel("Riders still willing to wait (Kaplan-Meier)")
ax.set_title(f"Patience curve: median {km_med:.1f} min", fontsize=11.5)
ax = axes[2]
qup = qu.to_pandas()
ax.errorbar(range(len(qup)), qup["mean_wait_min"], yerr=[qup["mean_wait_min"] - qup["ci_lo_min"], qup["ci_hi_min"] - qup["mean_wait_min"]],
            color=PALETTE["primary"], lw=2.6, marker="o", capsize=3)
ax.set_ylim(0, 6)
ax.set_xticks(range(len(qup)))
ax.set_xticklabels(qup["util_bin"], rotation=30)
ax.set_xlabel("Completed swaps per slot in the station-hour")
ax.set_ylabel("Mean wait of completed swaps (min, 95% CI)")
ax.text(0.2, 1.2, f"Flat at {qup['mean_wait_min'].min():.2f}-{qup['mean_wait_min'].max():.2f} min from the\nquietest to the busiest "
        f"station-hours\n({SH.filter(pl.col('comp') > 0).height / 1e6:.2f}M station-hours with a completed swap)", fontsize=10)
ax.set_title("Queue time does not rise with station load", fontsize=11.5)
headline(fig, f"Riders who abandon waited {qa['p50_s'] / qc['p50_s']:.1f}x longer than served riders, yet queue time is not a congestion problem",
         "Left: wait distribution by outcome. Middle: Kaplan-Meier patience curve (abandonment = event, completion = censored). "
         "Right: mean completed wait vs utilisation.", top=0.83)
save_fig(fig, "q2_01_queue_wait_patience")

# %% [markdown]
# ## 10. Key numbers recap

# %%
# sizing for recommendations (two summers in the data: Apr-Jun 2024 and Apr-Jun 2025)
g3h = seg.filter(pl.col("segment") == "Apr-Jun, hot-city Gen3").row(0, named=True)
avoid_g1 = g1["att"] * (g1["failure_rate"] - g3h["failure_rate"])
print(f"[SIZE] if hot-city Gen1 stations had failed at the hot-city Gen3 rate in Apr-Jun ({pct(g3h['failure_rate'], 2)} [Apr-Jun 2025, the only "
      f"summer with Gen3] vs {pct(g1['failure_rate'], 2)} [both summers]), failures avoided = {avoid_g1:,.0f} over two summers "
      f"(~{avoid_g1 / 2:,.0f} per summer on average; {pct(avoid_g1 / N_FAIL)} of all failures)")
AVOID_Y = {}
for y_ in (2024, 2025):
    s = SMCs.filter(pl.col("segment") == "Apr-Jun, hot-city Gen1", yr == y_)
    a_, f_ = s["att"].sum(), s["fail"].sum()
    AVOID_Y[y_] = a_ * (f_ / a_ - g3h["failure_rate"])
    print(f"[SIZE] Apr-Jun {y_} volume: hot-city Gen1 attempts {a_:,}, failure rate {pct(f_ / a_, 2)}; avoidable at the hot-city Gen3 rate = "
          f"{AVOID_Y[y_]:,.0f}")
# like-for-like alternative: within city-month MH risk ratios (Apr-Jun 2025), by city -> retrofit priority list
g1_st = st_all.filter(pl.col("charger_generation") == "Gen1").group_by("city").agg(pl.len().alias("gen1_stations"))
prio = (pl.from_pandas(gen_city[gen_city["city"].isin(HOT_CITIES + ["Pune"])])
        .join(g1_st, on="city", how="left")
        .with_columns((pl.col("gen1_failures") * (1 - 1 / pl.col("gen1_vs_gen3_rr"))).alias("avoidable_2025_mh"))
        .with_columns((pl.col("avoidable_2025_mh") / pl.col("gen1_stations")).alias("avoidable_2025_per_station"),
                      (pl.col("gen1_failures") / pl.col("gen1_attempts")).alias("gen1_failure_rate_AJ25"))
        .sort("avoidable_2025_per_station", descending=True))
prio.write_csv(TAB / "q2_retrofit_priority_by_city.csv")
ptable("Retrofit priority by city (Apr-Jun 2025, MH counterfactual: Gen1 failures x (1 - 1/RR))", prio)
mh_hot = gen_city[gen_city["city"] == "Hot cities pooled"].iloc[0]
print(f"[SIZE] MH counterfactual, Apr-Jun 2025, hot cities pooled: {mh_hot['gen1_failures']:,} Gen1 failures x (1 - 1/{mh_hot['gen1_vs_gen3_rr']:.2f}) = "
      f"{mh_hot['gen1_failures'] * (1 - 1 / mh_hot['gen1_vs_gen3_rr']):,.0f}; by city: " + ", ".join(
          f"{r['city']} {r['avoidable_2025_mh']:,.0f} ({r['gen1_stations']} sites, {r['avoidable_2025_per_station']:,.0f}/site, RR {r['gen1_vs_gen3_rr']:.2f})"
          for r in prio.iter_rows(named=True)))
h15 = hot15["station_id"].to_list()
h15_hot = SMCs.filter(pl.col("station_id").is_in(h15), pl.col("segment") == "Apr-Jun, hot-city Gen1")
avoid_h15 = h15_hot["att"].sum() * (h15_hot["fail"].sum() / h15_hot["att"].sum() - g3h["failure_rate"])
print(f"[SIZE] 15 largest hot-city Gen1 sites by excess: Apr-Jun attempts {h15_hot['att'].sum():,}, failure rate "
      f"{pct(h15_hot['fail'].sum() / h15_hot['att'].sum(), 2)}; avoidable at the hot-city Gen3 rate = {avoid_h15:,.0f} over two summers "
      f"(~{avoid_h15 / 2:,.0f} per summer) - large because they are big, not because they fail more than their city peers")
c3 = v_cool.filter(pl.col("vehicle_class") == "3W").row(0, named=True)
c2 = v_cool.filter(pl.col("vehicle_class") == "2W").row(0, named=True)
gap3 = c3["attempts"] * (c3["stockout_rate"] - c2["stockout_rate"])
n_cool_m = bm.filter(~pl.col("m").is_in(HOT_MONTHS)).height
print(f"[SIZE] 3W stockout gap vs 2W in Jul-Mar: {c3['attempts']:,} 3W attempts x ({pct(c3['stockout_rate'], 2)} - {pct(c2['stockout_rate'], 2)}) = "
      f"{gap3:,.0f} extra 3W stockouts over all {n_cool_m} cool months in the data (Jan-Mar 2024 + Jul 2024-Mar 2025)")
cs9 = (ev.filter(pl.col("month").is_between(pl.date(2024, 7, 1), pl.date(2025, 3, 1)))
       .group_by("vehicle_class").agg(pl.len().alias("n"), pl.col("is_stockout").mean().alias("so")).collect())
cs9d = {r["vehicle_class"]: r for r in cs9.iter_rows(named=True)}
GAP3_SEASON = cs9d["3W"]["n"] * (cs9d["3W"]["so"] - cs9d["2W"]["so"])
print(f"[SIZE] one cool season (Jul-24 to Mar-25, 9 months): {cs9d['3W']['n']:,} 3W attempts x ({pct(cs9d['3W']['so'], 2)} - "
      f"{pct(cs9d['2W']['so'], 2)}) = {GAP3_SEASON:,.0f} extra 3W stockouts per cool season")

# %% [markdown]
# ### 10b. What does a rider do after a failure? (is a failure a deferred swap or a lost one?)
# For every attempt, the gap to the same rider's next completed swap anywhere in the network. If failed riders came
# back soon (a retry), a failure would mostly be a delay; if not, it is a lost transaction. Attempts after
# 2025-06-01 are excluded so every attempt has at least a month of follow-up.

# %%
RD = (events(["rider_id", "event_ts", "event_type", "signup_date"]).sort("rider_id", "event_ts")
      .with_columns(pl.col("event_type").is_in(FAIL_TYPES).alias("fl"), (pl.col("event_type") == "swap_completed").alias("cp"))
      .with_columns(pl.when(pl.col("cp")).then(pl.col("event_ts")).alias("cts"))
      .with_columns(pl.col("cts").shift(-1).over("rider_id").alias("cts_n"), pl.int_range(pl.len()).over("rider_id").alias("k"))
      .with_columns(pl.col("cts_n").backward_fill().over("rider_id").alias("next_cts"))
      .with_columns((pl.col("next_cts") - pl.col("event_ts")).dt.total_minutes().alias("gap_min"))
      .filter(pl.col("fl") | pl.col("cp"))
      .select("fl", "cp", "k", "event_ts", "signup_date", "gap_min")
      .collect(engine="streaming"))
nx = (RD.filter(pl.col("event_ts") < pl.datetime(2025, 6, 1)).group_by("fl").agg(
    pl.len().alias("n"), (pl.col("gap_min") <= 60).mean().alias("w60"), (pl.col("gap_min") <= 1440).mean().alias("w24h"),
    (pl.col("gap_min").median() / 60).alias("med_h"), pl.col("gap_min").is_null().mean().alias("never")).sort("fl", descending=True))
NX = {r["fl"]: r for r in nx.iter_rows(named=True)}
for k_, lab in [(True, "after a service failure"), (False, "after a completed swap")]:
    r = NX[k_]
    print(f"[RETRY] {lab} (n={r['n']:,}): same rider completes a swap within 60 min {pct(r['w60'])}, within 24 h {pct(r['w24h'])}; "
          f"median gap to next completed swap {r['med_h']:.1f} h; never again {pct(r['never'])}")
fa = (RD.filter(pl.col("k") == 0, pl.col("event_ts") < pl.datetime(2025, 5, 1), pl.col("signup_date") >= pl.date(2024, 1, 1))
      .group_by("fl").agg(pl.len().alias("n"), (pl.col("gap_min") <= 30 * 1440).mean().alias("ret30")).sort("fl", descending=True))
print("[RETRY] new riders (signed up 2024+, first attempt before May-25), completed a further swap within 30 days: " + ", ".join(
    f"{'first attempt failed' if r['fl'] else 'first attempt completed'} {pct(r['ret30'])} (n={r['n']:,})" for r in fa.iter_rows(named=True))
      + "  [cross-link for Q6; short horizon only]")
del RD
REV_G1 = (events(["city", "charger_generation", "month", "is_completed", "amount_charged_inr"])
          .filter(pl.col("is_completed"), pl.col("city").is_in(HOT_CITIES), pl.col("charger_generation") == "Gen1",
                  pl.col("month") >= pl.date(2025, 4, 1))
          .select(pl.col("amount_charged_inr").mean()).collect().item())
print(f"[RETRY] mean revenue per completed swap, hot-city Gen1, Apr-Jun 2025: Rs {REV_G1:.1f}; avoidable failures at 2025 volume "
      f"({AVOID_Y[2025]:,.0f}) x revenue = Rs {AVOID_Y[2025] * REV_G1 / 1e6:.2f}M per summer of revenue (not margin)")

print("\n================ Q2 KEY NUMBERS ================")
print(f"Attempts {N_ATT:,}; service failures {N_FAIL:,} ({pct(FR_ALL, 2)}); 2W {pct(FR_2W, 2)}, 3W {pct(FR_3W, 2)}")
print(f"Jul-Mar baseline {pct(BASE_FR, 2)}; excess failures {tot_exc:,.0f} ({pct(tot_exc / N_FAIL)} of all failures)")
print(f"Apr-Jun hot-city Gen1 ({g1['stations']} stations): {pct(g1['share_aprjun_attempts'])} of Apr-Jun attempts -> {pct(g1['share_excess'])} "
      f"of excess failures; the same stations over 18 months: {pct(st36['att'].sum() / N_ATT)} of attempts, {pct(st36['fail'].sum() / N_FAIL)} of failures")
print(f"Gen1 station-days with peak cabinet >=45C: stockout {pct(SD_G1_HI, 2)} vs {pct(SD_G1_LO, 2)} when <40C (day level); hourly zero stock "
      f"within the same station-day: MH RR {HR_SD['rr']:.2f}, +window {HR_SDE['rr']:.2f} (no hour-level link)")
print(f"Avoidable failures at the hot-city Gen3 rate: 2024 volume {AVOID_Y[2024]:,.0f}, 2025 volume {AVOID_Y[2025]:,.0f}; "
      f"3W cool-season stockout gap {GAP3_SEASON:,.0f} (Jul-24 to Mar-25)")
print(f"Fixed Launch stations YoY risk ratio {rr_yoy[0]:.3f} (CI {rr_yoy[1]:.3f}-{rr_yoy[2]:.3f})")
print(f"3W vs 2W failure RR {rr3['fail'][0]:.2f}; median patience {km_med:.1f} min")
print(f"Runtime {time.time() - T0:.1f}s")
