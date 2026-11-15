# %% [markdown]
# # Q3 - Station & geographic patterns, expansion-wave effectiveness, competitor entry
#
# **Question.** Stations differ in charger generation, location type, commissioning time and are unevenly spread across
# cities. How do these relate to service performance, battery turnaround time and complaints? Did the two expansion
# waves improve service where riders needed it, or where sites were easiest to open? What did competitor entry do?
#
# **Approach.**
# 1. *Battery turnaround* - hourly telemetry (telemetry_status == 'ok' only): charge minutes vs cabinet temperature by
#    charger generation (piecewise-linear "hinge" fit), season exposure, quarantines, chargers online; then link hot
#    hours to stockouts against same-city same-day Gen2/Gen3 stations.
# 2. *Service & complaints by station attributes* - station-month panel, WLS with city and month fixed effects and a
#    utilisation control (attempts per slot-day), cluster-robust SE by station, so generation / location / host /
#    connectivity / wave are separated rather than read off confounded raw means.
# 3. *Uneven distribution* - supply (stations, slots) vs demand (riders by home zone, attempts, failures) by zone.
# 4. *Expansion waves* - placement vs pre-wave need (same-season need: failures are an Apr-Jun phenomenon);
#    what drives new-station utilisation (zone demand vs site type); utilisation and fixed cost per swap of new vs
#    legacy stations; DiD on legacy stations; network average vs busiest/hotspot stations, in rates AND counts, and
#    whether hotspot failure rates respond to load at all.
# 5. *Competitor entry* (10 stations, 2025-01-15) - DiD + event study vs same-city same-wave stations, permutation test,
#    station-specific-trend robustness.
#
# *Revision after verification (see outputs/findings/q3_verification.md)*: Wave-2 need is measured on the same-season
# Apr-Jun 2024 rate (the Jul-Dec rate is flat noise), the siting recommendation is based on a utilisation model of the
# 60 new stations, the headline separates "dilution" from "placement", and several labels/figure titles were corrected.

# %%
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # notebook / Colab
    ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
from common import *  # noqa: E402,F403

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402

warnings.filterwarnings("ignore")
set_style()
T0 = time.time()
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_rows(60)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)


def say(*a):
    print(f"[{time.time()-T0:5.1f}s]", *a, flush=True)


GEN_COLORS = {"Gen1": SERIES[1], "Gen2": SERIES[0], "Gen3": SERIES[2]}
GEN_MARK = {"Gen1": "o", "Gen2": "s", "Gen3": "^"}
WAVE_COLORS = {"Launch": PALETTE["muted"], "Wave1_2024H2": SERIES[0], "Wave2_2025H1": SERIES[3]}
WAVE_LABEL = {"Launch": "Legacy (Launch 2023)", "Wave1_2024H2": "Wave 1 (Aug-Oct 2024)", "Wave2_2025H1": "Wave 2 (Feb-Apr 2025)"}
HOT_C = 40.0  # cabinet-temperature threshold (validated by the hinge fit below)

st = stations().filter(~pl.col("is_test")).with_columns((pl.col("slots_2w") + pl.col("slots_3w")).alias("slots"))
say("real stations:", st.height, "| by wave:", dict(st.group_by("expansion_wave").len().iter_rows()))

# %% [markdown]
# ## 1. Battery turnaround: thermal derating by charger generation
# `avg_charge_minutes` is the turnaround of a pack in the cabinet. We bin cabinet temperature (1 C) and fit, per
# generation, `charge_min = a + b * max(0, T - k)` choosing the knot k by least squares.

# %%
H = (hourly(["station_id", "city", "charger_generation", "season", "month", "date", "hour", "telemetry_status",
             "cabinet_temp_c", "ambient_temp_c", "avg_charge_minutes", "packs_quarantined", "chargers_online",
             "slots_2w", "slots_3w", "outage_minutes"])
     .filter(pl.col("telemetry_status") == "ok").collect())
n_all = hourly(["station_id"]).select(pl.len()).collect().item()
say(f"telemetry rows ok: {H.height:,} of {n_all:,} ({H.height/n_all:.1%}); partial/missing excluded (never zero-filled)")

# binned curve
curve = (H.with_columns(pl.col("cabinet_temp_c").floor().alias("t_bin"))
         .group_by("charger_generation", "t_bin")
         .agg(pl.len().alias("station_hours"), pl.col("avg_charge_minutes").mean().alias("charge_min"),
              pl.col("avg_charge_minutes").quantile(0.9).alias("charge_min_p90"),
              (pl.col("packs_quarantined") > 0).mean().alias("quarantine_hour_share"))
         .sort("charger_generation", "t_bin"))
curve.write_csv(TAB / "q3_thermal_derating_curve.csv")

# hinge fit per generation
hinge_rows = []
for g in ["Gen1", "Gen2", "Gen3"]:
    d = H.filter(pl.col("charger_generation") == g)
    T = d["cabinet_temp_c"].to_numpy().astype(float); y = d["avg_charge_minutes"].to_numpy().astype(float)
    best = None
    for k in np.arange(30, 50.01, 0.5):
        X = np.column_stack([np.ones_like(T), np.clip(T - k, 0, None)])
        coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(((y - X @ coef) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, k, coef)
    sse, k, (a, b) = best
    r2 = 1 - sse / float(((y - y.mean()) ** 2).sum())
    hinge_rows.append({"generation": g, "station_hours": len(y), "knot_c": k, "base_charge_min": a, "slope_min_per_c": b,
                       "charge_min_at_45c": a + b * max(0, 45 - k), "charge_min_at_50c": a + b * max(0, 50 - k),
                       "capacity_at_50c_vs_base": a / (a + b * max(0, 50 - k)), "r2": r2})
hinge = pd.DataFrame(hinge_rows)
hinge.to_csv(TAB / "q3_thermal_hinge_fit.csv", index=False)
say("Hinge fit (charge minutes = base + slope x max(0, T - knot)):\n" + hinge.round(3).to_string(index=False))
# robustness: the Gen1 slope depends on the knot grid (slope and knot trade off); refit on a finer 0.25 C grid
d = H.filter(pl.col("charger_generation") == "Gen1")
T = d["cabinet_temp_c"].to_numpy().astype(float); y = d["avg_charge_minutes"].to_numpy().astype(float)
best = None
for k in np.arange(38, 45.01, 0.25):
    X = np.column_stack([np.ones_like(T), np.clip(T - k, 0, None)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    sse = float(((y - X @ coef) ** 2).sum())
    if best is None or sse < best[0]:
        best = (sse, k, coef)
sse25, k25, (a25, b25) = best
emp50 = float(y[(T >= 49) & (T <= 51)].mean())
say(f"Gen1 hinge on a 0.25 C knot grid: knot {k25:.2f} C, base {a25:.1f} min, slope {b25:.2f} min/C, {a25 + b25 * (50 - k25):.1f} min at 50 C "
    f"(R2 {1 - sse25 / float(((y - y.mean()) ** 2).sum()):.3f}) vs 0.5 C grid: knot {hinge.loc[0, 'knot_c']:.2f} C, slope "
    f"{hinge.loc[0, 'slope_min_per_c']:.2f} min/C -> quote 'about {b25:.1f}-{hinge.loc[0, 'slope_min_per_c']:.1f} min/C'; "
    f"empirical Gen1 mean at 49-51 C = {emp50:.1f} min (the 50 C level is robust)")

# exposure: share of station-hours above 40 C by generation x season, and by city in summer
expo = (H.group_by("charger_generation", "season")
        .agg(pl.len().alias("station_hours"), (pl.col("cabinet_temp_c") > HOT_C).mean().alias("hot_share"),
             pl.col("cabinet_temp_c").mean().alias("cabinet_mean"), pl.col("avg_charge_minutes").mean().alias("charge_min"))
        .sort("charger_generation", "season"))
say("Exposure by generation x season:\n" + str(expo))
expo_city = (H.filter(pl.col("season") == "Summer").group_by("city")
             .agg((pl.col("cabinet_temp_c") > HOT_C).mean().alias("summer_hot_share"),
                  pl.col("cabinet_temp_c").mean().alias("summer_cabinet_mean")).sort("summer_hot_share", descending=True))
expo_city_w = (H.filter(pl.col("season") == "Winter").group_by("city")
               .agg((pl.col("cabinet_temp_c") > HOT_C).mean().alias("winter_hot_share")))
expo_city = expo_city.join(expo_city_w, on="city")
expo_city.write_csv(TAB / "q3_thermal_exposure_city.csv")
say("Share of station-hours with cabinet > 40 C, summer vs winter, by city:\n" + str(expo_city))
gen1_summer = H.filter((pl.col("charger_generation") == "Gen1") & (pl.col("season") == "Summer"))
gen1_summer_hot = gen1_summer.filter(pl.col("cabinet_temp_c") > HOT_C)
say(f"Gen1 summer: {gen1_summer_hot.height/gen1_summer.height:.1%} of station-hours > 40 C; mean charge time in those hours "
    f"{gen1_summer_hot['avg_charge_minutes'].mean():.1f} min vs {gen1_summer.filter(pl.col('cabinet_temp_c') <= HOT_C)['avg_charge_minutes'].mean():.1f} min below 40 C")
# when in the day does it get hot
hot_by_hour = (H.filter(pl.col("season") == "Summer").group_by("hour").agg((pl.col("cabinet_temp_c") > HOT_C).mean().alias("hot_share")).sort("hour"))
pk = hot_by_hour.sort("hot_share", descending=True).head(1)
say(f"Summer hot-hour share peaks at hour {pk['hour'][0]} ({pk['hot_share'][0]:.1%}); at 06:00 = "
    f"{hot_by_hour.filter(pl.col('hour') == 6)['hot_share'][0]:.1%}")

# quarantines and chargers online
q = (H.with_columns((pl.col("cabinet_temp_c") > HOT_C).alias("hot"))
     .group_by("hot").agg(pl.len().alias("station_hours"), (pl.col("packs_quarantined") > 0).mean().alias("quarantine_hour_share"),
                          pl.col("packs_quarantined").sum().alias("quarantined_pack_hours"))).sort("hot")
say("Quarantines by hot / not hot hours (packs_quarantined is an hourly count, so its sum is quarantined PACK-HOURS, "
    "not distinct packs):\n" + str(q))
co = (H.group_by("charger_generation").agg(
    (pl.col("chargers_online") < pl.col("slots_2w") + pl.col("slots_3w")).mean().alias("share_hours_chargers_below_slots"),
    (pl.col("outage_minutes") > 0).mean().alias("share_hours_with_outage")).sort("charger_generation"))
say("Chargers online below installed slots / outages by generation:\n" + str(co))

# %% [markdown]
# ### 1b. Does heat-driven derating show up as stockouts?
# Station-day panel: for each city-day we compare a Gen1 station's stockout rate with the mean of Gen2/Gen3 stations in
# the same city on the same day (removes city-wide demand, weather and seasonality), as a function of the Gen1
# station's hours above 40 C that day.

# %%
SD = (events(["station_id", "date", "is_completed", "is_stockout", "is_abandoned", "is_system_error",
              "is_service_failure", "queue_wait_sec"])
      .group_by("station_id", "date")
      .agg(pl.len().alias("attempts"), pl.col("is_completed").sum().alias("completed"),
           pl.col("is_stockout").sum().alias("stockouts"), pl.col("is_abandoned").sum().alias("abandoned"),
           pl.col("is_system_error").sum().alias("sys_err"), pl.col("is_service_failure").sum().alias("failures"),
           pl.col("queue_wait_sec").filter(pl.col("is_completed")).sum().alias("qw_sum"))
      .collect(engine="streaming"))
HD = (H.group_by("station_id", "date").agg(pl.len().alias("ok_hours"), (pl.col("cabinet_temp_c") > HOT_C).sum().alias("hot_hours"),
                                           pl.col("avg_charge_minutes").mean().alias("charge_min")))
SD = (SD.join(st.select("station_id", "city", "zone", "charger_generation", "expansion_wave", "slots"), on="station_id", how="left")
      .join(HD, on=["station_id", "date"], how="left"))
say(f"station-day panel: {SD.height:,} rows; with telemetry {SD['ok_hours'].is_not_null().mean():.1%}")

ctrl = (SD.filter(pl.col("charger_generation") != "Gen1").group_by("city", "date")
        .agg((pl.col("stockouts").sum() / pl.col("attempts").sum()).alias("ctrl_stockout_rate"),
             (pl.col("failures").sum() / pl.col("attempts").sum()).alias("ctrl_failure_rate")))
g1 = (SD.filter((pl.col("charger_generation") == "Gen1") & pl.col("ok_hours").is_not_null() & (pl.col("ok_hours") >= 20))
      .join(ctrl, on=["city", "date"], how="inner")
      .with_columns(pl.when(pl.col("hot_hours") == 0).then(pl.lit("0 h"))
                    .when(pl.col("hot_hours") <= 3).then(pl.lit("1-3 h"))
                    .when(pl.col("hot_hours") <= 7).then(pl.lit("4-7 h"))
                    .otherwise(pl.lit("8+ h")).alias("hot_bucket")))
heat_stock = (g1.group_by("hot_bucket").agg(
    pl.len().alias("station_days"), (pl.col("stockouts").sum() / pl.col("attempts").sum()).alias("gen1_stockout_rate"),
    ((pl.col("ctrl_stockout_rate") * pl.col("attempts")).sum() / pl.col("attempts").sum()).alias("same_city_day_gen23_stockout_rate"),
    (pl.col("failures").sum() / pl.col("attempts").sum()).alias("gen1_failure_rate"),
    ((pl.col("ctrl_failure_rate") * pl.col("attempts")).sum() / pl.col("attempts").sum()).alias("same_city_day_gen23_failure_rate"),
    pl.col("charge_min").mean().alias("gen1_charge_min"))
    .with_columns((pl.col("gen1_stockout_rate") - pl.col("same_city_day_gen23_stockout_rate")).alias("excess_stockout_pp"))
    .sort("hot_bucket"))
heat_stock.write_csv(TAB / "q3_heat_vs_stockouts_gen1.csv")
say("Gen1 station-days: stockout rate vs same-city same-day Gen2/3 stations, by hours > 40 C:\n" + str(heat_stock))
# per-station test: regression of excess stockout on hot hours with station FE (within-station)
g1p = g1.with_columns((pl.col("stockouts") / pl.col("attempts") - pl.col("ctrl_stockout_rate")).alias("excess")).to_pandas()
m_heat = smf.wls("excess ~ hot_hours + C(station_id) + C(month)", data=g1p.assign(month=pd.to_datetime(g1p["date"]).dt.to_period("M").astype(str)),
                 weights=g1p["attempts"]).fit(cov_type="cluster", cov_kwds={"groups": g1p["station_id"]})
b_hh, se_hh = m_heat.params["hot_hours"], m_heat.bse["hot_hours"]
say(f"Within-station (station + month FE): each extra hour > 40 C adds {b_hh*100:.3f} pp excess stockout rate at a Gen1 station "
    f"(95% CI {100*(b_hh-1.96*se_hh):.3f} to {100*(b_hh+1.96*se_hh):.3f}; n={int(m_heat.nobs):,} station-days)")

# %% [markdown]
# ### Figure 1 (hero): thermal derating curves

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.45, 1]})
ax = axes[0]
cp = curve.filter(pl.col("station_hours") >= 200).to_pandas()
for g in ["Gen1", "Gen2", "Gen3"]:
    d = cp[cp.charger_generation == g]
    ax.plot(d.t_bin + 0.5, d.charge_min, color=GEN_COLORS[g], lw=2.2, marker=GEN_MARK[g], ms=4.5, label=g)
ax.axvline(HOT_C, color=PALETTE["muted"], ls="--", lw=1)
ax.text(HOT_C + 0.6, 272, "40 °C cabinet", ha="left", va="top", fontsize=9.5, color="#555555")
h1 = hinge.set_index("generation")
txt = "\n".join(f"{g}: {h1.loc[g].base_charge_min:.0f} min, +{h1.loc[g].slope_min_per_c:.1f} min/°C above {h1.loc[g].knot_c:.1f} °C"
                for g in ["Gen1", "Gen2", "Gen3"])
ax.text(0.02, 0.62, "Hinge fit (flat, then linear above knot):\n" + txt, transform=ax.transAxes, fontsize=9, color="#333333", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
ax.annotate(f"Gen1 at 50 °C: {h1.loc['Gen1'].charge_min_at_50c:.0f} min\n(capacity -{100*(1-h1.loc['Gen1'].capacity_at_50c_vs_base):.0f}%)",
            xy=(50.5, h1.loc["Gen1"].charge_min_at_50c), xytext=(51.5, 120), fontsize=9.2, color="#333333",
            arrowprops=dict(arrowstyle="-", color="#999999"))
ax.set_xlabel("Cabinet temperature (°C, 1-degree bins)")
ax.set_ylabel("Average charge time per pack (minutes)")
ax.set_ylim(0, 280); ax.set_xlim(4, 59)
ax.legend(title="Charger generation", loc="upper left")
ax.set_title("Gen1 charge time doubles by ~50 °C; Gen2 slows\nmodestly; Gen3 barely moves", fontsize=12.5)
ax = axes[1]
ec = expo_city.to_pandas().sort_values("summer_hot_share")
gen1_n = st_pd_tmp = st.group_by("city").agg((pl.col("charger_generation") == "Gen1").sum().alias("g1"), pl.len().alias("n")).to_pandas().set_index("city")
yc = np.arange(len(ec))
ax.barh(yc, 100 * ec.summer_hot_share, color=[PALETTE["accent"] if c in ["Delhi NCR", "Jaipur", "Hyderabad"] else PALETTE["muted"] for c in ec.city], height=0.65)
for yi, c, v in zip(yc, ec.city, ec.summer_hot_share):
    ax.text(100 * v + 0.5, yi, f"{100*v:.0f}%  |  Gen1: {int(gen1_n.loc[c,'g1'])} of {int(gen1_n.loc[c,'n'])} stations", va="center", fontsize=9.5, color="#333333")
ax.set_yticks(yc, ec.city)
ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
ax.set_xlim(0, 52)
ax.set_xlabel("% of summer (Mar-May) station-hours with cabinet > 40 °C")
ax.set_title("The hottest cities host most Gen1 chargers\n(winter: 0% of hours above 40 °C everywhere)", fontsize=12.5)
fig.suptitle("Battery turnaround: thermal derating by charger generation", x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.04)
fig.text(0.01, 0.985, f"Hourly telemetry, telemetry_status = ok only ({H.height:,} station-hours, Jan 2024 - Jun 2025). "
         "Curves = mean avg_charge_minutes per 1 °C bin (bins with >= 200 station-hours).", fontsize=9.5, color="#555555")
save_fig(fig, "q3_01_thermal_derating")
say("saved q3_01_thermal_derating")

# %% [markdown]
# ## 2. Service performance and complaints by station attributes
# Station-month panel (stations with >= 15 active days and >= 100 attempts in the month). Raw comparisons are
# confounded: Gen1 exists only at Launch stations, Wave2 is all Gen3 and opened in the 2025 summer, and busy sites fail
# more. So we fit WLS (weights = attempts) with **city and month fixed effects**, a **utilisation control**
# (log attempts per slot-day) and all attributes together; SEs are clustered by station. Spec B adds the station-month
# share of hours with cabinet > 40 C interacted with generation to test whether any Gen1 penalty is purely thermal.
# Complaints = support tickets linked to the station (95% of tickets carry a station_id) per 1,000 completed swaps.

# %%
SM = station_month().with_columns(
    (pl.col("stockouts") + pl.col("abandoned") + pl.col("system_errors")).alias("failures"),
    (pl.col("attempts") / (pl.col("active_days") * (pl.col("slots_2w") + pl.col("slots_3w")))).alias("att_per_slot_day"),
    (pl.col("slots_2w") + pl.col("slots_3w")).alias("slots"),
).join(st.select("station_id", "zone"), on="station_id", how="left")
hot_m = H.group_by("station_id", "month").agg((pl.col("cabinet_temp_c") > HOT_C).mean().alias("hot_share"),
                                              pl.col("avg_charge_minutes").mean().alias("charge_min"))
TK_all = tickets()
say(f"tickets carrying a station_id: {TK_all['station_id'].is_not_null().mean():.1%} of {TK_all.height:,}")
TK = TK_all.filter(pl.col("in_window") & ~pl.col("is_test") & pl.col("station_id").is_not_null())
tk_m = TK.group_by("station_id", "month").agg(
    pl.len().alias("tickets"),
    pl.col("category").is_in(["no_battery_available", "long_queue"]).sum().alias("tickets_service"),
    (pl.col("category") == "no_battery_available").sum().alias("tickets_nobattery"))
SM = (SM.join(hot_m, on=["station_id", "month"], how="left").join(tk_m, on=["station_id", "month"], how="left")
      .with_columns(pl.col("tickets").fill_null(0), pl.col("tickets_service").fill_null(0), pl.col("tickets_nobattery").fill_null(0)))
say(f"tickets in window with a real station: {TK.height:,}; matched to station-months: {int(SM['tickets'].sum()):,}")
P = (SM.filter((pl.col("active_days") >= 15) & (pl.col("attempts") >= 100))
     .with_columns((100 * pl.col("failures") / pl.col("attempts")).alias("fail_pp"),
                   (100 * pl.col("stockouts") / pl.col("attempts")).alias("stockout_pp"),
                   (1000 * pl.col("tickets") / pl.col("completed")).alias("tickets_per_1k"),
                   (1000 * pl.col("tickets_service") / pl.col("completed")).alias("svc_tickets_per_1k"),
                   pl.col("att_per_slot_day").log().alias("log_util"),
                   pl.col("month").cast(pl.String).alias("month_s"))
     .to_pandas())
say(f"station-month panel for models: {len(P):,} rows, {P.station_id.nunique()} stations")


# --- raw descriptives by attribute (pooled rates), full period and Apr-Jun 2025 (all 150 stations live)
def describe_by(df, col):
    g = df.groupby(col).agg(stations=("station_id", "nunique"), attempts=("attempts", "sum"), failures=("failures", "sum"),
                            stockouts=("stockouts", "sum"), sys_err=("system_errors", "sum"), completed=("completed", "sum"),
                            tickets=("tickets", "sum"), slot_days=("slots", lambda s: (s * df.loc[s.index, "active_days"]).sum()),
                            qw=("q_wait_completed_mean", lambda s: np.average(s.fillna(0), weights=df.loc[s.index, "completed"])))
    g["failure_rate_pct"] = 100 * g.failures / g.attempts
    g["stockout_rate_pct"] = 100 * g.stockouts / g.attempts
    g["system_error_rate_pct"] = 100 * g.sys_err / g.attempts
    g["tickets_per_1k_swaps"] = 1000 * g.tickets / g.completed
    g["attempts_per_slot_day"] = g.attempts / g.slot_days
    g = g.rename(columns={"qw": "queue_wait_sec"}).reset_index().rename(columns={col: "level"})
    g.insert(0, "attribute", col)
    return g[["attribute", "level", "stations", "attempts", "failure_rate_pct", "stockout_rate_pct", "system_error_rate_pct",
              "queue_wait_sec", "tickets_per_1k_swaps", "attempts_per_slot_day"]]


ATTRS = ["charger_generation", "location_type", "host_type", "connectivity_tier", "expansion_wave", "city"]
desc_all = pd.concat([describe_by(P, c) for c in ATTRS]).assign(period="Jan24-Jun25")
P_q2 = P[P.month_s >= "2025-04-01"]
desc_q2 = pd.concat([describe_by(P_q2, c) for c in ATTRS]).assign(period="Apr-Jun25")
desc = pd.concat([desc_all, desc_q2])
desc.to_csv(TAB / "q3_attribute_descriptives.csv", index=False)
say("Raw descriptives by attribute (full period):\n" + desc_all.round(2).to_string(index=False))
say("Raw descriptives by attribute (Apr-Jun 2025):\n" + desc_q2.round(2).to_string(index=False))
# confounding check: Gen1 vs Gen2 within Launch
lc = P[P.expansion_wave == "Launch"].groupby("charger_generation").apply(lambda d: pd.Series({
    "stations": d.station_id.nunique(), "attempts_per_slot_day": d.attempts.sum() / (d.slots * d.active_days).sum(),
    "share_commercial_or_market": d.drop_duplicates("station_id").location_type.isin(["commercial_hub", "market"]).mean(),
    "share_delhi_jaipur_hyd": d.drop_duplicates("station_id").city.isin(["Delhi NCR", "Jaipur", "Hyderabad"]).mean()}))
say("Confounding check within Launch stations (Gen1 vs Gen2):\n" + lc.round(3).to_string())

# --- multivariable models
RHS = ("C(charger_generation, Treatment('Gen2')) + C(location_type, Treatment('commercial_hub')) + "
       "C(host_type, Treatment('kirana')) + C(connectivity_tier, Treatment('good')) + C(expansion_wave, Treatment('Launch')) + "
       "log_util + C(city) + C(month_s)")


def fit(y, w, rhs=RHS, data=None):
    d = (P if data is None else data).dropna(subset=[y, w])
    return smf.wls(f"{y} ~ {rhs}", data=d, weights=d[w]).fit(cov_type="cluster", cov_kwds={"groups": d["station_id"]})


models = {"failure_rate_pp": fit("fail_pp", "attempts"), "stockout_rate_pp": fit("stockout_pp", "attempts"),
          "queue_wait_sec": fit("q_wait_completed_mean", "completed"), "tickets_per_1k": fit("tickets_per_1k", "completed"),
          "service_tickets_per_1k": fit("svc_tickets_per_1k", "completed")}
P["hot_pp"] = 100 * P["hot_share"]
m_hot = fit("fail_pp", "attempts", RHS + " + hot_pp + C(charger_generation, Treatment('Gen2')):hot_pp")


def clean_name(t):
    for a, b in [("C(charger_generation, Treatment('Gen2'))[T.", "Generation: "), ("C(location_type, Treatment('commercial_hub'))[T.", "Location: "),
                 ("C(host_type, Treatment('kirana'))[T.", "Host: "), ("C(connectivity_tier, Treatment('good'))[T.", "Connectivity: "),
                 ("C(expansion_wave, Treatment('Launch'))[T.", "Wave: "), ("C(city)[T.", "City: ")]:
        t = t.replace(a, b)
    return t.replace("]", "")


coef_rows = []
for name, m in list(models.items()) + [("failure_rate_pp_thermal_spec", m_hot)]:
    for t in m.params.index:
        if t.startswith("C(month_s)") or t == "Intercept":
            continue
        coef_rows.append({"outcome": name, "term": clean_name(t), "coef": m.params[t], "se": m.bse[t], "p": m.pvalues[t],
                          "ci_lo": m.params[t] - 1.96 * m.bse[t], "ci_hi": m.params[t] + 1.96 * m.bse[t],
                          "n": int(m.nobs), "r2": m.rsquared})
COEF = pd.DataFrame(coef_rows)
COEF.to_csv(TAB / "q3_station_attribute_model_coefs.csv", index=False)
show = COEF[COEF.outcome.isin(["failure_rate_pp", "queue_wait_sec", "tickets_per_1k"])].pivot_table(
    index="term", columns="outcome", values=["coef", "p"]).round(3)
say(f"Multivariable WLS (city + month FE, cluster SE by station; n={int(models['failure_rate_pp'].nobs):,} station-months):\n" + show.to_string())
th = COEF[COEF.outcome == "failure_rate_pp_thermal_spec"].set_index("term")
say("Thermal spec (failure pp; adds hot_pp = % of station-hours > 40 C, interacted with generation):\n"
    + th[["coef", "ci_lo", "ci_hi", "p"]].round(3).to_string())
s_g2 = th.loc["hot_pp", "coef"]; s_g1 = s_g2 + th.loc["Generation: Gen1:hot_pp", "coef"]; s_g3 = s_g2 + th.loc["Generation: Gen3:hot_pp", "coef"]
say(f"Failure-rate slope per +1 pp of hot station-hours: Gen1 {s_g1:.3f} pp, Gen2 {s_g2:.3f} pp, Gen3 {s_g3:.3f} pp "
    f"(Gen1/Gen2 ratio {s_g1/s_g2:.2f}x)")
gen_city = st.group_by("city", "charger_generation").len().pivot(on="charger_generation", index="city", values="len").fill_null(0).sort("city")
say("Stations by city x generation (Gen1 sits in the hottest cities):\n" + str(gen_city))
# implied Gen1 vs Gen2 gap at typical hot-share levels
for hs in [0, 10, 20, 30]:
    gap = th.loc["Generation: Gen1", "coef"] + th.loc["Generation: Gen1:hot_pp", "coef"] * hs
    say(f"  implied Gen1 - Gen2 failure gap at {hs}% hot station-hours: {gap:+.2f} pp")
say(f"R2 failure model {models['failure_rate_pp'].rsquared:.3f}; thermal spec {m_hot.rsquared:.3f}")
# tickets track failures: complaints per +1 pp failure rate
Pt = P.dropna(subset=["tickets_per_1k"])
m_tk = smf.wls("tickets_per_1k ~ fail_pp + C(city) + C(month_s)", data=Pt, weights=Pt["completed"]).fit(
    cov_type="cluster", cov_kwds={"groups": Pt["station_id"]})
say(f"Tickets per 1K swaps rise by {m_tk.params['fail_pp']:.3f} per +1 pp failure rate (95% CI {m_tk.conf_int().loc['fail_pp',0]:.3f} to "
    f"{m_tk.conf_int().loc['fail_pp',1]:.3f}); mean tickets per 1K = {1000*P.tickets.sum()/P.completed.sum():.2f}")
say(f"  => +1 pp failure rate = 10 extra failed attempts per 1,000 attempts, so this is about ONE extra ticket per "
    f"{10/m_tk.params['fail_pp']:.1f} extra failed attempts (not one ticket per failure)")
tk_att = P_q2.groupby("charger_generation").apply(lambda d: 1000 * d.tickets.sum() / d.attempts.sum())
say("Apr-Jun 2025 tickets per 1K ATTEMPTS (removes the completed-swaps denominator effect): "
    + ", ".join(f"{k} {v:.2f}" for k, v in tk_att.items()))
# counterfactual: hot-city Gen1 station-months if they had Gen2 (in-support) or Gen3 (extrapolated) thermal behaviour
cfp = P[(P.charger_generation == "Gen1") & P.city.isin(["Delhi NCR", "Jaipur", "Hyderabad"]) & P.month_s.str[5:7].isin(["04", "05", "06"])].copy()
cf_rows = []
for alt in ["Gen2", "Gen3"]:
    pred_act = m_hot.predict(cfp)
    pred_alt = m_hot.predict(cfp.assign(charger_generation=alt))
    for yr in ["2024", "2025"]:
        msk = cfp.month_s.str[:4] == yr
        avoided = float(((pred_act - pred_alt)[msk] * cfp.attempts[msk] / 100).sum())
        d_pp = float(np.average((pred_act - pred_alt)[msk], weights=cfp.attempts[msk]))
        cf_rows.append({"counterfactual": f"Gen1 -> {alt}", "period": f"Apr-Jun {yr}", "stations": cfp[msk].station_id.nunique(),
                        "attempts": int(cfp.attempts[msk].sum()), "actual_fail_pct": 100 * cfp.failures[msk].sum() / cfp.attempts[msk].sum(),
                        "failure_pp_reduction": d_pp, "failed_attempts_avoided": avoided,
                        "tickets_avoided": m_tk.params["fail_pp"] * d_pp * cfp.completed[msk].sum() / 1000})
CFT = pd.DataFrame(cf_rows)
# naive empirical anchor: actual Apr-Jun 2025 failure rates of Gen2 (Launch) and Gen3 stations in the SAME hot cities
# (Gen3 is inside heat support: 19 Gen3 stations operate in Delhi NCR / Jaipur / Hyderabad; only Launch-site support is missing)
smq = SM.filter((pl.col("month") >= pl.date(2025, 4, 1)) & pl.col("city").is_in(["Delhi NCR", "Jaipur", "Hyderabad"])).to_pandas()
net_fail_25 = float(SM.filter(pl.col("month") >= pl.date(2025, 4, 1))["failures"].sum())
g1q = smq[smq.charger_generation == "Gen1"]
r_g1 = 100 * g1q.failures.sum() / g1q.attempts.sum()
naive = {}
for alt, d in [("Gen2", smq[(smq.charger_generation == "Gen2") & (smq.expansion_wave == "Launch")]), ("Gen3", smq[smq.charger_generation == "Gen3"])]:
    r_alt = 100 * d.failures.sum() / d.attempts.sum()
    av = (r_g1 - r_alt) / 100 * g1q.attempts.sum()
    naive[alt] = (r_g1 - r_alt, av)
    say(f"Naive same-city check, Apr-Jun 2025: hot-city Gen1 ({g1q.station_id.nunique()} stations) {r_g1:.2f}% vs hot-city "
        f"{'Launch Gen2' if alt == 'Gen2' else 'Gen3'} ({d.station_id.nunique()} stations) {r_alt:.2f}% -> -{r_g1 - r_alt:.2f} pp, "
        f"{av:,.0f} failed attempts avoided ({av / net_fail_25:.1%} of network failures)")
for alt in ["Gen2", "Gen3"]:
    msk = (CFT.counterfactual == f"Gen1 -> {alt}") & (CFT.period == "Apr-Jun 2025")
    CFT.loc[msk, "naive_same_city_pp"] = naive[alt][0]
    CFT.loc[msk, "naive_same_city_avoided"] = naive[alt][1]
CFT.to_csv(TAB / "q3_gen1_upgrade_counterfactual.csv", index=False)
say("Counterfactual for the 36 hot-city Gen1 stations, Apr-Jun (thermal-spec model; naive same-city anchor for 2025):\n"
    + CFT.round(2).to_string(index=False))
qw_range = desc_all.groupby("attribute").queue_wait_sec.agg(["min", "max"])
say(f"Queue wait of completed swaps across all attribute levels: {desc_all.queue_wait_sec.min():.1f}-{desc_all.queue_wait_sec.max():.1f} s; "
    f"system-error rate {desc_all.system_error_rate_pct.min():.2f}-{desc_all.system_error_rate_pct.max():.2f}% (flat - not a differentiator)")

# %% [markdown]
# ### Figure 2: what actually separates stations - heat x generation, not location, host or connectivity

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.55})
ax = axes[0]
P["hot_bin"] = pd.cut(P.hot_pp, [-0.1, 0.5, 5, 10, 15, 20, 25, 30, 40, 60], labels=False)
for g in ["Gen1", "Gen2", "Gen3"]:
    d = P[P.charger_generation == g].groupby("hot_bin").apply(lambda x: pd.Series({
        "hot": np.average(x.hot_pp, weights=x.attempts), "fail": 100 * x.failures.sum() / x.attempts.sum(), "n": len(x)}))
    d = d[d.n >= 5]
    ax.plot(d.hot, d.fail, color=GEN_COLORS[g], marker=GEN_MARK[g], lw=2, ms=6, label=f"{g} ({P[P.charger_generation==g].station_id.nunique()} stations)")
ax.set_xlabel("% of station-hours in the month with cabinet > 40 °C")
ax.set_ylabel("Service-failure rate (% of attempts)")
ax.set_title("Failures climb with heat, steeply at Gen1 stations", fontsize=12.5)
ax.legend(loc="upper left", title="Charger generation (station-months, binned)")
b1 = th.loc["Generation: Gen1:hot_pp", "coef"]; b0 = th.loc["hot_pp", "coef"]
ax.text(0.03, 0.68, f"Model (city + month FE,\nutilisation control): per +1 pp\nof hot station-hours, failures rise\nGen1 +{b0+b1:.2f} pp, "
        f"Gen2 +{b0:.2f} pp,\nGen3 +{b0+th.loc['Generation: Gen3:hot_pp','coef']:.2f} pp; Gen1 with no hot\nhours: "
        f"{th.loc['Generation: Gen1','coef']:+.2f} pp vs Gen2", transform=ax.transAxes, ha="left", va="top", fontsize=9.3, color="#333333",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
ax = axes[1]
order = ["Generation: Gen1", "Generation: Gen3", "Wave: Wave1_2024H2", "Wave: Wave2_2025H1", "City: Delhi NCR", "City: Jaipur",
         "City: Hyderabad", "Location: market", "Location: residential", "Location: highway_fuel_pump", "Location: tech_park",
         "Location: transit_hub", "Host: fuel_retailer", "Host: mall_parking", "Host: owned_lot", "Connectivity: fair", "Connectivity: poor"]
yy = np.arange(len(order))[::-1].astype(float)
for spec, off, col, mk, lab in [("failure_rate_pp", 0.17, PALETTE["accent"], "o", "Base model"),
                                ("failure_rate_pp_thermal_spec", -0.17, PALETTE["primary"], "D", "+ heat x generation")]:
    cf = COEF[COEF.outcome == spec].set_index("term").loc[order].reset_index()
    sig = (cf.p < 0.05).values
    ax.errorbar(cf.coef, yy + off, xerr=1.96 * cf.se, fmt="none", ecolor=col, alpha=0.6, elinewidth=1.3, capsize=2.5)
    ax.scatter(cf.coef[sig], (yy + off)[sig], color=col, marker=mk, s=42, zorder=3, label=f"{lab}: p < 0.05")
    ax.scatter(cf.coef[~sig], (yy + off)[~sig], facecolor="white", edgecolor=col, marker=mk, s=42, zorder=3, label=f"{lab}: n.s.")
ax.axvline(0, color="#555555", lw=1)
labels = [t.replace("Generation: ", "Gen: ").replace("_2024H2", "").replace("_2025H1", "").replace("_", " ") for t in order]
ax.set_yticks(yy, labels); ax.set_ylim(-0.8, len(order) - 0.2)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Difference in failure rate vs reference (pp, 95% CI)\nrefs: Gen2, Launch, Bengaluru, commercial hub, kirana, good connectivity\n"
              "(Gen1 point in the heat model = Gen1 gap at 0% hot hours)")
ax.set_title("Gen1's penalty appears only in heat; Wave-2 and city\ngaps vanish once heat x generation is modelled", fontsize=12.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, fontsize=8.8)
fig.suptitle(f"Heat x charger generation explains the station failure gaps (model R² {models['failure_rate_pp'].rsquared:.2f} -> {m_hot.rsquared:.2f}); "
             f"location, host and connectivity do not", x=0.01, ha="left", fontsize=13.5, fontweight="bold", y=1.055)
fig.text(0.01, 0.975, f"Station-month WLS, {int(models['failure_rate_pp'].nobs):,} station-months (>= 15 active days, >= 100 attempts), weights = attempts, "
         "city + month fixed effects, log attempts per slot-day, SE clustered by station.", fontsize=9.5, color="#555555")
save_fig(fig, "q3_02_attributes_vs_failures")
say("saved q3_02_attributes_vs_failures")

# %% [markdown]
# ## 3. Uneven distribution: supply vs demand by city and zone
# Supply = stations and slots; demand = riders by **home zone** (latent demand, from riders.home_zone) and attempts
# at stations in the zone (realised demand, capped by where stations exist). "Need" before Wave 1 = Jan-Jun 2024.

# %%
st_pd = st.to_pandas()
zone = (st_pd.assign(L_slots=np.where(st_pd.expansion_wave == "Launch", st_pd.slots, 0),
                     W1_slots=np.where(st_pd.expansion_wave == "Wave1_2024H2", st_pd.slots, 0),
                     W2_slots=np.where(st_pd.expansion_wave == "Wave2_2025H1", st_pd.slots, 0),
                     L_st=(st_pd.expansion_wave == "Launch").astype(int), W1_st=(st_pd.expansion_wave == "Wave1_2024H2").astype(int),
                     W2_st=(st_pd.expansion_wave == "Wave2_2025H1").astype(int))
        .groupby(["city", "zone"])[["L_st", "W1_st", "W2_st", "L_slots", "W1_slots", "W2_slots"]].sum().reset_index())
RD = riders()
rz = (RD.group_by("home_zone").agg(pl.len().alias("riders_registered"),
                                   (pl.col("first_attempt_ts") < pl.datetime(2024, 7, 1)).sum().alias("riders_active_h1_24"),
                                   (pl.col("last_attempt_ts") >= pl.datetime(2025, 4, 1)).sum().alias("riders_active_q2_25"))
      .rename({"home_zone": "zone"}).to_pandas())
SMp = SM.to_pandas()
SMp["period"] = np.select([SMp.month < pd.Timestamp("2024-07-01"), (SMp.month >= pd.Timestamp("2024-07-01")) & (SMp.month < pd.Timestamp("2025-01-01")),
                           SMp.month >= pd.Timestamp("2025-04-01")], ["H1_24", "H2_24", "Q2_25"], "other")
SMp["slot_days"] = SMp.slots * SMp.active_days


def zone_period(df, per, tag):
    d = df[df.period == per].groupby("zone").agg(att=("attempts", "sum"), fail=("failures", "sum"), so=("stockouts", "sum"),
                                                 comp=("completed", "sum"), sd=("slot_days", "sum"))
    return pd.DataFrame({f"attempts_{tag}": d.att, f"failures_{tag}": d.fail, f"failure_rate_{tag}": 100 * d.fail / d.att,
                         f"stockout_rate_{tag}": 100 * d.so / d.att, f"att_per_slot_day_{tag}": d.att / d.sd, f"completed_{tag}": d.comp})


zone = (zone.merge(rz, on="zone", how="left")
        .merge(zone_period(SMp, "H1_24", "h1_24"), left_on="zone", right_index=True, how="left")
        .merge(zone_period(SMp, "H2_24", "h2_24"), left_on="zone", right_index=True, how="left")
        .merge(zone_period(SMp, "Q2_25", "q2_25"), left_on="zone", right_index=True, how="left"))
# Q2-24 legacy failure rate (same season as Q2-25) for the before/after view
q2_24 = SMp[(SMp.month >= pd.Timestamp("2024-04-01")) & (SMp.month < pd.Timestamp("2024-07-01"))]
zq = q2_24.groupby("zone").agg(a=("attempts", "sum"), f=("failures", "sum"))
zone["failure_rate_q2_24"] = zone.zone.map(100 * zq.f / zq.a)
zone["failures_q2_24"] = zone.zone.map(zq.f)
zone["new_slots"] = zone.W1_slots + zone.W2_slots
zone["riders_per_legacy_slot"] = np.where(zone.L_slots > 0, zone.riders_active_h1_24 / zone.L_slots.replace(0, np.nan), np.nan)
zone["share_riders_h1_24"] = zone.riders_active_h1_24 / zone.riders_active_h1_24.sum()
zone["share_legacy_slots"] = zone.L_slots / zone.L_slots.sum()
zone["share_failures_h1_24"] = zone.failures_h1_24.fillna(0) / zone.failures_h1_24.sum()
zone["share_new_slots"] = zone.new_slots / zone.new_slots.sum()
zone["share_W1_slots"] = zone.W1_slots / zone.W1_slots.sum()
zone["share_W2_slots"] = zone.W2_slots / zone.W2_slots.sum()
zone["supply_gap_pp"] = 100 * (zone.share_riders_h1_24 - zone.share_legacy_slots)
zone = zone.sort_values("failure_rate_h1_24", ascending=False, na_position="first")
zone.to_csv(TAB / "q3_zone_supply_demand.csv", index=False)
say("Zone supply-demand table (sorted by H1-2024 failure rate):\n" + zone[[
    "zone", "L_st", "W1_st", "W2_st", "L_slots", "new_slots", "riders_active_h1_24", "riders_per_legacy_slot", "failure_rate_h1_24",
    "att_per_slot_day_h1_24", "supply_gap_pp", "failure_rate_q2_24", "failure_rate_q2_25"]].round(2).to_string(index=False))
city = zone.groupby("city").agg(stations=("L_st", "sum"), W1=("W1_st", "sum"), W2=("W2_st", "sum"), legacy_slots=("L_slots", "sum"),
                                new_slots=("new_slots", "sum"), riders_h1_24=("riders_active_h1_24", "sum"), riders_q2_25=("riders_active_q2_25", "sum"),
                                attempts_h1_24=("attempts_h1_24", "sum"), failures_h1_24=("failures_h1_24", "sum"))
city["riders_per_legacy_slot"] = city.riders_h1_24 / city.legacy_slots
city["failure_rate_h1_24"] = 100 * city.failures_h1_24 / city.attempts_h1_24
city["share_riders"] = city.riders_h1_24 / city.riders_h1_24.sum(); city["share_legacy_slots"] = city.legacy_slots / city.legacy_slots.sum()
city["share_new_slots"] = city.new_slots / city.new_slots.sum(); city["share_failures_h1_24"] = city.failures_h1_24 / city.failures_h1_24.sum()
city.to_csv(TAB / "q3_city_supply_demand.csv")
say("City supply-demand:\n" + city.round(3).to_string())
no_station_zones = zone[zone.L_st == 0]
say(f"Zones with resident riders but NO Launch station: {no_station_zones.zone.tolist()} "
    f"({int(no_station_zones.riders_active_h1_24.sum())} riders active in H1-2024); new stations there: "
    f"W1={int(no_station_zones.W1_st.sum())}, W2={int(no_station_zones.W2_st.sum())}")

# %% [markdown]
# ## 4. Expansion waves: placed where riders needed them?
# **Need** before each wave = legacy failure rate and failures per zone. Wave 1 (opened Aug-Oct 2024) uses Jan-Jun 2024.
# Wave 2 (opened Feb-Apr 2025) uses the **same-season Apr-Jun 2024** rate: failures are an Apr-Jun phenomenon, and in the
# off-season (Jul-Dec 2024) every zone sits at about 3.9%, so ranking zones on that period ranks noise (checked below).
# We compare each wave's share of new slots going to the highest-need zones with those zones' share of failures, and
# rank-correlate need with new slots.

# %%
zl = zone[zone.L_st > 0].copy()
zl["need_tercile_h1"] = pd.qcut(zl.failure_rate_h1_24.rank(method="first"), 3, labels=["low", "mid", "high"])
zl["need_tercile_q2"] = pd.qcut(zl.failure_rate_q2_24.rank(method="first"), 3, labels=["low", "mid", "high"])
pl_rows = []
for wave, col, need, fcol, rcol, need_lbl in [("Wave1", "W1_slots", "need_tercile_h1", "failures_h1_24", "failure_rate_h1_24", "Jan-Jun 2024"),
                                              ("Wave2", "W2_slots", "need_tercile_q2", "failures_q2_24", "failure_rate_q2_24", "Apr-Jun 2024")]:
    for t in ["high", "mid", "low"]:
        sub = zl[zl[need] == t]
        pl_rows.append({"wave": wave, "need_period": need_lbl, "need_tercile": t, "zones": len(sub),
                        "share_of_failures": sub[fcol].sum() / zl[fcol].sum(),
                        "share_of_legacy_slots": sub.L_slots.sum() / zl.L_slots.sum(), "share_of_new_slots": sub[col].sum() / zone[col].sum(),
                        "new_slots": int(sub[col].sum())})
    rho, p = stats.spearmanr(zl[rcol], zl[col] / zl.L_slots)
    rho2, p2 = stats.spearmanr(zl[fcol], zl[col])
    say(f"{wave} (need = {need_lbl}): Spearman(zone failure rate, new slots per legacy slot) = {rho:.2f} (p={p:.2f}); "
        f"Spearman(zone failures, new slots) = {rho2:.2f} (p={p2:.2f}); n={len(zl)} zones with legacy stations")
    say(f"{wave}: slots placed in zones with no legacy station: {int(zone.loc[zone.L_st == 0, col].sum())} of {int(zone[col].sum())}")
placement = pd.DataFrame(pl_rows)
placement.to_csv(TAB / "q3_wave_placement_vs_need.csv", index=False)
say("Placement vs need terciles (Wave 2 need = same-season Apr-Jun 2024):\n" + placement.round(3).to_string(index=False))
# Wave-2 robustness, and why the off-season Jul-Dec 2024 rate is NOT used as Wave-2 need
r_h1, p_h1 = stats.spearmanr(zl.failure_rate_h1_24, zl.W2_slots / zl.L_slots)
r_h2, p_h2 = stats.spearmanr(zl.failure_rate_h2_24, zl.W2_slots / zl.L_slots)
r_h1h2 = stats.spearmanr(zl.failure_rate_h1_24, zl.failure_rate_h2_24)[0]
say(f"Wave2 robustness: need = Jan-Jun 2024 rate -> rho {r_h1:.2f} (p={p_h1:.2f}). NOT USED: the off-season Jul-Dec 2024 rate gives "
    f"rho {r_h2:.2f} (p={p_h2:.2f}) but zone rates there are flat noise: mean {zl.failure_rate_h2_24.mean():.2f}%, SD "
    f"{zl.failure_rate_h2_24.std():.2f} pp (Jan-Jun 2024 SD {zl.failure_rate_h1_24.std():.2f} pp, Apr-Jun 2024 SD "
    f"{zl.failure_rate_q2_24.std():.2f} pp); Spearman(Jan-Jun 2024 rate, Jul-Dec 2024 rate) = {r_h1h2:.2f}")

# site types: where waves went vs where legacy failures were
lt = (st_pd.groupby(["location_type", "expansion_wave"]).size().unstack(fill_value=0))
legacy_fail_by_loc = SMp[(SMp.expansion_wave == "Launch") & (SMp.period == "H1_24")].groupby("location_type").agg(
    a=("attempts", "sum"), f=("failures", "sum"), sd=("slot_days", "sum"))
lt["legacy_failure_rate_h1_24"] = 100 * legacy_fail_by_loc.f / legacy_fail_by_loc.a
lt["legacy_att_per_slot_day_h1_24"] = legacy_fail_by_loc.a / legacy_fail_by_loc.sd
lt["share_of_legacy_failures_h1_24"] = legacy_fail_by_loc.f / legacy_fail_by_loc.f.sum()
leg_slots_loc = st_pd[st_pd.expansion_wave == "Launch"].groupby("location_type").slots.sum()
lt["share_of_legacy_slots"] = leg_slots_loc / leg_slots_loc.sum()
lt.to_csv(TAB / "q3_wave_site_types.csv")
say("Site types by wave vs legacy demand/failures by site type (H1-2024):\n" + lt.round(3).to_string())
say("  NOTE: a site type's share of legacy failures mirrors its share of legacy slots (few legacy fuel-pump/residential sites), "
    f"while legacy failure RATES are similar across site types ({lt.legacy_failure_rate_h1_24.min():.1f}-{lt.legacy_failure_rate_h1_24.max():.1f}%) "
    "-> failure shares say nothing about need by site type")
rent = st_pd.assign(rent_per_slot=st_pd.monthly_rent_inr / st_pd.slots, opex=st_pd.monthly_rent_inr + st_pd.monthly_maintenance_inr).groupby("expansion_wave").agg(
    stations=("station_id", "count"), rent_mean=("monthly_rent_inr", "mean"), opex_mean=("opex", "mean"), rent_per_slot=("rent_per_slot", "mean"),
    slots_mean=("slots", "mean"))
say("Rent / opex by wave:\n" + rent.round(0).to_string())
hot_cities = ["Delhi NCR", "Jaipur", "Hyderabad"]
say(f"Hot cities (Delhi NCR, Jaipur, Hyderabad): {city.loc[hot_cities,'share_failures_h1_24'].sum():.1%} of H1-2024 failures, "
    f"{city.loc[hot_cities,'share_riders'].sum():.1%} of active riders, {city.loc[hot_cities,'share_legacy_slots'].sum():.1%} of legacy slots, "
    f"{city.loc[hot_cities,'share_new_slots'].sum():.1%} of Wave1+Wave2 slots (Wave 1 {zone[zone.city.isin(hot_cities)].W1_slots.sum()/zone.W1_slots.sum():.1%}, "
    f"Wave 2 {zone[zone.city.isin(hot_cities)].W2_slots.sum()/zone.W2_slots.sum():.1%}); Gen1 stations there: "
    f"{int(st.filter(pl.col('city').is_in(hot_cities) & (pl.col('charger_generation') == 'Gen1')).height)} of 51")

# %% [markdown]
# ### Figure 3 (hero): expansion placement vs need, by zone

# %%
zp = zone.copy()
zp["label"] = zp.zone.str.replace("-", " | ", n=1)
zp = pd.concat([zp[zp.L_st == 0], zp[zp.L_st > 0].sort_values("failure_rate_h1_24", ascending=False)])
fig, axes = plt.subplots(1, 3, figsize=(15, 10), sharey=True, gridspec_kw={"width_ratios": [1.1, 1, 1.25], "wspace": 0.08})
y = np.arange(len(zp))[::-1]
ax = axes[0]
cols = [PALETTE["accent"] if c in hot_cities else PALETTE["muted"] for c in zp.city]
ax.barh(y, zp.failure_rate_h1_24.fillna(0), color=cols, height=0.72)
for yi, v, ls in zip(y, zp.failure_rate_h1_24, zp.L_st):
    ax.text((0 if np.isnan(v) else v) + 0.2, yi, "no station at launch" if ls == 0 else f"{v:.1f}%", va="center", fontsize=9, color="#333333")
ax.set_yticks(y, zp.label, fontsize=9.5)
ax.set_xlabel("Service-failure rate, Jan-Jun 2024 (%)")
ax.set_title("Need before expansion:\nlegacy failure rate", fontsize=12)
from matplotlib.patches import Patch  # noqa: E402
ax.legend(handles=[Patch(color=PALETTE["accent"], label="DEL / JAI / HYD (hot)"), Patch(color=PALETTE["muted"], label="BLR / PUN / MUM")],
          loc="lower right", bbox_to_anchor=(1.02, 0.0), fontsize=8.5, title="City group", title_fontsize=8.8)
ax = axes[1]
rps = zp.riders_per_legacy_slot.fillna(0)
ax.barh(y, rps, color="#6C7A89", height=0.72)
for a_ in axes:
    a_.grid(axis="y", visible=False); a_.set_axisbelow(True)
for yi, v, r, ls in zip(y, rps, zp.riders_active_h1_24, zp.L_st):
    ax.text(v + 0.3, yi, f"{int(r)} riders, 0 slots" if ls == 0 else f"{v:.1f}", va="center", fontsize=9, color="#333333")
ax.set_xlabel("Active riders living in zone per legacy slot (H1-2024)")
ax.set_title("Demand vs supply:\nresident riders per slot", fontsize=12)
ax = axes[2]
ax.barh(y, zp.W1_slots, color=WAVE_COLORS["Wave1_2024H2"], height=0.72, label=WAVE_LABEL["Wave1_2024H2"])
ax.barh(y, zp.W2_slots, left=zp.W1_slots, color=WAVE_COLORS["Wave2_2025H1"], height=0.72, label=WAVE_LABEL["Wave2_2025H1"])
for yi, a, b in zip(y, zp.W1_slots, zp.W2_slots):
    ax.text(a + b + 1, yi, f"{int(a+b)}", va="center", fontsize=9, color="#333333")
ax.set_xlabel("New slots added (2W + 3W)")
ax.set_title("Where the new slots went", fontsize=12)
ax.legend(loc="lower right", fontsize=9)
w1r = placement.set_index(["wave", "need_tercile"])
fig.suptitle("New slots followed the existing footprint, not the failure hotspots (both waves)",
             x=0.01, ha="left", fontsize=14.5, fontweight="bold", y=1.01)
fig.text(0.01, 0.958, f"Highest-need third of zones: {w1r.loc[('Wave1','high'),'share_of_failures']:.0%} of Jan-Jun 2024 failures but "
         f"{w1r.loc[('Wave1','high'),'share_of_new_slots']:.0%} of Wave-1 slots; on same-season need, "
         f"{w1r.loc[('Wave2','high'),'share_of_failures']:.0%} of Apr-Jun 2024 failures but {w1r.loc[('Wave2','high'),'share_of_new_slots']:.0%} of Wave-2 slots.\n"
         f"Delhi NCR + Jaipur + Hyderabad held {city.loc[hot_cities,'share_failures_h1_24'].sum():.0%} of Jan-Jun 2024 failures and got "
         f"{city.loc[hot_cities,'share_new_slots'].sum():.0%} of all new slots. Zones sorted by Jan-Jun 2024 failure rate.",
         fontsize=10, color="#555555", va="bottom")
save_fig(fig, "q3_03_expansion_placement_vs_need")
say("saved q3_03_expansion_placement_vs_need")

# %% [markdown]
# ### 4b. Utilisation and fixed cost per swap: new vs legacy stations
# Apr-Jun 2025 (all stations live). Also a ramp curve: new-station utilisation relative to legacy stations in the same
# city and month, by months since commissioning (removes seasonality and city mix).

# %%
q2s = (SMp[SMp.period == "Q2_25"].groupby(["station_id", "expansion_wave", "city", "zone", "location_type", "charger_generation"])
       .agg(completed=("completed", "sum"), attempts=("attempts", "sum"), failures=("failures", "sum"), fixed_cost=("fixed_cost", "sum"),
            slot_days=("slot_days", "sum"), cm1=("cm1", "sum"), cm2=("cm2", "sum"), revenue=("revenue", "sum")).reset_index())
q2s["swaps_per_slot_day"] = q2s.completed / q2s.slot_days
q2s["fixed_cost_per_swap"] = q2s.fixed_cost / q2s.completed
q2s["cm1_per_swap"] = q2s.cm1 / q2s.completed
q2s["cm2_per_swap"] = q2s.cm2 / q2s.completed
q2s.to_csv(TAB / "q3_station_q2_2025_utilisation_cost.csv", index=False)
wv = q2s.groupby("expansion_wave").agg(stations=("station_id", "count"), completed=("completed", "sum"), fixed_cost=("fixed_cost", "sum"),
                                       slot_days=("slot_days", "sum"), cm1=("cm1", "sum"), cm2=("cm2", "sum"),
                                       median_util=("swaps_per_slot_day", "median"), median_fixed_per_swap=("fixed_cost_per_swap", "median"))
wv["swaps_per_slot_day"] = wv.completed / wv.slot_days
wv["fixed_cost_per_swap"] = wv.fixed_cost / wv.completed
wv["cm1_per_swap"] = wv.cm1 / wv.completed
wv["cm2_per_swap"] = wv.cm2 / wv.completed
wv["share_swaps"] = wv.completed / wv.completed.sum()
wv["share_fixed_cost"] = wv.fixed_cost / wv.fixed_cost.sum()
wv.to_csv(TAB / "q3_wave_economics_q2_2025.csv")
say("Apr-Jun 2025 economics by wave (under our usage-based battery-depreciation assumption):\n" + wv.round(3).to_string())
net_fps = q2s.fixed_cost.sum() / q2s.completed.sum()
leg = q2s[q2s.expansion_wave == "Launch"]
say(f"Network fixed cost per swap Apr-Jun 2025: Rs {net_fps:.2f}; legacy stations alone Rs {leg.fixed_cost.sum()/leg.completed.sum():.2f}; "
    f"new stations Rs {q2s[q2s.expansion_wave!='Launch'].fixed_cost.sum()/q2s[q2s.expansion_wave!='Launch'].completed.sum():.2f}")
new = q2s[q2s.expansion_wave != "Launch"]
say(f"New stations: {len(new)} of {len(q2s)} stations, {new.completed.sum()/q2s.completed.sum():.1%} of Q2-25 swaps, "
    f"{new.fixed_cost.sum()/q2s.fixed_cost.sum():.1%} of fixed cost; stations with fixed cost/swap > Rs 20: "
    f"new {int((new.fixed_cost_per_swap > 20).sum())}, legacy {int((leg.fixed_cost_per_swap > 20).sum())}")
loc_util = q2s[q2s.expansion_wave != "Launch"].groupby(["expansion_wave", "location_type"]).agg(
    stations=("station_id", "count"), completed=("completed", "sum"), slot_days=("slot_days", "sum"), fixed=("fixed_cost", "sum"))
loc_util["swaps_per_slot_day"] = loc_util.completed / loc_util.slot_days
loc_util["fixed_cost_per_swap"] = loc_util.fixed / loc_util.completed
say("New stations by wave x location type (Apr-Jun 2025):\n" + loc_util.round(2).to_string())
top_new = q2s[q2s.expansion_wave != "Launch"].sort_values("swaps_per_slot_day", ascending=False).head(6)
say("Busiest new stations (Apr-Jun 2025):\n" + top_new[["station_id", "expansion_wave", "zone", "location_type", "swaps_per_slot_day",
                                                       "fixed_cost_per_swap", "slot_days"]].round(2).to_string(index=False))
nz = q2s[q2s.zone.isin(no_station_zones.zone)]
say(f"New stations in the 3 previously unserved zones (n = {len(nz)} stations): {nz.completed.sum()/nz.slot_days.sum():.2f} swaps/slot-day vs "
    f"{new.completed.sum()/new.slot_days.sum():.2f} for all new stations (Apr-Jun 2025) = "
    f"{(nz.completed.sum()/nz.slot_days.sum())/(new.completed.sum()/new.slot_days.sum()):.2f}x")
say("  the unserved-zone stations individually (all highway fuel pumps -> small n, one outlier):\n"
    + nz[["station_id", "expansion_wave", "zone", "location_type", "swaps_per_slot_day", "fixed_cost_per_swap"]].round(2).to_string(index=False))


def util_cost(d):
    return pd.Series({"stations": d.station_id.nunique(), "swaps_per_slot_day": d.completed.sum() / d.slot_days.sum(),
                      "fixed_cost_per_swap": d.fixed_cost.sum() / d.completed.sum()})


# %% [markdown]
# ### 4b-ii. What drives new-station utilisation: unmet zone demand or site type?
# Wave 1 is mostly fuel pumps and residential sites and under-performs; but all 3 stars in formerly unserved zones are
# also fuel pumps. To separate the two we regress Apr-Jun 2025 swaps per slot-day of the 60 new stations on
# log(resident riders per zone slot) (riders active in H1-2024 living in the zone / all slots in the zone after both
# waves), site type (ref = market), wave and city; WLS weighted by slot-days, HC1 SEs. Cross-sectional: association.

# %%
NS = new.copy()
zone_i = zone.set_index("zone")
NS["riders_per_zone_slot"] = NS.zone.map(zone_i.riders_active_h1_24 / (zone_i.L_slots + zone_i.new_slots))
NS["unserved"] = NS.zone.isin(set(no_station_zones.zone)).astype(int)
fp_tab = NS[NS.location_type == "highway_fuel_pump"].groupby("unserved").apply(util_cost)
say("New highway-fuel-pump stations, Apr-Jun 2025, by zone status (1 = formerly unserved zone):\n" + fp_tab.round(2).to_string())
w1fr_ = NS[(NS.expansion_wave == "Wave1_2024H2") & NS.location_type.isin(["highway_fuel_pump", "residential"])]
say(f"Wave-1 fuel-pump + residential sites in already-served zones: {int((w1fr_.unserved == 0).sum())} of {len(w1fr_)} "
    f"(fuel pumps {int(((w1fr_.unserved == 0) & (w1fr_.location_type == 'highway_fuel_pump')).sum())} of "
    f"{int((w1fr_.location_type == 'highway_fuel_pump').sum())}, residential {int(((w1fr_.unserved == 0) & (w1fr_.location_type == 'residential')).sum())} of "
    f"{int((w1fr_.location_type == 'residential').sum())})")
SITE_RHS = "np.log(riders_per_zone_slot) + C(location_type, Treatment('market')) + C(expansion_wave) + C(city)"
m_site = smf.wls(f"swaps_per_slot_day ~ {SITE_RHS}", data=NS, weights=NS.slot_days).fit(cov_type="HC1")
m_site2 = smf.wls(f"swaps_per_slot_day ~ {SITE_RHS}", data=NS[NS.unserved == 0], weights=NS[NS.unserved == 0].slot_days).fit(cov_type="HC1")
site_rows = []
for nm, m in [("all_new_stations", m_site), ("already_served_zones_only", m_site2)]:
    for t in m.params.index:
        if t.startswith("C(city)") or t == "Intercept":
            continue
        site_rows.append({"sample": nm, "term": t.replace("C(location_type, Treatment('market'))[T.", "site: ").replace("C(expansion_wave)[T.", "wave: ")
                          .replace("]", ""), "coef": m.params[t], "ci_lo": m.conf_int().loc[t, 0], "ci_hi": m.conf_int().loc[t, 1],
                          "p": m.pvalues[t], "n": int(m.nobs), "r2": m.rsquared})
SITE = pd.DataFrame(site_rows)
SITE.to_csv(TAB / "q3_new_station_siting_model.csv", index=False)
say("New-station utilisation model (swaps/slot-day, Apr-Jun 2025; city FE not shown):\n" + SITE.round(3).to_string(index=False))
w2s = q2s[q2s.expansion_wave == "Wave2_2025H1"]
w1s = q2s[q2s.expansion_wave == "Wave1_2024H2"]
sub_econ = pd.DataFrame({
    "Legacy (Launch)": util_cost(q2s[q2s.expansion_wave == "Launch"]),
    "Wave 1 all": util_cost(w1s),
    "Wave 1 fuel pump + residential": util_cost(w1s[w1s.location_type.isin(["highway_fuel_pump", "residential"])]),
    "Wave 2 all": util_cost(w2s),
    "Wave 2 market only": util_cost(w2s[w2s.location_type == "market"]),
    "Wave 2 excl. 2 unserved-zone fuel pumps": util_cost(w2s[~w2s.station_id.isin(nz.station_id)]),
    "Unserved-zone stations (all fuel pumps)": util_cost(nz),
    "Fuel pumps in already-served zones": util_cost(NS[(NS.location_type == "highway_fuel_pump") & (NS.unserved == 0)])}).T
sub_econ.to_csv(TAB / "q3_new_station_subgroup_economics.csv")
say("Apr-Jun 2025 utilisation and fixed cost per swap by sub-group:\n" + sub_econ.round(2).to_string())

# ramp curve vs same-city legacy stations in the same month
ramp = SMp[(SMp.active_days >= 15)].copy()
ramp["util"] = ramp.completed / ramp.slot_days
legacy_cm = ramp[ramp.expansion_wave == "Launch"].groupby(["city", "month"]).apply(lambda d: pd.Series({
    "legacy_util": d.completed.sum() / d.slot_days.sum(), "legacy_att_util": d.attempts.sum() / d.slot_days.sum()}))
ramp = ramp.merge(legacy_cm.reset_index(), on=["city", "month"], how="left")
ramp["rel_util"] = ramp.util / ramp.legacy_util
ramp["age_m"] = ((pd.to_datetime(ramp.month).dt.year - pd.to_datetime(ramp.commissioned_date).dt.year) * 12
                 + pd.to_datetime(ramp.month).dt.month - pd.to_datetime(ramp.commissioned_date).dt.month)
# rel_util = completed swaps (capped by legacy stockouts in summer); rel_att = attempts (demand) - both vs same-city legacy
rc = (ramp[ramp.expansion_wave != "Launch"].groupby(["expansion_wave", "age_m"])
      .apply(lambda d: pd.Series({"stations": d.station_id.nunique(),
                                  "rel_util": (d.util * d.slot_days).sum() / (d.legacy_util * d.slot_days).sum(),
                                  "rel_att": d.attempts.sum() / (d.legacy_att_util * d.slot_days).sum()}))
      .reset_index())
rc.to_csv(TAB / "q3_new_station_ramp.csv", index=False)
say("New-station utilisation relative to same-city legacy stations, by months since commissioning:\n" + rc.round(3).to_string(index=False))
for w_, lo, hi in [("Wave1_2024H2", 1, 8), ("Wave2_2025H1", 1, 3)]:
    r_ = rc[(rc.expansion_wave == w_) & (rc.stations >= 20) & rc.age_m.between(lo, hi)]
    say(f"  {w_} months {lo}-{hi} (ages with >= 20 stations, as in Fig 4): completed basis {100*r_.rel_util.min():.0f}-{100*r_.rel_util.max():.0f}% "
        f"of same-city legacy; attempts basis {100*r_.rel_att.min():.0f}-{100*r_.rel_att.max():.0f}%")

# %% [markdown]
# ### Figure 4 (hero): utilisation vs fixed cost per swap - new vs legacy

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
for w in ["Launch", "Wave1_2024H2", "Wave2_2025H1"]:
    d = q2s[q2s.expansion_wave == w]
    ax.scatter(d.swaps_per_slot_day, d.fixed_cost_per_swap, s=46 if w != "Launch" else 30, color=WAVE_COLORS[w], alpha=0.85 if w != "Launch" else 0.6,
               edgecolor="white", linewidth=0.6, label=f"{WAVE_LABEL[w]}: median {d.swaps_per_slot_day.median():.2f} swaps/slot-day, "
                                                    f"Rs {d.fixed_cost_per_swap.median():.1f}/swap")
XMAX = 15
xs = np.linspace(1.2, XMAX, 100)
opex_slot_day = (q2s.fixed_cost.sum() / q2s.slot_days.sum())
ax.plot(xs, opex_slot_day / xs, color="#999999", lw=1.2, ls="--")
ax.annotate(f"network cost per slot-day\n(Rs {opex_slot_day:.0f}) / utilisation", xy=(1.3, opex_slot_day / 1.3), xytext=(2.6, 35),
            fontsize=9, color="#666666", va="center", arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8))
ax.axhline(net_fps, color=PALETTE["accent"], lw=1, ls=":")
ax.text(XMAX - 0.2, net_fps + 0.6, f"network Rs {net_fps:.1f}/swap", ha="right", fontsize=9, color=PALETTE["accent"])
off = q2s[q2s.swaps_per_slot_day > XMAX]
unserved_set = set(no_station_zones.zone)
for _, r_ in off.iterrows():
    tag = "fuel pump, unserved zone" if (r_.zone in unserved_set and r_.location_type == "highway_fuel_pump") else r_.location_type.replace("_", " ")
    ax.annotate(f"{r_.station_id} ({r_.zone}; {tag}):\n{r_.swaps_per_slot_day:.1f} swaps/slot-day, off scale ->", xy=(XMAX - 0.1, r_.fixed_cost_per_swap),
                xytext=(XMAX - 0.3, r_.fixed_cost_per_swap + 14), ha="right", fontsize=8.5, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#999999"))
ax.set_xlim(0.8, XMAX)
ax.set_xlabel("Completed swaps per slot per day (Apr-Jun 2025)")
ax.set_ylabel("Fixed cost per swap (Rs, rent + maintenance)")
ax.set_title("Per station: utilisation drives fixed cost per swap", fontsize=12.5)
ax.legend(loc="upper right", fontsize=8.6)
ax = axes[1]
for w, mk in [("Wave1_2024H2", "o"), ("Wave2_2025H1", "s")]:
    d = rc[(rc.expansion_wave == w) & (rc.stations >= 20)]
    ax.plot(d.age_m, 100 * d.rel_util, color=WAVE_COLORS[w], marker=mk, lw=2, ms=6, label=WAVE_LABEL[w])
ax.axhline(100, color="#555555", lw=1, ls="--")
ax.text(0.1, 101.5, "legacy stations, same city & month = 100", fontsize=9, color="#555555")
ax.set_xlabel("Months since commissioning (ages with >= 20 stations)")
ax.set_ylabel("Completed swaps per slot-day vs\nsame-city legacy stations (%)")
ax.set_title("Wave 2 beat legacy utilisation from month 1;\nWave 1 plateaued at ~83%", fontsize=12.5)
ax.legend(loc="upper right")
se_ = sub_econ
import textwrap  # noqa: E402
fig.suptitle(f"Wave-1 fuel-pump and residential sites cost Rs {se_.loc['Wave 1 fuel pump + residential','fixed_cost_per_swap']:.1f} per swap in "
             f"fixed cost vs Rs {se_.loc['Legacy (Launch)','fixed_cost_per_swap']:.1f} legacy;\nall Wave-2 sites "
             f"Rs {se_.loc['Wave 2 all','fixed_cost_per_swap']:.1f} (market sites alone Rs {se_.loc['Wave 2 market only','fixed_cost_per_swap']:.1f})",
             x=0.01, ha="left", va="bottom", fontsize=13.5, fontweight="bold", y=1.075)
sub4 = (f"Apr-Jun 2025; fixed cost = rent + maintenance. Utilisation sets cost per swap, and new-station utilisation "
        f"tracks unmet zone demand: the {int(se_.loc['Unserved-zone stations (all fuel pumps)','stations'])} stations in formerly unserved zones "
        f"(all fuel pumps) run {se_.loc['Unserved-zone stations (all fuel pumps)','swaps_per_slot_day']:.1f} swaps/slot-day. Without its 2 "
        f"unserved-zone fuel pumps, Wave 2 runs {se_.loc['Wave 2 excl. 2 unserved-zone fuel pumps','swaps_per_slot_day']:.2f} swaps/slot-day at "
        f"Rs {se_.loc['Wave 2 excl. 2 unserved-zone fuel pumps','fixed_cost_per_swap']:.2f}/swap, above legacy Rs {se_.loc['Legacy (Launch)','fixed_cost_per_swap']:.2f}.")
fig.text(0.01, 0.995, textwrap.fill(sub4, 172), fontsize=10, color="#555555", va="bottom")
save_fig(fig, "q3_04_new_vs_legacy_utilisation_cost")
say("saved q3_04_new_vs_legacy_utilisation_cost")

# %% [markdown]
# ### 4c. Did legacy hotspot stations improve where new stations opened? (DiD)
# **Design A - same-season long difference** (both waves): for every Launch station, change in failure rate
# Apr-Jun 2025 vs Apr-Jun 2024, regressed on the zone's new-slot intensity (new slots / legacy slots), controlling for
# the station's change in heat exposure x generation and city fixed effects.
# **Design B - Wave-1 TWFE**: Launch stations in zones that got a Wave-1 station vs Launch stations in zones that did not
# (Bengaluru Electronic City & South, Pune Central, Hyderabad West, Navi Mumbai), Jan 2024 - Jan 2025 (before Wave 2),
# with station FE, city x month FE and generation x month FE (Gen1 has its own heat seasonality).

# %%
L = SMp[(SMp.expansion_wave == "Launch")].copy()
L["fail_pp"] = 100 * L.failures / L.attempts
L["util"] = L.attempts / L.slot_days
L["hot_pp"] = 100 * L.hot_share
L = L.merge(zone[["zone", "L_slots", "W1_slots", "W2_slots", "new_slots"]], on="zone", how="left")
L["intensity"] = L.new_slots / L.L_slots
pre_q2 = L[(L.month >= pd.Timestamp("2024-04-01")) & (L.month < pd.Timestamp("2024-07-01"))]
post_q2 = L[(L.month >= pd.Timestamp("2025-04-01"))]


def agg_station(d):
    return d.groupby("station_id").agg(att=("attempts", "sum"), fail=("failures", "sum"), so=("stockouts", "sum"), sd=("slot_days", "sum"),
                                       comp=("completed", "sum"), hot=("hot_pp", "mean"))


a, b = agg_station(pre_q2), agg_station(post_q2)
LD = pd.DataFrame({"d_fail_pp": 100 * b.fail / b.att - 100 * a.fail / a.att, "d_util": b.att / b.sd - a.att / a.sd,
                   "d_hot_pp": b.hot - a.hot, "fail_q2_24": 100 * a.fail / a.att, "fail_q2_25": 100 * b.fail / b.att,
                   "util_q2_24": a.att / a.sd, "util_q2_25": b.att / b.sd}).reset_index()
LD = LD.merge(st_pd[["station_id", "city", "zone", "charger_generation"]], on="station_id").merge(
    zone[["zone", "new_slots", "L_slots"]], on="zone")
LD["intensity"] = LD.new_slots / LD.L_slots
h1 = SMp[(SMp.expansion_wave == "Launch") & (SMp.period == "H1_24")].groupby("station_id").agg(a=("attempts", "sum"), f=("failures", "sum"), sd=("slot_days", "sum"))
LD["fail_h1_24"] = LD.station_id.map(100 * h1.f / h1.a)
LD["util_h1_24"] = LD.station_id.map(h1.a / h1.sd)
LD["hotspot"] = (LD.fail_h1_24 >= LD.fail_h1_24.quantile(0.75)).astype(int)
LD["busiest"] = (LD.util_h1_24 >= LD.util_h1_24.quantile(0.90)).astype(int)
mA = smf.ols("d_fail_pp ~ intensity + d_hot_pp * C(charger_generation) + C(city)", data=LD).fit(cov_type="cluster", cov_kwds={"groups": LD["zone"]})
mA_u = smf.ols("d_util ~ intensity + C(city)", data=LD).fit(cov_type="cluster", cov_kwds={"groups": LD["zone"]})
mA_h = smf.ols("d_fail_pp ~ intensity * hotspot + d_hot_pp * C(charger_generation) + C(city)", data=LD).fit(cov_type="cluster", cov_kwds={"groups": LD["zone"]})
say(f"Design A (n={len(LD)} Launch stations, SE clustered by zone): change in failure rate per +1.0 new-slot intensity = "
    f"{mA.params['intensity']:.2f} pp (95% CI {mA.conf_int().loc['intensity',0]:.2f} to {mA.conf_int().loc['intensity',1]:.2f}, p={mA.pvalues['intensity']:.3f}); "
    f"change in attempts/slot-day = {mA_u.params['intensity']:.2f} (p={mA_u.pvalues['intensity']:.3f}); "
    f"extra for hotspot stations = {mA_h.params['intensity:hotspot']:.2f} pp (p={mA_h.pvalues['intensity:hotspot']:.3f})")
say(f"  median zone intensity = {LD.intensity.median():.2f} -> implied effect {mA.params['intensity']*LD.intensity.median():.2f} pp; "
    f"Launch-station mean d_fail Q2-25 vs Q2-24 = {LD.d_fail_pp.mean():.2f} pp, d_hot = {LD.d_hot_pp.mean():.2f} pp")
LD.to_csv(TAB / "q3_legacy_station_long_difference.csv", index=False)

# Design B: Wave-1 staggered TWFE on Launch stations, Jan 2024 - Jan 2025
w1_first = st_pd[st_pd.expansion_wave == "Wave1_2024H2"].groupby("zone").commissioned_date.min()
B = L[(L.month <= pd.Timestamp("2025-01-01")) & (L.active_days >= 15)].copy()
B["w1_date"] = pd.to_datetime(B.zone.map(w1_first))
B["treated_zone"] = B.w1_date.notna().astype(int)
B["post"] = ((B.treated_zone == 1) & (pd.to_datetime(B.month) > B.w1_date)).astype(int)  # months fully after first W1 opening in zone
B = B[~((B.treated_zone == 1) & (pd.to_datetime(B.month).dt.to_period("M") == B.w1_date.dt.to_period("M")))]  # drop opening month
B["month_s"] = B.month.astype(str)
B["cm"] = B.city + "_" + B.month_s
B["gm"] = B.charger_generation + "_" + B.month_s
did_rows = []
for yname, w in [("fail_pp", "attempts"), ("util", "slot_days")]:
    m = smf.wls(f"{yname} ~ post + C(station_id) + C(cm) + C(gm)", data=B, weights=B[w]).fit(cov_type="cluster", cov_kwds={"groups": B["station_id"]})
    did_rows.append({"design": "B_wave1_twfe", "outcome": yname, "coef": m.params["post"], "ci_lo": m.conf_int().loc["post", 0],
                     "ci_hi": m.conf_int().loc["post", 1], "p": m.pvalues["post"], "n": int(m.nobs),
                     "treated_stations": B[B.treated_zone == 1].station_id.nunique(), "control_stations": B[B.treated_zone == 0].station_id.nunique()})
for nm, m in [("fail_pp", mA), ("util", mA_u)]:
    did_rows.append({"design": "A_long_difference_q2", "outcome": nm, "coef": m.params["intensity"], "ci_lo": m.conf_int().loc["intensity", 0],
                     "ci_hi": m.conf_int().loc["intensity", 1], "p": m.pvalues["intensity"], "n": int(m.nobs)})
DID = pd.DataFrame(did_rows)
DID.to_csv(TAB / "q3_legacy_did_results.csv", index=False)
say("Legacy-station DiD results:\n" + DID.round(3).to_string(index=False))

# %% [markdown]
# ### 4d. Network average vs the busiest / hotspot stations (same season: Apr-Jun 2024 vs Apr-Jun 2025)

# %%
def rate(d):
    return 100 * d.failures.sum() / d.attempts.sum()


q2_24_all = SMp[(SMp.month >= pd.Timestamp("2024-04-01")) & (SMp.month < pd.Timestamp("2024-07-01"))]
q2_25_all = SMp[SMp.period == "Q2_25"]
busiest_ids = set(LD.loc[LD.busiest == 1, "station_id"]); hotspot_ids = set(LD.loc[LD.hotspot == 1, "station_id"])
gen1_hot_ids = set(st_pd[(st_pd.charger_generation == "Gen1") & st_pd.city.isin(hot_cities)].station_id)
groups = [("Network (all stations)", lambda d: d), ("Legacy stations", lambda d: d[d.expansion_wave == "Launch"]),
          ("New stations (Wave 1+2)", lambda d: d[d.expansion_wave != "Launch"]),
          ("Busiest 10% legacy (H1-24 util.)", lambda d: d[d.station_id.isin(busiest_ids)]),
          ("Hotspot 25% legacy (H1-24 failures)", lambda d: d[d.station_id.isin(hotspot_ids)]),
          ("Gen1 legacy in Delhi/Jaipur/Hyd", lambda d: d[d.station_id.isin(gen1_hot_ids)])]
def att_per_slot_day(d):
    return d.attempts.sum() / d.slot_days.sum() if len(d) else np.nan


nvb = pd.DataFrame([{"group": g, "stations_q2_25": f(q2_25_all).station_id.nunique(),
                     "fail_q2_24": rate(f(q2_24_all)) if len(f(q2_24_all)) else np.nan, "fail_q2_25": rate(f(q2_25_all)),
                     "failed_q2_24": int(f(q2_24_all).failures.sum()), "failed_q2_25": int(f(q2_25_all).failures.sum()),
                     "attempts_q2_24": int(f(q2_24_all).attempts.sum()), "attempts_q2_25": int(f(q2_25_all).attempts.sum()),
                     "att_per_slot_day_q2_24": att_per_slot_day(f(q2_24_all)), "att_per_slot_day_q2_25": att_per_slot_day(f(q2_25_all))}
                    for g, f in groups])
nvb["change_pp"] = nvb.fail_q2_25 - nvb.fail_q2_24
nvb["failed_change_pct"] = 100 * (nvb.failed_q2_25 / nvb.failed_q2_24.replace(0, np.nan) - 1)
nvb["attempts_change_pct"] = 100 * (nvb.attempts_q2_25 / nvb.attempts_q2_24.replace(0, np.nan) - 1)
nvb.to_csv(TAB / "q3_network_vs_busiest.csv", index=False)
say("Network average vs busiest/hotspot stations (Apr-Jun 2024 vs Apr-Jun 2025) - rates:\n"
    + nvb[["group", "stations_q2_25", "fail_q2_24", "fail_q2_25", "change_pp"]].round(2).to_string(index=False))
say("Same comparison - counts and load (failed attempts, attempts, attempts per slot-day):\n"
    + nvb[["group", "failed_q2_24", "failed_q2_25", "failed_change_pct", "attempts_q2_24", "attempts_q2_25", "attempts_change_pct",
           "att_per_slot_day_q2_24", "att_per_slot_day_q2_25"]].round(2).to_string(index=False))
# is the hotspot failure RATE load-sensitive at all? within-station Gen1 model (station + month FE, heat held fixed)
g1P = P[P.charger_generation == "Gen1"].dropna(subset=["hot_pp"])
m_load = smf.wls("fail_pp ~ hot_pp + log_util + C(station_id) + C(month_s)", data=g1P, weights=g1P["attempts"]).fit(
    cov_type="cluster", cov_kwds={"groups": g1P["station_id"]})
say(f"Load test, Gen1 station-months (n={int(m_load.nobs):,}, station + month FE, heat share held fixed): log attempts per slot-day "
    f"{m_load.params['log_util']:+.3f} pp (95% CI {m_load.conf_int().loc['log_util',0]:.2f} to {m_load.conf_int().loc['log_util',1]:.2f}, "
    f"p={m_load.pvalues['log_util']:.2f}); hot_pp {m_load.params['hot_pp']:+.3f} pp -> failure rates at Gen1 stations do not respond to load, "
    f"so relieving load (new stations nearby) cannot lower them")
say(f"Busiest-decile legacy stations: {len(busiest_ids)}; hotspot-quartile: {len(hotspot_ids)}; overlap hotspot & Gen1-hot-city: "
    f"{len(hotspot_ids & gen1_hot_ids)} of {len(hotspot_ids)}")
net_fail_q2_25 = int(q2_25_all.failures.sum())
g1h = q2_25_all[q2_25_all.station_id.isin(gen1_hot_ids)]
cf25 = CFT[CFT.period == "Apr-Jun 2025"].set_index("counterfactual")
say(f"Network failed attempts Apr-Jun 2025: {net_fail_q2_25:,}; the 36 hot-city Gen1 stations: {int(g1h.failures.sum()):,} "
    f"({g1h.failures.sum()/net_fail_q2_25:.1%}) from {g1h.attempts.sum()/q2_25_all.attempts.sum():.1%} of attempts; counterfactual avoids "
    f"{cf25.loc['Gen1 -> Gen2','failed_attempts_avoided']/net_fail_q2_25:.1%} (Gen2-like) to {cf25.loc['Gen1 -> Gen3','failed_attempts_avoided']/net_fail_q2_25:.1%} (Gen3-like) of network failures")
off_all = SMp[~pd.to_datetime(SMp.month).dt.month.isin([4, 5, 6])]
base_rate = off_all.failures.sum() / off_all.attempts.sum()
ex_net = q2_25_all.failures.sum() - base_rate * q2_25_all.attempts.sum()
ex_g1 = g1h.failures.sum() - base_rate * g1h.attempts.sum()
say(f"Excess failed attempts Apr-Jun 2025 above the {100*base_rate:.2f}% off-season (non Apr-Jun) network baseline: network {ex_net:,.0f}; "
    f"36 hot-city Gen1 stations {ex_g1:,.0f} ({ex_g1/ex_net:.1%} of the excess)")
w1 = wv.loc["Wave1_2024H2"]; lg = wv.loc["Launch"]
w1_excess = (w1.fixed_cost_per_swap - lg.fixed_cost_per_swap) * w1.completed
say(f"Wave-1 fixed-cost drag: Rs {w1.fixed_cost_per_swap:.2f} vs legacy Rs {lg.fixed_cost_per_swap:.2f} per swap -> "
    f"Rs {w1_excess/1e5:.2f} lakh excess fixed cost in Apr-Jun 2025 (Rs {w1_excess/3/1e5:.2f} lakh/month); "
    f"new stations lift network fixed cost/swap by Rs {net_fps - lg.fixed_cost_per_swap:.2f}")

# monthly failure-rate series by group (for figure 5)
SMp["grp"] = np.select([SMp.station_id.isin(hotspot_ids), SMp.expansion_wave == "Launch"],
                       ["Legacy hotspots (23 Gen1 stations, Delhi/Jaipur/Hyd)", "Other legacy stations (67)"], "New stations (Wave 1 + 2)")
mser = SMp.groupby(["month", "grp"]).apply(lambda d: 100 * d.failures.sum() / d.attempts.sum()).rename("fail").reset_index()
net_m = SMp.groupby("month").apply(lambda d: 100 * d.failures.sum() / d.attempts.sum()).rename("fail").reset_index().assign(grp="Network average")
mser = pd.concat([mser, net_m])
mser.to_csv(TAB / "q3_monthly_failure_by_group.csv", index=False)
off_season = SMp[~pd.to_datetime(SMp.month).dt.month.isin([4, 5, 6])].groupby("grp").apply(lambda d: 100 * d.failures.sum() / d.attempts.sum())
say("Failure rate outside Apr-Jun, by station group (%): " + "; ".join(f"{k}: {v:.2f}" for k, v in off_season.items()))

# %% [markdown]
# ### Figure 5: the network average improves, the hotspots do not

# %%
fig, ax = plt.subplots(figsize=(12.5, 5.8))
sty = {"Network average": (PALETTE["primary"], "-", 2.6), "Legacy hotspots (23 Gen1 stations, Delhi/Jaipur/Hyd)": (PALETTE["accent"], "-", 2.2),
       "Other legacy stations (67)": (PALETTE["muted"], "-", 1.8), "New stations (Wave 1 + 2)": (SERIES[5], "--", 1.8)}
for g_, (c_, ls_, lw_) in sty.items():
    d = mser[mser.grp == g_].sort_values("month")
    if g_ == "New stations (Wave 1 + 2)":
        d = d[pd.to_datetime(d.month) >= pd.Timestamp("2024-09-01")]
    ax.plot(pd.to_datetime(d.month), d.fail, color=c_, ls=ls_, lw=lw_, marker="o", ms=3.5, label=g_)
ax.set_ylim(2.5, 21.5)
ax.axvspan(pd.Timestamp("2024-08-01"), pd.Timestamp("2024-10-31"), color=WAVE_COLORS["Wave1_2024H2"], alpha=0.08)
ax.axvspan(pd.Timestamp("2025-02-01"), pd.Timestamp("2025-04-30"), color=WAVE_COLORS["Wave2_2025H1"], alpha=0.12)
ax.text(pd.Timestamp("2024-08-05"), 21.2, "Wave 1 opens", fontsize=9.5, color="#555555", va="top")
ax.text(pd.Timestamp("2025-02-05"), 21.2, "Wave 2 opens", fontsize=9.5, color="#555555", va="top")
nv = nvb.set_index("group")
HS_ = "Hotspot 25% legacy (H1-24 failures)"; NW_ = "Network (all stations)"
ax.annotate(f"Apr-Jun 2024 -> Apr-Jun 2025\nFailure rate: network {nv.loc[NW_,'fail_q2_24']:.1f}% -> {nv.loc[NW_,'fail_q2_25']:.1f}%,"
            f"\n  legacy {nv.loc['Legacy stations','fail_q2_24']:.1f}% -> {nv.loc['Legacy stations','fail_q2_25']:.1f}%, "
            f"hotspots {nv.loc[HS_,'fail_q2_24']:.1f}% -> {nv.loc[HS_,'fail_q2_25']:.1f}%,"
            f"\n  new stations {nv.loc['New stations (Wave 1+2)','fail_q2_25']:.1f}% (2025)"
            f"\nFailed attempts: network {nv.loc[NW_,'failed_q2_24']/1e3:.1f}K -> {nv.loc[NW_,'failed_q2_25']/1e3:.1f}K "
            f"({nv.loc[NW_,'failed_change_pct']:+.0f}%),\n  hotspots {nv.loc[HS_,'failed_q2_24']/1e3:.1f}K -> {nv.loc[HS_,'failed_q2_25']/1e3:.1f}K "
            f"({nv.loc[HS_,'failed_change_pct']:+.0f}%)\nHotspot load {nv.loc[HS_,'att_per_slot_day_q2_24']:.2f} -> {nv.loc[HS_,'att_per_slot_day_q2_25']:.2f} "
            f"attempts/slot-day,\n  failure rate unchanged",
            xy=(pd.Timestamp("2025-05-01"), nv.loc["Hotspot 25% legacy (H1-24 failures)", "fail_q2_25"] + 2), xytext=(pd.Timestamp("2024-08-05"), 17.5),
            fontsize=10, color="#333333", va="top", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"),
            arrowprops=dict(arrowstyle="-", color="#999999"))
import matplotlib.dates as mdates  # noqa: E402
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
ax.set_ylabel("Service-failure rate (% of attempts)")
ax.set_title("The network failure rate fell year-on-year only through dilution by low-failure new stations;\n"
             f"the 23 legacy hotspots stayed at {nv.loc[HS_,'fail_q2_25']:.1f}% and network failed attempts nearly doubled "
             f"({nv.loc[NW_,'failed_change_pct']:+.0f}%)", fontsize=12.5)
ax.legend(loc="upper left", fontsize=9.2)
save_fig(fig, "q3_05_network_vs_hotspots")
say("saved q3_05_network_vs_hotspots")

# %% [markdown]
# ## 5. Competitor entry (10 stations, competitor within 1.5 km from 2025-01-15)
# Weekly station panel 2024-10-28 .. 2025-06-23 (all comparison stations live; entry week dropped). Treated = 8 stations
# open before entry (5 Bengaluru Launch, 2 Bengaluru Wave-1, 1 Pune Wave-1); the 2 Bengaluru Wave-2 treated stations
# opened after entry and are compared descriptively. Controls = untreated stations of the **same city and wave**
# (both pilot cities, so the peak/off-peak pricing pilot affects both arms). TWFE: station FE + stratum x week FE,
# cluster SE by station; plus an event study (4-week bins) and a within-stratum permutation test (2,000 draws).

# %%
WK = (events(["station_id", "week", "is_completed", "is_service_failure", "rider_id"])
      .filter(pl.col("week") >= pl.date(2024, 10, 28))
      .group_by("station_id", "week")
      .agg(pl.len().alias("attempts"), pl.col("is_completed").sum().alias("completed"),
           pl.col("is_service_failure").sum().alias("failures"), pl.col("rider_id").n_unique().alias("riders"))
      .collect(engine="streaming").to_pandas())
ENTRY_WEEK = pd.Timestamp("2025-01-13")
comp_ids = st_pd[st_pd.competitor_within_1_5km_since.notna()].station_id.tolist()
WK["week"] = pd.to_datetime(WK.week)
WK = WK[(WK.week <= pd.Timestamp("2025-06-23")) & (WK.week != ENTRY_WEEK)].merge(
    st_pd[["station_id", "city", "expansion_wave", "commissioned_date", "competitor_within_1_5km_since"]], on="station_id")
WK["stratum"] = WK.city + "_" + WK.expansion_wave
WK["treated"] = WK.station_id.isin(comp_ids).astype(int)
strata = WK[(WK.treated == 1) & (pd.to_datetime(WK.commissioned_date) < pd.Timestamp("2024-10-28"))].stratum.unique()
CW = WK[WK.stratum.isin(strata) & (pd.to_datetime(WK.commissioned_date) < pd.Timestamp("2024-10-21"))].copy()
CW["post"] = (CW.week > ENTRY_WEEK).astype(int)
CW["tp"] = CW.treated * CW.post
CW["ln_swaps"] = np.log(CW.completed.clip(lower=1))
CW["ln_riders"] = np.log(CW.riders.clip(lower=1))
CW["fail_pp"] = 100 * CW.failures / CW.attempts
CW["sw"] = CW.stratum + "_" + CW.week.dt.strftime("%Y-%m-%d")
say(f"Competitor DiD sample: {CW[CW.treated==1].station_id.nunique()} treated, {CW[CW.treated==0].station_id.nunique()} control stations, "
    f"{CW.week.nunique()} weeks, strata {sorted(CW.stratum.unique())}")
comp_rows = []
for yname, lab in [("ln_swaps", "log completed swaps/week"), ("ln_riders", "log unique riders/week"), ("fail_pp", "failure rate (pp)")]:
    fml = f"{yname} ~ tp + C(station_id) + C(sw)"
    mod = smf.wls(fml, data=CW, weights=CW.attempts) if yname == "fail_pp" else smf.ols(fml, data=CW)
    m = mod.fit(cov_type="cluster", cov_kwds={"groups": CW["station_id"]})
    comp_rows.append({"outcome": lab, "did": m.params["tp"], "ci_lo": m.conf_int().loc["tp", 0], "ci_hi": m.conf_int().loc["tp", 1],
                      "p_cluster": m.pvalues["tp"], "pct_effect": (np.exp(m.params["tp"]) - 1) * 100 if yname != "fail_pp" else np.nan})
# permutation test on station-level pre/post differences, within strata
sd_ = CW.groupby(["station_id", "stratum", "treated", "post"]).agg(ln_swaps=("ln_swaps", "mean"), ln_riders=("ln_riders", "mean")).unstack("post")
rng = np.random.default_rng(59500)
for i, yname in enumerate(["ln_swaps", "ln_riders"]):
    dlt = (sd_[(yname, 1)] - sd_[(yname, 0)]).rename("d").reset_index()

    def did_est(lbl):
        tot, wts = 0.0, 0
        for s, g in dlt.assign(lbl=lbl).groupby("stratum"):
            nt = int(g.lbl.sum())
            if 0 < nt < len(g):
                tot += nt * (g.d[g.lbl == 1].mean() - g.d[g.lbl == 0].mean()); wts += nt
        return tot / wts

    actual = did_est(dlt.treated.values)
    perms = []
    for _ in range(2000):
        lbl = dlt.treated.values.copy()
        for s in dlt.stratum.unique():
            idx = np.where(dlt.stratum.values == s)[0]
            lbl[idx] = rng.permutation(lbl[idx])
        perms.append(did_est(lbl))
    perms = np.array(perms)
    comp_rows[i]["perm_did"] = actual
    comp_rows[i]["p_permutation"] = float((np.abs(perms) >= abs(actual) - 1e-12).mean())
COMPDID = pd.DataFrame(comp_rows)
COMPDID.to_csv(TAB / "q3_competitor_did.csv", index=False)
say("Competitor entry DiD (TWFE, station + stratum x week FE):\n" + COMPDID.round(4).to_string(index=False))
say(f"Swaps effect in %: {COMPDID.loc[0,'pct_effect']:.1f}% (95% CI {100*(np.exp(COMPDID.loc[0,'ci_lo'])-1):.1f}% to "
    f"{100*(np.exp(COMPDID.loc[0,'ci_hi'])-1):+.1f}%)")
se_sw = (COMPDID.loc[0, "ci_hi"] - COMPDID.loc[0, "ci_lo"]) / 3.92
say(f"Minimum detectable effect on swaps (80% power, 5% two-sided) ~ {100*(1-np.exp(-2.8*se_sw)):.1f}% decline")
# robustness: station-specific linear trends (does the late event-study drift survive?)
CW["t_wk"] = (CW.week - CW.week.min()).dt.days / 7
m_tr = smf.ols("ln_swaps ~ tp + C(station_id) + C(station_id):t_wk + C(sw)", data=CW).fit(cov_type="cluster", cov_kwds={"groups": CW["station_id"]})
trend_pct, trend_p = 100 * (np.exp(m_tr.params["tp"]) - 1), m_tr.pvalues["tp"]
say(f"Robustness - DiD with station-specific linear trends: swaps {trend_pct:+.1f}% (p={trend_p:.2f}) -> the post-entry drift is not robust")

# event study (4-week bins; ref = 4 weeks before entry)
CW["rel"] = ((CW.week - ENTRY_WEEK).dt.days // 7).astype(int)
CW["bin"] = np.where(CW.rel < 0, -((-CW.rel - 1) // 4) - 1, (CW.rel - 1) // 4)
bins = sorted(b for b in CW.bin.unique() if b != -1)
for b_ in bins:
    CW[f"tb_{'m' if b_ < 0 else 'p'}{abs(b_)}"] = ((CW.bin == b_) & (CW.treated == 1)).astype(int)
bcols = [c for c in CW.columns if c.startswith("tb_")]
es_rows = []
for yname in ["ln_swaps", "ln_riders"]:
    m = smf.ols(f"{yname} ~ {' + '.join(bcols)} + C(station_id) + C(sw)", data=CW).fit(cov_type="cluster", cov_kwds={"groups": CW["station_id"]})
    for c in bcols:
        b_ = -int(c[4:]) if c[3] == "m" else int(c[4:])
        es_rows.append({"outcome": yname, "bin": b_, "week_from": b_ * 4 + (1 if b_ >= 0 else 0), "coef": m.params[c],
                        "ci_lo": m.conf_int().loc[c, 0], "ci_hi": m.conf_int().loc[c, 1]})
    es_rows.append({"outcome": yname, "bin": -1, "week_from": -4, "coef": 0.0, "ci_lo": 0.0, "ci_hi": 0.0})
ES = pd.DataFrame(es_rows).sort_values(["outcome", "bin"])
ES["pct"] = 100 * (np.exp(ES.coef) - 1)
ES.to_csv(TAB / "q3_competitor_event_study.csv", index=False)
say("Event study (log completed swaps), coefficient by 4-week bin:\n" + ES[ES.outcome == "ln_swaps"].round(3).to_string(index=False))

# raw weekly index for the figure: treated vs control, normalised to pre-entry mean
raw = CW.groupby(["week", "treated"]).agg(completed=("completed", "mean")).reset_index()
pre_mean = raw[raw.week < ENTRY_WEEK].groupby("treated").completed.mean()
raw["idx"] = 100 * raw.completed / raw.treated.map(pre_mean)
# the two treated Wave-2 stations that opened after entry vs other Bengaluru Wave-2 stations
bw2 = q2s[(q2s.city == "Bengaluru") & (q2s.expansion_wave == "Wave2_2025H1")].assign(competitor=lambda d: d.station_id.isin(comp_ids))
bw2s = bw2.groupby("competitor").apply(lambda d: pd.Series({"stations": len(d), "swaps_per_slot_day": d.completed.sum() / d.slot_days.sum()}))
say("Bengaluru Wave-2 stations, Apr-Jun 2025, competitor nearby vs not:\n" + bw2s.round(3).to_string())

# rider-level: did riders of treated stations leave the network or just switch station?
RW = (events(["rider_id", "station_id", "week", "is_completed", "city"])
      .filter(pl.col("is_completed") & pl.col("city").is_in(["Bengaluru", "Pune"]) & (pl.col("week") >= pl.date(2024, 10, 28))
              & (pl.col("week") <= pl.date(2025, 6, 23)) & (pl.col("week") != pl.date(2025, 1, 13)))
      .with_columns((pl.col("week") > pl.date(2025, 1, 13)).alias("post"))
      .group_by("rider_id", "station_id", "post").agg(pl.len().alias("swaps"))
      .collect(engine="streaming").to_pandas())
pre_r = RW[~RW.post]
# most-used pre-entry station; ties broken by station_id so the result is deterministic (polars group_by order is not)
home = (pre_r.sort_values(["rider_id", "swaps", "station_id"], ascending=[True, False, True]).drop_duplicates("rider_id")
        [["rider_id", "station_id"]].rename(columns={"station_id": "home_station"}))
home = home.merge(st_pd[["station_id", "city", "expansion_wave"]].rename(columns={"station_id": "home_station"}), on="home_station")
home["stratum"] = home.city + "_" + home.expansion_wave
home = home[home.stratum.isin(strata) & home.home_station.isin(CW.station_id.unique())]
home["treated"] = home.home_station.isin(comp_ids).astype(int)
n_pre_w = CW[CW.post == 0].week.nunique(); n_post_w = CW[CW.post == 1].week.nunique()
tot = RW.groupby(["rider_id", "post"]).swaps.sum().unstack(fill_value=0)
at_home = RW.merge(home[["rider_id", "home_station"]], on="rider_id")
at_home = at_home[at_home.station_id == at_home.home_station].groupby(["rider_id", "post"]).swaps.sum().unstack(fill_value=0)
home = home.join(tot.rename(columns={False: "tot_pre", True: "tot_post"}), on="rider_id").join(
    at_home.rename(columns={False: "home_pre", True: "home_post"}), on="rider_id").fillna(0)
home["wk_pre"] = home.tot_pre / n_pre_w; home["wk_post"] = home.tot_post / n_post_w
home["active_post"] = (home.tot_post > 0).astype(int)
home["home_share_post"] = np.where(home.tot_post > 0, home.home_post / home.tot_post.replace(0, np.nan), np.nan)
home["home_share_pre"] = home.home_pre / home.tot_pre
rid = home.groupby("treated").agg(riders=("rider_id", "count"), swaps_wk_pre=("wk_pre", "mean"), swaps_wk_post=("wk_post", "mean"),
                                  still_active_post=("active_post", "mean"), home_share_pre=("home_share_pre", "mean"),
                                  home_share_post=("home_share_post", "mean"))
rid["network_swaps_change_pct"] = 100 * (rid.swaps_wk_post / rid.swaps_wk_pre - 1)
rid.to_csv(TAB / "q3_competitor_rider_level.csv")
home["d_wk"] = home.wk_post - home.wk_pre
m_r = smf.ols("d_wk ~ treated + C(stratum)", data=home).fit(cov_type="cluster", cov_kwds={"groups": home["home_station"]})
m_a = smf.ols("active_post ~ treated + C(stratum)", data=home).fit(cov_type="cluster", cov_kwds={"groups": home["home_station"]})
say("Rider-level (riders whose most-used pre-entry station is treated vs same-stratum control):\n" + rid.round(3).to_string())
say(f"  DiD network-wide swaps/week per rider = {m_r.params['treated']:+.3f} (95% CI {m_r.conf_int().loc['treated',0]:.3f} to "
    f"{m_r.conf_int().loc['treated',1]:.3f}); still active post-entry {m_a.params['treated']*100:+.1f} pp (p={m_a.pvalues['treated']:.3f})")

# %% [markdown]
# ### Figure 6: competitor entry event study

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
for t_, c_, lab in [(1, PALETTE["accent"], f"Competitor within 1.5 km ({CW[CW.treated==1].station_id.nunique()} stations)"),
                    (0, PALETTE["primary"], f"Same city & wave, no competitor ({CW[CW.treated==0].station_id.nunique()} stations)")]:
    d = raw[raw.treated == t_]
    ax.plot(d.week, d.idx, color=c_, lw=2, marker="o", ms=3, label=lab)
ax.axvline(pd.Timestamp("2025-01-15"), color="#555555", ls="--", lw=1)
ax.text(pd.Timestamp("2025-01-18"), ax.get_ylim()[0] + 3, "competitor opens\n15 Jan 2025", fontsize=9.5, color="#555555")
ax.set_ylabel("Completed swaps per station-week\n(index, pre-entry mean = 100)")
ax.set_title("Weekly swaps, treated vs comparable stations", fontsize=12.5)
ax.legend(loc="upper left", fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
ax = axes[1]
e = ES[ES.outcome == "ln_swaps"].sort_values("bin")
ax.errorbar(e.bin, 100 * (np.exp(e.coef) - 1), yerr=[100 * (np.exp(e.coef) - np.exp(e.ci_lo)), 100 * (np.exp(e.ci_hi) - np.exp(e.coef))],
            fmt="o", color=PALETTE["accent"], ecolor="#999999", capsize=3, ms=6)
ax.axhline(0, color="#555555", lw=1); ax.axvline(-0.5, color="#555555", ls="--", lw=1)
ax.set_xticks(e.bin, [f"{int(b)*4+1}..{int(b)*4+4}" if b >= 0 else f"{int(b)*4}..{int(b)*4+3}" for b in e.bin], rotation=45, fontsize=9)
ax.set_xlabel("Weeks relative to entry (4-week bins; ref = 4 weeks before)")
ax.set_ylabel("Effect on completed swaps (%, 95% CI)")
r0 = COMPDID.iloc[0]
ax.set_title(f"DiD: {r0.pct_effect:+.1f}% swaps (95% CI {100*(np.exp(r0.ci_lo)-1):+.1f}% to {100*(np.exp(r0.ci_hi)-1):+.1f}%)", fontsize=12.5)
r1 = COMPDID.iloc[1]
fig.suptitle(f"Competitor entry: no statistically detectable loss of swaps at the 8 affected stations "
             f"(permutation p = {r0.p_permutation:.2f});\nunique riders {r1.pct_effect:+.1f}% (p = {r1.p_cluster:.2f}), and their riders did not leave the network",
             x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.07)
fig.text(0.01, 0.955, f"Post-entry point estimates drift down, but every 95% CI includes 0 and the drift disappears once each station gets its own "
         f"linear trend ({trend_pct:+.1f}%, p = {trend_p:.2f}). Treat as a watch item, not a loss.", fontsize=10, color="#555555")
save_fig(fig, "q3_06_competitor_entry")
say("saved q3_06_competitor_entry")
say(f"TOTAL RUNTIME {time.time()-T0:.1f}s")
