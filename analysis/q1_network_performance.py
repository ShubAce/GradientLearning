# %% [markdown]
# # Q1 - Network performance over time (Jan-2024 to Jun-2025)
#
# **Question.** How have completed swaps, revenue, service-failure rates and contribution margin per swap trended,
# and do they tell the same story or diverge?
#
# **Approach.**
# 1. Monthly KPI panel from the clean event table (one row per swap attempt, test stations excluded).
# 2. Indexed divergence chart (H1-2024 monthly average = 100) for scale metrics, plus failure rate and Rs/swap economics.
# 3. Test the leadership claim *"service failures rose faster than swaps"*: naive start-vs-end vs same-season comparison,
#    then a fixed panel of Launch stations and station-demand deciles (dilution by new stations / Simpson's paradox).
# 4. Exact shift-share bridge (waterfall) of cm1 per swap H1-24 -> Jul-Aug-24 -> Oct-24..Mar-25, plus the same-season
#    Q1-24 -> Q1-25 bridge. Neither window contains a pack issued below the 70% end of life (EOL); see step 5.
# 5. Internal-consistency check of the wear model (section 5b): the per-cycle wear keeps being charged on Kyron packs
#    issued below 70% EOL (Apr-Jun-25 only), so their lifetime wear exceeds their purchase cost. Two consistent
#    variants: (a) wear stops at EOL; (b) post-EOL issues cost the Cellora wear of the same class.
# 6. Counterfactual: Kyron packs with Cellora-like wear. 7. Sensitivity of margin levels to the wear assumption.
#
# Shared cost model (src/clean.py): cm1 = revenue - energy - battery wear; cm2 = cm1 - station fixed cost.
# Levels depend on the battery-wear assumption, so they are quoted "under our usage-based battery-depreciation
# assumption". The analysis focuses on trends and differences, anchored in windows with no post-EOL issues, and
# section 7 shows which of them survive other assumptions.

# %%
import sys
import time
from datetime import date
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.ticker import FuncFormatter
from scipy import stats

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # running inside a notebook opened at the project root
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from common import *  # noqa: E402,F403  (paths, constants, loaders, plotting style)

set_style()
T0 = time.time()
RNG = np.random.default_rng(59500)
RNG2 = np.random.default_rng(59501)   # separate stream for the paired network bootstrap, so earlier CIs stay reproducible
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(230)
pl.Config.set_float_precision(3)
pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

RS = "₹"          # rupee sign
MINUS = "−"


def M(y, m):
    return date(y, m, 1)


PER = {
    "Jan-24": (M(2024, 1), M(2024, 1)), "May-24": (M(2024, 5), M(2024, 5)), "Jun-24": (M(2024, 6), M(2024, 6)),
    "May-25": (M(2025, 5), M(2025, 5)), "Jun-25": (M(2025, 6), M(2025, 6)),
    "Q1-24": (M(2024, 1), M(2024, 3)), "Q2-24": (M(2024, 4), M(2024, 6)),
    "Q1-25": (M(2025, 1), M(2025, 3)), "Q2-25": (M(2025, 4), M(2025, 6)),
    "H1-24": (M(2024, 1), M(2024, 6)), "Jul-Aug-24": (M(2024, 7), M(2024, 8)), "H1-25": (M(2025, 1), M(2025, 6)),
    # after the price rise, the Kyron ramp and the pilot, and before any pack was issued below 70% EOL (section 5b)
    "Oct-24..Mar-25": (M(2024, 10), M(2025, 3)),
}
DISP = {"Oct-24..Mar-25": "Oct-24 to Mar-25"}   # display names for figure labels


def in_per(p, col="month"):
    a, b = PER[p]
    return pl.col(col).is_between(a, b)


def rs(v, d=1, sign=True):
    """Rupee label with a real minus sign, e.g. '+Rs4.9' / '-Rs15.3'."""
    s = f"{abs(v):,.{d}f}"
    if not sign:
        return (MINUS if v < 0 else "") + RS + s
    return ("+" if v >= 0 else MINUS) + RS + s


def title_block(fig, title, subtitle, top=0.86):
    fig.subplots_adjust(top=top)
    fig.text(0.01, 0.995, title, ha="left", va="top", fontsize=15, fontweight="bold", color="#111111")
    fig.text(0.01, 0.995 - 0.045 * (11 / fig.get_figheight()), subtitle, ha="left", va="top", fontsize=11, color="#555555")


def month_axis(ax, every=2):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=every))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.tick_params(axis="x", labelsize=10)


print(f"[setup] ROOT={ROOT}")

# %% [markdown]
# ## 1. Monthly KPI panel
# One narrow, streamed scan per family of metrics (memory rule): counts and rupee sums; distinct riders/stations;
# queue wait of completed swaps (mean and p90). Per-swap metrics are always ratio-of-sums (swap-weighted).
#
# *Definitions.* Attempt = any swap event. Service failure = stockout (no charged battery) + abandoned queue + system
# error; **cancelled by rider is shown separately** (a rider choice, not a service failure). Active rider/station = at
# least one completed swap in the month. Queue wait = arrival-to-completion seconds of completed swaps.

# %%
c = pl.col("is_completed")
# post-EOL issue: a completed swap that handed out a pack already below the 70% end of life (see section 5b)
pe = c & (pl.col("soh_out_pct") < EOL_SOH)
mon = (events(["month", "is_completed", "is_stockout", "is_abandoned", "is_system_error", "is_cancelled",
               "is_service_failure", "revenue", "energy_cost", "wear_cost", "fixed_cost", "cm1", "cm2",
               "energy_to_recharge_kwh", "vehicle_class", "supplier_out", "partner_id", "city", "soh_out_pct"])
       .group_by("month").agg(
           pl.len().alias("attempts"), c.sum().alias("completed"),
           pl.col("is_service_failure").sum().alias("failures"), pl.col("is_stockout").sum().alias("stockouts"),
           pl.col("is_abandoned").sum().alias("abandoned"), pl.col("is_system_error").sum().alias("system_errors"),
           pl.col("is_cancelled").sum().alias("cancelled"),
           *[pl.col(x).cast(pl.Float64).sum().alias(x) for x in ["revenue", "energy_cost", "wear_cost", "fixed_cost", "cm1", "cm2"]],
           pl.col("energy_to_recharge_kwh").filter(c).cast(pl.Float64).sum().alias("kwh"),
           (c & (pl.col("supplier_out") == "Kyron")).sum().alias("kyron_swaps"),
           (c & (pl.col("vehicle_class") == "3W")).sum().alias("swaps_3w"),
           (c & (pl.col("partner_id") == ZIPDROP_ID)).sum().alias("zipdrop_swaps"),
           (c & pl.col("partner_id").is_not_null()).sum().alias("partner_swaps"),
           (c & pl.col("city").is_in(PILOT_CITIES)).sum().alias("pilot_city_swaps"),
           (c & (pl.col("supplier_out") == "Kyron") & (pl.col("vehicle_class") == "2W")).sum().alias("kyron_2w_swaps"),
           pe.sum().alias("post_eol_swaps"), (pe & (pl.col("vehicle_class") == "2W")).sum().alias("post_eol_2w"),
           (pe & (pl.col("vehicle_class") == "3W")).sum().alias("post_eol_3w"),
           pl.col("wear_cost").filter(pe).cast(pl.Float64).sum().alias("post_eol_wear"))
       .sort("month").collect(engine="streaming"))

uni = (events(["month", "is_completed", "rider_id", "station_id"]).filter(c)
       .group_by("month").agg(pl.col("rider_id").n_unique().alias("active_riders"),
                              pl.col("station_id").n_unique().alias("active_stations"))
       .sort("month").collect(engine="streaming"))

# queue wait of completed swaps: keep a narrow frame (month, Launch flag, seconds) for exact period quantiles
qw = (events(["month", "is_completed", "queue_wait_sec", "expansion_wave"]).filter(c)
      .select("month", (pl.col("expansion_wave") == "Launch").alias("is_launch"),
              pl.col("queue_wait_sec").cast(pl.Float32))
      .collect(engine="streaming"))
qm = qw.group_by("month").agg(pl.col("queue_wait_sec").mean().alias("qwait_mean_s"),
                              pl.col("queue_wait_sec").quantile(0.9).alias("qwait_p90_s")).sort("month")
print(f"[scan] completed swaps in queue-wait frame: {qw.height:,}; null queue waits: {qw['queue_wait_sec'].null_count():,}")

# 2W pack health at return (explains energy per swap): SoH / SoC of the returned pack, kWh to recharge, km since last swap
# also: what riders pay per km (amount charged / km since last swap, swaps with a valid km reading) and the health of
# returned non-Kyron packs (fleet ageing outside the Kyron cohort)
h2w = (events(["month", "is_completed", "vehicle_class", "soh_in_pct", "soc_in_pct", "energy_to_recharge_kwh", "km_since_last_swap",
               "supplier_in", "amount_charged_inr"])
       .filter(c & (pl.col("vehicle_class") == "2W"))
       .group_by("month").agg(pl.col("soh_in_pct").mean().alias("soh_in_2w"), pl.col("soc_in_pct").mean().alias("soc_in_2w"),
                              pl.col("energy_to_recharge_kwh").mean().alias("kwh_2w"), pl.col("km_since_last_swap").mean().alias("km_2w"),
                              pl.col("soh_in_pct").filter(pl.col("supplier_in") != "Kyron").mean().alias("soh_in_2w_nonkyron"),
                              pl.col("amount_charged_inr").filter(pl.col("km_since_last_swap").is_not_null()).cast(pl.Float64).sum().alias("rev_km_2w"),
                              pl.col("km_since_last_swap").cast(pl.Float64).sum().alias("km_sum_2w"))
       .sort("month").collect(engine="streaming"))

mon = mon.join(uni, on="month").join(qm, on="month").join(h2w, on="month").sort("month").with_columns(
    (pl.col("failures") / pl.col("attempts")).alias("fail_rate"),
    (pl.col("stockouts") / pl.col("attempts")).alias("stockout_rate"),
    (pl.col("abandoned") / pl.col("attempts")).alias("abandoned_rate"),
    (pl.col("system_errors") / pl.col("attempts")).alias("syserr_rate"),
    (pl.col("cancelled") / pl.col("attempts")).alias("cancel_rate"),
    (pl.col("completed") / pl.col("active_riders")).alias("swaps_per_rider"),
    (pl.col("completed") / pl.col("active_stations")).alias("swaps_per_station"),
    *[(pl.col(x) / pl.col("completed")).alias(f"{x}_ps") for x in ["revenue", "energy_cost", "wear_cost", "fixed_cost", "cm1", "cm2", "kwh"]],
    ((pl.col("energy_cost") + pl.col("wear_cost")) / pl.col("completed")).alias("varcost_ps"),
    (pl.col("kyron_swaps") / pl.col("completed")).alias("kyron_share"),
    (pl.col("swaps_3w") / pl.col("completed")).alias("share_3w"),
    (pl.col("zipdrop_swaps") / pl.col("completed")).alias("zipdrop_share"),
    (pl.col("partner_swaps") / pl.col("completed")).alias("partner_share"),
    (pl.col("pilot_city_swaps") / pl.col("completed")).alias("pilot_city_share"),
    (pl.col("qwait_mean_s") / 60).alias("qwait_mean_min"), (pl.col("qwait_p90_s") / 60).alias("qwait_p90_min"),
    (pl.col("kyron_2w_swaps") / (pl.col("completed") - pl.col("swaps_3w"))).alias("kyron_share_2w"),
    ((pl.col("kyron_swaps") - pl.col("kyron_2w_swaps")) / pl.col("swaps_3w")).alias("kyron_share_3w"),
    # wear-consistent variant (a): no wear on post-EOL issues (lifetime depreciation = purchase cost)
    (pl.col("post_eol_wear") / pl.col("completed")).alias("post_eol_ps"),
    ((pl.col("cm1") + pl.col("post_eol_wear")) / pl.col("completed")).alias("cm1_cap_ps"),
    ((pl.col("cm2") + pl.col("post_eol_wear")) / pl.col("completed")).alias("cm2_cap_ps"),
    ((pl.col("energy_cost") + pl.col("wear_cost") - pl.col("post_eol_wear")) / pl.col("completed")).alias("varcost_cap_ps"),
    (pl.col("post_eol_swaps") / pl.col("kyron_swaps")).alias("post_eol_share_of_kyron"),
    (pl.col("rev_km_2w") / pl.col("km_sum_2w")).alias("rs_per_km_2w"),
)
mon.write_csv(TAB / "q1_monthly_kpis.csv")
show = mon.select("month", "attempts", "completed", "active_riders", "active_stations",
                  pl.col("swaps_per_rider").round(1), (pl.col("revenue") / 1e6).round(2).alias("revenue_M"),
                  (pl.col("fail_rate") * 100).round(2).alias("fail%"), (pl.col("stockout_rate") * 100).round(2).alias("stockout%"),
                  (pl.col("abandoned_rate") * 100).round(2).alias("aband%"), (pl.col("syserr_rate") * 100).round(2).alias("syserr%"),
                  (pl.col("cancel_rate") * 100).round(2).alias("cancel%"), pl.col("qwait_mean_min").round(2).alias("q_mean_min"),
                  pl.col("qwait_p90_min").round(2).alias("q_p90_min"))
print("\n[1] Monthly KPI panel - volume and service")
print(show)
show2 = mon.select("month", *[pl.col(f"{x}_ps").round(2).alias(x.replace("_cost", "") + "_ps") for x in ["revenue", "energy_cost", "wear_cost", "fixed_cost", "cm1", "cm2"]],
                   pl.col("kwh_ps").round(3), (pl.col("cm1") / 1e6).round(2).alias("cm1_M"), (pl.col("cm2") / 1e6).round(2).alias("cm2_M"),
                   (pl.col("kyron_share") * 100).round(1).alias("kyron%"), (pl.col("zipdrop_share") * 100).round(1).alias("zipdrop%"),
                   (pl.col("partner_share") * 100).round(1).alias("partner%"), (pl.col("share_3w") * 100).round(1).alias("3W%"),
                   (pl.col("pilot_city_share") * 100).round(1).alias("pilotcity%"))
print("\n[1] Monthly KPI panel - unit economics (Rs per completed swap) and drivers")
print(show2)
print("\n[1] 2W pack health at return vs energy per swap (completed 2W swaps)")
print(mon.select("month", pl.col("soh_in_2w").round(1), pl.col("soc_in_2w").round(2), pl.col("kwh_2w").round(3), pl.col("km_2w").round(1)))
f0, f1 = mon.row(0, named=True), mon.row(-1, named=True)
print(f"[1] Jan-24 -> Jun-25: swaps x{f1['completed'] / f0['completed']:.2f}, revenue x{f1['revenue'] / f0['revenue']:.2f} "
      f"({f0['revenue'] / 1e6:.2f}M -> {f1['revenue'] / 1e6:.2f}M), active riders x{f1['active_riders'] / f0['active_riders']:.2f}, "
      f"active stations {f0['active_stations']} -> {f1['active_stations']}")
print(f"[1] 2W kWh/swap {f0['kwh_2w']:.3f} -> {f1['kwh_2w']:.3f} ({(f1['kwh_2w'] / f0['kwh_2w'] - 1) * 100:+.1f}%); returned-pack SoH "
      f"{f0['soh_in_2w']:.1f}% -> {f1['soh_in_2w']:.1f}%; SoC at return {f0['soc_in_2w']:.2f}% -> {f1['soc_in_2w']:.2f}%; km since last swap "
      f"{f0['km_2w']:.1f} -> {f1['km_2w']:.1f} ({(f1['km_2w'] / f0['km_2w'] - 1) * 100:+.1f}%)")
print(f"[1] ZipDrop share of completed swaps H1-25: {mon.filter(in_per('H1-25'))['zipdrop_swaps'].sum() / mon.filter(in_per('H1-25'))['completed'].sum() * 100:.1f}% "
      f"(= {mon.filter(in_per('H1-25'))['zipdrop_swaps'].sum() / mon.filter(in_per('H1-25'))['partner_swaps'].sum() * 100:.1f}% of partner-billed swaps)")


def rs_per_km(p):
    d = mon.filter(in_per(p))
    return d["rev_km_2w"].sum() / d["km_sum_2w"].sum()


print(f"[1] 2W price per km (amount charged / km since last swap, completed 2W swaps with a valid km reading): Jan-24 "
      f"{f0['rs_per_km_2w']:.3f} -> Jun-25 {f1['rs_per_km_2w']:.3f} Rs/km ({(f1['rs_per_km_2w'] / f0['rs_per_km_2w'] - 1) * 100:+.1f}%); "
      f"same season H1-24 {rs_per_km('H1-24'):.3f} -> H1-25 {rs_per_km('H1-25'):.3f} ({(rs_per_km('H1-25') / rs_per_km('H1-24') - 1) * 100:+.1f}%)")

ADD = ["attempts", "completed", "failures", "stockouts", "abandoned", "system_errors", "cancelled", "revenue",
       "energy_cost", "wear_cost", "fixed_cost", "cm1", "cm2", "kwh", "kyron_swaps", "swaps_3w", "zipdrop_swaps",
       "partner_swaps", "pilot_city_swaps", "post_eol_swaps", "post_eol_2w", "post_eol_3w", "post_eol_wear"]


def pagg(p):
    """Period totals (sums over months) and swap-weighted ratios."""
    d = mon.filter(in_per(p)).select([pl.col(x).sum() for x in ADD] + [pl.col("active_riders").mean(), pl.len().alias("n_months")]).row(0, named=True)
    n = d["completed"]
    d.update({f"{x}_ps": d[x] / n for x in ["revenue", "energy_cost", "wear_cost", "fixed_cost", "cm1", "cm2", "kwh"]})
    d["fail_rate"] = d["failures"] / d["attempts"]
    d["swaps_per_month"] = n / d["n_months"]
    d["post_eol_ps"] = d["post_eol_wear"] / n            # wear charged on packs issued below 70% EOL, per network swap
    d["cm1_cap_ps"] = d["cm1_ps"] + d["post_eol_ps"]     # variant (a): wear stops at EOL
    d["cm2_cap_ps"] = d["cm2_ps"] + d["post_eol_ps"]
    return d


P = {p: pagg(p) for p in PER}
print("\n[1] Period unit economics (Rs per completed swap; ratio of sums). post-EOL = wear charged on packs issued below 70% EOL")
for p in ["H1-24", "Q1-24", "Q2-24", "Jul-Aug-24", "Oct-24..Mar-25", "Q1-25", "Q2-25", "H1-25"]:
    d = P[p]
    print(f"  {p:<14} swaps/month={d['swaps_per_month']:>9,.0f}  rev/swap={d['revenue_ps']:6.2f}  energy={d['energy_cost_ps']:6.2f}  "
          f"wear={d['wear_cost_ps']:6.2f} (post-EOL {d['post_eol_ps']:5.2f})  fixed={d['fixed_cost_ps']:6.2f}  cm1={d['cm1_ps']:6.2f}  cm2={d['cm2_ps']:6.2f}  "
          f"cm1 total/month={d['cm1'] / d['n_months'] / 1e6:6.2f}M  cm2 total/month={d['cm2'] / d['n_months'] / 1e6:6.2f}M  fail={d['fail_rate'] * 100:5.2f}%")

# %% [markdown]
# ## 2. Do the metrics tell the same story? Indexed divergence (hero figure 1)
# Scale metrics are indexed to their H1-2024 monthly average (=100). cm1 per swap cannot be indexed (its H1-2024 base
# is close to Rs 0, so an index would explode), so the bottom panel shows revenue per swap against variable cost per
# swap (energy + wear) in rupees; the shaded gap between them *is* cm1 per swap.

# %%
sm = station_month().with_columns(
    (pl.col("stockouts") + pl.col("abandoned") + pl.col("system_errors")).alias("failures"),
    pl.when(pl.col("expansion_wave") == "Launch").then(pl.lit("Launch")).otherwise(pl.lit("New")).alias("grp"))
chk_att = sm["attempts"].sum() - mon["attempts"].sum()
print(f"[2] station_month vs event scan attempts difference: {chk_att} (should be 0)")
grp_m = (sm.group_by("month", "grp").agg(pl.col("attempts").sum(), pl.col("failures").sum(), pl.col("completed").sum(),
                                        pl.col("station_id").n_unique().alias("stations"))
         .with_columns((pl.col("failures") / pl.col("attempts")).alias("fail_rate")).sort("month", "grp"))
print(grp_m.with_columns((pl.col("fail_rate") * 100).round(2)).pivot(on="grp", index="month", values=["fail_rate", "attempts", "stations"]))

base = mon.filter(in_per("H1-24")).select([pl.col(x).mean() for x in ["completed", "revenue", "active_riders", "failures", "attempts", "revenue_ps", "varcost_ps"]]).row(0, named=True)
mon = mon.with_columns(*[(pl.col(x) / base[x] * 100).alias(f"idx_{x}") for x in ["completed", "revenue", "active_riders", "failures", "attempts"]])
last = mon.row(-1, named=True)
kyron_ramp = mon.filter(pl.col("kyron_share") >= 0.10)["month"].min()
print(f"[2] Index (H1-24 avg = 100) in Jun-25: swaps {last['idx_completed']:.0f}, revenue {last['idx_revenue']:.0f}, "
      f"active riders {last['idx_active_riders']:.0f}, failures {last['idx_failures']:.0f}, attempts {last['idx_attempts']:.0f}")
print(f"[2] Kyron share of issued packs first >=10% in {kyron_ramp}; Kyron share by month (%): "
      + ", ".join(f"{m:%b-%y}={v * 100:.1f}" for m, v in zip(mon['month'], mon['kyron_share'])))
print("[2] Kyron share of 2W issues by month (%): " + ", ".join(f"{m:%b-%y}={v * 100:.1f}" for m, v in zip(mon['month'], mon['kyron_share_2w'])))
print("[2] Kyron share of 3W issues by month (%): " + ", ".join(f"{m:%b-%y}={v * 100:.1f}" for m, v in zip(mon['month'], mon['kyron_share_3w'])))

pdm = mon.to_pandas()
pdm["month"] = pd.to_datetime(pdm["month"])
gm = grp_m.to_pandas()
gm["month"] = pd.to_datetime(gm["month"])
fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True, gridspec_kw={"height_ratios": [1.2, 1, 1.05]})
ax = axes[0]
lines = [("idx_revenue", "Revenue", SERIES[5], "-"), ("idx_completed", "Completed swaps", PALETTE["primary"], "-"),
         ("idx_active_riders", "Active riders", SERIES[4], "--")]
for col, lab, colr, ls in lines:
    ax.plot(pdm["month"], pdm[col], color=colr, lw=2.4, ls=ls, label=lab)
    ax.annotate(f"{lab} {pdm[col].iloc[-1]:.0f}", (pdm["month"].iloc[-1], pdm[col].iloc[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=10.5, color="#333333")
ax.axhline(100, color="#999999", lw=1)
ax.set_ylabel("Index, H1-2024 monthly avg = 100")
ax.set_ylim(60, 365)
ax.set_title(f"Scale keeps climbing: Jun-25 swaps {last['idx_completed']:.0f}, revenue {last['idx_revenue']:.0f}, "
             f"active riders {last['idx_active_riders']:.0f} (H1-24 = 100)", fontsize=12, loc="left")
ax.legend(loc="upper left")
events_lab = [(BASE_PRICE_CHANGE, f"Base price +{RS}5 (2W) / +{RS}10 (3W)"), (kyron_ramp, "Kyron packs >10% of issues"),
              (PILOT_START, "Peak/off-peak pilot (BLR, PUN)"), (ZIPDROP_AMEND, "ZipDrop discount 12% -> 28%")]
for k, (mth_d, lab) in enumerate(events_lab):
    for a in axes:
        a.axvline(pd.Timestamp(mth_d), color="#777777", lw=0.9, ls=(0, (3, 3)), zorder=0)
    ax.text(pd.Timestamp(mth_d), 358 - k * 18, " " + lab, fontsize=9.8, color="#444444", va="top", ha="left")
ax = axes[1]
net = pdm[["month", "fail_rate"]]
la = gm[gm.grp == "Launch"]
nw = gm[gm.grp == "New"]
for yr in (2024, 2025):
    ax.axvspan(pd.Timestamp(yr, 3, 20), pd.Timestamp(yr, 6, 15), color=PALETTE["warn"], alpha=0.13, lw=0)
ax.plot(net["month"], net["fail_rate"] * 100, color=PALETTE["accent"], lw=2.6, label="Network average", marker="o", ms=4)
ax.plot(la["month"], la["fail_rate"] * 100, color=PALETTE["primary"], lw=1.9, ls="--", label="Launch stations (same 90)")
ax.plot(nw["month"], nw["fail_rate"] * 100, color=SERIES[5], lw=1.9, ls=":", label="New stations (Wave 1 + 2)", marker="s", ms=3)
v = {p: (float(net.loc[net.month == pd.Timestamp(PER[p][0]), "fail_rate"].iloc[0]) * 100,
         float(la.loc[la.month == pd.Timestamp(PER[p][0]), "fail_rate"].iloc[0]) * 100) for p in ("May-24", "May-25")}
v_new25 = float(nw.loc[nw.month == pd.Timestamp(PER["May-25"][0]), "fail_rate"].iloc[0]) * 100
ax.annotate(f"May-24: {v['May-24'][0]:.1f}%\n(all 90 stations are Launch)", (pd.Timestamp(PER["May-24"][0]), v["May-24"][0]),
            xytext=(18, 2), textcoords="offset points", fontsize=10, va="top", color="#222222")
ax.annotate(f"May-25: Launch {v['May-25'][1]:.1f}%, network {v['May-25'][0]:.1f}%,\nnew stations {v_new25:.1f}%",
            (pd.Timestamp(PER["May-25"][0]), v["May-25"][1]), xytext=(-12, 2), textcoords="offset points", fontsize=10, va="top",
            ha="right", color="#222222")
ax.set_ylabel("Service-failure rate, % of attempts")
ax.set_ylim(0, 14)
ax.set_title("Reliability follows the season (Apr-Jun shaded), not the trend; the 2025 'improvement' is the new stations",
             fontsize=12, loc="left")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3)
ax = axes[2]
ax.plot(pdm["month"], pdm["revenue_ps"], color=PALETTE["primary"], lw=2.6, label="Revenue per swap")
ax.plot(pdm["month"], pdm["varcost_ps"], color=PALETTE["accent"], lw=2.6, ls="--", label="Variable cost per swap (energy + battery wear)")
ax.fill_between(pdm["month"], pdm["revenue_ps"], pdm["varcost_ps"], where=pdm["revenue_ps"] >= pdm["varcost_ps"],
                color=PALETTE["good"], alpha=0.25, interpolate=True, label="Gap = cm1 > 0")
ax.fill_between(pdm["month"], pdm["revenue_ps"], pdm["varcost_ps"], where=pdm["revenue_ps"] < pdm["varcost_ps"],
                color=PALETTE["accent"], alpha=0.22, interpolate=True, label="Gap = cm1 < 0")
capm = pdm[pdm.month >= pd.Timestamp(2025, 3, 1)]       # identical to the dashed line until Mar-25
ax.plot(capm["month"], capm["varcost_cap_ps"], color=PALETTE["muted"], lw=2.2, ls=":", marker="o", ms=3.5,
        label="Variable cost if wear stops at 70% end of life")
for p, off in [("H1-24", (0, 22)), ("Jul-Aug-24", (-6, -92)), ("Oct-24..Mar-25", (0, -64))]:
    a, b = PER[p]
    mid = pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2
    yv = float(pdm.loc[(pdm.month >= pd.Timestamp(a)) & (pdm.month <= pd.Timestamp(b)), "revenue_ps"].mean())
    ax.annotate(f"{DISP.get(p, p)}\ncm1 {rs(P[p]['cm1_ps'])}/swap", (mid, yv), xytext=off, textcoords="offset points", ha="center",
                fontsize=10.5, fontweight="bold", color="#222222")
ax.annotate(f"Q2-25 cm1: {rs(P['Q2-25']['cm1_ps'])}/swap per-cycle model,\n{rs(P['Q2-25']['cm1_cap_ps'])} if wear stops at EOL",
            (pd.Timestamp(2025, 6, 1), float(capm["varcost_cap_ps"].iloc[-1])), xytext=(-8, -26), textcoords="offset points",
            ha="right", fontsize=9.8, color="#333333")
ax.set_ylabel(f"{RS} per completed swap")
ax.set_ylim(40, 80)
ax.set_title(f"Unit economics: the Jul-24 price rise lifted cm1 to {rs(P['Jul-Aug-24']['cm1_ps'])}/swap; battery wear per swap then jumped "
             f"{RS}{P['Jul-Aug-24']['wear_cost_ps']:.1f} -> {RS}{P['Oct-24..Mar-25']['wear_cost_ps']:.1f} (Oct-24 to Mar-25)", fontsize=12, loc="left")
ax.legend(loc="upper left", ncol=2, fontsize=9.8)
ticks = pd.date_range("2024-01-01", "2025-06-01", freq="MS")
ax.set_xticks(ticks)
ax.set_xticklabels([t.strftime("%b\n%y") if t.month in (1, 4, 7, 10) else t.strftime("%b") for t in ticks], fontsize=10)
ax.set_xlim(pd.Timestamp(2023, 12, 12), pd.Timestamp(2025, 9, 10))
title_block(fig, "Swaps and revenue nearly tripled while margin per swap turned negative; failure rates follow the season",
            f"VoltRelay network, monthly, Jan-2024 to Jun-2025. cm1 = revenue - energy - battery wear "
            f"(under our usage-based battery-depreciation assumption). Test stations excluded.\n"
            f"From Apr-25 the model also charges wear on Kyron packs issued below 70% end of life ({mon.filter(pl.col('month') == M(2025, 6))['post_eol_share_of_kyron'].item() * 100:.0f}% of "
            f"Kyron issues in Jun-25); the dotted line removes that charge.", top=0.9)
fig.subplots_adjust(bottom=0.06)
save_fig(fig, "q1_01_kpi_divergence")
print("[fig] q1_01_kpi_divergence saved")

# %% [markdown]
# ### Full monthly KPI panel (supporting figure)

# %%
fig, axs = plt.subplots(3, 3, figsize=(16, 12.5), sharex=True)
axs = axs.ravel()
a = axs[0]
a.plot(pdm.month, pdm.attempts / 1e3, color=PALETTE["muted"], lw=2, label="Attempts")
a.plot(pdm.month, pdm.completed / 1e3, color=PALETTE["primary"], lw=2.2, label="Completed swaps")
a.set_title("Attempts and completed swaps (K/month)", fontsize=11.5)
a.legend(loc="upper left")
a = axs[1]
a.plot(pdm.month, pdm.active_riders / 1e3, color=SERIES[4], lw=2.2)
a.set_title("Active riders (K, >=1 completed swap)", fontsize=11.5)
a = axs[2]
a.step(pdm.month, pdm.active_stations, where="mid", color=PALETTE["primary"], lw=2.2, label="Active stations")
a.set_title("Active stations (>=1 completed swap)", fontsize=11.5)
a2v = pdm.swaps_per_rider
a = axs[3]
a.plot(pdm.month, a2v, color=SERIES[4], lw=2.2)
a.set_title("Completed swaps per active rider (per month)", fontsize=11.5)
a = axs[4]
for col, lab, colr, ls in [("stockout_rate", "Stockout", PALETTE["accent"], "-"), ("abandoned_rate", "Abandoned queue", PALETTE["warn"], "--"),
                           ("syserr_rate", "System error", PALETTE["primary"], "-."), ("cancel_rate", "Cancelled (not a failure)", PALETTE["muted"], ":")]:
    a.plot(pdm.month, pdm[col] * 100, color=colr, lw=2, ls=ls, label=lab)
a.set_title("Failure rate by type, % of attempts", fontsize=11.5)
a.legend(loc="upper right", fontsize=9)
a = axs[5]
a.plot(pdm.month, pdm.qwait_mean_min, color=PALETTE["primary"], lw=2.2, label="Mean")
a.plot(pdm.month, pdm.qwait_p90_min, color=PALETTE["accent"], lw=2, ls="--", label="p90")
a.set_title("Queue wait of completed swaps (minutes)", fontsize=11.5)
a.set_ylim(0, 8.5)
a.legend(loc="lower right", ncol=2)
a.text(0.02, 0.22, "Flat every month, every hour and at the\nmost stressed stations: this field carries\nno load signal "
       "(survivorship is only a\nhypothesis), so it is a weak KPI",
       transform=a.transAxes, fontsize=9.5, color="#444444")
a = axs[6]
for col, lab, colr, ls in [("revenue_ps", "Revenue", PALETTE["primary"], "-"), ("wear_cost_ps", "Battery wear", PALETTE["accent"], "--"),
                           ("energy_cost_ps", "Energy", PALETTE["warn"], "-."), ("fixed_cost_ps", "Station fixed", PALETTE["muted"], ":")]:
    a.plot(pdm.month, pdm[col], color=colr, lw=2.2, ls=ls, label=lab)
a.set_title(f"Revenue and costs per swap ({RS})", fontsize=11.5)
a.set_ylim(0, 72)
a.legend(loc="upper left", bbox_to_anchor=(0.0, 0.80), fontsize=9)
a = axs[7]
a.axhline(0, color="#555555", lw=1)
a.plot(pdm.month, pdm.cm1_ps, color=PALETTE["primary"], lw=2.4, marker="o", ms=3.5, label="cm1 / swap")
a.plot(pdm.month, pdm.cm2_ps, color=PALETTE["accent"], lw=2.2, ls="--", marker="s", ms=3, label="cm2 / swap")
a.plot(capm.month, capm.cm1_cap_ps, color=PALETTE["muted"], lw=2, ls=":", marker="o", ms=3, label="cm1 / swap, wear stops at EOL")
a.set_title(f"Contribution margin per swap ({RS})", fontsize=11.5)
a.legend(loc="upper left")
a = axs[8]
a.axhline(0, color="#555555", lw=1)
a.bar(pdm.month - pd.Timedelta(days=6), pdm.cm1 / 1e6, width=12, color=PALETTE["primary"], label="cm1 total")
a.bar(pdm.month + pd.Timedelta(days=6), pdm.cm2 / 1e6, width=12, color=PALETTE["accent"], label="cm2 total")
a.set_title(f"Total contribution margin ({RS} M/month)", fontsize=11.5)
a.legend(loc="lower left")
for a in axs:
    month_axis(a, every=3)
title_block(fig, "Volume tripled and per-swap station cost fell, but battery wear per swap rose and margins turned negative",
            "Monthly KPI panel, Jan-2024 to Jun-2025 (ratio-of-sums; costs under our usage-based battery-depreciation assumption). "
            "Apr-Jun-25 margins include wear charged on Kyron packs\nissued below 70% end of life; the dotted line in the margin panel removes it.",
            top=0.9)
fig.subplots_adjust(bottom=0.06)
save_fig(fig, "q1_02_kpi_panel")
print("[fig] q1_02_kpi_panel saved")

# %% [markdown]
# ## 3. "Service failures rose faster than swaps" - naive vs same-season
# The naive comparison takes the first and last months of the window (Jan-24, a low-failure winter month, vs Jun-25,
# a peak-season month). Failures are strongly seasonal, so the fair test compares the same months a year apart.

# %%
def sm_tot(p, launch_only=False):
    d = sm.filter(in_per(p))
    if launch_only:
        d = d.filter(pl.col("grp") == "Launch")
    return d.select(pl.col("attempts").sum(), pl.col("completed").sum(), pl.col("failures").sum(), pl.col("stockouts").sum(),
                    pl.col("abandoned").sum(), pl.col("system_errors").sum(), pl.col("cancelled").sum()).row(0, named=True)


COMPS = [("Naive: Jan-24 vs Jun-25", "Jan-24", "Jun-25", False),
         ("Same month: Jun-24 vs Jun-25", "Jun-24", "Jun-25", False),
         ("Same season: Q1-24 vs Q1-25", "Q1-24", "Q1-25", False),
         ("Same season: Q2-24 vs Q2-25", "Q2-24", "Q2-25", False),
         ("Same season: H1-24 vs H1-25", "H1-24", "H1-25", False),
         ("Launch stations: Q1-24 vs Q1-25", "Q1-24", "Q1-25", True),
         ("Launch stations: Q2-24 vs Q2-25", "Q2-24", "Q2-25", True)]
rows = []
for lab, p0, p1, lo in COMPS:
    a0, a1 = sm_tot(p0, lo), sm_tot(p1, lo)
    rows.append({"comparison": lab, "period_0": p0, "period_1": p1, "scope": "Launch (90 stations)" if lo else "Network",
                 "swaps_growth_x": a1["completed"] / a0["completed"], "attempts_growth_x": a1["attempts"] / a0["attempts"],
                 "failures_growth_x": a1["failures"] / a0["failures"],
                 "fail_rate_0_pct": a0["failures"] / a0["attempts"] * 100, "fail_rate_1_pct": a1["failures"] / a1["attempts"] * 100})
fg = pd.DataFrame(rows)
fg["fail_rate_change_pp"] = fg.fail_rate_1_pct - fg.fail_rate_0_pct
fg["failures_faster_than_swaps"] = fg.failures_growth_x > fg.swaps_growth_x
fg.to_csv(TAB / "q1_failure_growth_comparisons.csv", index=False)
print("\n[3] Growth multiples: does the leadership claim hold?")
print(fg.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 6.6))
yy = np.arange(len(fg))[::-1]
h = 0.36
ax.barh(yy + h / 2, fg.swaps_growth_x, height=h, color=PALETTE["primary"], label="Completed swaps")
ax.barh(yy - h / 2, fg.failures_growth_x, height=h, color=PALETTE["accent"], label="Service failures")
for y_, s_, f_, r0, r1 in zip(yy, fg.swaps_growth_x, fg.failures_growth_x, fg.fail_rate_0_pct, fg.fail_rate_1_pct):
    ax.text(s_ + 0.05, y_ + h / 2, f"{s_:.2f}x", va="center", fontsize=10, color="#333333")
    ax.text(f_ + 0.05, y_ - h / 2, f"{f_:.2f}x   (failure rate {r0:.1f}% -> {r1:.1f}%)", va="center", fontsize=10, color="#333333")
ax.set_yticks(yy)
ax.set_yticklabels(fg.comparison, fontsize=10.5)
ax.axvline(1, color="#888888", lw=1)
ax.set_xlabel("Growth multiple, later period / earlier period")
ax.set_xlim(0, max(fg.failures_growth_x.max(), fg.swaps_growth_x.max()) + 2.2)
ax.legend(loc="lower right")
ax.grid(axis="y", visible=False)
ax.axhline(yy[0] - 0.5, color="#BBBBBB", lw=0.8)
ax.axhline(yy[4] - 0.5, color="#BBBBBB", lw=0.8)
title_block(fig, "\"Failures rose faster than swaps\" only holds for a winter-vs-summer comparison",
            "Same-season, failures grew slower than swaps network-wide; at the 90 Launch stations they grew in line with swaps",
            top=0.86)
save_fig(fig, "q1_03_failure_claim_naive_vs_same_season")
print("[fig] q1_03 saved")

# %% [markdown]
# ## 4. Does the network average hide deterioration? Launch-station panel and demand deciles
# New stations (Wave 1 from Aug-24, Wave 2 from Feb-25) start with low utilisation and low failure rates, so they pull
# the network average down even if established stations are no better. Checks:
# * **Decomposition** of the YoY change in the network rate into a within-Launch change and a new-station mix effect.
# * **Fixed panel** of the 90 Launch stations: station-level paired YoY differences (bootstrap CI over stations,
#   Wilcoxon signed-rank), so clustering by station is respected.
# * **Station-demand deciles** (attempts per active day, ranked within each period) - failure rate YoY by decile.
# CIs for network-level differences use a station-cluster bootstrap (2,000 resamples), not a naive binomial test.

# %%
def station_period(p):
    return (sm.filter(in_per(p)).group_by("station_id", "expansion_wave", "grp", "city").agg(
        *[pl.col(x).sum() for x in ["attempts", "completed", "failures", "stockouts", "abandoned", "system_errors", "cancelled", "active_days"]],
        (pl.col("q_wait_completed_mean") * pl.col("completed")).sum().alias("qsum"))
        .filter(pl.col("attempts") > 0)
        .with_columns((pl.col("failures") / pl.col("attempts")).alias("fail_rate"),
                      (pl.col("attempts") / pl.col("active_days")).alias("demand_per_day"),
                      (pl.col("qsum") / pl.col("completed") / 60).alias("q_mean_min"))
        .sort("station_id").to_pandas())   # deterministic row order -> reproducible bootstrap and decile tie-breaks


SP = {p: station_period(p) for p in ["Q1-24", "Q2-24", "Q1-25", "Q2-25"]}


def boot_ratio(num, den, B=2000):
    n = len(num)
    idx = RNG.integers(0, n, (B, n))
    return num[idx].sum(1) / den[idx].sum(1)


yoy_rows, decomp = [], {}
for q in ("Q1", "Q2"):
    s0, s1 = SP[f"{q}-24"], SP[f"{q}-25"]
    r0 = s0.failures.sum() / s0.attempts.sum()
    r1 = s1.failures.sum() / s1.attempts.sum()
    bnet = boot_ratio(s1.failures.values, s1.attempts.values) - boot_ratio(s0.failures.values, s0.attempts.values)
    w = s1.groupby("grp").attempts.sum() / s1.attempts.sum()
    rg = s1.groupby("grp").failures.sum() / s1.groupby("grp").attempts.sum()
    within = w["Launch"] * (rg["Launch"] - r0)
    dilution = w.get("New", 0) * (rg.get("New", r0) - r0)
    decomp[q] = dict(r0=r0, r1=r1, within=within, dilution=dilution, w_new=w.get("New", 0), r_launch=rg["Launch"], r_new=rg.get("New", np.nan))
    print(f"\n[4] {q}: network failure rate {r0 * 100:.2f}% -> {r1 * 100:.2f}% ({(r1 - r0) * 100:+.2f} pp; station-cluster bootstrap 95% CI "
          f"{np.percentile(bnet, 2.5) * 100:+.2f} to {np.percentile(bnet, 97.5) * 100:+.2f} pp)")
    print(f"     decomposition: within-Launch {within * 100:+.2f} pp + new-station dilution {dilution * 100:+.2f} pp "
          f"(new stations = {w.get('New', 0) * 100:.1f}% of attempts at {rg.get('New', np.nan) * 100:.2f}% vs Launch {rg['Launch'] * 100:.2f}%)")
    # Launch fixed panel, paired by station
    L = s0[s0.grp == "Launch"].merge(s1[s1.grp == "Launch"], on="station_id", suffixes=("_24", "_25"))
    dd = (L.fail_rate_25 - L.fail_rate_24).values
    idx = RNG.integers(0, len(L), (2000, len(L)))
    bl = (L.failures_25.values[idx].sum(1) / L.attempts_25.values[idx].sum(1)) - (L.failures_24.values[idx].sum(1) / L.attempts_24.values[idx].sum(1))
    wil = stats.wilcoxon(dd)
    rl0 = L.failures_24.sum() / L.attempts_24.sum()
    rl1 = L.failures_25.sum() / L.attempts_25.sum()
    dem_g = L.demand_per_day_25 / L.demand_per_day_24 - 1
    rho = stats.spearmanr(dem_g, dd)
    qb = (boot_ratio(L.qsum_25.values, L.completed_25.values) - boot_ratio(L.qsum_24.values, L.completed_24.values)) / 60
    # Paired station-cluster bootstrap for the NETWORK change: the 2024 network is exactly the Launch panel, so each
    # Launch station is resampled together with both of its years; new stations are resampled separately.
    N1 = s1[s1.grp == "New"]
    assert len(L) == len(s0) and L.attempts_25.sum() + N1.attempts.sum() == s1.attempts.sum(), "panel does not cover the network"
    iL = RNG2.integers(0, len(L), (2000, len(L)))
    iN = RNG2.integers(0, len(N1), (2000, len(N1)))
    bpair = ((L.failures_25.values[iL].sum(1) + N1.failures.values[iN].sum(1)) / (L.attempts_25.values[iL].sum(1) + N1.attempts.values[iN].sum(1))
             - L.failures_24.values[iL].sum(1) / L.attempts_24.values[iL].sum(1))
    print(f"     paired station-cluster bootstrap 95% CI for the network change: {np.percentile(bpair, 2.5) * 100:+.2f} to "
          f"{np.percentile(bpair, 97.5) * 100:+.2f} pp (the unpaired CI above resamples the two years independently, so it is wider)")
    print(f"     Launch panel (n={len(L)} stations): pooled {rl0 * 100:.2f}% -> {rl1 * 100:.2f}% ({(rl1 - rl0) * 100:+.2f} pp, paired bootstrap "
          f"95% CI {np.percentile(bl, 2.5) * 100:+.2f} to {np.percentile(bl, 97.5) * 100:+.2f}); median station change {np.median(dd) * 100:+.2f} pp; "
          f"{(dd > 0).sum()}/{len(dd)} stations worse, {(dd <= 0).sum()} not worse; Wilcoxon p={wil.pvalue:.3g}")
    print(f"     Launch demand per station-day {L.demand_per_day_24.mean():.0f} -> {L.demand_per_day_25.mean():.0f} attempts "
          f"({(L.attempts_25.sum() / L.attempts_24.sum() - 1) * 100:+.1f}% attempts); Spearman(demand growth, failure-rate change) "
          f"rho={rho.statistic:.2f}, p={rho.pvalue:.3g}")
    for scope, d0, d1 in [("Network", s0, s1), ("Launch", s0[s0.grp == "Launch"], s1[s1.grp == "Launch"]), ("New stations", None, s1[s1.grp == "New"])]:
        for metric, col in [("Service-failure rate", "failures"), ("Stockout rate", "stockouts"), ("Abandoned rate", "abandoned"),
                            ("System-error rate", "system_errors"), ("Cancelled rate (not a failure)", "cancelled")]:
            v0 = d0[col].sum() / d0.attempts.sum() * 100 if d0 is not None else np.nan
            v1 = d1[col].sum() / d1.attempts.sum() * 100
            row = {"quarter": q, "scope": scope, "metric": metric, "2024": v0, "2025": v1, "change": v1 - v0, "unit": "% of attempts"}
            if metric == "Service-failure rate" and scope == "Network":
                row.update(ci_lo=np.percentile(bnet, 2.5) * 100, ci_hi=np.percentile(bnet, 97.5) * 100,
                           ci_paired_lo=np.percentile(bpair, 2.5) * 100, ci_paired_hi=np.percentile(bpair, 97.5) * 100)
            if metric == "Service-failure rate" and scope == "Launch":
                row.update(ci_lo=np.percentile(bl, 2.5) * 100, ci_hi=np.percentile(bl, 97.5) * 100, p_wilcoxon=wil.pvalue)
            yoy_rows.append(row)
    # queue waits: means from station-month (weighted), p90 from the event frame
    for scope, flt in [("Network", None), ("Launch", True), ("New stations", False)]:
        def qstats(p):
            d = qw.filter(in_per(p))
            if flt is not None:
                d = d.filter(pl.col("is_launch") == flt)
            if d.height == 0:
                return np.nan, np.nan
            return d["queue_wait_sec"].mean() / 60, d["queue_wait_sec"].quantile(0.9) / 60
        m0, p90_0 = qstats(f"{q}-24")
        m1, p90_1 = qstats(f"{q}-25")
        row = {"quarter": q, "scope": scope, "metric": "Queue wait, mean", "2024": m0, "2025": m1, "change": m1 - m0, "unit": "minutes"}
        if scope == "Launch":
            row.update(ci_lo=np.percentile(qb, 2.5), ci_hi=np.percentile(qb, 97.5))
        yoy_rows.append(row)
        yoy_rows.append({"quarter": q, "scope": scope, "metric": "Queue wait, p90", "2024": p90_0, "2025": p90_1, "change": p90_1 - p90_0, "unit": "minutes"})
    L[["station_id", "city_24", "attempts_24", "attempts_25", "fail_rate_24", "fail_rate_25", "demand_per_day_24", "demand_per_day_25",
       "q_mean_min_24", "q_mean_min_25"]].assign(fail_rate_change_pp=dd * 100).to_csv(TAB / f"q1_launch_panel_station_yoy_{q.lower()}.csv", index=False)

yoy = pd.DataFrame(yoy_rows)
yoy.to_csv(TAB / "q1_yoy_service_metrics.csv", index=False)
print("\n[4] Same-season YoY service metrics (rates in % of attempts, waits in minutes)")
print(yoy.round(3).to_string(index=False))

# %% [markdown]
# ### Failure rate by station-demand decile, YoY
# Deciles are formed within each period on attempts per active station-day (decile 10 = busiest stations). The 2025
# deciles contain new stations; the share of new stations per decile shows where dilution comes from.

# %%
dec_rows = []
for q in ("Q1", "Q2"):
    for yr in ("24", "25"):
        s = SP[f"{q}-{yr}"].copy()
        s["decile"] = pd.qcut(s.demand_per_day.rank(method="first"), 10, labels=range(1, 11)).astype(int)
        g = s.groupby("decile").agg(stations=("station_id", "size"), attempts=("attempts", "sum"), failures=("failures", "sum"),
                                    stockouts=("stockouts", "sum"), abandoned=("abandoned", "sum"),
                                    median_demand_per_day=("demand_per_day", "median"), new_station_share=("grp", lambda x: (x == "New").mean()))
        g["fail_rate_pct"] = g.failures / g.attempts * 100
        g["stockout_rate_pct"] = g.stockouts / g.attempts * 100
        g["abandoned_rate_pct"] = g.abandoned / g.attempts * 100
        for gname, gl in [("Launch", "launch"), ("New", "new")]:   # within-decile split (Simpson check)
            sg = s[s.grp == gname].groupby("decile").agg(n=("station_id", "size"), a=("attempts", "sum"), f=("failures", "sum"))
            g[f"n_{gl}"] = sg.n.reindex(g.index).fillna(0).astype(int)
            g[f"fail_rate_{gl}_pct"] = (sg.f / sg.a * 100).reindex(g.index)
        g["quarter"], g["year"] = q, 2000 + int(yr)
        dec_rows.append(g.reset_index())
dec = pd.concat(dec_rows)
dec.to_csv(TAB / "q1_failure_by_demand_decile.csv", index=False)
pv = dec.pivot_table(index="decile", columns=["quarter", "year"], values="fail_rate_pct")
print("\n[4] Service-failure rate (%) by station-demand decile (10 = busiest), same quarter YoY")
print(pv.round(2))
print("\n[4] Median attempts per station-day by decile")
print(dec.pivot_table(index="decile", columns=["quarter", "year"], values="median_demand_per_day").round(0))
print("\n[4] Share of new-wave stations in each 2025 decile")
print(dec[dec.year == 2025].pivot_table(index="decile", columns="quarter", values="new_station_share").round(2))
print("\n[4] Q2-2025 failure rate (%) within each demand decile, Launch vs new stations (n = stations)")
print(dec[(dec.year == 2025) & (dec.quarter == "Q2")][["decile", "stations", "n_launch", "fail_rate_launch_pct", "n_new", "fail_rate_new_pct", "fail_rate_pct"]]
      .round(2).to_string(index=False))
for q in ("Q1", "Q2"):
    for yr in (2024, 2025):
        d = dec[(dec.quarter == q) & (dec.year == yr)]
        rho = stats.spearmanr(d.median_demand_per_day, d.fail_rate_pct)
        print(f"[4] {q}-{yr}: Spearman(decile demand, decile failure rate) rho={rho.statistic:.2f} (p={rho.pvalue:.3f}); "
              f"top decile {d.loc[d.decile == 10, 'fail_rate_pct'].iloc[0]:.2f}% vs bottom {d.loc[d.decile == 1, 'fail_rate_pct'].iloc[0]:.2f}%")

# Launch fixed panel by 2024 demand quintile (avoids re-ranking; same stations both years)
lq_rows = []
for q in ("Q1", "Q2"):
    s0, s1 = SP[f"{q}-24"], SP[f"{q}-25"]
    L = s0[s0.grp == "Launch"].merge(s1[s1.grp == "Launch"], on="station_id", suffixes=("_24", "_25"))
    L["quintile_2024"] = pd.qcut(L.demand_per_day_24.rank(method="first"), 5, labels=range(1, 6)).astype(int)
    g = L.groupby("quintile_2024").agg(stations=("station_id", "size"), att24=("attempts_24", "sum"), att25=("attempts_25", "sum"),
                                       f24=("failures_24", "sum"), f25=("failures_25", "sum"),
                                       dem24=("demand_per_day_24", "mean"), dem25=("demand_per_day_25", "mean"))
    g["fail_rate_2024_pct"] = g.f24 / g.att24 * 100
    g["fail_rate_2025_pct"] = g.f25 / g.att25 * 100
    g["change_pp"] = g.fail_rate_2025_pct - g.fail_rate_2024_pct
    g["attempt_growth_pct"] = (g.att25 / g.att24 - 1) * 100
    g["quarter"] = q
    lq_rows.append(g.reset_index())
lq = pd.concat(lq_rows)
lq.to_csv(TAB / "q1_launch_panel_by_2024_demand_quintile.csv", index=False)
print("\n[4] Launch stations (fixed panel) by 2024 demand quintile (5 = busiest)")
print(lq[["quarter", "quintile_2024", "stations", "dem24", "dem25", "attempt_growth_pct", "fail_rate_2024_pct", "fail_rate_2025_pct", "change_pp"]].round(2).to_string(index=False))

# %% [markdown]
# ### Why do new stations fail less? Load vs equipment (Simpson check)
# Hypothesis from the brief: new stations are *low-utilisation* and dilute the average. Test it: compare utilisation
# (attempts per slot-day) and failure rates by **charger generation** and **city**, Launch vs new, same quarter.
# New stations were built with Gen2/Gen3 chargers only; 51 of the 90 Launch stations are Gen1.

# %%
sm_q = sm.with_columns(pl.col("month").dt.year().alias("yr"), pl.col("month").dt.month().alias("mo"),
                       (pl.col("active_days") * (pl.col("slots_2w") + pl.col("slots_3w"))).alias("slot_days"))
util = (sm_q.filter(in_per("Q2-25")).group_by("grp").agg(pl.col("station_id").n_unique().alias("stations"),
        (pl.col("attempts").sum() / pl.col("slot_days").sum()).alias("attempts_per_slot_day"),
        (pl.col("attempts").sum() / pl.col("active_days").sum()).alias("attempts_per_station_day"),
        (pl.col("failures").sum() / pl.col("attempts").sum() * 100).alias("fail_rate_pct")).sort("grp"))
print("\n[4b] Q2-2025 utilisation and failure rate, Launch vs new stations")
print(util)
cg_rows = []
for q, (m0, m1) in {"Q1": (1, 3), "Q2": (4, 6)}.items():
    t = (sm_q.filter(pl.col("mo").is_between(m0, m1)).group_by("yr", "grp", "charger_generation")
         .agg(pl.col("station_id").n_unique().alias("stations"), pl.col("attempts").sum(), pl.col("failures").sum(), pl.col("stockouts").sum(),
              (pl.col("attempts").sum() / pl.col("slot_days").sum()).alias("attempts_per_slot_day"))
         .with_columns((pl.col("failures") / pl.col("attempts") * 100).alias("fail_rate_pct"),
                       (pl.col("stockouts") / pl.col("attempts") * 100).alias("stockout_rate_pct"), pl.lit(q).alias("quarter"))
         .sort("grp", "charger_generation", "yr"))
    cg_rows.append(t)
cg = pl.concat(cg_rows).to_pandas()
cg.to_csv(TAB / "q1_failure_by_charger_generation.csv", index=False)
print("\n[4b] Failure rate by station group x charger generation, same quarter YoY")
print(cg[["quarter", "yr", "grp", "charger_generation", "stations", "attempts", "attempts_per_slot_day", "fail_rate_pct", "stockout_rate_pct"]].round(2).to_string(index=False))
cc = (sm_q.filter(pl.col("mo").is_between(4, 6)).group_by("yr", "city", "grp", "charger_generation")
      .agg(pl.col("station_id").n_unique().alias("stations"), (pl.col("failures").sum() / pl.col("attempts").sum() * 100).alias("fail_rate_pct"))
      .sort("city", "charger_generation", "grp", "yr").to_pandas())
cc.to_csv(TAB / "q1_q2_failure_by_city_generation_group.csv", index=False)
ccp = cc.pivot_table(index=["city", "charger_generation"], columns=["grp", "yr"], values="fail_rate_pct").round(1)
print("\n[4b] Q2 failure rate (%) by city x charger generation: Launch (2024, 2025) vs new (2025) - like-for-like cells")
print(ccp.to_string())
q2_25 = sm_q.filter(in_per("Q2-25"))
g1 = q2_25.filter((pl.col("grp") == "Launch") & (pl.col("charger_generation") == "Gen1"))
g1_share_st = g1["station_id"].n_unique() / q2_25["station_id"].n_unique()
g1_share_att = g1["attempts"].sum() / q2_25["attempts"].sum()
g1_share_fail = g1["failures"].sum() / q2_25["failures"].sum()
print(f"[4b] Q2-25: the {g1['station_id'].n_unique()} Gen1 Launch stations = {g1_share_st * 100:.0f}% of stations, {g1_share_att * 100:.0f}% of attempts, "
      f"{g1_share_fail * 100:.0f}% of service failures ({g1['failures'].sum():,} of {q2_25['failures'].sum():,})")
HOT = ["Delhi NCR", "Jaipur", "Hyderabad"]
g1h = q2_25.filter((pl.col("charger_generation") == "Gen1") & pl.col("city").is_in(HOT))
print(f"[4b] Q2-25: Gen1 stations in Delhi NCR/Jaipur/Hyderabad = {g1h['station_id'].n_unique()} stations, "
      f"{g1h['failures'].sum() / q2_25['failures'].sum() * 100:.0f}% of network service failures, failure rate "
      f"{g1h['failures'].sum() / g1h['attempts'].sum() * 100:.1f}%")
# What-if (association, city-controlled): Gen1 stations at the Q2-25 failure rate of Gen2 stations in the SAME city
g2_city = q2_25.filter(pl.col("charger_generation") == "Gen2").group_by("city").agg((pl.col("failures").sum() / pl.col("attempts").sum()).alias("r_gen2_city"))
g1_city = (q2_25.filter(pl.col("charger_generation") == "Gen1").group_by("city").agg(pl.col("attempts").sum(), pl.col("failures").sum())
           .join(g2_city, on="city", how="left").with_columns((pl.col("failures") - pl.col("attempts") * pl.col("r_gen2_city")).alias("avoidable"))
           .sort("city"))
print("[4b] Gen1 vs same-city Gen2, Q2-25:")
print(g1_city.with_columns((pl.col("failures") / pl.col("attempts") * 100).round(2).alias("gen1_rate_pct"), (pl.col("r_gen2_city") * 100).round(2).alias("gen2_rate_pct")))
avoid = g1_city["avoidable"].sum()
tot_f, tot_a = q2_25["failures"].sum(), q2_25["attempts"].sum()
print(f"[4b] What-if: Gen1 stations failing at their city's Gen2 rate -> {avoid:,.0f} fewer service failures in Q2-25 "
      f"({avoid / 3:,.0f}/month, {avoid / tot_f * 100:.0f}% of Q2-25 failures); network Q2-25 rate {tot_f / tot_a * 100:.2f}% -> {(tot_f - avoid) / tot_a * 100:.2f}%")
# Upper scenario: Gen1 at the same-city Gen3 rate. Every Gen3 station is new (Wave 1/2), so this is confounded with site age.
g3_city = (q2_25.filter(pl.col("charger_generation") == "Gen3").group_by("city")
           .agg((pl.col("failures").sum() / pl.col("attempts").sum()).alias("r_gen3_city")))
g1_city = g1_city.join(g3_city, on="city", how="left").with_columns(
    (pl.col("failures") - pl.col("attempts") * pl.col("r_gen3_city")).alias("avoidable_gen3"))
avoid3 = g1_city["avoidable_gen3"].sum()
print("[4b] Q2-25 same-city failure rate, Gen2 vs Gen3 (%): " + ", ".join(
    f"{r['city']} {r['r_gen2_city'] * 100:.2f} vs {r['r_gen3_city'] * 100:.2f}" for r in g1_city.iter_rows(named=True) if r["r_gen3_city"] is not None))
print(f"[4b] What-if (upper, confounded with site age): Gen1 at the same-city Gen3 rate -> {avoid3:,.0f} fewer failures in Q2-25 "
      f"({avoid3 / tot_f * 100:.0f}% of failures); network Q2-25 rate -> {(tot_f - avoid3) / tot_a * 100:.2f}%. Gen1 cities with no Gen3 station "
      f"(left at their own rate): {g1_city.filter(pl.col('r_gen3_city').is_null())['city'].to_list()}")

# %%
# Figure 4: same-month YoY failure rate by station group + Q2 failure rate by charger generation
fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.6), gridspec_kw={"width_ratios": [1, 1.05]})
mo = np.arange(1, 7)
mlab = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]


def grp_rate(yr, grp=None):
    d = sm.filter(pl.col("month").dt.year() == yr)
    if grp:
        d = d.filter(pl.col("grp") == grp)
    d = d.group_by(pl.col("month").dt.month().alias("m")).agg(pl.col("failures").sum(), pl.col("attempts").sum()).sort("m")
    d = d.filter(pl.col("m") <= 6)
    return d["m"].to_numpy(), (d["failures"] / d["attempts"] * 100).to_numpy()


x, y24 = grp_rate(2024)
_, y25 = grp_rate(2025)
_, y25l = grp_rate(2025, "Launch")
_, y25n = grp_rate(2025, "New")
a1.plot(x, y24, color=PALETTE["muted"], lw=3.2, marker="o", ms=7, label="2024, all stations (= the 90 Launch)")
a1.plot(x, y25l, color=PALETTE["primary"], lw=2.2, ls="--", marker="D", ms=5, label="2025, the same 90 Launch stations")
a1.plot(x, y25, color=PALETTE["accent"], lw=2.6, marker="o", label="2025, network average (150 stations)")
a1.plot(x, y25n, color=SERIES[5], lw=2.2, ls=":", marker="s", label="2025, new stations (Wave 1 + 2)")
a1.annotate(f"May: {y24[4]:.1f}% (2024) vs {y25l[4]:.1f}% (2025)\nat the same 90 Launch stations", (5, y24[4]), xytext=(1.25, 9.3),
            textcoords="data", fontsize=10.5, color="#222222", arrowprops=dict(arrowstyle="-", color="#999999"))
a1.annotate(f"{y25[4]:.1f}% network", (5, y25[4]), xytext=(10, -16), textcoords="offset points", fontsize=10, color="#222222")
a1.annotate(f"{y25n[4]:.1f}% new", (5, y25n[4]), xytext=(10, -14), textcoords="offset points", fontsize=10, color="#222222")
a1.set_xticks(mo)
a1.set_xticklabels(mlab)
a1.set_ylabel("Service-failure rate, % of attempts")
a1.set_ylim(0, 14.5)
a1.set_xlim(0.7, 6.6)
a1.set_title("Same month, a year apart", fontsize=12, loc="left")
a1.legend(loc="upper left", fontsize=9.5)
cats = [("Launch", "Gen1"), ("Launch", "Gen2"), ("New", "Gen2"), ("New", "Gen3")]
cq = cg[cg.quarter == "Q2"]
xc = np.arange(len(cats))
wc = 0.36
for j, (yr, colr) in enumerate([(2024, PALETTE["muted"]), (2025, PALETTE["accent"])]):
    for i, (gname, gen) in enumerate(cats):
        r = cq[(cq.yr == yr) & (cq.grp == gname) & (cq.charger_generation == gen)]
        if len(r) == 0:
            continue
        xv = xc[i] + (j - 0.5) * wc if gname == "Launch" else xc[i]
        a2.bar(xv, r.fail_rate_pct.iloc[0], width=wc, color=colr, edgecolor="white", linewidth=1.5, label=f"Q2-{yr}" if i == 0 else None)
        a2.text(xv, r.fail_rate_pct.iloc[0] + 0.2, f"{r.fail_rate_pct.iloc[0]:.1f}%", ha="center", va="bottom", fontsize=10.5, color="#222222")
nst = {(r.grp, r.charger_generation): int(r.stations) for r in cq[cq.yr == 2025].itertuples()}
a2.set_xticks(xc)
a2.set_xticklabels([f"{g} {gen}\n({nst[(g, gen)]} stations)" for g, gen in cats], fontsize=10.5)
a2.set_ylabel("Q2 service-failure rate, % of attempts")
a2.set_ylim(0, 14.5)
a2.grid(axis="x", visible=False)
a2.legend(loc="upper left", bbox_to_anchor=(0.30, 1.0))
q1min, q1max = cg[cg.quarter == "Q1"].fail_rate_pct.min(), cg[cg.quarter == "Q1"].fail_rate_pct.max()
a2.set_title(f"Q2 failure rate by charger generation (Q1: every group {q1min:.1f}-{q1max:.1f}%)", fontsize=12, loc="left")
u_new = util.filter(pl.col("grp") == "New")["attempts_per_slot_day"].item()
u_l = util.filter(pl.col("grp") == "Launch")["attempts_per_slot_day"].item()
lchg = cq[cq.grp == "Launch"].pivot_table(index="charger_generation", columns="yr", values="fail_rate_pct")
lmax = (lchg[2025] - lchg[2024]).abs().max()
a2.text(0.98, 0.80, f"Launch Gen1 and Gen2 rates moved by\nat most {lmax:.2f} pp year on year.\nGen1 = {g1_share_st * 100:.0f}% of stations but {g1_share_fail * 100:.0f}% of\n"
        f"Q2-25 failures. New stations are not\nlighter-loaded ({u_new:.1f} vs {u_l:.1f} attempts per\nslot-day); none of them is Gen1.",
        transform=a2.transAxes, ha="right", va="top", fontsize=10, color="#333333")
d_q2 = decomp["Q2"]
title_block(fig, "The summer-2025 'improvement' is a mix effect: the same Launch stations failed as often as in 2024",
            f"Q2 network rate {d_q2['r0'] * 100:.1f}% -> {d_q2['r1'] * 100:.1f}% = within-Launch {d_q2['within'] * 100:+.2f} pp + new Gen2/Gen3 stations "
            f"{d_q2['dilution'] * 100:+.2f} pp (they carry {d_q2['w_new'] * 100:.0f}% of attempts at {d_q2['r_new'] * 100:.1f}% vs {d_q2['r_launch'] * 100:.1f}% at Launch)",
            top=0.84)
save_fig(fig, "q1_04_failure_dilution_launch_vs_network")
print("[fig] q1_04 saved")

# %%
# Figure 8: station-demand deciles (within-decile Launch vs new) and the Launch fixed panel by 2024 demand quintile
fig, (a1, a2) = plt.subplots(1, 2, figsize=(17.5, 6.8), gridspec_kw={"width_ratios": [1.3, 1]})
q2 = dec[dec.quarter == "Q2"]
xd = np.arange(1, 11)
wd = 0.27
b24 = q2[q2.year == 2024].set_index("decile").reindex(xd)
b25 = q2[q2.year == 2025].set_index("decile").reindex(xd)
a1.bar(xd - wd, b24.fail_rate_pct, width=wd, color=PALETTE["muted"], edgecolor="white", label="Q2-2024, all 90 stations (Launch)")
a1.bar(xd, b25.fail_rate_launch_pct, width=wd, color=PALETTE["primary"], edgecolor="white", label="Q2-2025, Launch stations in decile")
a1.bar(xd + wd, b25.fail_rate_new_pct, width=wd, color=SERIES[5], edgecolor="white", hatch="//", label="Q2-2025, new stations in decile")
a1.set_xticks(xd)
a1.set_xticklabels([f"D{i}\n{b24.median_demand_per_day.loc[i]:.0f} / {b25.median_demand_per_day.loc[i]:.0f}" for i in xd], fontsize=9.8)
a1.set_xlabel("Station-demand decile within each year (D10 = busiest)\ntick: median attempts per station-day, 2024 / 2025")
a1.set_ylabel("Q2 service-failure rate, % of attempts")
a1.set_ylim(0, 17)
a1.grid(axis="x", visible=False)
a1.legend(loc="upper left", fontsize=9.5, ncol=1)
a1.set_title("Within every demand decile, new stations fail less than Launch stations", fontsize=12, loc="left")
lq2 = lq[lq.quarter == "Q2"].set_index("quintile_2024")
xq = np.arange(1, 6)
a2.bar(xq - 0.19, lq2.fail_rate_2024_pct, width=0.38, color=PALETTE["muted"], edgecolor="white", label="Q2-2024")
a2.bar(xq + 0.19, lq2.fail_rate_2025_pct, width=0.38, color=PALETTE["primary"], edgecolor="white", label="Q2-2025")
for i in xq:
    a2.text(i, max(lq2.fail_rate_2024_pct.loc[i], lq2.fail_rate_2025_pct.loc[i]) + 0.3,
            f"{lq2.change_pp.loc[i]:+.1f} pp\nvol. {lq2.attempt_growth_pct.loc[i]:+.0f}%", ha="center", fontsize=9.8, color="#222222")
a2.set_xticks(xq)
a2.set_xticklabels([f"Q{i}" + ("\nquietest" if i == 1 else "\nbusiest" if i == 5 else "") for i in xq], fontsize=10.5)
a2.set_xlabel("Launch stations by 2024 demand quintile\n(18 each, same stations both years)\nlabel: failure-rate change; vol. = growth in attempts")
a2.set_ylabel("Q2 service-failure rate, % of attempts")
a2.set_ylim(0, 17)
a2.grid(axis="x", visible=False)
a2.legend(loc="upper left")
a2.set_title("Fixed Launch panel: flat at every demand level", fontsize=12, loc="left")
launch_growth = (SP["Q2-25"].loc[SP["Q2-25"].grp == "Launch", "attempts"].sum() / SP["Q2-24"]["attempts"].sum() - 1) * 100
title_block(fig, "No hidden deterioration at the busiest stations: Launch failure rates are flat YoY at every demand level",
            f"Lower 2025 decile rates come from the new stations inside each decile; the 90 Launch stations absorbed "
            f"{launch_growth:.0f}% more Q2 attempts at an unchanged failure rate", top=0.84)
save_fig(fig, "q1_08_failure_by_demand_decile_and_launch_panel")
print("[fig] q1_08 saved")

# %% [markdown]
# ### Is queue wait a load signal at all? (testing the survivorship explanation)
# The completed-swap wait is flat every month. One explanation is survivorship: stressed riders abandon instead of
# waiting longer. If that were the whole story, the wait should still rise somewhat with load (by hour of day, at the
# most stressed stations) and abandoned riders should wait longer in peak months. Check all three.

# %%
qx = (events(["month", "hour", "event_type", "queue_wait_sec", "city", "charger_generation"])
      .group_by("month", "hour", "city", "charger_generation", "event_type")
      .agg(pl.len().alias("n"), pl.col("queue_wait_sec").count().alias("nq"), pl.col("queue_wait_sec").cast(pl.Float64).sum().alias("s"))
      .collect(engine="streaming"))
qcomp = qx.filter(pl.col("event_type") == "swap_completed")
q_hour = qcomp.group_by("hour").agg((pl.col("s").sum() / pl.col("nq").sum() / 60).alias("wait_min"), pl.col("n").sum().alias("swaps")).sort("hour")
q_hour_main = q_hour.filter(pl.col("swaps") >= 0.01 * q_hour["swaps"].sum())      # hours carrying >= 1% of swaps
q_ab = (qx.filter(pl.col("event_type") == "abandoned_queue").group_by("month")
        .agg((pl.col("s").sum() / pl.col("nq").sum() / 60).alias("wait_min")).sort("month"))
may25 = qx.filter(pl.col("month") == M(2025, 5)).with_columns(
    (pl.col("city").is_in(HOT) & (pl.col("charger_generation") == "Gen1")).alias("hot_gen1"))


def cell_stats(d):
    dc = d.filter(pl.col("event_type") == "swap_completed")
    return dict(wait_min=dc["s"].sum() / dc["nq"].sum() / 60, swaps=dc["n"].sum(),
                fail_rate_pct=d.filter(pl.col("event_type").is_in(FAIL_TYPES))["n"].sum() / d["n"].sum() * 100)


hot_c = cell_stats(may25.filter(pl.col("hot_gen1")))
other_cells = pd.DataFrame([{"city": k[0], "charger_generation": k[1], **cell_stats(g)}
                            for k, g in may25.filter(~pl.col("hot_gen1")).group_by("city", "charger_generation")])
other_cells = other_cells[other_cells.swaps >= 5000]
print(f"[4c] Completed-swap wait by hour of day (hours with >=1% of swaps): {q_hour_main['wait_min'].min():.3f}-{q_hour_main['wait_min'].max():.3f} min "
      f"(all 24 hours: {q_hour['wait_min'].min():.3f}-{q_hour['wait_min'].max():.3f})")
print(f"[4c] May-25, Gen1 stations in Delhi NCR/Jaipur/Hyderabad: failure rate {hot_c['fail_rate_pct']:.1f}%, completed wait {hot_c['wait_min']:.3f} min; "
      f"other city x generation cells (>=5,000 swaps): failure rate {other_cells.fail_rate_pct.min():.1f}-{other_cells.fail_rate_pct.max():.1f}%, "
      f"completed wait {other_cells.wait_min.min():.3f}-{other_cells.wait_min.max():.3f} min")
print(f"[4c] Abandoned-queue wait by month: {q_ab['wait_min'].min():.2f}-{q_ab['wait_min'].max():.2f} min "
      f"(May-24 {q_ab.filter(pl.col('month') == M(2024, 5))['wait_min'].item():.2f}, Oct-24 {q_ab.filter(pl.col('month') == M(2024, 10))['wait_min'].item():.2f})")
pd.concat([q_hour.to_pandas().assign(slice="completed wait by hour"),
           other_cells.assign(slice="May-25 completed wait, other city x generation cells"),
           pd.DataFrame([{**hot_c, "slice": "May-25 completed wait, hot-city Gen1"}]),
           q_ab.to_pandas().assign(slice="abandoned wait by month")]).to_csv(TAB / "q1_queue_wait_load_check.csv", index=False)

# %% [markdown]
# ### Is a failed attempt a delayed swap or a lost one?
# For every Q2-25 service failure, the time until the same rider's next completed swap (any station). A short gap
# means the swap was only delayed; a long gap or none means riding time, and probably revenue, was lost.

# %%
rr = (events(["rider_id", "event_ts", "month", "is_completed", "is_service_failure"]).filter(in_per("Q2-25"))
      .select("rider_id", "event_ts", "is_completed", "is_service_failure").collect(engine="streaming").sort("rider_id", "event_ts"))
rr = rr.with_columns(pl.when(pl.col("is_completed")).then(pl.col("event_ts")).alias("cts")).with_columns(
    pl.col("cts").backward_fill().over("rider_id").alias("next_completed"))
gap = rr.filter(pl.col("is_service_failure")).select(((pl.col("next_completed") - pl.col("event_ts")).dt.total_minutes()).alias("gap_min"))["gap_min"]
lost_ub = P["Q2-25"]["failures"] * P["Q2-25"]["revenue_ps"]
within = {m: (gap <= m).fill_null(False).mean() * 100 for m in (60, 180, 1440)}   # denominator = all failures
print(f"[4d] Q2-25 service failures ({gap.len():,}) followed by the same rider's completed swap within 60 min: {within[60]:.1f}%; within 3 h: "
      f"{within[180]:.1f}%; within 24 h: {within[1440]:.1f}%; never in Q2-25: {gap.is_null().mean() * 100:.1f}%")
print(f"[4d] Upper bound on Q2-25 revenue lost to failures (failures x revenue per swap): Rs{lost_ub / 1e6:.2f}M "
      f"({P['Q2-25']['failures']:,} failures x Rs{P['Q2-25']['revenue_ps']:.2f})")
del rr, gap

# %% [markdown]
# ## 5. Contribution-margin bridge (hero figure 2)
# **Exact shift-share.** Segments s = vehicle class (2W/3W) x billing (independent / ZipDrop / other partner) x pilot city
# (Bengaluru+Pune vs other). With w = segment share of swaps:
# * **Mix** = sum_s (w1 - w0) x cm1_0,s (segment composition, e.g. more 3W or partner swaps).
# * **Rate effects** (weighted by w1) split exactly, per segment, using revenue = list x (1 - d) + pilot adjustment
#   (d = discount / list; the pilot adjustment = amount charged - list + discount, i.e. +Rs15 peak / -Rs8 off-peak):
#   - base list price = dList x (1 - d0) (net of partner discount at the old rate);
#   - partner discount rate = -List1 x dd, split into **ZipDrop** (12% -> 28% amendment) and other partners;
#   - pilot = d(pilot adjustment);
#   - energy = kWh/swap effect (dkWh x tariff0) + grid-tariff effect (kWh1 x dtariff);
#   - battery wear = **supplier-share (Kyron) effect** = sum_k du_k x reference wear_k (full-window wear per swap of
#     supplier k, same vehicle class) + within-supplier change (the rest).
# * Fixed cost (for cm2) is taken at network level: d(cm2) = d(cm1) - d(fixed cost per swap).
# The components sum exactly to the change in cm1 per swap (checked below).
#
# **Window choice.** The headline bridges end in Mar-25: H1-24 -> Jul-Aug-24 -> Oct-24..Mar-25, and the same-season
# Q1-24 -> Q1-25. From Apr-25 the per-cycle wear is also charged on Kyron packs issued below 70% EOL (section 5b), which
# inflates the Kyron term, so the H1-25 and Q2-25 bridges are printed for transparency only and flagged.

# %%
seg = (events(["month", "is_completed", "vehicle_class", "partner_id", "city", "supplier_out", "list_price_inr", "discount_inr",
               "amount_charged_inr", "energy_cost", "energy_to_recharge_kwh", "wear_cost", "fixed_cost", "soh_out_pct"])
       .filter(c)
       .with_columns(pl.when(pl.col("partner_id").is_null()).then(pl.lit("Independent"))
                     .when(pl.col("partner_id") == ZIPDROP_ID).then(pl.lit("ZipDrop")).otherwise(pl.lit("Other partner")).alias("billing"),
                     pl.col("city").is_in(PILOT_CITIES).alias("pilot_city"),
                     (pl.col("soh_out_pct") < EOL_SOH).fill_null(False).alias("post_eol"))
       .group_by("month", "vehicle_class", "billing", "pilot_city", "supplier_out", "post_eol")
       .agg(pl.len().alias("n"),
            pl.col("list_price_inr").cast(pl.Float64).sum().alias("L"), pl.col("discount_inr").cast(pl.Float64).sum().alias("D"),
            pl.col("amount_charged_inr").cast(pl.Float64).sum().alias("A"), pl.col("energy_cost").cast(pl.Float64).sum().alias("E"),
            pl.col("energy_to_recharge_kwh").cast(pl.Float64).sum().alias("K"), pl.col("wear_cost").cast(pl.Float64).sum().alias("W"),
            pl.col("fixed_cost").cast(pl.Float64).sum().alias("F"))
       .collect(engine="streaming"))
SEG = ["vehicle_class", "billing", "pilot_city"]
chk = seg.group_by("month").agg((pl.col("A") - pl.col("E") - pl.col("W")).sum().alias("cm1")).sort("month").join(mon.select("month", "cm1"), on="month")
print(f"[5] segment table rows={seg.height}; max |cm1 check| vs monthly panel = {float((chk['cm1'] - chk['cm1_right']).abs().max()):.4f}")
gref = (seg.group_by("vehicle_class", "supplier_out").agg((pl.col("W").sum() / pl.col("n").sum()).alias("g_ref"), pl.col("n").sum())
        .sort("vehicle_class", "supplier_out"))
print("[5] reference wear per swap by supplier x class (full window):")
print(gref)
GREF = {(r["vehicle_class"], r["supplier_out"]): r["g_ref"] for r in gref.iter_rows(named=True)}
print(f"[5] Kyron 2W wear per swap = {GREF[('2W', 'Kyron')] / GREF[('2W', 'Cellora')]:.2f}x Cellora 2W "
      f"({GREF[('2W', 'Kyron')]:.1f} vs {GREF[('2W', 'Cellora')]:.1f} Rs/swap)")


def seg_stats(p):
    g = seg.filter(in_per(p))
    s = g.group_by(SEG).agg([pl.col(x).sum() for x in ["n", "L", "D", "A", "E", "K", "W", "F"]]).to_pandas().set_index(SEG)
    s["Lm"], s["Dm"], s["Am"], s["Em"], s["Km"], s["Wm"] = s.L / s.n, s.D / s.n, s.A / s.n, s.E / s.n, s.K / s.n, s.W / s.n
    s["d"] = s.Dm / s.Lm
    s["Pm"] = s.Am - s.Lm + s.Dm
    s["tau"] = s.E / s.K
    s["cm1m"] = s.Am - s.Em - s.Wm
    u = (g.group_by(SEG + ["supplier_out"]).agg(pl.col("n").sum()).to_pandas()
         .pivot_table(index=SEG, columns="supplier_out", values="n", aggfunc="sum").fillna(0))
    u = u.div(u.sum(axis=1), axis=0)
    return s, u, g["n"].sum(), g["F"].sum()


def bridge(p0, p1):
    s0, u0, N0, F0 = seg_stats(p0)
    s1, u1, N1, F1 = seg_stats(p1)
    idx = s0.index.union(s1.index)
    s0, s1 = s0.reindex(idx), s1.reindex(idx)
    w0, w1 = (s0.n / N0).fillna(0), (s1.n / N1).fillna(0)
    both = s0.n.notna() & s1.n.notna()
    comp = {}
    mix = ((w1 - w0) * s0.cm1m)[both].sum() + (w1 * s1.cm1m)[~both & s1.n.notna()].sum() - (w0 * s0.cm1m)[~both & s0.n.notna()].sum()
    b = both
    comp["Base list price"] = (w1 * (s1.Lm - s0.Lm) * (1 - s0.d))[b].sum()
    zd = np.array([i[1] == "ZipDrop" for i in idx])
    op = np.array([i[1] == "Other partner" for i in idx])
    disc = -(w1 * s1.Lm * (s1.d - s0.d))
    comp["ZipDrop discount"] = disc[b & zd].sum()
    comp["Other partner discounts"] = disc[b & op].sum()
    comp["Peak/off-peak pilot"] = (w1 * (s1.Pm - s0.Pm))[b].sum()
    comp["Energy: kWh per swap"] = -(w1 * (s1.Km - s0.Km) * s0.tau)[b].sum()
    comp["Energy: grid tariff"] = -(w1 * s1.Km * (s1.tau - s0.tau))[b].sum()
    sup = sorted(set(u0.columns) | set(u1.columns))
    u0r, u1r = u0.reindex(index=idx, columns=sup).fillna(0), u1.reindex(index=idx, columns=sup).fillna(0)
    gr = pd.DataFrame({k: [GREF.get((i[0], k), np.nan) for i in idx] for k in sup}, index=idx).fillna(0)
    share_eff = ((u1r - u0r) * gr).sum(axis=1)
    comp["Wear: Kyron share of packs"] = -(w1 * share_eff)[b].sum()
    comp["Wear: within supplier"] = -(w1 * ((s1.Wm - s0.Wm) - share_eff))[b].sum()
    comp["Segment mix"] = mix
    cm1_0 = (s0.A.sum() - s0.E.sum() - s0.W.sum()) / N0
    cm1_1 = (s1.A.sum() - s1.E.sum() - s1.W.sum()) / N1
    f0, f1 = F0 / N0, F1 / N1
    out = pd.Series(comp)
    resid = (cm1_1 - cm1_0) - out.sum()
    return dict(p0=p0, p1=p1, cm1_0=cm1_0, cm1_1=cm1_1, comp=out, resid=resid, f0=f0, f1=f1,
                cm2_0=cm1_0 - f0, cm2_1=cm1_1 - f1, N1=N1, months1=g_months(p1))


def g_months(p):
    a, b = PER[p]
    return (b.year - a.year) * 12 + b.month - a.month + 1


# clean windows first (no post-EOL issue in either period), then the H1-25 / Q2-25 bridges for transparency
BRIDGES = [("H1-24", "Jul-Aug-24"), ("Jul-Aug-24", "Oct-24..Mar-25"), ("Q1-24", "Q1-25"),
           ("Jul-Aug-24", "H1-25"), ("H1-24", "H1-25"), ("Q2-24", "Q2-25")]
BR = {k: bridge(*k) for k in BRIDGES}
br_rows = []
for (p0, p1), r in BR.items():
    vol = r["N1"] / r["months1"]
    pe0, pe1 = P[p0]["post_eol_ps"], P[p1]["post_eol_ps"]
    flag = ("clean window: no pack issued below 70% EOL" if pe0 == 0 and pe1 == 0 else
            f"NOT CLEAN: {p1} includes {pe1:.2f} Rs/swap of wear on packs issued below 70% EOL, inside the Kyron term (section 5b)")
    print(f"\n[5] cm1/swap bridge {p0} -> {p1}: {r['cm1_0']:.2f} -> {r['cm1_1']:.2f} (change {r['cm1_1'] - r['cm1_0']:+.2f}); "
          f"components sum exactly (|residual| < 1e-9: {abs(r['resid']) < 1e-9}); {flag}")
    for k, v in r["comp"].sort_values().items():
        print(f"     {k:<28} {v:+7.2f} Rs/swap  ~ {v * vol / 1e6:+6.2f} M Rs/month at {p1} volume ({vol:,.0f} swaps/month)")
        br_rows.append({"from": p0, "to": p1, "component": k, "rs_per_swap": v, "rs_per_month_at_end_volume": v * vol,
                        "end_period_post_eol_wear_ps": pe1})
    print(f"     fixed cost/swap {r['f0']:.2f} -> {r['f1']:.2f} => cm2/swap {r['cm2_0']:.2f} -> {r['cm2_1']:.2f} "
          f"(fixed-cost effect {-(r['f1'] - r['f0']):+.2f} Rs/swap ~ {-(r['f1'] - r['f0']) * vol / 1e6:+.2f} M Rs/month)")
    br_rows += [{"from": p0, "to": p1, "component": "cm1_per_swap_start", "rs_per_swap": r["cm1_0"]},
                {"from": p0, "to": p1, "component": "cm1_per_swap_end", "rs_per_swap": r["cm1_1"]},
                {"from": p0, "to": p1, "component": "Fixed cost per swap (cm2 only)", "rs_per_swap": -(r["f1"] - r["f0"]),
                 "rs_per_month_at_end_volume": -(r["f1"] - r["f0"]) * vol},
                {"from": p0, "to": p1, "component": "cm2_per_swap_start", "rs_per_swap": r["cm2_0"]},
                {"from": p0, "to": p1, "component": "cm2_per_swap_end", "rs_per_swap": r["cm2_1"]}]
pd.DataFrame(br_rows).to_csv(TAB / "q1_cm_bridge.csv", index=False)

# %% [markdown]
# ### Direct monthly rupee impact of the biggest items, cross-checking the bridge
# * Kyron excess wear = sum over Kyron-issued swaps of (wear - Cellora wear per swap of the same class). Shown for the
#   clean Q1-25 window and for H1-25, where it is split into pre-EOL issues and the charges on post-EOL issues.
# * ZipDrop amendment = ZipDrop swaps x list price x (28% - 12%) (H1-25).
# * Base price rise = swaps x list-price step x (1 - discount rate), at H1-2025 volume.
# * Pilot = net surcharge collected in Bengaluru + Pune (peak +Rs15, off-peak -Rs8) - gross, before any volume response.

# %%
h125 = seg.filter(in_per("H1-25"))
nm = g_months("H1-25")


def kyron_excess(p, cls, post=None):
    """Kyron excess wear over the Cellora wear of the same class: (Rs per month, a cost is negative; Kyron swaps per month)."""
    k = seg.filter(in_per(p) & (pl.col("supplier_out") == "Kyron") & (pl.col("vehicle_class") == cls))
    if post is not None:
        k = k.filter(pl.col("post_eol") == post)
    return -(k["W"].sum() - k["n"].sum() * GREF[(cls, "Cellora")]) / g_months(p), k["n"].sum() / g_months(p)


ky2_h1, n2_h1 = kyron_excess("H1-25", "2W")
ky2_pre, n2_pre = kyron_excess("H1-25", "2W", post=False)
ky2_q1, n2_q1 = kyron_excess("Q1-25", "2W")
ky3_h1, _ = kyron_excess("H1-25", "3W")
pe2_h1 = -h125.filter((pl.col("vehicle_class") == "2W") & pl.col("post_eol"))["W"].sum() / nm
zdp = h125.filter(pl.col("billing") == "ZipDrop").select(pl.col("n").sum(), pl.col("L").sum(), pl.col("D").sum()).row(0, named=True)
zd_month = zdp["L"] * (0.28 - 0.12) / nm
pil = h125.filter(pl.col("pilot_city")).select((pl.col("A") - pl.col("L") + pl.col("D")).sum()).item() / nm
lst = h125.group_by("vehicle_class").agg(pl.col("n").sum(), pl.col("L").sum(), pl.col("D").sum()).to_pandas()
lst["step"] = lst.vehicle_class.map({"2W": 5.0, "3W": 10.0})
price_month = (lst.n * lst.step * (1 - lst.D / lst.L)).sum() / nm
imp = pd.DataFrame([
    {"item": "Kyron 2W excess battery wear vs Cellora", "window": "Q1-25 (no post-EOL issues)", "rs_per_month": ky2_q1,
     "basis": f"{n2_q1:,.0f} Kyron-2W swaps/month x (Kyron wear - Cellora {GREF[('2W', 'Cellora')]:.1f})"},
    {"item": "Kyron 2W excess battery wear vs Cellora, per-cycle model", "window": "H1-25", "rs_per_month": ky2_h1,
     "basis": f"{n2_h1:,.0f} Kyron-2W swaps/month; includes post-EOL issues"},
    {"item": "Kyron 2W excess battery wear vs Cellora, pre-EOL issues only", "window": "H1-25", "rs_per_month": ky2_pre,
     "basis": f"{n2_pre:,.0f} Kyron-2W pre-EOL swaps/month"},
    {"item": "Wear charged on Kyron 2W packs issued below 70% EOL (beyond purchase cost)", "window": "H1-25", "rs_per_month": pe2_h1,
     "basis": "all of it is depreciation beyond the purchase cost"},
    {"item": "Kyron 3W excess battery wear vs Cellora", "window": "H1-25", "rs_per_month": ky3_h1, "basis": "3W packs"},
    {"item": "ZipDrop discount 12% -> 28%", "window": "H1-25", "rs_per_month": -zd_month, "basis": f"{zdp['n'] / nm:,.0f} ZipDrop swaps/month x list x 16 pts"},
    {"item": "Base price rise (Jul-24), net of partner discounts", "window": "H1-25", "rs_per_month": price_month, "basis": "2W +Rs5, 3W +Rs10 x H1-25 swaps"},
    {"item": "Peak/off-peak pilot net surcharge (gross of volume response)", "window": "H1-25", "rs_per_month": pil, "basis": "Bengaluru + Pune"},
])
imp["rs_M_per_month"] = imp.rs_per_month / 1e6
imp.to_csv(TAB / "q1_cm_monthly_impact.csv", index=False)
print("\n[5] Direct monthly rupee impact, H1-2025 average month")
print(imp.round(2).to_string(index=False))

# %%
# Hero figure 2: chained waterfall H1-24 -> Jul-Aug-24 -> H1-25 (cm1 per swap)
SHORT = {"Base list price": "Base price\nrise (Jul-24)", "ZipDrop discount": "ZipDrop\ndiscount\n12% -> 28%",
         "Other partner discounts": "Other partner\ndiscounts", "Peak/off-peak pilot": "Peak/off-peak\npilot",
         "Energy: kWh per swap": "Energy: fewer\nkWh per swap\n(ageing packs)", "Energy: grid tariff": "Energy:\ngrid tariff",
         "Wear: Kyron share of packs": "Battery wear:\nKyron share\nof packs", "Wear: within supplier": "Battery wear:\nwithin supplier",
         "Segment mix": "Segment mix\n(2W/3W, partner,\ncity)"}


def plab(p):
    return DISP.get(p, p).replace(" to ", " to\n")


def legs_to_steps(legs, min_abs=0.35, max_items=5):
    steps = []
    for (p0, p1) in legs:
        r = BR[(p0, p1)]
        if not steps:
            steps.append((f"{plab(p0)}\ncm1/swap", r["cm1_0"], "total", None))
        comp = r["comp"].copy()
        keep = comp[comp.abs() >= min_abs].abs().sort_values(ascending=False).index[:max_items]
        rest = comp.drop(keep)
        vol = r["N1"] / r["months1"]
        for k in comp.loc[keep].sort_values(key=lambda s: -s.abs()).index:
            steps.append((SHORT[k], comp[k], "delta", comp[k] * vol))
        if len(rest):
            steps.append((f"Other\n({len(rest)} items)", rest.sum(), "delta", rest.sum() * vol))
        steps.append((f"{plab(p1)}\ncm1/swap", r["cm1_1"], "total", None))
    return steps


def draw_waterfall(ax, steps, show_month=True, breaks=()):
    """breaks: indices after which no connector is drawn (the next bar starts a new bridge)."""
    cum = 0.0
    for i, (lab, v, kind, per_month) in enumerate(steps):
        if kind == "total":
            bottom, height, colr = 0.0, v, PALETTE["primary"]
            cum = v
        else:
            bottom, height = cum, v
            colr = PALETTE["good"] if v >= 0 else PALETTE["accent"]
            cum += v
        ax.bar(i, height, bottom=bottom, width=0.66, color=colr, edgecolor="white", linewidth=1.5, zorder=3,
               hatch=None if (kind == "total" or v >= 0) else "///")
        top, bot = max(bottom, bottom + height), min(bottom, bottom + height)
        txt = (rs(v, sign=False) if kind == "total" else rs(v))
        if kind == "delta" and per_month is not None and show_month and abs(per_month) >= 0.2e6:
            txt += f"\n({'+' if per_month >= 0 else MINUS}{RS}{abs(per_month) / 1e6:.1f}M/mo)"
        if (kind == "total" and v >= 0) or (kind == "delta" and v >= 0):
            ax.text(i, top + 0.35, txt, ha="center", va="bottom", fontsize=10, fontweight="bold" if kind == "total" else None, color="#222222")
        else:
            ax.text(i, bot - 0.35, txt, ha="center", va="top", fontsize=10, fontweight="bold" if kind == "total" else None, color="#222222")
        if i < len(steps) - 1 and i not in breaks:
            ax.plot([i + 0.33, i + 0.67], [cum, cum], color="#888888", lw=0.9, ls=":", zorder=2)
    ax.axhline(0, color="#444444", lw=1)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], fontsize=10)
    ax.grid(axis="x", visible=False)


def wf_range(steps):
    cum, lows, highs = 0.0, [], []
    for _, v, kind, _ in steps:
        if kind == "total":
            cum = v
            lows.append(min(0, v))
            highs.append(max(0, v))
        else:
            lows.append(min(cum, cum + v))
            highs.append(max(cum, cum + v))
            cum += v
    return min(lows), max(highs)


LEG2_END = "Oct-24..Mar-25"     # last window before any pack was issued below 70% EOL
steps = legs_to_steps([("H1-24", "Jul-Aug-24"), ("Jul-Aug-24", LEG2_END)])
fig, ax = plt.subplots(figsize=(16, 7.6))
draw_waterfall(ax, steps)
ax.set_ylabel(f"cm1 per completed swap ({RS})")
lo, hi = wf_range(steps)
ax.set_ylim(lo - 4.5, hi + 3.5)
i_mid = next(i for i, s in enumerate(steps) if s[0].startswith("Jul-Aug-24"))
ax.text(i_mid / 2, hi + 2.4, "Leg 1: price rise (H1-24 -> Jul-Aug-24)", ha="center", fontsize=10.5, color="#555555", style="italic")
ax.text((i_mid + len(steps) - 1) / 2, hi + 2.4, f"Leg 2: Kyron, ZipDrop, pilot (Jul-Aug-24 -> {DISP[LEG2_END]})", ha="center", fontsize=10.5,
        color="#555555", style="italic")
r2 = BR[("Jul-Aug-24", LEG2_END)]
rq1 = BR[("Q1-24", "Q1-25")]
title_block(fig, f"Kyron battery wear erased the July-2024 price gain: cm1 per swap went {rs(BR[('H1-24', 'Jul-Aug-24')]['cm1_0'])} -> "
                 f"{rs(r2['cm1_0'])} -> {rs(r2['cm1_1'])}",
            f"Exact shift-share bridge of cm1 per completed swap (revenue - energy - battery wear), under our usage-based battery-depreciation "
            f"assumption.\nBracketed: monthly rupee impact at the end period's volume. Same season, Q1-24 -> Q1-25: cm1/swap {rs(rq1['cm1_0'])} -> "
            f"{rs(rq1['cm1_1'])}, cm2/swap {rs(rq1['cm2_0'])} -> {rs(rq1['cm2_1'])} (fig q1_06).\nLeg 2 ends in Mar-25: from Apr-25 the model also "
            f"charges wear on Kyron packs issued below 70% end of life ({rs(P['Q2-25']['post_eol_ps'], sign=False)}/swap in Q2-25), "
            f"which would inflate the Kyron bar.", top=0.8)
save_fig(fig, "q1_05_cm1_bridge_waterfall")
print("[fig] q1_05_cm1_bridge_waterfall saved")

# %%
# Same-season bridge Q1-24 -> Q1-25 (supporting figure), as two chained bridges: cm1 (drivers), then cm2 (cm1 change +
# change in station fixed cost per swap). Q1 replaces the first version's Q2 bridge: Q2-25 contains packs issued below
# 70% EOL (section 5b), and the Q2 result flips sign if their wear is capped, while Q1 has no post-EOL issue at all.
# (The first version also added the YoY change in fixed cost to the Q2-25 cm1 level, so its cm2 bars did not chain.)
(FIG / "q1_06_cm_bridge_q2_same_season.png").unlink(missing_ok=True)      # superseded figure
rq = BR[("Q1-24", "Q1-25")]
volq = rq["N1"] / rq["months1"]
stq = legs_to_steps([("Q1-24", "Q1-25")], min_abs=0.35, max_items=5)
n_cm1 = len(stq)
stq = stq + [("Q1-24\ncm2/swap", rq["cm2_0"], "total", None),
             ("cm1 change\n(left bridge)", rq["cm1_1"] - rq["cm1_0"], "delta", (rq["cm1_1"] - rq["cm1_0"]) * volq),
             ("Station fixed\ncost per swap\n(utilisation)", -(rq["f1"] - rq["f0"]), "delta", -(rq["f1"] - rq["f0"]) * volq),
             ("Q1-25\ncm2/swap", rq["cm2_1"], "total", None)]
fig, ax = plt.subplots(figsize=(19, 7.6))
draw_waterfall(ax, stq, breaks=(n_cm1 - 1,))
ax.axvline(n_cm1 - 0.5, color="#BBBBBB", lw=1, ls="--")
ax.set_ylabel(f"{RS} per completed swap")
lo, hi = wf_range(stq)
ax.set_ylim(lo - 4.5, hi + 4.5)
ax.text((n_cm1 - 1) / 2, hi + 2.6, "cm1 per swap:\nrevenue - energy - battery wear", ha="center", fontsize=10.5, color="#555555", style="italic")
ax.text(n_cm1 + 1.5, hi + 2.6, "cm2 per swap:\ncm1 - station rent and maintenance", ha="center", fontsize=10.5, color="#555555", style="italic")
ky_q = rq["comp"]["Wear: Kyron share of packs"]
pr_q = rq["comp"]["Base list price"]
title_block(fig, f"Same season, Q1-24 -> Q1-25: Kyron wear ({rs(ky_q)}/swap) wiped out the price rise ({rs(pr_q)}); lower fixed cost "
                 f"per swap ({rs(-(rq['f1'] - rq['f0']))}) kept cm2 roughly flat",
            f"Exact shift-share bridge, under our usage-based battery-depreciation assumption. Bracketed: monthly rupee impact at Q1-25 volume "
            f"({volq:,.0f} swaps/month).\nNo pack was issued below 70% end of life in either quarter. A Q2 bridge is not robust: if wear stops at "
            f"end of life, Q2-25 cm1 is {rs(P['Q2-25']['cm1_cap_ps'])} vs {rs(P['Q2-24']['cm1_ps'])} in Q2-24 (section 5b).", top=0.83)
save_fig(fig, "q1_06_cm_bridge_q1_same_season")
print("[fig] q1_06 saved")

# %% [markdown]
# ## 5b. Wear charged after end of life: an internal-consistency check of the cost model
# The shared model charges every issue `purchase cost x SoH loss per cycle / (initial SoH - 70)`. Nothing stops the
# charge once a pack is below the 70% end of life (EOL), so a pack that keeps being issued past EOL is depreciated by
# more than its purchase cost. That contradicts the model's own assumption (70% EOL, no salvage). We flag every
# completed swap that issued a pack below 70% SoH (`soh_out_pct < EOL_SOH`) and add two consistent variants:
# * **(a) cap**: post-EOL issues carry no wear, so lifetime depreciation equals the purchase cost;
# * **(b) replacement**: post-EOL issues carry the Cellora reference wear of the same class (the cost of the pack that
#   should have been issued instead).
#
# Months without any post-EOL issue (every month up to Mar-25) are unaffected; that is why the bridges above end there.

# %%
lw = (batteries().group_by("supplier", "vehicle_class")
      .agg(pl.len().alias("packs"), pl.col("purchase_cost_inr").sum().alias("cost_total"), pl.col("purchase_cost_inr").mean().alias("cost_mean"),
           pl.col("n_cycles").mean().alias("cycles_issued_mean"), pl.col("expected_cycles_to_eol").mean().alias("cycles_to_eol_mean"),
           ((pl.col("n_cycles") * pl.col("wear_cost_per_cycle")).sum() / pl.col("purchase_cost_inr").sum()).alias("wear_charged_over_cost"),
           pl.col("current_soh_pct").mean().alias("soh_30jun25_mean"))
      .sort("supplier", "vehicle_class"))
print("[5b] Wear charged over each pack's issues in the window / purchase cost, by supplier x class")
print(lw)
k2 = lw.filter((pl.col("supplier") == "Kyron") & (pl.col("vehicle_class") == "2W")).row(0, named=True)
c2 = lw.filter((pl.col("supplier") == "Cellora") & (pl.col("vehicle_class") == "2W")).row(0, named=True)
ky2_w_events = seg.filter((pl.col("supplier_out") == "Kyron") & (pl.col("vehicle_class") == "2W"))["W"].sum()
print(f"[5b] Kyron 2W: lifetime wear charged = {k2['wear_charged_over_cost']:.2f}x purchase cost ({k2['cycles_issued_mean']:.1f} cycles issued vs "
      f"{k2['cycles_to_eol_mean']:.1f} cycles to 70% EOL; event-level check {ky2_w_events / k2['cost_total']:.2f}x). "
      f"Cellora 2W so far {c2['wear_charged_over_cost']:.2f}x ({c2['cycles_issued_mean']:.0f} of {c2['cycles_to_eol_mean']:.0f} cycles).")

pem = (seg.filter(pl.col("supplier_out") == "Kyron").group_by("month")
       .agg(pl.col("n").sum().alias("kyron_issues"), pl.col("n").filter(pl.col("post_eol")).sum().alias("post_eol_issues"),
            pl.col("W").filter(pl.col("post_eol")).sum().alias("post_eol_wear"))
       .join(mon.select("month", "completed"), on="month").sort("month")
       .with_columns((pl.col("post_eol_issues") / pl.col("kyron_issues") * 100).alias("pct_of_kyron_issues"),
                     (pl.col("post_eol_wear") / pl.col("completed")).alias("post_eol_wear_per_network_swap")))
print("[5b] Kyron issues below 70% EOL by month (Jan-25 on)")
print(pem.filter(pl.col("month") >= M(2025, 1)))
non_ky_pe = seg.filter((pl.col("supplier_out") != "Kyron") & pl.col("post_eol"))["n"].sum()
pq2 = pem.filter(in_per("Q2-25"))
print(f"[5b] {pq2['post_eol_issues'].sum():,} Kyron issues below 70% EOL in Apr-Jun-25 = "
      + " / ".join(f"{v:.1f}%" for v in pq2["pct_of_kyron_issues"]) + " of Kyron issues in Apr / May / Jun; first month with any: "
      f"{pem.filter(pl.col('post_eol_issues') > 0)['month'].min():%b-%y}; non-Kyron issues below 70% in the whole window: {non_ky_pe}")

# the two EOL-consistent variants, at segment level (so bridges and counterfactuals can use them)
GC = {"2W": GREF[("2W", "Cellora")], "3W": GREF[("3W", "Cellora")]}
seg = seg.with_columns(pl.col("vehicle_class").replace_strict(GC, return_dtype=pl.Float64).alias("gC"))
seg = seg.with_columns(pl.when(pl.col("post_eol")).then(0.0).otherwise(pl.col("W")).alias("W_cap"),
                       pl.when(pl.col("post_eol")).then(pl.col("n") * pl.col("gC")).otherwise(pl.col("W")).alias("W_rep"))
print(f"[5b] Variant (b) prices a post-EOL issue at the Cellora reference wear per swap: 2W Rs{GC['2W']:.2f}, 3W Rs{GC['3W']:.2f}")
VARP = ["H1-24", "Q1-24", "Q2-24", "Jul-Aug-24", "Oct-24..Mar-25", "Q1-25", "Q2-25", "H1-25", "Jun-25"]
VARS = [("base", "W"), ("rep", "W_rep"), ("cap", "W_cap")]


def var_levels(p):
    g = seg.filter(in_per(p))
    t = g.select([pl.col(x).sum() for x in ["n", "A", "E", "W", "W_cap", "W_rep", "F"]]).row(0, named=True)
    k = g.filter(pl.col("supplier_out") == "Kyron")
    n, nmo = t["n"], g_months(p)
    out = {"period": p, "swaps_per_month": n / nmo, "post_eol_wear_ps": (t["W"] - t["W_cap"]) / n,
           "post_eol_wear_M_per_month": (t["W"] - t["W_cap"]) / nmo / 1e6}
    for lab, wcol in VARS:
        out[f"cm1_{lab}"] = (t["A"] - t["E"] - t[wcol]) / n
        out[f"cm2_{lab}"] = (t["A"] - t["E"] - t[wcol] - t["F"]) / n
        out[f"kyron_excess_ps_{lab}"] = -(k[wcol].sum() - (k["n"] * k["gC"]).sum()) / n     # vs Cellora wear, per network swap
    return out


VL = pd.DataFrame([var_levels(p) for p in VARP]).set_index("period")
VL.to_csv(TAB / "q1_post_eol_wear_variants.csv")
print("\n[5b] cm1 / cm2 per swap: per-cycle model (base), (b) post-EOL issues at Cellora wear (rep), (a) wear stops at EOL (cap);")
print("     kyron_excess_ps = Kyron wear above Cellora wear of the same class, per network swap")
print(VL.round(2).to_string())
for p in ["H1-25", "Q2-25", "Jun-25"]:
    print(f"[5b] {p}: post-EOL wear {VL.loc[p, 'post_eol_wear_ps']:.2f} Rs per network swap ({VL.loc[p, 'post_eol_wear_M_per_month']:.2f}M Rs/month); "
          f"cm1/swap {VL.loc[p, 'cm1_base']:+.2f} (per-cycle model) / {VL.loc[p, 'cm1_rep']:+.2f} (post-EOL at Cellora wear) / "
          f"{VL.loc[p, 'cm1_cap']:+.2f} (wear stops at EOL); cm2/swap {VL.loc[p, 'cm2_base']:+.2f} / {VL.loc[p, 'cm2_rep']:+.2f} / {VL.loc[p, 'cm2_cap']:+.2f}")
print("[5b] cm1/swap change, base / (b) Cellora wear / (a) cap:")
for p0, p1 in [("Q1-24", "Q1-25"), ("Jul-Aug-24", "Oct-24..Mar-25"), ("H1-24", "H1-25"), ("Jul-Aug-24", "H1-25"), ("Q2-24", "Q2-25")]:
    print(f"     {p0} -> {p1}: " + " / ".join(f"{VL.loc[p1, f'cm1_{v}'] - VL.loc[p0, f'cm1_{v}']:+.2f}" for v, _ in VARS))
for v, _ in VARS:
    dk = VL.loc["H1-25", f"kyron_excess_ps_{v}"] - VL.loc["Jul-Aug-24", f"kyron_excess_ps_{v}"]
    print(f"[5b] Kyron excess wear, change Jul-Aug-24 -> H1-25 ({v}): {dk:+.2f} Rs/swap (~{dk * P['H1-25']['swaps_per_month'] / 1e6:+.2f}M Rs/month)")

# Kyron packs issued after the retirement date recorded in the battery master
bt_ret = (batteries().filter(pl.col("retired_date").is_not_null())
          .select("battery_id", "supplier", "vehicle_class", pl.col("retired_date").cast(pl.Date)))
after = (events(["date", "is_completed", "battery_out_id", "soh_out_pct"]).filter(c)
         .join(bt_ret.lazy(), left_on="battery_out_id", right_on="battery_id", how="inner")
         .group_by("supplier", "vehicle_class", (pl.col("date") > pl.col("retired_date")).alias("issued_after_retirement"))
         .agg(pl.len().alias("issues"), pl.col("soh_out_pct").mean().alias("issued_soh_mean"), pl.col("date").max().alias("last_issue"),
              pl.col("retired_date").min().alias("retired_min"), pl.col("retired_date").max().alias("retired_max"))
         .collect(engine="streaming").sort("supplier", "vehicle_class", "issued_after_retirement"))
print("\n[5b] Issues of packs the battery master marks as retired, before vs after their recorded retirement date")
print(after)
a2w = after.filter((pl.col("vehicle_class") == "2W") & pl.col("issued_after_retirement")).row(0, named=True)
print(f"[5b] {a2w['issues']:,} Kyron 2W issues after the recorded retirement date ({a2w['retired_min']}), mean issued SoH "
      f"{a2w['issued_soh_mean']:.1f}%, last issue {a2w['last_issue']}")

# what a rider gets from a worn pack: km and kWh per 2W swap by SoH band of the RETURNED pack (H1-25, association)
kb = (events(["month", "is_completed", "vehicle_class", "soh_in_pct", "energy_to_recharge_kwh", "km_since_last_swap"])
      .filter(c & (pl.col("vehicle_class") == "2W") & in_per("H1-25"))
      .with_columns((pl.col("soh_in_pct") // 5 * 5).alias("soh_band"))
      .group_by("soh_band").agg(pl.len().alias("swaps"), pl.col("energy_to_recharge_kwh").mean().alias("kwh_per_swap"),
                                pl.col("km_since_last_swap").mean().alias("km_per_swap"))
      .collect(engine="streaming").sort("soh_band"))
kb.write_csv(TAB / "q1_range_by_returned_soh_band.csv")
print("\n[5b] H1-25 completed 2W swaps by SoH band of the returned pack (lower bound of a 5-point band)")
print(kb)
kbd = {r["soh_band"]: r for r in kb.iter_rows(named=True) if r["soh_band"] is not None and r["swaps"] >= 1000}
print("[5b] km per 2W swap by SoH band of the returned pack (bands with >=1,000 swaps): "
      + ", ".join(f"{b:.0f}-{b + 5:.0f}% {kbd[b]['km_per_swap']:.1f} km / {kbd[b]['kwh_per_swap']:.2f} kWh (n={kbd[b]['swaps']:,})" for b in sorted(kbd)))

# would a 2-3-month supplier gate have caught Kyron? Mean SoH of issued 2W packs, first months of Kyron use
ed = (events(["month", "is_completed", "vehicle_class", "supplier_out", "soh_out_pct"])
      .filter(c & (pl.col("vehicle_class") == "2W") & pl.col("month").is_between(M(2024, 9), M(2025, 1)))
      .group_by("month", "supplier_out").agg(pl.col("soh_out_pct").mean().alias("soh_out"))
      .collect(engine="streaming").pivot(on="supplier_out", index="month", values="soh_out").sort("month"))
print("\n[5b] Mean SoH of issued 2W packs by supplier, first months of Kyron use")
print(ed)
e_sep = ed.filter(pl.col("month") == M(2024, 9)).row(0, named=True)
e_nov = ed.filter(pl.col("month") == M(2024, 11)).row(0, named=True)
print(f"[5b] Sep-24 -> Nov-24 issued SoH: Kyron {e_sep['Kyron']:.1f} -> {e_nov['Kyron']:.1f}% ({e_nov['Kyron'] - e_sep['Kyron']:+.1f} pts), "
      f"Cellora {e_sep['Cellora']:.1f} -> {e_nov['Cellora']:.1f}% ({e_nov['Cellora'] - e_sep['Cellora']:+.1f} pts)")

# warranty / price-claim basis: one-off life shortfall of the Kyron 2W cohort vs a Cellora-grade life
shortfall = k2["cost_total"] * (1 - k2["cycles_to_eol_mean"] / c2["cycles_to_eol_mean"])
print(f"[5b] One-off lifetime shortfall, Kyron 2W vs Cellora-grade life: {k2['packs']:,} packs x Rs{k2['cost_mean']:,.0f} x "
      f"(1 - {k2['cycles_to_eol_mean']:.0f}/{c2['cycles_to_eol_mean']:,.0f}) = Rs{shortfall / 1e6:.1f}M (of Rs{k2['cost_total'] / 1e6:.1f}M purchase cost)")

# the rest of the fleet is ageing in step: when does the non-Kyron 2W fleet reach 70%?
nk = batteries().filter((pl.col("supplier") != "Kyron") & (pl.col("vehicle_class") == "2W"))
tr = mon.filter(pl.col("month").is_between(M(2025, 2), M(2025, 6)))["soh_in_2w_nonkyron"]
slope = (tr[-1] - tr[0]) / (len(tr) - 1)
m_to_eol = (tr[-1] - EOL_SOH) / -slope
eol_month = pd.Timestamp(2025, 6, 1) + pd.DateOffset(months=int(round(m_to_eol)))
print(f"[5b] Non-Kyron 2W fleet: {nk.height:,} packs, SoH on 30-Jun-25 mean {nk['current_soh_pct'].mean():.1f}% "
      f"({(nk['current_soh_pct'] <= 80).mean() * 100:.0f}% at or below 80%, {(nk['current_soh_pct'] <= 75).mean() * 100:.0f}% at or below 75%). "
      f"Returned non-Kyron 2W SoH {tr[0]:.1f}% (Feb-25) -> {tr[-1]:.1f}% (Jun-25) = {slope:+.2f} pts/month; at that pace the fleet reaches 70% "
      f"in about {m_to_eol:.0f} months (~{eol_month:%b-%Y}), nearly all packs at once")

# %% [markdown]
# ## 6. Counterfactual: Kyron packs with Cellora-like wear
# Replace each Kyron-issued swap's wear with the Cellora wear per swap of the same vehicle class (2W ~ Rs32, 3W ~ Rs76).
# Where no pack was issued past EOL (to Mar-25), the uplift is the **forward per-swap cost gap** between buying
# Kyron-grade and Cellora-grade packs. In H1-25 the per-cycle "actual" also contains post-EOL charges, so the uplift is
# also shown against the two EOL-consistent actuals (section 5b).
# Second-order effects are ignored on purpose and stated: healthier packs would also take more energy to recharge
# (estimated below from returned-pack supplier) and would deliver more km per swap (fewer swaps - see Q4).

# %%
cf_m = (seg.with_columns(pl.when(pl.col("supplier_out") == "Kyron").then(pl.col("n") * pl.col("gC")).otherwise(pl.col("W")).alias("W_cf"))
        .group_by("month").agg(pl.col("n").sum(), (pl.col("A") - pl.col("E") - pl.col("W")).sum().alias("cm1"),
                               (pl.col("A") - pl.col("E") - pl.col("W_cap")).sum().alias("cm1_cap"),
                               (pl.col("A") - pl.col("E") - pl.col("W_rep")).sum().alias("cm1_rep"),
                               (pl.col("A") - pl.col("E") - pl.col("W_cf")).sum().alias("cm1_cf"), pl.col("F").sum())
        .sort("month").with_columns((pl.col("cm1") / pl.col("n")).alias("cm1_ps"), (pl.col("cm1_cf") / pl.col("n")).alias("cm1_cf_ps"),
                                    ((pl.col("cm1_cf") - pl.col("F")) / pl.col("n")).alias("cm2_cf_ps")))
eoff = (events(["month", "is_completed", "vehicle_class", "supplier_in", "energy_cost", "energy_to_recharge_kwh", "soh_in_pct"])
        .filter(c & in_per("H1-25") & (pl.col("vehicle_class") == "2W"))
        .group_by("supplier_in").agg(pl.len().alias("n"), pl.col("energy_cost").mean().alias("energy_ps"),
                                     pl.col("energy_to_recharge_kwh").mean().alias("kwh_ps"), pl.col("soh_in_pct").mean().alias("soh_in"))
        .sort("supplier_in").collect(engine="streaming"))
print("[6] 2W energy per swap by supplier of the RETURNED pack, H1-2025:")
print(eoff)
eo = {r["supplier_in"]: r for r in eoff.iter_rows(named=True)}
energy_offset_ps = eo["Kyron"]["n"] * (eo["Cellora"]["energy_ps"] - eo["Kyron"]["energy_ps"]) / P["H1-25"]["completed"]
cf_rows = []
for p in ["H1-24", "Jul-Aug-24", "Oct-24..Mar-25", "Q1-25", "H1-25", "Q2-25", "Jun-25"]:
    d = cf_m.filter(in_per(p)).select([pl.col(x).sum() for x in ["n", "cm1", "cm1_cap", "cm1_rep", "cm1_cf", "F"]]).row(0, named=True)
    n, nmo = d["n"], g_months(p)
    cf_rows.append({"period": p, "cm1_ps_actual": d["cm1"] / n, "cm1_ps_actual_rep": d["cm1_rep"] / n, "cm1_ps_actual_cap": d["cm1_cap"] / n,
                    "cm1_ps_cellora_like_kyron": d["cm1_cf"] / n,
                    "cm2_ps_actual": (d["cm1"] - d["F"]) / n, "cm2_ps_cellora_like_kyron": (d["cm1_cf"] - d["F"]) / n,
                    "uplift_ps": (d["cm1_cf"] - d["cm1"]) / n, "uplift_rs_per_month": (d["cm1_cf"] - d["cm1"]) / nmo,
                    "uplift_vs_rep_ps": (d["cm1_cf"] - d["cm1_rep"]) / n, "uplift_vs_rep_rs_per_month": (d["cm1_cf"] - d["cm1_rep"]) / nmo,
                    "uplift_vs_cap_ps": (d["cm1_cf"] - d["cm1_cap"]) / n, "uplift_vs_cap_rs_per_month": (d["cm1_cf"] - d["cm1_cap"]) / nmo})
cf = pd.DataFrame(cf_rows)
cf["energy_offset_ps_h125_estimate"] = np.where(cf.period == "H1-25", -energy_offset_ps, np.nan)
cf.to_csv(TAB / "q1_kyron_counterfactual.csv", index=False)
print("\n[6] Counterfactual cm per swap if Kyron packs wore like Cellora (Rs/swap); actual under base / (b) rep / (a) cap")
print(cf.round(2).to_string(index=False))
cfi = cf.set_index("period")
print(f"[6] Forward per-swap cost gap, Kyron-grade vs Cellora-grade packs (windows with no post-EOL issue): Oct-24..Mar-25 "
      f"{cfi.loc['Oct-24..Mar-25', 'uplift_ps']:+.2f} Rs/swap ({cfi.loc['Oct-24..Mar-25', 'uplift_rs_per_month'] / 1e6:.2f}M Rs/month), Q1-25 "
      f"{cfi.loc['Q1-25', 'uplift_ps']:+.2f} ({cfi.loc['Q1-25', 'uplift_rs_per_month'] / 1e6:.2f}M Rs/month); H1-25 under the per-cycle model "
      f"{cfi.loc['H1-25', 'uplift_ps']:+.2f} ({cfi.loc['H1-25', 'uplift_rs_per_month'] / 1e6:.2f}M Rs/month)")
print(f"[6] H1-25 uplift against an EOL-consistent actual: {cfi.loc['H1-25', 'uplift_vs_cap_ps']:+.2f} Rs/swap "
      f"({cfi.loc['H1-25', 'uplift_vs_cap_rs_per_month'] / 1e6:.2f}M Rs/month, wear stops at EOL) to {cfi.loc['H1-25', 'uplift_vs_rep_ps']:+.2f} "
      f"({cfi.loc['H1-25', 'uplift_vs_rep_rs_per_month'] / 1e6:.2f}M Rs/month, post-EOL issues at Cellora wear)")
print(f"[6] Energy offset if Kyron returns had Cellora-like energy per swap: {-energy_offset_ps:+.2f} Rs/swap in H1-25 "
      f"(net counterfactual cm1 ~ {cf.loc[cf.period == 'H1-25', 'cm1_ps_cellora_like_kyron'].iloc[0] - energy_offset_ps:.2f})")

# %% [markdown]
# ## 7. Sensitivity of margin levels to the battery-wear assumption
# Wear per cycle = purchase cost x (1 - salvage share) x SoH loss per cycle / (initial SoH - end-of-life SoH).
# We vary EOL (70% base vs 60%) and salvage value (0% base vs 20% of purchase cost) and recompute every month, and add
# the two EOL-consistent treatments from section 5b (wear capped at EOL; post-EOL issues at Cellora wear).
# The point: *levels* move a lot. The *direction* of the decline holds under every assumption in the windows before any
# pack passed EOL (Q1-24 -> Q1-25 and Jul-Aug-24 -> Oct-24..Mar-25), but not for the H1 or Q2 same-season comparisons
# once post-EOL wear is capped.

# %%
bo = (events(["month", "is_completed", "battery_out_id", "soh_out_pct"]).filter(c)
      .with_columns((pl.col("soh_out_pct") < EOL_SOH).fill_null(False).alias("post_eol"),
                    (pl.col("soh_out_pct") < 60).fill_null(False).alias("below60"))
      .group_by("month", "battery_out_id", "post_eol", "below60").agg(pl.len().alias("n"))
      .collect(engine="streaming"))
bt = batteries().select("battery_id", "supplier", "vehicle_class", "purchase_cost_inr", "loss_per_cycle_used", "initial_soh_pct")
bo = bo.join(bt, left_on="battery_out_id", right_on="battery_id", how="left").with_columns(
    pl.col("vehicle_class").replace_strict(GC, return_dtype=pl.Float64, default=None).alias("gC"))
print(f"[7] issued-pack rows={bo.height:,}; unmatched packs={bo['supplier'].null_count()}")
SCN = [("EOL 70%, salvage 0% (base)", 70, 0.0, None), ("EOL 70%, salvage 20%", 70, 0.2, None),
       ("EOL 60%, salvage 0%", 60, 0.0, None), ("EOL 60%, salvage 20%", 60, 0.2, None),
       ("EOL 70%, wear capped at EOL", 70, 0.0, "cap"), ("EOL 70%, post-EOL issues at Cellora wear", 70, 0.0, "rep")]
sens_m = mon.select("month", "completed", "revenue", "energy_cost", "fixed_cost", "wear_cost")
for lab, eol, salv, mode in SCN:
    wexpr = pl.col("purchase_cost_inr") * (1 - salv) * pl.col("loss_per_cycle_used") / (pl.col("initial_soh_pct") - eol) * pl.col("n")
    if mode == "cap":
        wexpr = pl.when(pl.col("post_eol")).then(0.0).otherwise(wexpr)
    elif mode == "rep":
        wexpr = pl.when(pl.col("post_eol")).then(pl.col("gC") * pl.col("n")).otherwise(wexpr)
    w = bo.with_columns(wexpr.alias("w")).group_by("month").agg(pl.col("w").sum().alias(f"W|{lab}"))
    sens_m = sens_m.join(w, on="month")
sens_m = sens_m.sort("month")          # joins do not guarantee order; plots and period sums need calendar order
base_lab = SCN[0][0]
print(f"[7] base-scenario wear vs event wear, max abs monthly diff: {float((sens_m[f'W|{base_lab}'] - sens_m['wear_cost']).abs().max()):.2f} Rs")
b60 = bo.filter(pl.col("below60")).with_columns(
    (pl.col("purchase_cost_inr") * pl.col("loss_per_cycle_used") / (pl.col("initial_soh_pct") - 60) * pl.col("n")).alias("w60"))
print(f"[7] The EOL-60% scenarios are not capped: {b60['n'].sum():,} issues below 60% SoH (months: "
      f"{', '.join(f'{m:%b-%y}' for m in b60['month'].unique().sort())}) carry Rs{b60['w60'].sum() / P['H1-25']['completed']:.2f} per H1-25 swap "
      f"of wear beyond the 60% EOL (EOL 60%, no salvage) - small next to the Rs{P['H1-25']['post_eol_ps']:.2f} beyond 70% in the base")
SENS_P = ["H1-24", "Q1-24", "Q2-24", "Jul-Aug-24", "Oct-24..Mar-25", "Q1-25", "Q2-25", "H1-25", "Jun-25"]
srows = []
for lab, eol, salv, mode in SCN:
    for p in SENS_P:
        d = sens_m.filter(in_per(p)).select(pl.col("completed").sum(), pl.col("revenue").sum(), pl.col("energy_cost").sum(),
                                            pl.col("fixed_cost").sum(), pl.col(f"W|{lab}").sum().alias("W")).row(0, named=True)
        cm1 = (d["revenue"] - d["energy_cost"] - d["W"]) / d["completed"]
        srows.append({"scenario": lab, "eol_soh": eol, "salvage": salv, "post_eol_treatment": mode or "per-cycle (uncapped)", "period": p,
                      "wear_ps": d["W"] / d["completed"], "cm1_ps": cm1, "cm2_ps": cm1 - d["fixed_cost"] / d["completed"]})
sens = pd.DataFrame(srows)
order = [s[0] for s in SCN]
sp = sens.pivot_table(index="scenario", columns="period", values="cm1_ps")[SENS_P].reindex(order)
CHG = [("Q1-24", "Q1-25"), ("Jul-Aug-24", "Oct-24..Mar-25"), ("H1-24", "H1-25"), ("Jul-Aug-24", "H1-25"), ("Q2-24", "Q2-25")]
for p0, p1 in CHG:
    sp[f"chg {p0} -> {p1}"] = sp[p1] - sp[p0]
CM2P = ["H1-24", "Q1-24", "Q1-25", "Jul-Aug-24", "Oct-24..Mar-25", "H1-25"]
sp2 = sens.pivot_table(index="scenario", columns="period", values="cm2_ps")[CM2P].reindex(order)
sp2.columns = [f"cm2 {x}" for x in sp2.columns]
sens_tab = sp.join(sp2)
sens_tab.to_csv(TAB / "q1_wear_sensitivity.csv")
print("\n[7] cm1 per swap (Rs) by wear assumption")
print(sens_tab[SENS_P].round(2).to_string())
print("\n[7] Change in cm1 per swap between periods (Rs)")
print(sens_tab[[f"chg {a} -> {b}" for a, b in CHG]].round(2).to_string())
print("\n[7] cm2 per swap (Rs)")
print(sens_tab[list(sp2.columns)].round(2).to_string())

# %%
smp = sens_m.to_pandas()
smp["month"] = pd.to_datetime(smp["month"])
cfp = cf_m.to_pandas()
cfp["month"] = pd.to_datetime(cfp["month"])
fig, ax = plt.subplots(figsize=(13.5, 7.0))
ax.axhline(0, color="#444444", lw=1)
styles = [(PALETTE["primary"], "-", 2.8), (SERIES[5], "--", 2.0), (SERIES[4], "-.", 2.0), (PALETTE["muted"], ":", 2.4), (PALETTE["warn"], "-", 2.6)]
end_labels = []
for (lab, eol, salv, mode), (colr, ls, lw) in zip(SCN[:5], styles):      # the 'rep' variant is in the table only
    y = (smp.revenue - smp.energy_cost - smp[f"W|{lab}"]) / smp.completed
    if mode == "cap":   # identical to the base line until Mar-25, so draw it only where it differs
        keep = smp.month >= pd.Timestamp(2025, 3, 1)
        ax.plot(smp.month[keep], y[keep], color=colr, ls=ls, lw=lw, marker="o", ms=3.5, label=f"cm1/swap, {lab} (= base until Mar-25)")
    else:
        ax.plot(smp.month, y, color=colr, ls=ls, lw=lw, label=f"cm1/swap, {lab}")
    end_labels.append((float(y.iloc[-1]), f"{rs(y.iloc[-1])}  {lab}", False))
ax.plot(cfp.month, cfp.cm1_cf_ps, color=PALETTE["good"], lw=2.6, ls=(0, (6, 2)), marker="o", ms=3.5,
        label="Base assumption, if Kyron packs wore like Cellora")
end_labels.append((float(cfp.cm1_cf_ps.iloc[-1]), f"{rs(cfp.cm1_cf_ps.iloc[-1])}  Base, Kyron at Cellora wear", True))
# right-hand labels = Jun-25 values, spread vertically so they never overlap
end_labels.sort()
ypos = [end_labels[0][0]]
for yv, _, _ in end_labels[1:]:
    ypos.append(max(yv, ypos[-1] + 1.8))
x_end = smp.month.iloc[-1]
x_txt = mdates.date2num(x_end + pd.Timedelta(days=14))
for (yv, txt, bold), yp in zip(end_labels, ypos):
    ax.annotate(txt, (mdates.date2num(x_end), yv), xytext=(x_txt, yp), textcoords="data", va="center", fontsize=9.8, color="#333333",
                fontweight="bold" if bold else None, arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.6))
ax.text(x_txt, ypos[-1] + 2.3, "Jun-25 values:", fontsize=10, color="#555555", style="italic", va="center")
ax.axvline(pd.Timestamp(kyron_ramp), color="#777777", lw=0.9, ls=(0, (3, 3)))
y_top = max(24, ypos[-1] + 4)
ax.text(pd.Timestamp(kyron_ramp), y_top - 0.5, " Kyron >10% of issued packs", fontsize=9.8, color="#444444", va="top")
ax.axvspan(pd.Timestamp(2025, 3, 16), pd.Timestamp(2025, 6, 16), color=PALETTE["warn"], alpha=0.10, lw=0)
ax.text(pd.Timestamp(2025, 3, 20), -12.8, "Kyron packs issued\nbelow 70% EOL", fontsize=9.5, color="#7A5A00", va="bottom")
ax.set_xticks(pd.date_range("2024-01-01", "2025-06-01", freq="2MS"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.set_xlim(pd.Timestamp(2023, 12, 10), pd.Timestamp(2026, 1, 15))
ax.set_ylabel(f"cm1 per completed swap ({RS})")
ax.set_ylim(-14, y_top)
ax.legend(loc="lower left", fontsize=9.5)
bq, bj = sens_tab["chg Q1-24 -> Q1-25"], sens_tab["chg Jul-Aug-24 -> Oct-24..Mar-25"]
cap_lab = SCN[4][0]
title_block(fig, "The margin fall up to Mar-25 holds under every wear assumption; later levels hinge on post-EOL wear",
            "Same season Q1-24 -> Q1-25, cm1/swap change: " + " / ".join(rs(bq[s[0]]) for s in SCN[:4]) + " (base -> EOL 60% + 20% salvage); "
            f"Jul-Aug-24 -> Oct-24 to Mar-25: {rs(bj[SCN[0][0]])} to {rs(bj[SCN[3][0]])}.\n"
            f"No pack was issued below 70% end of life before Apr-25. If wear stops at end of life (amber), Q2-25 cm1 is "
            f"{rs(sens_tab.loc[cap_lab, 'Q2-25'])} vs {rs(sens_tab.loc[cap_lab, 'Q2-24'])} in Q2-24.\n"
            f"Green dashed: base assumption with Kyron packs costing Cellora-like wear.", top=0.8)
save_fig(fig, "q1_07_wear_sensitivity_and_kyron_counterfactual")
print("[fig] q1_07 saved")

# %% [markdown]
# ## 8. Summary of key numbers (everything quoted in outputs/findings/q1.md is printed above or here)

# %%
h1a, h1b = P["H1-24"], P["H1-25"]
print("\n[summary] H1-24 -> H1-25 (same season):")
print(f"  attempts x{h1b['attempts'] / h1a['attempts']:.2f}, swaps x{h1b['completed'] / h1a['completed']:.2f}, failures x{h1b['failures'] / h1a['failures']:.2f}, "
      f"revenue x{h1b['revenue'] / h1a['revenue']:.2f}, active riders (monthly avg) x{h1b['active_riders'] / h1a['active_riders']:.2f}")
print(f"  revenue/swap {h1a['revenue_ps']:.2f} -> {h1b['revenue_ps']:.2f}; cm1/swap {h1a['cm1_ps']:.2f} -> {h1b['cm1_ps']:.2f}; "
      f"cm2/swap {h1a['cm2_ps']:.2f} -> {h1b['cm2_ps']:.2f}")
print(f"  cm1 total per month {h1a['cm1'] / 6 / 1e6:.2f}M -> {h1b['cm1'] / 6 / 1e6:.2f}M; cm2 total per month {h1a['cm2'] / 6 / 1e6:.2f}M -> {h1b['cm2'] / 6 / 1e6:.2f}M")
print(f"  failure rate {h1a['fail_rate'] * 100:.2f}% -> {h1b['fail_rate'] * 100:.2f}%")
print("[summary] Margin in windows with no pack issued below 70% EOL (the headline anchors):")
for p0, p1 in [("Q1-24", "Q1-25"), ("Jul-Aug-24", "Oct-24..Mar-25")]:
    r = BR[(p0, p1)]
    kyv = r["comp"]["Wear: Kyron share of packs"]
    print(f"  {p0} -> {p1}: cm1/swap {r['cm1_0']:+.2f} -> {r['cm1_1']:+.2f} ({r['cm1_1'] - r['cm1_0']:+.2f}); Kyron share effect {kyv:+.2f} Rs/swap "
          f"(~{kyv * r['N1'] / r['months1'] / 1e6:+.2f}M Rs/month); cm2/swap {r['cm2_0']:+.2f} -> {r['cm2_1']:+.2f}; "
          f"total cm2/month {P[p0]['cm2'] / P[p0]['n_months'] / 1e6:+.2f}M -> {P[p1]['cm2'] / P[p1]['n_months'] / 1e6:+.2f}M")
print(f"  H1-25 cm1/swap {P['H1-25']['cm1_ps']:+.2f} (per-cycle model) / {VL.loc['H1-25', 'cm1_rep']:+.2f} (post-EOL at Cellora wear) / "
      f"{VL.loc['H1-25', 'cm1_cap']:+.2f} (wear stops at EOL)")
print(f"[summary] runtime {time.time() - T0:.1f}s")
