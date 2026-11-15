# %% [markdown]
# # Q5 - Pricing and partner economics (+ fleet-partner value deep dive)
# **Question.** How do the base price change (2024-07-01), the peak/off-peak pilot (Bengaluru + Pune from 2024-10-01) and
# the individual fleet-partner contract terms relate to swap timing, revenue and contribution margin? Is the effect the
# same for every rider segment? Is the largest partner the most valuable?
#
# **Approach.**
# 1. Verify the pilot tariff schedule and partner surcharge billing *in the data* (not from documentation).
# 2. Base price change: difference-in-differences of price-paying independents vs partner-billed riders (whose employer
#    pays), rider + month fixed effects, Jan-Aug 2024 (stops before the Kyron roll-out of Sep-2024 and the pilot).
# 3. Pilot: difference-in-differences / event study, treated = Bengaluru + Pune, control = the other four cities;
#    rider-city-month panel for rider outcomes, station-month panel for congestion; city-permutation inference
#    (only 6 cities); robustness for summer-2025 heat and for the Kyron battery roll-out, which hit three of the four
#    control cities and contaminates any naive margin comparison.
# 4. Partner economics: revenue, discount, energy, battery wear, station fixed cost, cm1/cm2 and working-capital cost
#    for each of the 12 partners and independents; mix-standardised comparison; ZipDrop amendment before/after.
# 5. Budget proposals (c) network-wide pricing rollout and (d) long-term exclusive with the largest partner.
#
# All margin levels are *under our usage-based battery-depreciation assumption* (no salvage value, 70% SoH end of life).

# %%
import sys
import datetime as dt
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # running inside a notebook
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from common import *  # noqa: E402,F401

set_style()
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(230); pl.Config.set_fmt_str_lengths(40)
pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40)

QID = "q5"
COC = 0.12                       # annual cost of capital on receivables (OUR assumption; not in the data or brief)
M = lambda y, m: dt.date(y, m, 1)  # noqa: E731
PRE_PILOT = [M(2024, 7), M(2024, 8), M(2024, 9)]                 # after the base price change, before the pilot
POST_MAIN = [M(2024, m) for m in (10, 11, 12)] + [M(2025, m) for m in (1, 2, 3)]   # Oct-24..Mar-25 (no summer heat)
POST_FULL = POST_MAIN + [M(2025, 4), M(2025, 5), M(2025, 6)]
CUR = (M(2024, 11), M(2025, 6))  # "current terms" window = after the ZipDrop amendment
CONTROL_CITIES = [c for c in CITIES if c not in PILOT_CITIES]
GRP_ORDER = ["Independent", "Partner Y (surcharge billable)", "Partner N (not billable)"]


def money(x):
    return fmt_inr(x)


def p(msg):
    print(msg, flush=True)

# %% [markdown]
# ## 1. Load the clean swap events (projected columns only) and tag segments
# Segment = independent (pays the tariff itself) vs partner-billed, the latter split by whether the contract makes the
# pilot's peak surcharge billable (`peak_surcharge_billable` Y/N).

# %%
PARTNERS = partners()
COLS = ["rider_id", "station_id", "month", "hour", "event_type", "is_completed", "is_stockout", "is_abandoned",
        "is_service_failure", "queue_wait_sec", "vehicle_class", "city", "tariff_code", "list_price_inr", "discount_inr",
        "amount_charged_inr", "partner_id", "is_independent", "plan_type", "revenue", "energy_cost", "wear_cost",
        "fixed_cost", "cm1", "cm2", "supplier_out", "season"]
MONEY = ["list_price_inr", "discount_inr", "amount_charged_inr", "revenue", "energy_cost", "wear_cost", "fixed_cost", "cm1", "cm2"]
LF = (events(COLS)
      .with_columns([pl.col(c).cast(pl.Float64) for c in MONEY])
      .join(PARTNERS.lazy().select("partner_id", "partner_name", "partner_segment", "peak_surcharge_billable",
                                   "payment_terms_days"), on="partner_id", how="left")
      .with_columns(
          pl.when(pl.col("is_independent")).then(pl.lit(GRP_ORDER[0]))
            .when(pl.col("peak_surcharge_billable") == "Y").then(pl.lit(GRP_ORDER[1]))
            .otherwise(pl.lit(GRP_ORDER[2])).alias("grp"),
          pl.col("city").is_in(PILOT_CITIES).alias("treated"),
          (pl.col("supplier_out") == "Kyron").alias("kyron"),
          pl.col("partner_id").fill_null("IND").alias("pid")))

# %% [markdown]
# ## 2. What did the pilot actually charge? (verify in the data)
# The brief describes PEAK as roughly 12:00-22:59. We read the schedule from `tariff_code` of independent swaps in pilot
# cities after 2024-10-01, and check what partner riders were billed at peak (tariff_code is always PARTNER for them, so
# the surcharge can only be seen as `amount_charged - (list - discount)`).

# %%
sched = (LF.filter(pl.col("is_completed") & pl.col("treated") & pl.col("is_independent") & (pl.col("month") >= PILOT_START))
         .group_by("hour").agg(pl.len().alias("swaps"),
                               (pl.col("tariff_code") == "PEAK").mean().alias("share_PEAK"),
                               (pl.col("tariff_code") == "OFFPEAK").mean().alias("share_OFFPEAK"),
                               (pl.col("tariff_code") == "STD").mean().alias("share_STD"),
                               (pl.col("amount_charged_inr") - pl.col("list_price_inr")).mean().alias("charged_minus_list"))
         .sort("hour").collect())
PEAK_H = sched.filter(pl.col("share_PEAK") > 0.95)["hour"].to_list()
OFF_H = sched.filter(pl.col("share_OFFPEAK") > 0.95)["hour"].to_list()
STD_H = [h for h in range(24) if h not in PEAK_H + OFF_H]
# sub-bands of the two discontinuous tariff windows (derived from the data-verified schedule)
MID_H = [h for h in PEAK_H if h < 17]          # PEAK midday 12-13h
EVE_H = [h for h in PEAK_H if h >= 17]         # PEAK evening 19-22h
AFT_H = [h for h in OFF_H if 12 <= h <= 18]    # OFFPEAK afternoon 14-16h
NIGHT_H = [h for h in OFF_H if h not in AFT_H]  # OFFPEAK night 23-05h
sched.write_csv(TAB / f"{QID}_pilot_tariff_schedule.csv")
p(sched)
p(f"PEAK hours (+Rs15): {PEAK_H}\nOFFPEAK hours (-Rs8): {OFF_H}\nSTD hours: {STD_H}")
peak_rev_ind = sched.filter(pl.col("hour").is_in(PEAK_H))["charged_minus_list"].mean()
off_rev_ind = sched.filter(pl.col("hour").is_in(OFF_H))["charged_minus_list"].mean()
p(f"Independents: charged - list at PEAK = {peak_rev_ind:+.2f}, at OFFPEAK = {off_rev_ind:+.2f}")

band = (pl.when(pl.col("hour").is_in(PEAK_H)).then(pl.lit("peak"))
        .when(pl.col("hour").is_in(OFF_H)).then(pl.lit("offpeak")).otherwise(pl.lit("std")))
LF = LF.with_columns(band.alias("band"))

# Where is congestion actually concentrated? (network-wide, all months)
hg = (pl.when(pl.col("hour").is_in([12, 13])).then(pl.lit("1 PEAK midday 12-13"))
      .when(pl.col("hour").is_in([19, 20, 21, 22])).then(pl.lit("2 PEAK evening 19-22"))
      .when(pl.col("hour").is_in(OFF_H)).then(pl.lit("3 OFFPEAK")).otherwise(pl.lit("4 STD")))
band_kpi = (LF.with_columns(hg.alias("hour_group")).group_by("hour_group")
            .agg(pl.len().alias("attempts"), (100 * pl.col("is_service_failure").mean()).alias("failure_rate_pct"),
                 (100 * pl.col("is_stockout").mean()).alias("stockout_rate_pct"), (100 * pl.col("is_abandoned").mean()).alias("abandon_rate_pct"),
                 pl.col("queue_wait_sec").filter(pl.col("is_completed")).mean().alias("qwait_completed_mean_s"),
                 pl.col("queue_wait_sec").filter(pl.col("is_completed")).quantile(0.9).alias("qwait_completed_p90_s"))
            .with_columns((100 * pl.col("attempts") / pl.col("attempts").sum()).alias("share_of_attempts_pct"))
            .sort("hour_group").collect())
band_kpi.write_csv(TAB / f"{QID}_congestion_by_tariff_band.csv")
p("Congestion by tariff band (all cities, Jan-24..Jun-25):")
p(band_kpi.with_columns(pl.col(pl.Float64).round(2)))

# %% [markdown]
# **Is the evening excess network-wide? (Simpson check.)** The network gap between the evening (19-22h) and midday
# (12-13h) PEAK windows could come from a few city-seasons. We split evening-minus-midday failure rates by city x season.

# %%
EVM = (LF.filter(pl.col("hour").is_in(MID_H + EVE_H))
       .with_columns(pl.when(pl.col("hour").is_in(EVE_H)).then(pl.lit("eve")).otherwise(pl.lit("mid")).alias("hb"))
       .group_by("city", "season", "hb").agg(pl.len().alias("att"), pl.col("is_service_failure").sum().alias("fail"))
       .collect())


def eve_minus_mid(keys):
    t = (EVM.group_by(keys + ["hb"]).agg(pl.col("att").sum(), pl.col("fail").sum())
         .with_columns((100 * pl.col("fail") / pl.col("att")).alias("fail_pct"))
         .pivot(on="hb", index=keys, values=["fail_pct", "att"])
         .select(keys + ["fail_pct_mid", "fail_pct_eve", "att_mid", "att_eve"]))   # fixed column order
    return (t.with_columns((pl.col("fail_pct_eve") - pl.col("fail_pct_mid")).alias("eve_minus_mid_pp"))
            .sort(keys))


EVM_CS = eve_minus_mid(["city", "season"])
EVM_C = eve_minus_mid(["city"])
EVM_CS.write_csv(TAB / f"{QID}_evening_vs_midday_failure_city_season.csv")
p("Failure rate, evening 19-22h vs midday 12-13h, by city x season (pp):")
p(EVM_CS.with_columns(pl.col(pl.Float64).round(2)))
p("... by city, all months (pp): " + ", ".join(f"{c} {v:+.2f}" for c, v in EVM_C.select("city", "eve_minus_mid_pp").iter_rows()))
HOT = ["Delhi NCR", "Hyderabad", "Jaipur"]
hot_ss = EVM_CS.filter(pl.col("city").is_in(HOT) & pl.col("season").is_in(["Summer", "Monsoon"]))["eve_minus_mid_pp"]
cool_ss = EVM_CS.filter(~pl.col("city").is_in(HOT))["eve_minus_mid_pp"]
wint = EVM_CS.filter(pl.col("season").is_in(["Winter", "Post-monsoon"]))["eve_minus_mid_pp"]
p(f"Evening-minus-midday failure: Delhi NCR/Hyderabad/Jaipur in Summer+Monsoon {hot_ss.min():+.2f} to {hot_ss.max():+.2f} pp; "
  f"Bengaluru/Mumbai/Pune all seasons {cool_ss.min():+.2f} to {cool_ss.max():+.2f} pp; Winter+Post-monsoon, every city "
  f"{wint.min():+.2f} to {wint.max():+.2f} pp; smallest cell {min(EVM_CS['att_mid'].min(), EVM_CS['att_eve'].min()):,} attempts")

# %% [markdown]
# ### Figure: where is the evening excess? (city x season)

# %%
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SEASONS = ["Winter", "Summer", "Monsoon", "Post-monsoon"]
city_order = EVM_C.sort("eve_minus_mid_pp", descending=True)["city"].to_list()
mat = np.array([[EVM_CS.filter((pl.col("city") == c) & (pl.col("season") == s))["eve_minus_mid_pp"][0] for s in SEASONS]
                + [EVM_C.filter(pl.col("city") == c)["eve_minus_mid_pp"][0]] for c in city_order])
div_cmap = LinearSegmentedColormap.from_list("q5_div", [PALETTE["primary"], "#F2F2F2", PALETTE["accent"]])
lim = np.ceil(np.abs(mat).max())
fig, ax = plt.subplots(figsize=(10, 5.2))
im = ax.imshow(mat, cmap=div_cmap, norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), aspect="auto")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=11.5, fontweight="bold" if abs(v) >= 1 else "normal",
                color="white" if abs(v) > 0.55 * lim else "#333333")
ax.axvline(len(SEASONS) - 0.5, color="white", lw=6)
ax.set_xticks(range(len(SEASONS) + 1)); ax.set_xticklabels(SEASONS + ["All months"])
ax.set_yticks(range(len(city_order)))
ax.set_yticklabels([f"{c} (pilot)" if c in PILOT_CITIES else c for c in city_order])
ax.tick_params(length=0); ax.grid(False)
for s_ in ax.spines.values():
    s_.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label("Failure rate 19-22h minus 12-13h (pp)")
nk = band_kpi.filter(pl.col("hour_group").str.contains("evening"))["failure_rate_pct"][0]
nm_ = band_kpi.filter(pl.col("hour_group").str.contains("midday"))["failure_rate_pct"][0]
ax.set_title("Evening congestion is a hot-city, hot-season problem; the pilot cities barely have it", fontsize=13.5, pad=40)
ax.text(0.0, 1.02, f"Service-failure rate in the evening PEAK window (19-22h) minus the midday PEAK window (12-13h), pp, all swap attempts Jan-2024..Jun-2025.\n"
        f"Network-wide it is {nk:.2f}% vs {nm_:.2f}%, but the gap comes from Delhi NCR, Hyderabad and Jaipur in Summer/Monsoon.",
        transform=ax.transAxes, fontsize=9.5, color="#555555", va="bottom")
fig.tight_layout()
save_fig(fig, f"{QID}_08_evening_congestion_city_season")

surch = (LF.filter(pl.col("is_completed") & ~pl.col("is_independent"))
         .with_columns((pl.col("month") >= PILOT_START).alias("post"),
                       (pl.col("amount_charged_inr") - pl.col("list_price_inr") + pl.col("discount_inr")).alias("surcharge"))
         .group_by("peak_surcharge_billable", "treated", "post", "band")
         .agg(pl.len().alias("swaps"), pl.col("surcharge").mean().round(2).alias("mean_surcharge_inr"),
              (pl.col("surcharge").abs() < 0.5).mean().round(4).alias("share_no_surcharge"))
         .sort("peak_surcharge_billable", "treated", "post", "band").collect())
surch.write_csv(TAB / f"{QID}_partner_surcharge_check.csv")
p(surch)
sY = surch.filter(pl.col("treated") & pl.col("post") & (pl.col("peak_surcharge_billable") == "Y"))
sN = surch.filter(pl.col("treated") & pl.col("post") & (pl.col("peak_surcharge_billable") == "N"))
p(f"Pilot cities, post: billable-Y partners pay {sY.filter(pl.col('band')=='peak')['mean_surcharge_inr'][0]:+.2f} at peak and "
  f"{sY.filter(pl.col('band')=='offpeak')['mean_surcharge_inr'][0]:+.2f} off-peak; billable-N partners pay "
  f"{sN.filter(pl.col('band')=='peak')['mean_surcharge_inr'][0]:+.2f} at peak and {sN.filter(pl.col('band')=='offpeak')['mean_surcharge_inr'][0]:+.2f} off-peak")

# %% [markdown]
# ## 3. Analysis panels
# * **rider-city-month** (completed swaps): swaps, peak / off-peak counts, revenue, cm1, Kyron packs received.
# * **station-month** (all attempts in pilot PEAK hours): stockouts, abandonment, failures, queue wait.
# * **partner cells** (partner x month x city x vehicle class x tariff band): revenue and cost components.

# %%
RM = (LF.filter(pl.col("is_completed"))
      .group_by("rider_id", "city", "month")
      .agg(pl.len().alias("swaps"), (pl.col("band") == "peak").sum().alias("n_peak"),
           (pl.col("band") == "offpeak").sum().alias("n_off"),
           pl.col("hour").is_in(MID_H).sum().alias("n_mid"), pl.col("hour").is_in(EVE_H).sum().alias("n_eve"),
           pl.col("hour").is_in(AFT_H).sum().alias("n_aft"), pl.col("hour").is_in(NIGHT_H).sum().alias("n_night"),
           pl.col("revenue").sum().alias("rev"),
           pl.col("cm1").sum().alias("cm1"), pl.col("list_price_inr").sum().alias("list_rev"),
           pl.col("kyron").sum().alias("n_ky"), pl.col("grp").first(), pl.col("pid").first(),
           pl.col("plan_type").first(), pl.col("vehicle_class").mode().first().alias("vc"),
           pl.col("treated").first())
      .collect(engine="streaming").to_pandas())
RM["unit"] = RM["rider_id"] + "|" + RM["city"]
RM["peak_share"] = RM["n_peak"] / RM["swaps"]
RM["off_share"] = RM["n_off"] / RM["swaps"]
for _k in ("mid", "eve", "aft", "night"):
    RM[f"{_k}_share"] = RM[f"n_{_k}"] / RM["swaps"]
RM["rev_ps"] = RM["rev"] / RM["swaps"]
RM["cm1_ps"] = RM["cm1"] / RM["swaps"]
RM["ky_share"] = RM["n_ky"] / RM["swaps"]
RM["lsw"] = np.log(RM["swaps"])  # log swaps per active rider-month (proportional effects)
RM["month"] = pd.to_datetime(RM["month"]).dt.date
p(f"rider-city-month panel: {len(RM):,} rows, {RM['unit'].nunique():,} rider-city units")

def station_month_panel(hours):
    d = (LF.filter(pl.col("hour").is_in(hours))
         .group_by("station_id", "city", "month")
         .agg(pl.len().alias("att"), pl.col("is_stockout").sum().alias("so"), pl.col("is_abandoned").sum().alias("ab"),
              pl.col("is_service_failure").sum().alias("fail"), pl.col("is_completed").sum().alias("done"),
              pl.col("queue_wait_sec").filter(pl.col("is_completed")).mean().alias("qwait"),
              pl.col("kyron").filter(pl.col("is_completed")).mean().alias("ky_share"))
         .collect().to_pandas())
    d["month"] = pd.to_datetime(d["month"]).dt.date
    d["so_rate"] = d["so"] / d["att"]; d["ab_rate"] = d["ab"] / d["att"]; d["fail_rate"] = d["fail"] / d["att"]
    d["treated"] = d["city"].isin(PILOT_CITIES)
    return d


SMP = station_month_panel(PEAK_H)        # all PEAK-hour attempts (12-13h + 19-22h)
SMP_EVE = station_month_panel(EVE_H)     # evening PEAK window only (19-22h)
_nm = SMP.groupby("station_id")["month"].nunique()
BAL_ST = _nm[_nm == SMP["month"].nunique()].index   # stations with PEAK attempts in every one of the 18 months
p(f"station-month PEAK panel: {len(SMP):,} rows, {SMP['station_id'].nunique()} stations; balanced (all "
  f"{SMP['month'].nunique()} months): {len(BAL_ST)} stations")

CELLS = (LF.filter(pl.col("is_completed"))
         .group_by("pid", "month", "city", "vehicle_class", "band")
         .agg(pl.len().alias("swaps"), pl.col("revenue").sum().alias("revenue"), pl.col("list_price_inr").sum().alias("list_rev"),
              pl.col("discount_inr").sum().alias("disc"), pl.col("energy_cost").sum().alias("energy"),
              pl.col("wear_cost").sum().alias("wear"), pl.col("fixed_cost").sum().alias("fixed"),
              pl.col("cm1").sum().alias("cm1"), pl.col("cm2").sum().alias("cm2"),
              pl.col("kyron").sum().alias("ky_swaps"), pl.col("wear_cost").filter(~pl.col("kyron")).sum().alias("wear_nonky"),
              pl.col("energy_cost").filter(~pl.col("kyron")).sum().alias("energy_nonky"))
         .collect(engine="streaming"))
p(f"partner cells: {CELLS.height:,} rows; completed swaps {CELLS['swaps'].sum():,}; revenue {money(CELLS['revenue'].sum())}")

# %% [markdown]
# ### Estimation helpers
# Two-way fixed effects via within-unit (weighted) demeaning of the outcome, the treatment terms *and* the month dummies
# (exact by Frisch-Waugh-Lovell for unbalanced panels); standard errors clustered by unit. Because the pilot was assigned
# to 2 of only 6 cities, unit-clustered CIs overstate precision; we therefore add a **city-permutation test**: the DiD is
# recomputed for all 15 ways of picking 2 "treated" cities, and p = share of placebo effects at least as large.

# %%
def twfe(df, y, xvars, unit="unit", time="month", w=None, cluster=None):
    d = df.dropna(subset=[y] + xvars).reset_index(drop=True)
    T = pd.get_dummies(d[time].astype(str), prefix="m", drop_first=True, dtype=float)
    X = pd.concat([d[xvars].astype(float), T], axis=1)
    g = d[unit]
    if w is not None:
        ww = d[w].astype(float)
        sw = ww.groupby(g).transform("sum")
        dm = lambda s: s - (s * ww).groupby(g).transform("sum") / sw  # noqa: E731
    else:
        ww = None
        dm = lambda s: s - s.groupby(g).transform("mean")  # noqa: E731
    Xd = X.apply(dm)
    Xd = Xd.loc[:, Xd.abs().sum() > 1e-9]
    yd = dm(d[y].astype(float))
    mod = sm.WLS(yd, Xd, weights=ww) if w is not None else sm.OLS(yd, Xd)
    groups = pd.factorize(d[cluster or unit])[0]
    return mod.fit(cov_type="cluster", cov_kwds={"groups": groups})


def coef(res, name):
    b = res.params[name]; se = res.bse[name]
    return dict(est=b, se=se, lo=b - 1.96 * se, hi=b + 1.96 * se, p=res.pvalues[name])


def perm_did(cell, col, pre, post, weight=None):
    """DiD on city x month cells, equal city weights; permutation over all 2-of-6 treated-city assignments."""
    def city_mean(months):
        s = cell[cell["month"].isin(months)]
        if weight:
            return s.groupby("city").apply(lambda g: np.average(g[col], weights=g[weight]), include_groups=False)
        return s.groupby("city")[col].mean()
    d = city_mean(post) - city_mean(pre)
    eff = lambda tr: d[list(tr)].mean() - d.drop(list(tr)).mean()  # noqa: E731
    actual = eff(PILOT_CITIES)
    plac = [eff(c) for c in combinations(CITIES, 2)]
    pval = np.mean([abs(x) >= abs(actual) - 1e-12 for x in plac])
    return actual, pval

# %% [markdown]
# ## 4. Base price change (2024-07-01: 2W list Rs60 -> 65, 3W Rs100 -> 110)
# Independents pay the tariff themselves; partner-billed riders' swaps are paid by their employer, so the rider faces
# no price change. Comparing the two within the same months removes common shocks (the Apr-Jun 2024 heat-driven stockout
# spike, monsoon). Window Jan-Aug 2024: Sep-2024 onwards is contaminated by the Kyron roll-out (degraded packs force
# more frequent swaps) and Oct onwards by the pilot.

# %%
bp = RM[(RM["month"] <= M(2024, 8))].copy()
bp["ind"] = (bp["grp"] == GRP_ORDER[0]).astype(float)
bp["post"] = (bp["month"] >= BASE_PRICE_CHANGE).astype(float)
bp["ind_post"] = bp["ind"] * bp["post"]
BP_ROWS = []
for vc, dp in (("2W", 5 / 60), ("3W", 10 / 100)):
    s = bp[bp["vc"] == vc]
    pre_mean = s[(s["ind"] == 1) & (s["post"] == 0)]["swaps"].mean()
    c = coef(twfe(s, "lsw", ["ind_post"]), "ind_post")          # log swaps: proportional usage change
    clev = coef(twfe(s, "swaps", ["ind_post"]), "ind_post")     # same in swaps per rider-month
    crev = coef(twfe(s, "rev_ps", ["ind_post"], w="swaps"), "ind_post")
    lp = np.log(1 + dp)
    BP_ROWS.append(dict(vehicle_class=vc, price_change_pct=dp * 100, ind_pre_swaps_per_rider_month=pre_mean,
                        did_swaps_per_rider_level=clev["est"], did_log_swaps=c["est"], lo=c["lo"], hi=c["hi"], p=c["p"],
                        did_pct=100 * (np.exp(c["est"]) - 1), did_pct_lo=100 * (np.exp(c["lo"]) - 1), did_pct_hi=100 * (np.exp(c["hi"]) - 1),
                        elasticity=c["est"] / lp, elasticity_lo=c["lo"] / lp, elasticity_hi=c["hi"] / lp,
                        did_pct_level=100 * clev["est"] / pre_mean, p_level=clev["p"],
                        elasticity_level=(clev["est"] / pre_mean) / dp, elasticity_level_lo=(clev["lo"] / pre_mean) / dp,
                        elasticity_level_hi=(clev["hi"] / pre_mean) / dp,
                        did_rev_per_swap=crev["est"], n_rider_months=len(s)))
BP = pd.DataFrame(BP_ROWS)
p("\nBase price change - DiD independents vs partner-billed riders (rider + month FE, Jan-Aug 2024):")
p(BP.round(3).to_string(index=False))

# event study for the figure (2W, reference month Jun-2024)
s2 = bp[bp["vc"] == "2W"].copy()
bp_months = sorted(s2["month"].unique())
ev_bp = []
for m_ in bp_months:
    if m_ == M(2024, 6):
        continue
    s2[f"ix_{m_}"] = s2["ind"] * (s2["month"] == m_)
xv = [f"ix_{m_}" for m_ in bp_months if m_ != M(2024, 6)]
r_es = twfe(s2, "lsw", xv)
for m_ in bp_months:
    if m_ == M(2024, 6):
        ev_bp.append(dict(month=m_, est=0, lo=0, hi=0)); continue
    c = coef(r_es, f"ix_{m_}"); ev_bp.append(dict(month=m_, est=100 * c["est"], lo=100 * c["lo"], hi=100 * c["hi"]))
EV_BP = pd.DataFrame(ev_bp)

# extensive margin: still active next month (independents vs partner riders)
act = RM.groupby(["rider_id", "month"], as_index=False).agg(grp=("grp", "first"), vc=("vc", "first"), treated=("treated", "first"))
act["next"] = act["month"].map(lambda d: (pd.Timestamp(d) + pd.DateOffset(months=1)).date())
keys = set(zip(act["rider_id"], act["month"]))
act["active_next"] = [((r, n) in keys) for r, n in zip(act["rider_id"], act["next"])]
act["active_next"] = act["active_next"].astype(float)
a = act[(act["next"] >= M(2024, 2)) & (act["next"] <= M(2024, 8))].copy()
a["ind"] = (a["grp"] == GRP_ORDER[0]).astype(float)
a["post"] = (a["next"] >= BASE_PRICE_CHANGE).astype(float)
a["ind_post"] = a["ind"] * a["post"]
Xc = pd.concat([a[["ind", "ind_post"]], pd.get_dummies(a["month"].astype(str), prefix="m", drop_first=True, dtype=float)], axis=1)
Xc = sm.add_constant(Xc)
rc = sm.OLS(a["active_next"], Xc).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(a["rider_id"])[0]})
BP_CONT = coef(rc, "ind_post")
cont_pre_ind = a[(a["ind"] == 1) & (a["post"] == 0)]["active_next"].mean()
p(f"Continuation (active next month) DiD independents vs partners, transitions into Jul-Aug 2024 vs Feb-Jun 2024: "
  f"{100*BP_CONT['est']:+.2f} pp (95% CI {100*BP_CONT['lo']:+.2f} to {100*BP_CONT['hi']:+.2f}; p={BP_CONT['p']:.2f}); "
  f"independents' pre-period continuation {100*cont_pre_ind:.1f}%")

# revenue per swap and cm1 per swap before/after by segment
seg_bp = (CELLS.filter(pl.col("month") <= M(2024, 8))
          .with_columns(pl.when(pl.col("pid") == "IND").then(pl.lit("Independent")).otherwise(pl.lit("Partner")).alias("seg"),
                        (pl.col("month") >= BASE_PRICE_CHANGE).alias("post"))
          .group_by("seg", "vehicle_class", "month")
          .agg(pl.col("swaps").sum(), (pl.col("revenue").sum() / pl.col("swaps").sum()).alias("rev_ps"),
               (pl.col("cm1").sum() / pl.col("swaps").sum()).alias("cm1_ps"))
          .sort("seg", "vehicle_class", "month"))
seg_bp.write_csv(TAB / f"{QID}_base_price_monthly.csv")
bp_sum = (CELLS.filter(pl.col("month").is_between(M(2024, 4), M(2024, 8)))
          .with_columns(pl.when(pl.col("pid") == "IND").then(pl.lit("Independent")).otherwise(pl.lit("Partner")).alias("seg"),
                        pl.when(pl.col("month") >= BASE_PRICE_CHANGE).then(pl.lit("Jul-Aug 24")).otherwise(pl.lit("Apr-Jun 24")).alias("period"))
          .group_by("seg", "vehicle_class", "period")
          .agg((pl.col("revenue").sum() / pl.col("swaps").sum()).alias("rev_ps"), (pl.col("cm1").sum() / pl.col("swaps").sum()).alias("cm1_ps"),
               (pl.col("energy").sum() / pl.col("swaps").sum()).alias("energy_ps"), (pl.col("wear").sum() / pl.col("swaps").sum()).alias("wear_ps"))
          .sort("seg", "vehicle_class", "period"))
p(bp_sum)
BP.to_csv(TAB / f"{QID}_base_price_did.csv", index=False)

# %% [markdown]
# ## 5. Peak/off-peak pilot: difference-in-differences and event study
# Outcomes for **independent** riders (rider-city-month panel, rider-city + month FE, swap-weighted for shares and
# per-swap money): share of swaps in PEAK / OFFPEAK hours, swaps per active rider-month, revenue per swap, cm1 per swap.
# Congestion (station-month, all riders, PEAK hours): stockout rate, abandonment rate, queue wait.
# Pre = Jul-Sep 2024 (same list price as post); main post = Oct-24..Mar-25 (avoids summer-2025 heat, which differs
# strongly across cities); full post = Oct-24..Jun-25 as robustness.
# **Trap:** Kyron packs (fast-degrading, wear ~Rs104/swap) went to Delhi NCR, Hyderabad and Jaipur (~38% of issued packs
# from Oct-24) but only ~10% in Bengaluru, Pune and Mumbai. Naive DiD on cm1 or swaps/rider would credit the pilot with
# the control cities' battery problem. We control for each rider-month's Kyron share and re-estimate with Mumbai (same
# Kyron exposure as the pilot cities) as the only control.

# %%
ky_city = (CELLS.group_by("city", "month").agg((pl.col("ky_swaps").sum() / pl.col("swaps").sum()).alias("kyron_share"))
           .filter(pl.col("month").is_in(POST_MAIN)).group_by("city").agg(pl.col("kyron_share").mean()).sort("kyron_share"))
p(ky_city)

pil = RM[RM["month"] >= M(2024, 1)].copy()
pil["post"] = pil["month"] >= PILOT_START
pil["tp"] = (pil["treated"] & pil["post"]).astype(float)

OUTS = {  # name: (column, weight, label, scale)
    "peak_share": ("peak_share", "swaps", "Share of swaps in PEAK hours (pp)", 100),
    "off_share": ("off_share", "swaps", "Share of swaps in OFFPEAK hours (pp)", 100),
    "swaps_per_rider": ("swaps", None, "Swaps per active rider-month", 1),
    "log_swaps": ("lsw", None, "log swaps per active rider-month (x100 ~ %)", 100),
    "rev_per_swap": ("rev_ps", "swaps", "Revenue per swap (Rs)", 1),
    "cm1_per_swap": ("cm1_ps", "swaps", "cm1 per swap (Rs)", 1),
    # sub-bands of the PEAK and OFFPEAK windows (where did the timing shift come from / go to?)
    "mid_share": ("mid_share", "swaps", "Share of swaps at 12-13h (PEAK midday, pp)", 100),
    "eve_share": ("eve_share", "swaps", "Share of swaps at 19-22h (PEAK evening, pp)", 100),
    "aft_share": ("aft_share", "swaps", "Share of swaps at 14-16h (OFFPEAK afternoon, pp)", 100),
    "night_share": ("night_share", "swaps", "Share of swaps at 23-05h (OFFPEAK night, pp)", 100),
}
MAIN_OUTS = ["peak_share", "off_share", "swaps_per_rider", "log_swaps", "rev_per_swap", "cm1_per_swap"]
SUB_OUTS = ["mid_share", "eve_share", "aft_share", "night_share"]


def did_rider(df, out, months_post, extra=(), control_cities=None, pre=PRE_PILOT):
    col, w, _, sc = OUTS[out]
    s = df[df["month"].isin(pre + months_post)]
    if control_cities is not None:
        s = s[s["city"].isin(PILOT_CITIES + control_cities)]
    r = twfe(s, col, ["tp", *extra], w=w)
    c = coef(r, "tp")
    pre_t = s[(s["treated"]) & (~s["post"]) & s["month"].isin(PRE_PILOT)]
    base = np.average(pre_t[col], weights=pre_t[w]) if w else pre_t[col].mean()
    return dict(est=c["est"] * sc, lo=c["lo"] * sc, hi=c["hi"] * sc, p=c["p"], pre_treated_mean=base * sc, n=len(s))


DID_ROWS = []
ind = pil[pil["grp"] == GRP_ORDER[0]]
cellc = (ind.groupby(["city", "month"]).apply(lambda g: pd.Series({
    "peak_share": np.average(g["peak_share"], weights=g["swaps"]), "off_share": np.average(g["off_share"], weights=g["swaps"]),
    "swaps_per_rider": g["swaps"].mean(), "log_swaps": 100 * g["lsw"].mean(), "rev_per_swap": np.average(g["rev_ps"], weights=g["swaps"]),
    "cm1_per_swap": np.average(g["cm1_ps"], weights=g["swaps"])}), include_groups=False).reset_index())
for out in MAIN_OUTS:
    for wname, months in (("Oct24-Mar25", POST_MAIN), ("Oct24-Jun25", POST_FULL)):
        row = dict(segment=GRP_ORDER[0], outcome=out, post_window=wname, spec="rider-city + month FE", **did_rider(ind, out, months))
        row["perm_p_city"] = perm_did(cellc, out, PRE_PILOT, months)[1]
        DID_ROWS.append(row)
    if out in ("swaps_per_rider", "log_swaps", "cm1_per_swap"):
        DID_ROWS.append(dict(segment=GRP_ORDER[0], outcome=out, post_window="Oct24-Mar25", spec="+ Kyron-share control",
                             **did_rider(ind, out, POST_MAIN, extra=("ky_share",))))
    DID_ROWS.append(dict(segment=GRP_ORDER[0], outcome=out, post_window="Oct24-Mar25", spec="Mumbai-only control",
                         **did_rider(ind, out, POST_MAIN, control_cities=["Mumbai"])))
    DID_ROWS.append(dict(segment=GRP_ORDER[0], outcome=out, post_window="Oct24-Mar25", spec="long pre (Jan-Sep 2024)",
                         **did_rider(ind, out, POST_MAIN, pre=[M(2024, m) for m in range(1, 10)])))

# Triple difference: Partner-N riders (ZipDrop, Swiggle Go) face no price change at all, so any "pilot effect" on them
# is a city-level shock. DDD = independents' DiD minus Partner-N riders' DiD (same cities, same months).
ddd_s = pil[pil["grp"].isin([GRP_ORDER[0], GRP_ORDER[2]]) & pil["month"].isin(PRE_PILOT + POST_MAIN)].copy()
ddd_s["ind"] = (ddd_s["grp"] == GRP_ORDER[0]).astype(float)
ddd_s["ind_post"] = ddd_s["ind"] * ddd_s["post"].astype(float)
ddd_s["tp_ind"] = ddd_s["tp"] * ddd_s["ind"]
DDD = {}
for out in ("log_swaps", "swaps_per_rider", "peak_share", "off_share", "rev_per_swap"):
    col, w, _, sc = OUTS[out]
    r = twfe(ddd_s, col, ["tp", "ind_post", "tp_ind", "ky_share"], w=w)
    c = coef(r, "tp_ind"); c0 = coef(r, "tp")
    DDD[out] = c
    DID_ROWS.append(dict(segment=GRP_ORDER[0], outcome=out, post_window="Oct24-Mar25", spec="DDD vs Partner-N riders (+Kyron)",
                         est=c["est"] * sc, lo=c["lo"] * sc, hi=c["hi"] * sc, p=c["p"], n=len(ddd_s),
                         pre_treated_mean=did_rider(ind, out, POST_MAIN)["pre_treated_mean"]))
    p(f"DDD {out}: independents vs Partner-N {c['est']*sc:+.3f} (95% CI {c['lo']*sc:+.3f} to {c['hi']*sc:+.3f}; p={c['p']:.3f}); "
      f"city-level shock common to both (tp) {c0['est']*sc:+.3f}")
# heterogeneity: partner riders by surcharge billability, independents by vehicle class and plan
for gname in GRP_ORDER[1:]:
    for out in ("peak_share", "off_share", "rev_per_swap", "log_swaps"):
        DID_ROWS.append(dict(segment=gname, outcome=out, post_window="Oct24-Mar25", spec="rider-city + month FE",
                             **did_rider(pil[pil["grp"] == gname], out, POST_MAIN, extra=("ky_share",) if out == "log_swaps" else ())))
for lab, mask in (("Independent 2W", (ind["vc"] == "2W")), ("Independent 3W", (ind["vc"] == "3W")),
                  ("Independent pay-as-you-go", ind["plan_type"] == "pay_as_you_go"), ("Independent prepaid pack", ind["plan_type"] == "prepaid_pack")):
    for out in ("peak_share", "off_share", "rev_per_swap", "log_swaps"):
        DID_ROWS.append(dict(segment=lab, outcome=out, post_window="Oct24-Mar25", spec="rider-city + month FE",
                             **did_rider(ind[mask], out, POST_MAIN, extra=("ky_share",) if out == "log_swaps" else ())))
# where did the peak-share drop come from and go to? (needed to price the 12-13h band separately)
for gname, dfg in ((GRP_ORDER[0], ind), (GRP_ORDER[1], pil[pil["grp"] == GRP_ORDER[1]])):
    for out in SUB_OUTS:
        DID_ROWS.append(dict(segment=gname, outcome=out, post_window="Oct24-Mar25", spec="rider-city + month FE",
                             **did_rider(dfg, out, POST_MAIN)))

# congestion (station-month, PEAK hours, all riders)
SMP["tp"] = (SMP["treated"] & (SMP["month"] >= PILOT_START)).astype(float)
SMP["post"] = SMP["month"] >= PILOT_START
cong_cell = SMP.groupby(["city", "month"]).apply(lambda g: pd.Series({
    "so_rate": g["so"].sum() / g["att"].sum(), "ab_rate": g["ab"].sum() / g["att"].sum(),
    "fail_rate": g["fail"].sum() / g["att"].sum(),
    "qwait": np.average(g["qwait"].fillna(g["qwait"].mean()), weights=g["done"].clip(lower=1))}), include_groups=False).reset_index()
for out, lab, sc in (("so_rate", "peak_stockout_rate_pp", 100), ("ab_rate", "peak_abandon_rate_pp", 100),
                     ("fail_rate", "peak_failure_rate_pp", 100), ("qwait", "peak_queue_wait_sec", 1)):
    for wname, months in (("Oct24-Mar25", POST_MAIN), ("Oct24-Jun25", POST_FULL)):
        s = SMP[SMP["month"].isin(PRE_PILOT + months)].copy()
        wcol = "att" if out != "qwait" else "done"
        r = twfe(s, out, ["tp"], unit="station_id", w=wcol)
        c = coef(r, "tp")
        pre_t = s[s["treated"] & ~s["post"]]
        base = np.average(pre_t[out].fillna(0), weights=pre_t[wcol])
        DID_ROWS.append(dict(segment="All riders at station (PEAK hours)", outcome=lab, post_window=wname, spec="station + month FE",
                             est=c["est"] * sc, lo=c["lo"] * sc, hi=c["hi"] * sc, p=c["p"], pre_treated_mean=base * sc, n=len(s),
                             perm_p_city=perm_did(cong_cell, out, PRE_PILOT, months)[1]))
# Robustness of the congestion null (Oct-24..Mar-25): Mumbai-only control (same ~10% Kyron share as the pilot cities),
# the balanced panel of stations open in all 18 months, and the evening window alone (19-22h, where relief would show).
for spec_lab, panel, keep in (("Mumbai-only control", SMP, SMP["city"].isin(PILOT_CITIES + ["Mumbai"])),
                              ("balanced 90-station panel", SMP, SMP["station_id"].isin(BAL_ST)),
                              ("evening 19-22h only", SMP_EVE, pd.Series(True, index=SMP_EVE.index))):
    s0 = panel[keep & panel["month"].isin(PRE_PILOT + POST_MAIN)].copy()
    s0["post"] = s0["month"] >= PILOT_START
    s0["tp"] = (s0["treated"] & s0["post"]).astype(float)
    for out, lab, sc in (("so_rate", "peak_stockout_rate_pp", 100), ("fail_rate", "peak_failure_rate_pp", 100)):
        c = coef(twfe(s0, out, ["tp"], unit="station_id", w="att"), "tp")
        pre_t = s0[s0["treated"] & ~s0["post"]]
        DID_ROWS.append(dict(segment="All riders at station (PEAK hours)", outcome=lab, post_window="Oct24-Mar25", spec=spec_lab,
                             est=c["est"] * sc, lo=c["lo"] * sc, hi=c["hi"] * sc, p=c["p"],
                             pre_treated_mean=np.average(pre_t[out], weights=pre_t["att"]) * sc, n=len(s0)))
DID = pd.DataFrame(DID_ROWS)
DID.to_csv(TAB / f"{QID}_pilot_did.csv", index=False)
p("\nPilot DiD estimates (treated = Bengaluru+Pune; pre = Jul-Sep 2024):")
p(DID.round(4).to_string(index=False))


def did_get(seg, out, win="Oct24-Mar25", spec="rider-city + month FE"):
    r = DID[(DID["segment"] == seg) & (DID["outcome"] == out) & (DID["post_window"] == win) & (DID["spec"] == spec)]
    return r.iloc[0]

# %% [markdown]
# ### Event study (monthly treated x month coefficients, reference = Sep-2024)

# %%
ES = []
es_months = sorted(pil["month"].unique())
REF = M(2024, 9)
ind_es = ind.copy()
for m_ in es_months:
    if m_ != REF:
        ind_es[f"tx_{m_}"] = ind_es["treated"].astype(float) * (ind_es["month"] == m_)
xv = [f"tx_{m_}" for m_ in es_months if m_ != REF]
for out in ("peak_share", "off_share", "rev_per_swap", "log_swaps"):
    col, w, lab, sc = OUTS[out]
    extra = ["ky_share"] if out == "log_swaps" else []
    r = twfe(ind_es, col, xv + extra, w=w)
    for m_ in es_months:
        if m_ == REF:
            ES.append(dict(outcome=out, month=m_, est=0.0, lo=0.0, hi=0.0)); continue
        c = coef(r, f"tx_{m_}")
        ES.append(dict(outcome=out, month=m_, est=c["est"] * sc, lo=c["lo"] * sc, hi=c["hi"] * sc))
# Stockouts: use the BALANCED panel of the 90 stations with PEAK attempts in all 18 months. In the unbalanced panel
# Wave1/Wave2 stations enter part-way (mostly into the summer-2025 months) and their station effects are estimated on
# different months, which drags every post-pilot coefficient below zero. On the fixed station set we report the
# attempt-weighted treated-minus-control gap by month (treated-group + month effects, saturated) with station-clustered
# CIs; the station-FE version on the same 90 stations and the old unbalanced version are printed as checks.
smp_es = SMP[SMP["station_id"].isin(BAL_ST)].copy()
smp_es["grp_unit"] = np.where(smp_es["treated"], "treated", "control")
smp_ub = SMP.copy()
for d_ in (smp_es, smp_ub):
    for m_ in es_months:
        if m_ != REF:
            d_[f"tx_{m_}"] = d_["treated"].astype(float) * (d_["month"] == m_)
r = twfe(smp_es, "so_rate", xv, unit="grp_unit", w="att", cluster="station_id")
r_bal_st = twfe(smp_es, "so_rate", xv, unit="station_id", w="att")
r_unbal = twfe(smp_ub, "so_rate", xv, unit="station_id", w="att")
for m_ in es_months:
    if m_ == REF:
        ES.append(dict(outcome="peak_stockout_rate", month=m_, est=0.0, lo=0.0, hi=0.0)); continue
    c = coef(r, f"tx_{m_}")
    ES.append(dict(outcome="peak_stockout_rate", month=m_, est=c["est"] * 100, lo=c["lo"] * 100, hi=c["hi"] * 100))
_pm = [f"tx_{m_}" for m_ in POST_MAIN]
p(f"Peak stockout event study, Oct-24..Mar-25 coefficients (pp vs Sep-24): balanced {len(BAL_ST)} stations, group gap: "
  + ", ".join(f"{100*r.params[k]:+.2f}" for k in _pm)
  + f" | balanced, station FE: " + ", ".join(f"{100*r_bal_st.params[k]:+.2f}" for k in _pm)
  + f" | unbalanced {SMP['station_id'].nunique()} stations, station FE (previous figure): " + ", ".join(f"{100*r_unbal.params[k]:+.2f}" for k in _pm))
ES = pd.DataFrame(ES)
ES.to_csv(TAB / f"{QID}_pilot_event_study.csv", index=False)
pre_es = ES[(ES["month"] < M(2024, 9))].groupby("outcome")["est"].agg(lambda s: s.abs().max())
p("Largest absolute pre-period (Jan-Aug 2024) event-study coefficient by outcome (parallel-trends check):")
p(pre_es.round(3).to_string())
es_w = ES.assign(m=lambda d: pd.to_datetime(d["month"]).dt.strftime("%y-%m")).pivot(index="m", columns="outcome", values="est")
p("Event-study coefficients (treated - control vs Sep-2024):")
p(es_w.round(2).to_string())
es_pre = ES[ES["month"] < M(2024, 9)].groupby("outcome")["est"].mean()
es_post = ES[ES["month"].isin(POST_MAIN)].groupby("outcome")["est"].agg(["mean", "min", "max"])
p("Mean pre (Jan-Aug 24) vs post (Oct-24..Mar-25) coefficients:")
p(pd.concat([es_pre.rename("pre_mean"), es_post], axis=1).round(2).to_string())
p("Base-price event study (2W independents vs partners, log-points x100 ~ %, ref Jun-24): " +
  ", ".join(f"{m_:%b}: {e:+.1f}%" for m_, e in EV_BP[["month", "est"]].values))
p("Summer peak stockout gaps, balanced panel (treated - control, pp, vs Sep-24): " +
  ", ".join(f"{m_:%b-%y} {v:+.2f}" for m_, v in ES[(ES['outcome'] == 'peak_stockout_rate') & ES['month'].isin(
      [M(2024, 4), M(2024, 5), M(2024, 6), M(2025, 4), M(2025, 5), M(2025, 6)])][["month", "est"]].values))

# pilot continuation (activity) response of independents - retention detail is q6's job
ai = act[(act["grp"] == GRP_ORDER[0]) & (act["month"].isin(PRE_PILOT[:2] + POST_MAIN))].copy()
# transitions m -> m+1 with m in Jul, Aug (pre) and Oct..Mar (post: next month also under the pilot)
ai["post"] = ai["month"] >= PILOT_START
ai["tp"] = (ai["treated"] & ai["post"]).astype(float)
Xa = sm.add_constant(pd.concat([ai[["tp"]].astype(float), ai[["treated"]].astype(float),
                                pd.get_dummies(ai["month"].astype(str), prefix="m", drop_first=True, dtype=float)], axis=1))
ra = sm.OLS(ai["active_next"], Xa).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(ai["rider_id"])[0]})
PIL_CONT = coef(ra, "tp")
p(f"Pilot: independents' month-to-month continuation DiD = {100*PIL_CONT['est']:+.2f} pp "
  f"(95% CI {100*PIL_CONT['lo']:+.2f} to {100*PIL_CONT['hi']:+.2f}; p={PIL_CONT['p']:.2f})")

# %% [markdown]
# ### Figure: pilot event study (hero)

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))
spec = [("peak_share", "Peak-hour share of independents' swaps\n(treated - control, pp)", PALETTE["primary"]),
        ("rev_per_swap", "Revenue per independent swap\n(treated - control, Rs)", PALETTE["good"]),
        ("peak_stockout_rate", f"Peak-hour stockout rate, all riders\n(treated - control, pp; {len(BAL_ST)} stations open all 18 months)", PALETTE["accent"])]
for ax, (out, lab, colr) in zip(axes, spec):
    d = ES[ES["outcome"] == out]
    x = pd.to_datetime(d["month"])
    ax.axvspan(pd.Timestamp("2024-10-01") - pd.Timedelta(days=15), pd.Timestamp("2025-06-15"), color=PALETTE["light"], alpha=0.5, lw=0)
    ax.axvspan(pd.Timestamp("2025-03-16"), pd.Timestamp("2025-06-15"), color=PALETTE["warn"], alpha=0.12, lw=0)
    ax.axvspan(pd.Timestamp("2024-03-16"), pd.Timestamp("2024-06-15"), color=PALETTE["warn"], alpha=0.12, lw=0)
    ax.axhline(0, color="#555555", lw=1)
    ax.fill_between(x, d["lo"], d["hi"], color=colr, alpha=0.18, lw=0)
    ax.plot(x, d["est"], "o-", color=colr, ms=4, lw=1.8)
    ax.set_title(lab, fontsize=11.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
pk = did_get(GRP_ORDER[0], "peak_share"); rv = did_get(GRP_ORDER[0], "rev_per_swap")
so = DID[(DID["outcome"] == "peak_stockout_rate_pp") & (DID["post_window"] == "Oct24-Mar25") & (DID["spec"] == "station + month FE")].iloc[0]
so_mum = DID[(DID["outcome"] == "peak_stockout_rate_pp") & (DID["spec"] == "Mumbai-only control")].iloc[0]
axes[0].annotate(f"DiD {pk['est']:+.1f} pp\n(95% CI {pk['lo']:+.1f} to {pk['hi']:+.1f})", xy=(0.03, 0.06), xycoords="axes fraction", fontsize=10.5, color=PALETTE["primary"], fontweight="bold")
axes[1].annotate(f"DiD +Rs{rv['est']:.1f}/swap\n(+{100*rv['est']/rv['pre_treated_mean']:.0f}% vs Rs{rv['pre_treated_mean']:.1f})", xy=(0.03, 0.80), xycoords="axes fraction", fontsize=10.5, color=PALETTE["good"], fontweight="bold")
axes[2].annotate(f"Oct-Mar DiD {so['est']:+.2f} pp\n(95% CI {so['lo']:+.2f} to {so['hi']:+.2f});\nMumbai-only control {so_mum['est']:+.2f} pp",
                 xy=(0.33, 0.30), xycoords="axes fraction", va="bottom", fontsize=10.5, color=PALETTE["accent"], fontweight="bold")
for xt in ("2024-05-01", "2025-05-01"):
    axes[2].text(pd.Timestamp(xt), axes[2].get_ylim()[1] - 0.15, "summer\nheat", ha="center", va="top", fontsize=9, color="#9a6a00")
fig.suptitle("Peak pricing raised revenue ~12% per independent swap but moved few swaps and did not ease peak congestion",
             x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.04)
fig.text(0.01, 0.965, "Event study: Bengaluru + Pune vs Delhi NCR, Hyderabad, Mumbai, Jaipur; reference month Sep-2024; 95% CIs (rider-city- or station-clustered). "
         "Shaded = pilot period; amber = summer heat.", fontsize=9.5, color="#555555", ha="left")
fig.tight_layout()
save_fig(fig, f"{QID}_01_pilot_event_study")

# %% [markdown]
# ### Figure: who responded? DiD by segment

# %%
segs = [GRP_ORDER[0], "Independent 2W", "Independent 3W", "Independent pay-as-you-go", "Independent prepaid pack", GRP_ORDER[1], GRP_ORDER[2]]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
for ax, (out, lab, colr) in zip(axes, [("peak_share", "PEAK-hour share (pp)", PALETTE["primary"]), ("off_share", "OFFPEAK-hour share (pp)", PALETTE["good"]),
                                      ("rev_per_swap", "Revenue per swap (Rs)", PALETTE["warn"])]):
    d = pd.DataFrame([did_get(s, out) for s in segs])
    y = np.arange(len(segs))[::-1]
    ax.errorbar(d["est"], y, xerr=[d["est"] - d["lo"], d["hi"] - d["est"]], fmt="o", color=colr, ecolor=colr, capsize=3, ms=6)
    ax.axvline(0, color="#555555", lw=1)
    for yy, (_, rr) in zip(y, d.iterrows()):
        ax.text(rr["hi"] + (abs(d["hi"]).max() * 0.04 + 0.05), yy, f"{rr['est']:+.1f}", va="center", fontsize=9.5)
    ax.set_title(lab, fontsize=11.5)
    ax.set_yticks(y); ax.set_yticklabels(segs)
fig.suptitle("Partners whose contracts bill the surcharge shifted timing as much as independents; non-billable partners barely moved",
             x=0.01, ha="left", fontsize=13.5, fontweight="bold", y=1.03)
fig.text(0.01, 0.955, "Pilot DiD (Oct-24..Mar-25 vs Jul-Sep-24), rider-city + month FE, 95% CI. Partner N = ZipDrop (FP-03) and Swiggle Go (FP-05).",
         fontsize=9.5, color="#555555", ha="left")
fig.tight_layout()
save_fig(fig, f"{QID}_02_pilot_segments")

# %% [markdown]
# ### Figure: base price change

# %%
fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.7))
sb = seg_bp.to_pandas()
for (seg, vc), colr, ls in ((("Independent", "2W"), PALETTE["primary"], "-"), (("Partner", "2W"), PALETTE["primary"], "--"),
                            (("Independent", "3W"), PALETTE["accent"], "-"), (("Partner", "3W"), PALETTE["accent"], "--")):
    d = sb[(sb["seg"] == seg) & (sb["vehicle_class"] == vc)]
    axes[0].plot(pd.to_datetime(d["month"]), d["rev_ps"], ls, marker="o", ms=4, color=colr, label=f"{seg} {vc}")
axes[0].axvline(pd.Timestamp(BASE_PRICE_CHANGE), color="#555555", lw=1, ls=":")
axes[0].set_title("Revenue per swap (Rs)", fontsize=11.5); axes[0].legend(ncol=2, loc="center right")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
x = pd.to_datetime(EV_BP["month"])
axes[1].fill_between(x, EV_BP["lo"], EV_BP["hi"], color=PALETTE["primary"], alpha=0.18, lw=0)
axes[1].plot(x, EV_BP["est"], "o-", color=PALETTE["primary"])
axes[1].axhline(0, color="#555555", lw=1); axes[1].axvline(pd.Timestamp(BASE_PRICE_CHANGE), color="#555555", lw=1, ls=":")
b2 = BP[BP["vehicle_class"] == "2W"].iloc[0]; b3 = BP[BP["vehicle_class"] == "3W"].iloc[0]
axes[1].set_title("Independent minus partner usage, 2W riders\n(log swaps per active rider-month x100 ~ %, ref Jun-24, 95% CI)", fontsize=11.5)
axes[1].annotate(f"Jul-Aug DiD 2W: {b2['did_pct']:+.1f}% in logs (elasticity {b2['elasticity']:+.2f}, CI {b2['elasticity_lo']:+.2f} to {b2['elasticity_hi']:+.2f});\n"
                 f"   {b2['did_pct_level']:+.1f}% in levels (elasticity {b2['elasticity_level']:+.2f}, CI {b2['elasticity_level_lo']:+.2f} to {b2['elasticity_level_hi']:+.2f})\n"
                 f"3W: {b3['did_pct']:+.1f}% in logs, {b3['did_pct_level']:+.1f}% in levels (not significant)",
                 xy=(0.02, 0.04), xycoords="axes fraction", fontsize=9.5, color=PALETTE["primary"])
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
fig.suptitle("The Jul-2024 base price rise passed straight into revenue; independents barely changed usage",
             x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.03)
fig.tight_layout()
save_fig(fig, f"{QID}_03_base_price_change")

# %% [markdown]
# ## 6. Partner economics (12 partners + independents)
# Per completed swap: revenue, energy, battery wear, station fixed cost -> cm1, cm2. Working-capital cost =
# revenue x 12% x payment_terms_days / 365 (independents pay at the swap: 0). "Steady-state" cm1 replaces Kyron wear and
# energy with the non-Kyron average (all Kyron packs were retired on 2025-06-15), i.e. the margin a contract signed today
# would earn. Main window = **current terms** (Nov-2024..Jun-2025, after the ZipDrop amendment); full window also saved.

# %%
riders_by = {}
for lab, (m0, m1) in {"current": CUR, "full": (M(2024, 1), M(2025, 6))}.items():
    riders_by[lab] = (LF.filter(pl.col("is_completed") & pl.col("month").is_between(m0, m1))
                      .group_by("pid").agg(pl.col("rider_id").n_unique().alias("riders")).collect())
PINFO = PARTNERS.select(pl.col("partner_id").alias("pid"), "partner_name", "partner_segment", "contract_type", "discount_pct",
                        "discount_pct_after_amendment", "peak_surcharge_billable", "payment_terms_days")


def partner_table(m0, m1, rid):
    c = CELLS.filter(pl.col("month").is_between(m0, m1))
    n_months = len({d for d in c["month"].to_list()})
    tot = c["swaps"].sum()
    pilot_peak = (pl.col("city").is_in(PILOT_CITIES)) & (pl.col("month") >= PILOT_START) & (pl.col("band") == "peak")
    g = (c.group_by("pid").agg(
        pl.col("swaps").sum(), pl.col("revenue").sum(), pl.col("list_rev").sum(), pl.col("disc").sum(),
        pl.col("energy").sum(), pl.col("wear").sum(), pl.col("fixed").sum(), pl.col("cm1").sum(), pl.col("cm2").sum(),
        pl.col("ky_swaps").sum(), pl.col("wear_nonky").sum(), pl.col("energy_nonky").sum(),
        pl.col("swaps").filter(pl.col("band") == "peak").sum().alias("peak_swaps"),
        pl.col("swaps").filter(pilot_peak).sum().alias("pilot_peak_swaps"),
        pl.col("swaps").filter(pl.col("vehicle_class") == "3W").sum().alias("swaps_3w"),
        pl.col("swaps").filter(pl.col("city").is_in(PILOT_CITIES)).sum().alias("pilot_city_swaps"))
         .join(rid, on="pid", how="left").join(PINFO, on="pid", how="left")
         .with_columns(pl.when(pl.col("pid") == "IND").then(pl.lit("Independents")).otherwise(pl.col("partner_name")).alias("partner_name"),
                       pl.when(pl.col("pid") == "IND").then(pl.lit("independent")).otherwise(pl.col("partner_segment")).alias("partner_segment"),
                       pl.col("payment_terms_days").fill_null(0))
         .with_columns((pl.col("swaps") / tot).alias("share_network_swaps"),
                       (pl.col("disc") / pl.col("list_rev")).alias("discount_pct_realised"),
                       *[(pl.col(k) / pl.col("swaps")).alias(f"{k}_ps") for k in ("revenue", "list_rev", "energy", "wear", "fixed", "cm1", "cm2")],
                       (pl.col("swaps_3w") / pl.col("swaps")).alias("share_3w"),
                       (pl.col("peak_swaps") / pl.col("swaps")).alias("peak_share"),
                       (pl.col("revenue") * COC * pl.col("payment_terms_days") / 365).alias("wc_cost"))
         .with_columns((pl.col("wc_cost") / pl.col("swaps")).alias("wc_ps"),
                       (pl.col("cm2") - pl.col("wc_cost")).alias("cm2_after_wc"),
                       (pl.col("revenue_ps") - pl.col("energy_nonky") / (pl.col("swaps") - pl.col("ky_swaps"))
                        - pl.col("wear_nonky") / (pl.col("swaps") - pl.col("ky_swaps"))).alias("cm1_ps_steady"),
                       pl.when(pl.col("peak_surcharge_billable") == "N").then(pl.col("pilot_peak_swaps") * 15.0).otherwise(0.0).alias("surcharge_forgone_pilot"),
                       pl.lit(n_months).alias("n_months"))
         .with_columns((pl.col("cm2_after_wc") / pl.col("swaps")).alias("cm2_after_wc_ps"),
                       (pl.col("cm1_ps_steady") - pl.col("fixed_ps") - pl.col("wc_ps")).alias("cm2wc_ps_steady"),
                       (pl.col("swaps") / n_months).alias("swaps_per_month"))
         .sort("swaps", descending=True))
    return g


PT_CUR = partner_table(*CUR, riders_by["current"])
PT_FULL = partner_table(M(2024, 1), M(2025, 6), riders_by["full"])
keep = ["pid", "partner_name", "partner_segment", "riders", "swaps", "share_network_swaps", "revenue", "discount_pct_realised",
        "revenue_ps", "energy_ps", "wear_ps", "fixed_ps", "cm1_ps", "cm2_ps", "cm1", "cm2", "payment_terms_days", "wc_cost",
        "wc_ps", "cm2_after_wc", "cm2_after_wc_ps", "cm1_ps_steady", "cm2wc_ps_steady", "peak_share", "share_3w",
        "peak_surcharge_billable", "surcharge_forgone_pilot", "swaps_per_month"]
PT_CUR.select(keep).write_csv(TAB / f"{QID}_partner_economics_current_terms.csv")
PT_FULL.select(keep).write_csv(TAB / f"{QID}_partner_economics_full_window.csv")
show = ["pid", "partner_name", "riders", "swaps", "share_network_swaps", "revenue", "discount_pct_realised", "revenue_ps", "energy_ps",
        "wear_ps", "fixed_ps", "cm1_ps", "cm2_ps", "cm1", "cm2", "wc_ps", "cm2_after_wc", "cm1_ps_steady", "peak_share", "share_3w"]
p("\nPartner economics - CURRENT TERMS (Nov-2024..Jun-2025):")
p(PT_CUR.select(show).with_columns(pl.col(pl.Float64).round(3)))
p("\nPartner economics - FULL WINDOW (Jan-2024..Jun-2025):")
p(PT_FULL.select(show).with_columns(pl.col(pl.Float64).round(3)))

# who is the largest partner, and is it the most valuable?
for lab, t in (("current terms", PT_CUR), ("full window", PT_FULL)):
    pt = t.filter(pl.col("pid") != "IND")
    big_sw = pt.sort("swaps", descending=True)["partner_name"][0]
    big_rev = pt.sort("revenue", descending=True)["partner_name"][0]
    big_rid = pt.sort("riders", descending=True)["partner_name"][0]
    rank_cm1 = pt.sort("cm1", descending=True).with_row_index("rank_cm1", offset=1)
    rank_cm2 = pt.sort("cm2_after_wc", descending=True).with_row_index("rank_cm2wc", offset=1)
    zd = rank_cm1.filter(pl.col("pid") == ZIPDROP_ID)
    zd2 = rank_cm2.filter(pl.col("pid") == ZIPDROP_ID)
    p(f"[{lab}] largest partner by swaps: {big_sw}; by revenue: {big_rev}; by riders: {big_rid}. "
      f"ZipDrop share of network swaps {100*zd['share_network_swaps'][0]:.1f}%, of partner swaps {100*zd['swaps'][0]/pt['swaps'].sum():.1f}%; "
      f"ZipDrop rank by total cm1 = {zd['rank_cm1'][0]}/12 (cm1 {money(zd['cm1'][0])}), by cm2 after WC = {zd2['rank_cm2wc'][0]}/12 "
      f"({money(zd2['cm2_after_wc'][0])})")
    p(f"[{lab}] best partner by total cm1: {rank_cm1['partner_name'][0]} ({money(rank_cm1['cm1'][0])}); by cm1/swap: "
      f"{pt.sort('cm1_ps', descending=True)['partner_name'][0]} (Rs{pt.sort('cm1_ps', descending=True)['cm1_ps'][0]:.2f}); "
      f"by cm2 after WC: {rank_cm2['partner_name'][0]} ({money(rank_cm2['cm2_after_wc'][0])})")

# Per-swap ranks differ from total ranks: the two 3W cargo partners lose more per swap than ZipDrop.
PT_RANK = PT_CUR.filter(pl.col("pid") != "IND").join(
    PARTNERS.select(pl.col("partner_id").alias("pid"), pl.col("vehicle_class").alias("contract_vc")), on="pid", how="left")
ZD_RANK = {}
for k, lab_k in (("cm1_ps", "cm1/swap"), ("cm2_after_wc_ps", "cm2 after WC/swap"), ("cm1_ps_steady", "steady-state cm1/swap")):
    rk = PT_RANK.sort(k, descending=True).with_row_index("rk", offset=1)
    zrow_ = rk.filter(pl.col("pid") == ZIPDROP_ID)
    lower = rk.filter(pl.col(k) < zrow_[k][0])
    rk2w = PT_RANK.filter(pl.col("contract_vc") != "3W").sort(k, descending=True).with_row_index("rk", offset=1)  # 2W + 'mixed' (MetroMove)
    ZD_RANK[k] = dict(rank=zrow_["rk"][0], n=rk.height, rank_2w=rk2w.filter(pl.col("pid") == ZIPDROP_ID)["rk"][0], n_2w=rk2w.height,
                      lower=lower["partner_name"].to_list())
    p(f"[current terms] ZipDrop rank by {lab_k}: {ZD_RANK[k]['rank']}/{rk.height} (Rs{zrow_[k][0]:.2f}); among the partners with 2W or mixed "
      f"fleets {ZD_RANK[k]['rank_2w']}/{rk2w.height}; lower per swap: "
      + ", ".join(f"{n} ({v_:.2f}, {vc_})" for n, v_, vc_ in lower.select("partner_name", k, "contract_vc").iter_rows()))

# %% [markdown]
# ### Mix-standardised comparison: is it the contract or the cost to serve?
# For each partner, compare its revenue and cost per swap with the *other partners'* average in the same
# city x month x vehicle-class x tariff-band cell. Positive cost gap = more expensive to serve than peers.

# %%
pc = CELLS.filter((pl.col("pid") != "IND") & pl.col("month").is_between(*CUR))
keys_ = ["month", "city", "vehicle_class", "band"]
tot_c = pc.group_by(keys_).agg([pl.col(k).sum().alias(f"T_{k}") for k in ("swaps", "revenue", "energy", "wear", "fixed", "cm1")])
mixstd = (pc.join(tot_c, on=keys_)
          .with_columns([((pl.col(f"T_{k}") - pl.col(k)) / (pl.col("T_swaps") - pl.col("swaps"))).alias(f"peer_{k}_ps")
                         for k in ("revenue", "energy", "wear", "fixed", "cm1")])
          .filter(pl.col("T_swaps") > pl.col("swaps"))
          .group_by("pid").agg(pl.col("swaps").sum(),
                               *[((pl.col(k) - pl.col("swaps") * pl.col(f"peer_{k}_ps")).sum() / pl.col("swaps").sum()).alias(f"gap_{k}_ps")
                                 for k in ("revenue", "energy", "wear", "fixed", "cm1")])
          .with_columns((pl.col("gap_energy_ps") + pl.col("gap_wear_ps") + pl.col("gap_fixed_ps")).alias("gap_cost_ps"))
          .join(PINFO.select("pid", "partner_name"), on="pid").sort("gap_cm1_ps"))
mixstd.write_csv(TAB / f"{QID}_partner_mix_standardised_gaps.csv")
p("\nMix-standardised gaps vs peer partners in the same city x month x class x band cell (Rs/swap, current terms):")
p(mixstd.with_columns(pl.col(pl.Float64).round(2)))

# segment-level view
SEG = (PT_CUR.group_by("partner_segment").agg(
    pl.col("riders").sum(), pl.col("swaps").sum(), pl.col("revenue").sum(), pl.col("list_rev").sum(), pl.col("disc").sum(),
    pl.col("energy").sum(), pl.col("wear").sum(), pl.col("fixed").sum(), pl.col("cm1").sum(), pl.col("cm2").sum(),
    pl.col("wc_cost").sum(), pl.col("peak_swaps").sum(), pl.col("swaps_3w").sum())
       .with_columns((pl.col("swaps") / pl.col("swaps").sum()).alias("share_swaps"),
                     (pl.col("disc") / pl.col("list_rev")).alias("discount_pct"),
                     *[(pl.col(k) / pl.col("swaps")).alias(f"{k}_ps") for k in ("revenue", "energy", "wear", "fixed", "cm1", "cm2")],
                     ((pl.col("cm2") - pl.col("wc_cost")) / pl.col("swaps")).alias("cm2_after_wc_ps"),
                     (pl.col("peak_swaps") / pl.col("swaps")).alias("peak_share"), (pl.col("swaps_3w") / pl.col("swaps")).alias("share_3w"))
       .sort("cm1_ps", descending=True))
SEG.write_csv(TAB / f"{QID}_segment_economics.csv")
p("\nSegment economics (current terms):")
p(SEG.select("partner_segment", "riders", "swaps", "share_swaps", "discount_pct", "revenue_ps", "energy_ps", "wear_ps", "cm1_ps", "cm2_ps",
             "cm2_after_wc_ps", "cm1", "peak_share", "share_3w").with_columns(pl.col(pl.Float64).round(3)))

# 2W vs 3W unit economics (all riders, current terms, steady-state wear)
VC = (CELLS.filter(pl.col("month").is_between(*CUR)).group_by("vehicle_class")
      .agg(pl.col("swaps").sum(), (pl.col("revenue").sum() / pl.col("swaps").sum()).alias("rev_ps"),
           (pl.col("list_rev").sum() / pl.col("swaps").sum()).alias("list_ps"),
           (pl.col("energy").sum() / pl.col("swaps").sum()).alias("energy_ps"), (pl.col("wear").sum() / pl.col("swaps").sum()).alias("wear_ps"),
           (pl.col("wear_nonky").sum() / (pl.col("swaps").sum() - pl.col("ky_swaps").sum())).alias("wear_ps_nonkyron"),
           (pl.col("energy_nonky").sum() / (pl.col("swaps").sum() - pl.col("ky_swaps").sum())).alias("energy_ps_nonkyron"),
           (pl.col("fixed").sum() / pl.col("swaps").sum()).alias("fixed_ps"), (pl.col("cm1").sum() / pl.col("swaps").sum()).alias("cm1_ps"),
           pl.col("cm1").sum().alias("cm1"))
      .with_columns((pl.col("rev_ps") - pl.col("energy_ps_nonkyron") - pl.col("wear_ps_nonkyron")).alias("cm1_ps_steady"))
      .sort("vehicle_class"))
v3 = VC.filter(pl.col("vehicle_class") == "3W").to_dicts()[0]
be3 = v3["energy_ps_nonkyron"] + v3["wear_ps_nonkyron"]
p(f"3W steady-state energy+wear = Rs{be3:.2f}/swap vs list Rs{v3['list_ps']:.0f}; at the current realised/list ratio "
  f"{v3['rev_ps']/v3['list_ps']:.3f} the cm1 break-even 3W list price is Rs{be3/(v3['rev_ps']/v3['list_ps']):.0f}")

VC.write_csv(TAB / f"{QID}_vehicle_class_economics.csv")
p("\n2W vs 3W unit economics (current terms):")
p(VC.with_columns(pl.col(pl.Float64).round(2)))

# %% [markdown]
# ### Figure: partner value bubble chart (hero)

# %%
SEG_COL = {"quick_commerce": SERIES[1], "food_delivery": SERIES[3], "ecom_logistics": SERIES[0], "cargo_3w": SERIES[4],
           "bike_taxi": SERIES[5], "independent": SERIES[2]}
bt = PT_CUR.to_pandas()
fig, ax = plt.subplots(figsize=(13, 7.4))
smax = bt["revenue"].max()
OFFS = {"IND": (-22, 16, "right"), ZIPDROP_ID: (16, -6, "left"), "FP-07": (8, 9, "left"), "FP-06": (8, -24, "left"),
        "FP-08": (-10, 12, "right"), "FP-12": (-8, -26, "right"), "FP-11": (-10, 4, "right"), "FP-10": (9, -16, "left")}
for _, r in bt.iterrows():
    ax.scatter(r["swaps"] / 1e3, r["cm1_ps"], s=1800 * r["revenue"] / smax + 40, color=SEG_COL[r["partner_segment"]], alpha=0.55,
               edgecolor="white", lw=1.2, zorder=3)
    lbl = f"{r['partner_name']}\n{r['cm1_ps']:+.1f}/swap, total {money(r['cm1'])}"
    dx, dy, ha = OFFS.get(r["pid"], (8, 8, "left"))
    ax.annotate(lbl, (r["swaps"] / 1e3, r["cm1_ps"]), xytext=(dx, dy), textcoords="offset points", fontsize=8.8,
                ha=ha, color="#333333", fontweight="bold" if r["pid"] in (ZIPDROP_ID, "IND") else "normal")
ax.axhline(0, color="#555555", lw=1)
for segname, colr in SEG_COL.items():
    ax.scatter([], [], s=90, color=colr, alpha=0.6, label=segname.replace("_", " "))
ax.legend(title="Segment (bubble area = revenue)", loc="upper center", ncol=3, fontsize=9, title_fontsize=9.5)
ax.set_xscale("log")
ax.set_xlim(22, 1500); ax.set_ylim(-27, 11)
ax.set_xticks([30, 50, 100, 200, 400, 800]); ax.set_xticklabels(["30", "50", "100", "200", "400", "800"])
ax.set_xlabel("Completed swaps, Nov-2024 to Jun-2025 (thousands, log scale)")
ax.set_ylabel("cm1 per swap (Rs) = revenue - energy - battery wear")
zrow = bt[bt["pid"] == ZIPDROP_ID].iloc[0]
ax.set_title("The largest fleet partner (ZipDrop) is the least valuable: biggest volume, biggest total loss", fontsize=13.5, pad=38)
ax.text(0.0, 1.015, f"Current contract terms. ZipDrop = {100*zrow['share_network_swaps']:.0f}% of network swaps at a {100*zrow['discount_pct_realised']:.0f}% discount and no peak surcharge; "
        f"worst cm1/swap of the {ZD_RANK['cm1_ps']['n_2w']} partners with 2W or mixed fleets.\nPer swap only the 3W cargo partners ({', '.join(ZD_RANK['cm1_ps']['lower'])}) are lower. "
        "Levels under our usage-based battery-depreciation assumption.", transform=ax.transAxes, fontsize=9.5, color="#555555")
save_fig(fig, f"{QID}_04_partner_value_bubble")

# %% [markdown]
# ### Figure: per-swap unit economics by partner (revenue vs cost stack)

# %%
bt2 = bt.sort_values("cm1_ps")
fig, ax = plt.subplots(figsize=(13, 6.4))
y = np.arange(len(bt2))
left = np.zeros(len(bt2))
for k, colr, lab in (("energy_ps", SERIES[5], "energy"), ("wear_ps", SERIES[1], "battery wear"), ("fixed_ps", PALETTE["muted"], "station fixed cost"),
                     ("wc_ps", SERIES[4], "working capital (12%/yr)")):
    ax.barh(y, bt2[k], left=left, color=colr, alpha=0.75, label=lab, height=0.62)
    left += bt2[k].values
ax.scatter(bt2["revenue_ps"], y, marker="D", color="black", zorder=4, s=36, label="revenue per swap")
for yy, (_, r) in zip(y, bt2.iterrows()):
    ax.text(max(r["revenue_ps"], left[yy]) + 2, yy, f"cm1 {r['cm1_ps']:+.1f} | cm2 {r['cm2_after_wc_ps']:+.1f}", va="center", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels([f"{n} ({100*d:.0f}% disc.)" for n, d in zip(bt2["partner_name"], bt2["discount_pct_realised"])])
ax.set_xlabel("Rs per completed swap (Nov-2024 to Jun-2025)")
ax.set_xlim(0, max(left.max(), bt2["revenue_ps"].max()) + 38)
ax.legend(loc="upper right", fontsize=9)
ax.set_title("Cost to serve is almost identical across 2W partners; the discount decides the margin", fontsize=13.5, pad=24)
ax.text(0.0, 1.015, "Stacked bars = cost per swap; diamond = revenue per swap; cm2 shown after working-capital cost. 3W partners (CargoTuk, HaulKing) carry ~2x energy and wear.",
        transform=ax.transAxes, fontsize=9.3, color="#555555")
save_fig(fig, f"{QID}_05_partner_unit_economics")

# %% [markdown]
# ## 7. ZipDrop amendment (2024-11-01: discount 12% -> 28%, peak surcharge not billable)
# Before/after for ZipDrop 2W swaps vs other partners' 2W swaps (difference-in-differences removes the network-wide
# Kyron wear shock that started in Sep-2024). Pre = Jul-Oct 2024 (new list price already in force); post = Nov-24..Jun-25.
#
# **Concurrent-treatment trap.** From Oct-2024 the surcharge-billable partners in Bengaluru and Pune started paying the
# pilot's +Rs15 / -Rs8, which lifts the comparison group's revenue per swap in the post period. The price and margin DiD is
# therefore estimated in the **four non-pilot cities only** (both groups), where nobody pays a surcharge; the all-city
# version is printed for comparison. Volume and the Rs-per-month cost of the amendment use all cities.

# %%
def zip_monthly(cities=None):
    c = CELLS.filter((pl.col("pid") != "IND") & (pl.col("vehicle_class") == "2W") & (pl.col("month") >= M(2024, 1)))
    if cities is not None:
        c = c.filter(pl.col("city").is_in(cities))
    return (c.with_columns(pl.when(pl.col("pid") == ZIPDROP_ID).then(pl.lit("ZipDrop")).otherwise(pl.lit("Other partners")).alias("who"))
            .group_by("who", "month").agg(pl.col("swaps").sum(), pl.col("revenue").sum(), pl.col("list_rev").sum(), pl.col("disc").sum(),
                                          pl.col("cm1").sum(), pl.col("cm2").sum(), pl.col("energy").sum(), pl.col("wear").sum())
            .with_columns((pl.col("revenue") / pl.col("swaps")).alias("rev_ps"), (pl.col("cm1") / pl.col("swaps")).alias("cm1_ps"),
                          (pl.col("disc") / pl.col("list_rev")).alias("disc_pct"),
                          ((pl.col("energy") + pl.col("wear")) / pl.col("swaps")).alias("var_cost_ps"))
            .sort("who", "month"))


def zd_prepost(zm_):
    zp_ = zm_.to_pandas()
    zp_["month"] = pd.to_datetime(zp_["month"]).dt.date
    zp_["post"] = zp_["month"] >= ZIPDROP_AMEND
    pre_, post_ = zp_[(zp_["month"] >= M(2024, 7)) & (~zp_["post"])], zp_[zp_["post"]]
    res = {}
    for who in ("ZipDrop", "Other partners"):
        for per, d in (("pre", pre_), ("post", post_)):
            dd = d[d["who"] == who]
            res[(who, per)] = dict(rev_ps=dd["revenue"].sum() / dd["swaps"].sum(), cm1_ps=dd["cm1"].sum() / dd["swaps"].sum(),
                                   var_cost_ps=(dd["energy"].sum() + dd["wear"].sum()) / dd["swaps"].sum(),
                                   disc_pct=dd["disc"].sum() / dd["list_rev"].sum(), swaps_pm=dd["swaps"].mean())
    dlt = lambda k: (res[("ZipDrop", "post")][k] - res[("ZipDrop", "pre")][k]) - (res[("Other partners", "post")][k] - res[("Other partners", "pre")][k])  # noqa: E731
    return zp_, res, dlt("rev_ps"), dlt("cm1_ps")


zm = zip_monthly()                     # all cities (volume, cost of the amendment)
zm_np = zip_monthly(CONTROL_CITIES)    # four non-pilot cities: nobody pays a pilot surcharge (price / margin DiD)
pl.concat([zm.with_columns(pl.lit("all cities").alias("scope")),
           zm_np.with_columns(pl.lit("4 non-pilot cities").alias("scope"))]).write_csv(TAB / f"{QID}_zipdrop_monthly.csv")
zp, ZD, did_rev_all, did_cm1_all = zd_prepost(zm)
zp_np, ZD_NP, did_rev, did_cm1 = zd_prepost(zm_np)
zd_all = CELLS.filter((pl.col("pid") == ZIPDROP_ID) & (pl.col("month") >= ZIPDROP_AMEND))
zd_months = zd_all["month"].n_unique()
zd_list_pm = zd_all["list_rev"].sum() / zd_months
zd_disc_pm = zd_all["disc"].sum() / zd_months
monthly_cost = zd_list_pm * (0.28 - 0.12)
zd_post_swaps_pm = zd_all["swaps"].sum() / zd_months
growth = {who: ZD[(who, "post")]["swaps_pm"] / ZD[(who, "pre")]["swaps_pm"] - 1 for who in ("ZipDrop", "Other partners")}
share_pre = ZD[("ZipDrop", "pre")]["swaps_pm"] / (ZD[("ZipDrop", "pre")]["swaps_pm"] + ZD[("Other partners", "pre")]["swaps_pm"])
share_post = ZD[("ZipDrop", "post")]["swaps_pm"] / (ZD[("ZipDrop", "post")]["swaps_pm"] + ZD[("Other partners", "post")]["swaps_pm"])
p(f"\nZipDrop 2W: realised discount {100*ZD[('ZipDrop','pre')]['disc_pct']:.1f}% -> {100*ZD[('ZipDrop','post')]['disc_pct']:.1f}%; "
  f"revenue/swap Rs{ZD[('ZipDrop','pre')]['rev_ps']:.2f} -> Rs{ZD[('ZipDrop','post')]['rev_ps']:.2f}; cm1/swap Rs{ZD[('ZipDrop','pre')]['cm1_ps']:.2f} -> Rs{ZD[('ZipDrop','post')]['cm1_ps']:.2f}")
p(f"Other partners 2W (all cities): revenue/swap Rs{ZD[('Other partners','pre')]['rev_ps']:.2f} -> Rs{ZD[('Other partners','post')]['rev_ps']:.2f} "
  f"({ZD[('Other partners','post')]['rev_ps'] - ZD[('Other partners','pre')]['rev_ps']:+.2f}: pilot surcharge paid by billable partners in Bengaluru/Pune); "
  f"cm1/swap Rs{ZD[('Other partners','pre')]['cm1_ps']:.2f} -> Rs{ZD[('Other partners','post')]['cm1_ps']:.2f}; var cost/swap ZipDrop "
  f"Rs{ZD[('ZipDrop','pre')]['var_cost_ps']:.2f}->{ZD[('ZipDrop','post')]['var_cost_ps']:.2f}, others Rs{ZD[('Other partners','pre')]['var_cost_ps']:.2f}->{ZD[('Other partners','post')]['var_cost_ps']:.2f}")
p(f"4 non-pilot cities: ZipDrop 2W revenue/swap Rs{ZD_NP[('ZipDrop','pre')]['rev_ps']:.2f} -> Rs{ZD_NP[('ZipDrop','post')]['rev_ps']:.2f}, cm1 "
  f"Rs{ZD_NP[('ZipDrop','pre')]['cm1_ps']:.2f} -> Rs{ZD_NP[('ZipDrop','post')]['cm1_ps']:.2f}; other partners revenue/swap "
  f"Rs{ZD_NP[('Other partners','pre')]['rev_ps']:.2f} -> Rs{ZD_NP[('Other partners','post')]['rev_ps']:.2f}, cm1 "
  f"Rs{ZD_NP[('Other partners','pre')]['cm1_ps']:.2f} -> Rs{ZD_NP[('Other partners','post')]['cm1_ps']:.2f}")
p(f"Aggregate DiD ZipDrop vs other partners: 4 non-pilot cities revenue/swap {did_rev:+.2f}, cm1/swap {did_cm1:+.2f} Rs | "
  f"all cities (comparison group moved by the pilot surcharge) revenue/swap {did_rev_all:+.2f}, cm1/swap {did_cm1_all:+.2f} Rs")
p(f"Monthly cost of the amendment (all ZipDrop swaps, Nov-24..Jun-25): list revenue {money(zd_list_pm)}/month x 16 pp = {money(monthly_cost)}/month "
  f"({money(monthly_cost*12)}/yr); ZipDrop swaps {zd_post_swaps_pm:,.0f}/month")
p(f"Volume: ZipDrop 2W swaps/month {ZD[('ZipDrop','pre')]['swaps_pm']:,.0f} -> {ZD[('ZipDrop','post')]['swaps_pm']:,.0f} ({100*growth['ZipDrop']:+.1f}%) vs other partners "
  f"{100*growth['Other partners']:+.1f}%; ZipDrop share of partner 2W swaps {100*share_pre:.1f}% -> {100*share_post:.1f}%")
growth_np = {who: ZD_NP[(who, "post")]["swaps_pm"] / ZD_NP[(who, "pre")]["swaps_pm"] - 1 for who in ("ZipDrop", "Other partners")}
p(f"Volume robustness, 4 non-pilot cities only: ZipDrop {100*growth_np['ZipDrop']:+.1f}% vs other partners {100*growth_np['Other partners']:+.1f}%")
# swaps-weighted DiD on the rider panel with CIs: main = 4 non-pilot cities; all cities shown for comparison
zr = RM[(RM["grp"] != GRP_ORDER[0]) & (RM["vc"] == "2W") & (RM["month"] >= M(2024, 7))].copy()
zr["zp"] = ((zr["pid"] == ZIPDROP_ID) & (zr["month"] >= ZIPDROP_AMEND)).astype(float)
ZD_REG = {k: coef(twfe(zr[~zr["treated"]], k, ["zp"], w="swaps"), "zp") for k in ("rev_ps", "cm1_ps")}
ZD_REG_ALL = {k: coef(twfe(zr, k, ["zp"], w="swaps"), "zp") for k in ("rev_ps", "cm1_ps")}
p(f"Rider-level DiD (rider-city + month FE), 4 non-pilot cities [MAIN]: revenue/swap {ZD_REG['rev_ps']['est']:+.2f} "
  f"(CI {ZD_REG['rev_ps']['lo']:+.2f} to {ZD_REG['rev_ps']['hi']:+.2f}); cm1/swap {ZD_REG['cm1_ps']['est']:+.2f} "
  f"(CI {ZD_REG['cm1_ps']['lo']:+.2f} to {ZD_REG['cm1_ps']['hi']:+.2f}); mechanical 16 pp x Rs65 list = Rs{0.16*65:.2f}")
p(f"Rider-level DiD, all cities (inflated by the pilot surcharge in the comparison group): revenue/swap {ZD_REG_ALL['rev_ps']['est']:+.2f} "
  f"(CI {ZD_REG_ALL['rev_ps']['lo']:+.2f} to {ZD_REG_ALL['rev_ps']['hi']:+.2f}); cm1/swap {ZD_REG_ALL['cm1_ps']['est']:+.2f} "
  f"(CI {ZD_REG_ALL['cm1_ps']['lo']:+.2f} to {ZD_REG_ALL['cm1_ps']['hi']:+.2f}); inflation revenue "
  f"{ZD_REG_ALL['rev_ps']['est'] - ZD_REG['rev_ps']['est']:+.2f}, cm1 {ZD_REG_ALL['cm1_ps']['est'] - ZD_REG['cm1_ps']['est']:+.2f} Rs/swap")
# volume per rider response (did the deeper discount buy more swaps per rider?)
zr["swaps_f"] = zr["swaps"].astype(float)
ZD_VOL = coef(twfe(zr, "swaps_f", ["zp", "ky_share"]), "zp")
zpre_spr = zr[(zr["pid"] == ZIPDROP_ID) & (zr["month"] < ZIPDROP_AMEND)]["swaps"].mean()
p(f"Swaps per active ZipDrop rider-month DiD: {ZD_VOL['est']:+.2f} (CI {ZD_VOL['lo']:+.2f} to {ZD_VOL['hi']:+.2f}) on a base of {zpre_spr:.1f}")

# %% [markdown]
# ### Figure: ZipDrop before/after (hero)

# %%
fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))
for who, colr in (("ZipDrop", PALETTE["accent"]), ("Other partners", PALETTE["primary"])):
    d = zp_np[zp_np["who"] == who]          # price / margin: four non-pilot cities
    x = pd.to_datetime(d["month"])
    axes[0].plot(x, d["rev_ps"], "o-", ms=4, color=colr, label=who)
    axes[1].plot(x, d["cm1_ps"], "o-", ms=4, color=colr, label=who)
    d = zp[zp["who"] == who]                # volume: all cities
    axes[2].plot(pd.to_datetime(d["month"]), d["swaps"] / 1e3, "o-", ms=4, color=colr, label=f"{who} 2W swaps")
for ax in axes:
    ax.axvline(pd.Timestamp(ZIPDROP_AMEND), color="#555555", lw=1, ls=":")
    ax.axvline(pd.Timestamp(BASE_PRICE_CHANGE), color="#BBBBBB", lw=1, ls=":")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
axes[1].axhline(0, color="#555555", lw=1)
axes[0].set_title("Revenue per 2W swap, 4 non-pilot cities (Rs)", fontsize=11.5)
axes[1].set_title("cm1 per 2W swap, 4 non-pilot cities (Rs)", fontsize=11.5)
axes[2].set_title("Monthly 2W swaps, all cities (thousands)", fontsize=11.5)
axes[0].legend(loc="upper left")
axes[0].annotate(f"Discount 12% -> 28%:\nDiD {ZD_REG['rev_ps']['est']:+.2f} Rs/swap\n= 16 pp x Rs65 list\n= {money(monthly_cost)}/month",
                 xy=(0.03, 0.14), xycoords="axes fraction", va="bottom", fontsize=10, color=PALETTE["accent"], fontweight="bold")
axes[1].annotate(f"DiD {ZD_REG['cm1_ps']['est']:+.2f} Rs/swap\n(95% CI {ZD_REG['cm1_ps']['lo']:+.2f} to {ZD_REG['cm1_ps']['hi']:+.2f})",
                 xy=(0.03, 0.06), xycoords="axes fraction", fontsize=10, color=PALETTE["accent"], fontweight="bold")
axes[2].annotate(f"ZipDrop {100*growth['ZipDrop']:+.0f}% vs others {100*growth['Other partners']:+.0f}%\n(Nov-Jun vs Jul-Oct avg);\nshare of partner 2W swaps\n{100*share_pre:.1f}% -> {100*share_post:.1f}%",
                 xy=(0.56, 0.36), xycoords="axes fraction", va="bottom", fontsize=9.5, color="#333333")
axes[2].set_ylim(bottom=0)
fig.suptitle(f"ZipDrop's deeper discount cut revenue by Rs{-ZD_REG['rev_ps']['est']:.1f} per 2W swap and bought no extra volume",
             x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.06)
fig.text(0.01, 0.94, "ZipDrop (FP-03) 2W swaps vs all other partners' 2W swaps; DiD = rider-city + month FE, pre Jul-Oct 2024, post Nov-24..Jun-25. "
         "Price and margin panels use the four non-pilot cities: in Bengaluru/Pune\nbillable partners began paying the pilot surcharge in Oct-24, "
         f"which would inflate the all-city DiD to {ZD_REG_ALL['rev_ps']['est']:+.1f} Rs. Dotted lines: base price change (Jul-24), amendment (Nov-24). "
         "The Sep-Oct-24 cm1 fall for both groups is Kyron battery wear.", fontsize=9.5, color="#555555", ha="left")
fig.tight_layout()
save_fig(fig, f"{QID}_06_zipdrop_amendment")

# %% [markdown]
# ## 8. Budget proposals
# **(c) Network-wide pricing rollout.** Apply the pilot DiD estimates (Oct-24..Mar-25) to the four control cities'
# Jan-Jun 2025 monthly volumes. Revenue effect = DiD revenue/swap x swaps for independents and billable-Y partners
# (non-billable partners pay nothing extra); cost per swap does not change with price, so cm1 moves one-for-one,
# plus the (statistically null) volume response. Congestion = DiD on peak stockout rate x peak attempts. The surcharge not
# collected from non-billable partners is shown **net** of the -Rs8 off-peak credit the same tariff would give them. We also
# price the option of dropping the (uncongested) 12-13h window from the surcharge.
# **(d) Long-term exclusive with the largest partner** = what ZipDrop's current terms lock in, annualised. A contract signed
# now should be judged at **steady state** (non-Kyron energy and wear; all Kyron packs were retired on 2025-06-15); the
# Nov-24..Jun-25 run-rate is shown as history. cm2 includes allocated station cost that an exclusive does not avoid, so cm1
# is the marginal test. FeastFly (largest partner by revenue and riders) is shown for comparison.

# %%
base = (CELLS.filter(pl.col("city").is_in(CONTROL_CITIES) & pl.col("month").is_between(M(2025, 1), M(2025, 6)))
        .join(PINFO.select("pid", "peak_surcharge_billable"), on="pid", how="left")
        .with_columns(pl.when(pl.col("pid") == "IND").then(pl.lit(GRP_ORDER[0]))
                      .when(pl.col("peak_surcharge_billable") == "Y").then(pl.lit(GRP_ORDER[1])).otherwise(pl.lit(GRP_ORDER[2])).alias("grp"))
        .group_by("grp").agg((pl.col("swaps").sum() / 6).alias("swaps_pm"), (pl.col("cm1").sum() / pl.col("swaps").sum()).alias("cm1_ps"),
                             (pl.col("swaps").filter(pl.col("band") == "peak").sum() / 6).alias("peak_swaps_pm"),
                             (pl.col("swaps").filter(pl.col("band") == "offpeak").sum() / 6).alias("off_swaps_pm"))
        .sort("grp").to_pandas().set_index("grp"))
peak_att_pm = SMP[SMP["city"].isin(CONTROL_CITIES) & (SMP["month"] >= M(2025, 1))]["att"].sum() / 6
ROLL = []
for g in GRP_ORDER[:2]:
    r = did_get(g, "rev_per_swap")
    n = base.loc[g, "swaps_pm"]
    ROLL.append(dict(item=f"Revenue/cm1: {g}", swaps_pm=n, did_rs_per_swap=r["est"], lo=r["lo"], hi=r["hi"],
                     monthly_rs=n * r["est"], monthly_lo=n * r["lo"], monthly_hi=n * r["hi"]))
vol = did_get(GRP_ORDER[0], "log_swaps", spec="DDD vs Partner-N riders (+Kyron)")
vol_pct = vol["est"] / 100
ROLL.append(dict(item="Volume response (independents) x cm1/swap", swaps_pm=base.loc[GRP_ORDER[0], "swaps_pm"], did_rs_per_swap=np.nan,
                 lo=np.nan, hi=np.nan, monthly_rs=base.loc[GRP_ORDER[0], "swaps_pm"] * vol_pct * base.loc[GRP_ORDER[0], "cm1_ps"],
                 monthly_lo=base.loc[GRP_ORDER[0], "swaps_pm"] * (vol["lo"] / 100) * base.loc[GRP_ORDER[0], "cm1_ps"],
                 monthly_hi=base.loc[GRP_ORDER[0], "swaps_pm"] * (vol["hi"] / 100) * base.loc[GRP_ORDER[0], "cm1_ps"]))
ROLL[-1]["monthly_lo"], ROLL[-1]["monthly_hi"] = sorted([ROLL[-1]["monthly_lo"], ROLL[-1]["monthly_hi"]])
nb_peak = base.loc[GRP_ORDER[2], "peak_swaps_pm"]
nb_off = base.loc[GRP_ORDER[2], "off_swaps_pm"]
nb_gross = nb_peak * 15.0
nb_net = nb_gross - nb_off * 8.0     # the same tariff gives billable partners the -Rs8 off-peak credit
ROLL.append(dict(item="Not collected: ZipDrop + Swiggle Go peak swaps x Rs15 net of off-peak swaps x Rs8 (if made billable)",
                 swaps_pm=nb_peak, did_rs_per_swap=np.nan, lo=np.nan, hi=np.nan, monthly_rs=nb_net, monthly_lo=np.nan, monthly_hi=np.nan))
ROLL = pd.DataFrame(ROLL)
so_m = DID[(DID["outcome"] == "peak_stockout_rate_pp") & (DID["post_window"] == "Oct24-Mar25") & (DID["spec"] == "station + month FE")].iloc[0]
pk_m = did_get(GRP_ORDER[0], "peak_share")
roll_rev = ROLL.iloc[:2]["monthly_rs"].sum(); roll_lo = ROLL.iloc[:2]["monthly_lo"].sum(); roll_hi = ROLL.iloc[:2]["monthly_hi"].sum()
roll_cm1 = roll_rev + ROLL.iloc[2]["monthly_rs"]
p("\n(c) Network-wide pricing rollout to Delhi NCR, Hyderabad, Mumbai, Jaipur (Jan-Jun 2025 base volumes):")
p(ROLL.round(2).to_string(index=False))
p(f"Projected revenue uplift {money(roll_rev)}/month (CI {money(roll_lo)} to {money(roll_hi)}) = {money(12*roll_rev)}/yr; "
  f"cm1 uplift incl. volume response {money(roll_cm1)}/month")
p(f"Projected peak congestion change: stockout rate {so_m['est']:+.2f} pp (CI {so_m['lo']:+.2f} to {so_m['hi']:+.2f}) x {peak_att_pm:,.0f} peak attempts/month "
  f"= {so_m['est']/100*peak_att_pm:+,.0f} stockouts/month (CI {so_m['lo']/100*peak_att_pm:+,.0f} to {so_m['hi']/100*peak_att_pm:+,.0f}); "
  f"independents' peak share {pk_m['est']:+.1f} pp shifts {abs(pk_m['est'])/100*base.loc[GRP_ORDER[0],'swaps_pm']:,.0f} swaps/month out of peak")
p(f"Not collected from non-billable partners (ZipDrop + Swiggle Go, 4 control cities): {nb_peak:,.0f} peak swaps/month x Rs15 = "
  f"Rs{nb_gross/1e5:.2f} L gross; net of {nb_off:,.0f} off-peak swaps/month x Rs8 credit = Rs{nb_net/1e5:.2f} L/month")

# Cross-check: re-price the control cities' OWN timing mix mechanically (after the pilot's timing shift) instead of
# transferring the pilot cities' Rs/swap.
mix_ctrl = (LF.filter(pl.col("is_completed") & pl.col("city").is_in(CONTROL_CITIES) & pl.col("month").is_between(M(2025, 1), M(2025, 6))
                      & pl.col("grp").is_in(GRP_ORDER[:2]))
            .group_by("grp").agg((pl.len() / 6).alias("swaps_pm"), pl.col("hour").is_in(MID_H).mean().alias("mid_share"),
                                 (pl.col("band") == "peak").mean().alias("peak_share"), (pl.col("band") == "offpeak").mean().alias("off_share"))
            .collect().to_pandas().set_index("grp"))
mech_roll = sum(mix_ctrl.loc[g, "swaps_pm"] * (15 * (mix_ctrl.loc[g, "peak_share"] + did_get(g, "peak_share")["est"] / 100)
                                               - 8 * (mix_ctrl.loc[g, "off_share"] + did_get(g, "off_share")["est"] / 100)) for g in GRP_ORDER[:2])
p(f"Mechanical rollout cross-check (control cities' own timing mix after the pilot's shift): {money(mech_roll)}/month "
  f"({100*(mech_roll/roll_rev - 1):+.1f}% vs the DiD-based {money(roll_rev)})")

# What would dropping the 12-13h window from the surcharge give up? (12-13h is the least congested PEAK window)
mid_i = did_get(GRP_ORDER[0], "mid_share")
mid_post_ind = (mid_i["pre_treated_mean"] + mid_i["est"]) / 100
drop_mid_ps = 15 * mid_post_ind
drop_mid_pm = sum(mix_ctrl.loc[g, "swaps_pm"] * 15 * (mix_ctrl.loc[g, "mid_share"] + did_get(g, "mid_share")["est"] / 100) for g in GRP_ORDER[:2])
p(f"Dropping 12-13h from the surcharge: pilot-city independents' 12-13h share after the pilot = {100*mid_post_ind:.1f}% "
  f"(pre {mid_i['pre_treated_mean']:.1f}% {mid_i['est']:+.2f} pp) x Rs15 = Rs{drop_mid_ps:.2f} of the Rs{rv['est']:.2f}/swap; "
  f"in the rollout it gives up {money(drop_mid_pm)}/month of the {money(roll_rev)}/month "
  f"(control-city 12-13h shares IND {100*mix_ctrl.loc[GRP_ORDER[0], 'mid_share']:.1f}%, billable partners {100*mix_ctrl.loc[GRP_ORDER[1], 'mid_share']:.1f}%, "
  f"less the pilot's 12-13h shift)")
so_eve = DID[(DID["outcome"] == "peak_stockout_rate_pp") & (DID["spec"] == "evening 19-22h only")].iloc[0]
p(f"Pilot DiD on evening-only (19-22h) stockouts: {so_eve['est']:+.2f} pp (95% CI {so_eve['lo']:+.2f} to {so_eve['hi']:+.2f}); "
  f"base {so_eve['pre_treated_mean']:.2f}%")

z = PT_CUR.filter(pl.col("pid") == ZIPDROP_ID).to_dicts()[0]
others = PT_CUR.filter((pl.col("pid") != "IND") & (pl.col("pid") != ZIPDROP_ID))
oth_2w = (CELLS.filter((pl.col("pid") != "IND") & (pl.col("pid") != ZIPDROP_ID) & (pl.col("vehicle_class") == "2W") & pl.col("month").is_between(*CUR)))
zd2w = CELLS.filter((pl.col("pid") == ZIPDROP_ID) & (pl.col("vehicle_class") == "2W") & pl.col("month").is_between(*CUR))
steady_cost_2w = ((zd2w["energy_nonky"].sum() + zd2w["wear_nonky"].sum()) / (zd2w["swaps"].sum() - zd2w["ky_swaps"].sum()))
fixed_2w = zd2w["fixed"].sum() / zd2w["swaps"].sum()
list_2w = zd2w["list_rev"].sum() / zd2w["swaps"].sum()
be_cm1 = 1 - steady_cost_2w / list_2w
be_cm2 = 1 - (steady_cost_2w + fixed_2w) / (list_2w * (1 - COC * 45 / 365))
annual = 12 / z["n_months"]
ff = PT_CUR.filter(pl.col("partner_name") == "FeastFly").to_dicts()[0]   # largest partner by revenue and riders
EXCL = pd.DataFrame([
    dict(metric="ZipDrop swaps per year (current run-rate)", value=z["swaps"] * annual),
    dict(metric="share of network swaps", value=z["share_network_swaps"]),
    dict(metric="revenue per year (Rs)", value=z["revenue"] * annual),
    dict(metric="realised discount", value=z["discount_pct_realised"]),
    dict(metric="STEADY STATE (non-Kyron energy + wear; all Kyron packs retired 2025-06-15): cm1 per swap (Rs)", value=z["cm1_ps_steady"]),
    dict(metric="steady-state cm1 per year (Rs)", value=z["cm1_ps_steady"] * z["swaps"] * annual),
    dict(metric="steady-state cm2 after WC per swap (Rs; cm2 includes allocated station cost an exclusive does not avoid)", value=z["cm2wc_ps_steady"]),
    dict(metric="steady-state cm2 after WC per year (Rs)", value=z["cm2wc_ps_steady"] * z["swaps"] * annual),
    dict(metric="HISTORICAL Nov-24..Jun-25 run-rate (incl. wear on retired Kyron packs): cm1 per swap (Rs)", value=z["cm1_ps"]),
    dict(metric="historical cm2 after working capital per swap (Rs)", value=z["cm2_after_wc_ps"]),
    dict(metric="historical cm1 per year (Rs)", value=z["cm1"] * annual),
    dict(metric="historical cm2 after working capital per year (Rs)", value=z["cm2_after_wc"] * annual),
    dict(metric="working-capital cost per year (45-day terms, Rs)", value=z["wc_cost"] * annual),
    dict(metric="peak-hour share of swaps", value=z["peak_share"]),
    dict(metric="amendment revenue given up per year (16 pp of list, Rs)", value=monthly_cost * 12),
    dict(metric="break-even discount for steady-state cm1 = 0 (2W)", value=be_cm1),
    dict(metric="break-even discount for steady-state cm2 after WC = 0 (2W)", value=be_cm2),
    dict(metric="other partners' cm1 per swap (avg, Rs)", value=others["cm1"].sum() / others["swaps"].sum()),
    dict(metric="other partners' steady-state cm1 per swap (avg, Rs)",
         value=(others["revenue"].sum() - (others["energy_nonky"].sum() + others["wear_nonky"].sum()) / (others["swaps"].sum() - others["ky_swaps"].sum()) * others["swaps"].sum()) / others["swaps"].sum()),
    dict(metric="FeastFly (largest by revenue and riders) share of network swaps", value=ff["share_network_swaps"]),
    dict(metric="FeastFly revenue per year (Rs)", value=ff["revenue"] * annual),
    dict(metric="FeastFly steady-state cm1 per swap (Rs)", value=ff["cm1_ps_steady"]),
    dict(metric="FeastFly steady-state cm1 per year (Rs)", value=ff["cm1_ps_steady"] * ff["swaps"] * annual),
    dict(metric="FeastFly steady-state cm2 after WC per swap (Rs)", value=ff["cm2wc_ps_steady"]),
    dict(metric="FeastFly historical cm1 per swap, Nov-24..Jun-25 (Rs)", value=ff["cm1_ps"]),
])
EXCL.to_csv(TAB / f"{QID}_budget_exclusive_zipdrop.csv", index=False)
ROLL.to_csv(TAB / f"{QID}_budget_pricing_rollout.csv", index=False)
p("\n(d) Long-term exclusive with the largest partner (ZipDrop) - what current terms lock in:")
p(EXCL.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
zd_peak_hours_share = (CELLS.filter(pl.col("month").is_between(*CUR) & (pl.col("band") == "peak"))
                       .select((pl.col("swaps").filter(pl.col("pid") == ZIPDROP_ID).sum() / pl.col("swaps").sum())).item())
p(f"ZipDrop share of all PEAK-hour swaps (current terms) = {100*zd_peak_hours_share:.1f}%")
zd_ss_yr = z["cm1_ps_steady"] * z["swaps"] * annual
zd_ss2_yr = z["cm2wc_ps_steady"] * z["swaps"] * annual
p(f"(d) ZipDrop exclusive, STEADY STATE: cm1 Rs{z['cm1_ps_steady']:.2f}/swap = {money(zd_ss_yr)}/yr; cm2 after WC Rs{z['cm2wc_ps_steady']:.2f}/swap = "
  f"{money(zd_ss2_yr)}/yr. HISTORICAL Nov-24..Jun-25 run-rate (incl. Kyron wear): cm1 {money(z['cm1'] * annual)}/yr, cm2 after WC "
  f"{money(z['cm2_after_wc'] * annual)}/yr (historical cm1 is {z['cm1'] * annual / zd_ss_yr:.1f}x the steady-state loss)")
p(f"FeastFly (largest by revenue and riders): steady-state cm1 Rs{ff['cm1_ps_steady']:+.2f}/swap = {money(ff['cm1_ps_steady'] * ff['swaps'] * annual)}/yr; "
  f"steady-state cm2 after WC Rs{ff['cm2wc_ps_steady']:+.2f}/swap; historical cm1 Rs{ff['cm1_ps']:+.2f}/swap; "
  f"{ff['payment_terms_days']}-day terms, {100*ff['discount_pct_realised']:.0f}% discount, peak surcharge billable = {ff['peak_surcharge_billable']}")

# %% [markdown]
# ### Figure: the two pricing-related budget proposals side by side

# %%
fig, axes = plt.subplots(1, 2, figsize=(17, 5.8), gridspec_kw=dict(width_ratios=[1, 1.35]))
items = ["Independents\n(DiD +Rs/swap)", "Billable partners\n(DiD +Rs/swap)", "Volume response\n(independents)",
         "Not collected, net of\n-Rs8 off-peak credit:\nZipDrop + Swiggle Go"]
vals = ROLL["monthly_rs"].values / 1e5
lo = ROLL["monthly_lo"].values / 1e5; hi = ROLL["monthly_hi"].values / 1e5
cols = [PALETTE["good"], PALETTE["good"], PALETTE["muted"], PALETTE["warn"]]
yy = np.arange(len(items))[::-1]
axes[0].barh(yy, vals, color=cols, alpha=0.85, height=0.6)
for i in range(3):
    if not np.isnan(lo[i]) and lo[i] <= vals[i] <= hi[i]:
        axes[0].errorbar(vals[i], yy[i], xerr=[[vals[i] - lo[i]], [hi[i] - vals[i]]], color="#333333", capsize=4)
for i, v in enumerate(vals):
    axes[0].text(max(v, 0) + 0.3, yy[i] + 0.22, f"Rs {v:.1f} L/month", fontsize=10)
axes[0].axvline(0, color="#555555", lw=1)
axes[0].set_yticks(yy); axes[0].set_yticklabels(items)
axes[0].set_xlabel("Rs lakh per month (4 control cities, Jan-Jun 2025 volumes)")
axes[0].set_title(f"(c) Rollout: +{money(roll_rev)}/month revenue, congestion\nunchanged ({so_m['est']:+.2f} pp peak stockouts)", fontsize=11.5)
axes[0].text(0.98, 0.37, f"Evening-only bands (dropping 12-13h)\nwould give up {money(drop_mid_pm)} of the {money(roll_rev)}/month",
             transform=axes[0].transAxes, ha="right", va="center", fontsize=9.5, color="#555555")
ff_ss_yr = ff["cm1_ps_steady"] * ff["swaps"] * annual
ex_items = ["ZipDrop\ncm1\nsteady state", "ZipDrop\ncm2 after WC\nsteady state*", "ZipDrop\ncm1\nhistorical",
            "ZipDrop\ncm2 after WC\nhistorical", "Amendment\ncost\n(16 pp)", "FeastFly\ncm1\nsteady state"]
ex_vals = np.array([zd_ss_yr, zd_ss2_yr, z["cm1"] * annual, z["cm2_after_wc"] * annual, -monthly_cost * 12, ff_ss_yr]) / 1e5
ex_cols = [PALETTE["accent"], PALETTE["accent"], PALETTE["muted"], PALETTE["muted"], PALETTE["warn"], PALETTE["good"]]
bars = axes[1].bar(np.arange(len(ex_vals)), ex_vals, color=ex_cols, alpha=0.85, width=0.62)
for b_ in (bars[2], bars[3]):
    b_.set_hatch("//"); b_.set_edgecolor("white")
for i, v in enumerate(ex_vals):
    axes[1].text(i, v - 4 if v < 0 else v + 2, f"Rs {v:+.1f} L", ha="center", va="top" if v < 0 else "bottom", fontsize=10.5, fontweight="bold")
axes[1].axhline(0, color="#555555", lw=1)
axes[1].set_ylim(min(ex_vals.min() * 1.15, -10), max(ex_vals.max() * 1.35, 10))
axes[1].set_xticks(np.arange(len(ex_vals))); axes[1].set_xticklabels(ex_items, fontsize=9.5)
axes[1].set_ylabel("Rs lakh per year")
axes[1].set_title(f"(d) ZipDrop exclusive at a {100*z['discount_pct_realised']:.0f}% discount loses {money(-zd_ss_yr)}/yr cm1 even with healthy\n"
                  f"batteries (cm1 break-even discount {100*be_cm1:.0f}%); FeastFly, largest by revenue, earns {money(ff_ss_yr)}/yr", fontsize=11.5)
fig.text(0.40, -0.02, "(d) Steady state = non-Kyron energy and wear (all Kyron packs retired 2025-06-15); hatched grey = historical Nov-24..Jun-25 run-rate incl. Kyron wear.\n"
         f"*cm2 adds allocated station rent and maintenance that an exclusive does not avoid, so its {100*be_cm2:.0f}% break-even discount is a full-cost, not a marginal, threshold.",
         fontsize=9, color="#555555", ha="left", va="top")
fig.suptitle("Budget: roll out peak pricing for revenue (not congestion); do not lock in ZipDrop's current terms",
             x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.03)
fig.tight_layout()
save_fig(fig, f"{QID}_07_budget_pricing_partner")

# %% [markdown]
# ## 9. Key numbers for the findings file

# %%
p("\n===== KEY NUMBERS =====")
for out in ("peak_share", "off_share", "rev_per_swap", "swaps_per_rider", "log_swaps", "cm1_per_swap"):
    for spec_ in ("rider-city + month FE", "+ Kyron-share control", "Mumbai-only control", "long pre (Jan-Sep 2024)", "DDD vs Partner-N riders (+Kyron)"):
        rr = DID[(DID["segment"] == GRP_ORDER[0]) & (DID["outcome"] == out) & (DID["spec"] == spec_)]
        for _, r in rr.iterrows():
            extra = f"; city-permutation p={r['perm_p_city']:.3f}" if pd.notna(r.get("perm_p_city", np.nan)) else ""
            p(f"IND {out:16s} [{r['post_window']}, {spec_}]: {r['est']:+.3f} (95% CI {r['lo']:+.3f} to {r['hi']:+.3f}; p={r['p']:.4f}{extra}); "
              f"pre treated mean {r['pre_treated_mean']:.3f}")
for _, r in DID[DID["segment"].str.startswith("All riders")].iterrows():
    perm_txt = f"; perm p={r['perm_p_city']:.3f}" if pd.notna(r.get("perm_p_city", np.nan)) else ""
    p(f"Congestion {r['outcome']:24s} [{r['post_window']}, {r['spec']}]: {r['est']:+.3f} (95% CI {r['lo']:+.3f} to {r['hi']:+.3f}; "
      f"p={r['p']:.3f}{perm_txt}); pre {r['pre_treated_mean']:.3f}")
for g in GRP_ORDER[:2]:
    p(f"{g} timing-shift decomposition (pp, Oct-24..Mar-25): " + ", ".join(
        f"{o} {did_get(g, o)['est']:+.2f} (CI {did_get(g, o)['lo']:+.2f} to {did_get(g, o)['hi']:+.2f}; base {did_get(g, o)['pre_treated_mean']:.1f}%)" for o in SUB_OUTS))
fade = ES[(ES["outcome"] == "peak_share") & ES["month"].isin([M(2025, 5), M(2025, 6)])]["est"]
p(f"Independents' peak-share gap fades to {fade.iloc[0]:+.2f} / {fade.iloc[1]:+.2f} pp in May/Jun-25 (Oct-Mar mean "
  f"{ES[(ES['outcome'] == 'peak_share') & ES['month'].isin(POST_MAIN)]['est'].mean():+.2f} pp)")
for s_ in segs[1:]:
    for out in ("peak_share", "off_share", "rev_per_swap", "log_swaps"):
        r = did_get(s_, out)
        p(f"{s_:32s} {out:16s}: {r['est']:+.3f} (95% CI {r['lo']:+.3f} to {r['hi']:+.3f}); pre treated mean {r['pre_treated_mean']:.3f}")
rel_shift = -pk_m["est"] / pk_m["pre_treated_mean"]
p(f"Relative reduction of independents' peak-hour swaps share: {100*rel_shift:.1f}%")
cm_naive = did_get(GRP_ORDER[0], "cm1_per_swap"); cm_ky = did_get(GRP_ORDER[0], "cm1_per_swap", spec="+ Kyron-share control")
p(f"Naive cm1 DiD / Kyron-adjusted cm1 DiD = {cm_naive['est']:.2f} / {cm_ky['est']:.2f} = {cm_naive['est']/cm_ky['est']:.1f}x")
p(f"Attempts analysed for congestion bands: {band_kpi['attempts'].sum():,}")
rv_y = did_get(GRP_ORDER[1], "rev_per_swap")
p(f"Pilot revenue uplift: independents {100*rv['est']/rv['pre_treated_mean']:.1f}% of pre Rs{rv['pre_treated_mean']:.2f}/swap; "
  f"billable partners {100*rv_y['est']/rv_y['pre_treated_mean']:.1f}% of pre Rs{rv_y['pre_treated_mean']:.2f}/swap")
p(f"Mechanical check: Rs15 x {pk_m['pre_treated_mean'] + pk_m['est']:.1f}% peak - Rs8 x {did_get(GRP_ORDER[0], 'off_share')['pre_treated_mean'] + did_get(GRP_ORDER[0], 'off_share')['est']:.1f}% off-peak = "
  f"Rs{0.15*(pk_m['pre_treated_mean'] + pk_m['est']) - 0.08*(did_get(GRP_ORDER[0], 'off_share')['pre_treated_mean'] + did_get(GRP_ORDER[0], 'off_share')['est']):.2f}/swap; "
  f"at pre-pilot timing it would be Rs{0.15*pk_m['pre_treated_mean'] - 0.08*did_get(GRP_ORDER[0], 'off_share')['pre_treated_mean']:.2f}/swap")
p("done.")
