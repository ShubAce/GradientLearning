# %% [markdown]
# # Q6 - Root cause of new-rider retention
# **Question.** Which factors are most strongly associated with new riders not returning? Which drivers are
# **primary** and which are **secondary**?
#
# **Design (set before looking at results).**
# * **New rider** = signed up on or after 2024-01-01 and made at least 1 attempt. Day 0 = first attempt.
# * **Primary outcome**: *not retained* = **no completed swap in days 31-60** after the first attempt. We use it
#   because riders swap almost daily (median gap 1 day, 99th pct 4 days), so a month with no swap is a clear
#   sign the rider has left. The window also starts after the 14-day feature window. Only riders with at least
#   60 days of follow-up count (first attempt before 2025-05-01). That avoids the censoring trap where recent
#   signups look "lost" only because we have not watched them long enough.
# * **Robustness checks**: activity in days 15-28 (R14), activity in days 91-120 (R90), and survival analysis,
#   where churn = the last active day before 30 or more days without a swap. Each rider's follow-up stops
#   30 days before the data ends (administrative censoring), so a churn event is never counted before it can
#   be observed.
# * **Early experience** is measured only in days 0-13, so it always comes before the outcome window.
#
# Per-rider aggregates are built in polars from projected lazy scans, so the full event table is never loaded.
#
# **Revised after independent verification** (`outputs/findings/q6_verification.md`): km per swap and log
# attempts are centred within vehicle class (fixes an inflated 3W effect), the hero counterfactual now removes the
# modelled failure effect from *observed* retention, the "tolerated first failure" reading is tested against
# exposure (Simpson's paradox), the season-mixed comparison is replaced by a season-matched one, and the share of
# the summer dip explained by early failures is bracketed by four methods.

# %%
import sys
import datetime as dt
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
from common import *  # noqa: E402,F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402

warnings.filterwarnings("ignore")
set_style()
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)
pl.Config.set_tbl_rows(40)
RNG = np.random.default_rng(59500)

DATA_END = dt.datetime(2025, 7, 1)          # exclusive end of the event log (last day 2025-06-30)
EARLY_DAYS = 14                             # early-experience window: days 0-13 after first attempt
OUT_LO, OUT_HI = 30, 60                     # outcome window: day index 30..59 (= days 31-60)
ELIG_CUTOFF = dt.datetime(2025, 5, 1)       # first attempt before this -> >= 60 days follow-up
CHURN_GAP = 30                              # survival: churn = last active day before >= 30 idle days


def wilson(k, n, z=1.96):
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def title_sub(ax, title, sub=None):
    """Insight title plus a grey method/subtitle line that never collides with the title."""
    ax.set_title(title, pad=(26 + 13 * sub.count("\n")) if sub else 8)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9.5, color="#555555", va="bottom", ha="left")


def no_minor_labels(ax):
    ax.xaxis.set_minor_formatter(plt.NullFormatter())


# %% [markdown]
# ## 1. Build the new-rider base, outcomes and survival times

# %%
rid = riders()
base = rid.filter(pl.col("signup_date") >= dt.date(2024, 1, 1))
n_signups = base.height
t0 = (events(["rider_id", "event_ts"]).group_by("rider_id").agg(t0=pl.col("event_ts").min())
      .collect(engine="streaming"))
base = base.join(t0, on="rider_id", how="inner")
B = base.select("rider_id", "t0").lazy()
EV_COLS = ["rider_id", "event_ts", "date", "hour", "station_id", "event_type", "is_completed", "is_stockout",
           "is_abandoned", "is_system_error", "is_cancelled", "is_service_failure", "queue_wait_sec",
           "supplier_out", "soh_out_pct", "km_since_last_swap", "tariff_code", "amount_charged_inr",
           "competitor_active", "city"]
E = events(EV_COLS).join(B, on="rider_id", how="inner").with_columns(
    d=(pl.col("event_ts") - pl.col("t0")).dt.total_seconds() / 86400.0)
early = E.filter(pl.col("d") < EARLY_DAYS).collect(engine="streaming")
act = (E.filter(pl.col("is_completed")).select("rider_id", day=pl.col("d").floor().cast(pl.Int32))
       .unique().collect(engine="streaming").sort(["rider_id", "day"]))

end_day = base.select("rider_id", end_day=(pl.lit(DATA_END) - pl.col("t0")).dt.total_seconds() / 86400.0)
outc = act.group_by("rider_id").agg(
    r17=pl.col("day").is_between(1, 7).any(),
    r14=pl.col("day").is_between(14, 27).any(),
    r3060=pl.col("day").is_between(OUT_LO, OUT_HI - 1).any(),
    r90=pl.col("day").is_between(90, 119).any(),
    any_after30=(pl.col("day") >= 30).any(),
    n_active_days=pl.len())
# survival: first active day followed by >= 30 idle days (gap to next active day, or to data end)
a = act.join(end_day, on="rider_id").with_columns(nxt=pl.col("day").shift(-1).over("rider_id"))
a = a.with_columns(gap=pl.when(pl.col("nxt").is_null()).then(pl.col("end_day") - pl.col("day"))
                   .otherwise(pl.col("nxt") - pl.col("day")))
churn = (a.filter((pl.col("gap") >= CHURN_GAP) & (pl.col("day") <= pl.col("end_day") - CHURN_GAP))
         .group_by("rider_id").agg(churn_day=pl.col("day").min(),
                                   came_back=pl.col("nxt").sort_by("day").first().is_not_null()))
base = (base.join(outc, on="rider_id", how="left").join(churn, on="rider_id", how="left")
        .join(end_day, on="rider_id", how="left")
        .with_columns(pl.col("r17", "r14", "r3060", "r90", "any_after30").fill_null(False),
                      coh=pl.col("t0").dt.truncate("1mo").dt.date())
        .with_columns(surv_E=pl.col("churn_day").is_not_null(),
                      surv_T=pl.coalesce(pl.col("churn_day").cast(pl.Float64), pl.col("end_day") - CHURN_GAP)))
n_with_attempt = base.height
n_elig = base.filter(pl.col("t0") < ELIG_CUTOFF).height
print(f"New riders (signup >= 2024-01-01): {n_signups:,}; with >= 1 attempt: {n_with_attempt:,}; "
      f"with >= 60 days follow-up (first attempt < 2025-05-01): {n_elig:,}")
gaps = a.filter(pl.col("nxt").is_not_null()).select((pl.col("nxt") - pl.col("day")).alias("g"))["g"]
print(f"Gap between consecutive active days: median {gaps.median():.0f} d, p99 {gaps.quantile(0.99):.0f} d")
n_churn_ev = int(base["surv_E"].sum())
n_came_back = int(base["came_back"].fill_null(False).sum())
print(f"Survival: {n_churn_ev:,} riders had a 30-day churn event; {n_came_back:,} "
      f"({pct(n_came_back / n_churn_ev)}) of them came back later (win-back exists, so churn is not always final)")

# %% [markdown]
# ## 2. Early-experience features (days 0-13 only)
# Includes service failures by type, queue waits, the pack received (supplier, SoH, km range), price paid and
# peak-tariff exposure, competitor presence, support tickets, main-station attributes and rider profile.

# %%
QLONG = 600  # a "long wait" = more than 10 minutes in the queue
feat = early.group_by("rider_id").agg(
    e_att=pl.len(), e_comp=pl.col("is_completed").sum(), e_days=pl.col("date").n_unique(),
    e_fail=pl.col("is_service_failure").sum(), e_stockout=pl.col("is_stockout").sum(),
    e_abandon=pl.col("is_abandoned").sum(), e_syserr=pl.col("is_system_error").sum(),
    e_cancel=pl.col("is_cancelled").sum(),
    first_fail=pl.col("is_service_failure").sort_by("event_ts").first(),
    q_mean=pl.col("queue_wait_sec").filter(pl.col("is_completed")).mean() / 60.0,
    q_long_n=(pl.col("queue_wait_sec") > QLONG).sum(),
    kyron_share=(pl.col("supplier_out") == "Kyron").filter(pl.col("is_completed")).mean(),
    soh_out=pl.col("soh_out_pct").filter(pl.col("is_completed")).mean(),
    km_mean=pl.col("km_since_last_swap").mean(),
    price_mean=pl.col("amount_charged_inr").filter(pl.col("is_completed")).mean(),
    peak_share=(pl.col("tariff_code") == "PEAK").filter(pl.col("is_completed")).mean(),
    comp_share=pl.col("competitor_active").cast(pl.Float64).mean(),
    main_station=pl.col("station_id").mode().sort().first(),
    n_stations=pl.col("station_id").n_unique(),
    eve_share=pl.col("hour").is_between(17, 21).mean(),
)
# tickets raised in the first 14 days
tk = tickets().filter(~pl.col("is_test")).join(base.select("rider_id", "t0"), on="rider_id", how="inner")
tk = tk.filter((pl.col("created_ts") >= pl.col("t0")) &
               (pl.col("created_ts") < pl.col("t0") + pl.duration(days=EARLY_DAYS)))
tkf = tk.group_by("rider_id").agg(tk_n=pl.len(), tk_unres=(pl.col("resolution_status") == "unresolved").sum(),
                                  tk_battery=pl.col("category").is_in(["no_battery_available", "low_range"]).sum())
st = stations().select("station_id", main_wave="expansion_wave", main_gen="charger_generation",
                       main_city="city", main_conn="connectivity_tier", main_loc="location_type",
                       comp_since="competitor_within_1_5km_since")
df = (base.join(feat, on="rider_id", how="left").join(tkf, on="rider_id", how="left")
      .join(st, left_on="main_station", right_on="station_id", how="left")
      .with_columns(pl.col("tk_n", "tk_unres", "tk_battery").fill_null(0)))
df = df.to_pandas().sort_values("rider_id").reset_index(drop=True)   # deterministic row order
df["y"] = (~df["r3060"]).astype(int)                       # 1 = NOT retained (primary outcome)
df["elig"] = df["t0"] < ELIG_CUTOFF
df["coh"] = pd.to_datetime(df["coh"])
df["coh_s"] = df["coh"].dt.strftime("%Y-%m")
df["coh_q"] = df["coh"].dt.to_period("Q").astype(str)
df["coh_idx"] = (df["coh"].dt.year - 2024) * 12 + df["coh"].dt.month - 1
df["fail_grp"] = pd.cut(df["e_fail"].fillna(0), [-1, 0, 1, 2, 999], labels=["0", "1", "2", "3+"]).astype(str)
df["fail2p"] = (df["e_fail"] >= 2).astype(int)
# stockouts + abandoned queues only (the failure types that drive exits; excludes kiosk errors)
df["sa2p"] = ((df["e_stockout"] + df["e_abandon"]) >= 2).astype(int)
# exposure-free dose: share of early attempts that failed
df["fail_rate"] = df["e_fail"] / df["e_att"]
df["frate_bin"] = pd.cut(df["fail_rate"], [-0.01, 0, 0.05, 0.10, 0.20, 1.01],
                         labels=["0", "0-5%", "5-10%", "10-20%", ">20%"]).astype(str)
df["first_fail"] = df["first_fail"].fillna(False).astype(int)
df["comp_any"] = (df["comp_share"].fillna(0) > 0).astype(int)
df["tk_any"] = (df["tk_n"] > 0).astype(int)
df["tk_unres_any"] = (df["tk_unres"] > 0).astype(int)
df["kyron_share"] = df["kyron_share"].fillna(0)
df["peak_share"] = df["peak_share"].fillna(0)
n_km_missing = int(df["km_mean"].isna().sum())
df["km_mean"] = df["km_mean"].fillna(df["km_mean"].median())
df["q_mean"] = df["q_mean"].fillna(df["q_mean"].median())
df["soh_out"] = df["soh_out"].fillna(df["soh_out"].median())
df["price_mean"] = df["price_mean"].fillna(df["price_mean"].median())
df["q_long_any"] = (df["q_long_n"] > 0).astype(int)
df["w3"] = (df["vehicle_class"] == "3W").astype(int)
# km per swap mostly reflects pack size (3W packs are 4.8 kWh vs 2.1 kWh for 2W), so raw km would make 3W riders
# look like "high-range" riders and the 3W dummy would then be read at a 2W-typical km. Centring km within vehicle
# class is an exact reparameterisation whenever w3 is in the model: every other coefficient is unchanged and the
# 3W term becomes the total 3W effect at the class-typical km per swap. Same for log early attempts (3W riders
# swap less often because their packs are bigger).
km_cls = df.groupby("vehicle_class")["km_mean"].transform("mean")
df["km10c"] = (df["km_mean"] - km_cls) / 10
df["log_att"] = np.log(df["e_att"])
df["log_att_c"] = df["log_att"] - df.groupby("vehicle_class")["log_att"].transform("mean")
df["segment"] = np.where(df["partner_id"].isna(), np.where(df["plan_type"] == "prepaid_pack", "Indep-prepaid", "Indep-PAYG"),
                         np.where(df["partner_id"] == ZIPDROP_ID, "ZipDrop", "Other partner"))
df["indep"] = df["partner_id"].isna().astype(int)
df["partner"] = df["partner_id"].fillna("Independent")
df["age_band"] = df["age_band"].fillna("unknown")
df["declared_shift"] = df["declared_shift"].fillna("unknown")
df["kyc_no"] = (~df["kyc_verified"].astype(bool)).astype(int)
df["pilot_city"] = df["main_city"].isin(PILOT_CITIES).astype(int)
for c in ["main_wave", "main_gen", "main_conn", "main_loc", "main_city"]:
    df[c] = df[c].fillna("unknown")
D = df[df["elig"]].copy()
overall_nr = D["y"].mean()
print(f"\nEligible riders: {len(D):,}; NOT retained (no swap in days 31-60): {int(D['y'].sum()):,} "
      f"= {pct(overall_nr)}; retained {pct(1 - overall_nr)}")
print(f"Early window (days 0-13): median {D['e_att'].median():.0f} attempts, mean service failures/rider "
      f"{D['e_fail'].mean():.2f}; riders with >=2 failures: {int(D['fail2p'].sum()):,} ({pct(D['fail2p'].mean())})")
print(f"Mean km per swap in first 14 days: 2W {D.loc[D['w3'] == 0, 'km_mean'].mean():.1f} km vs 3W "
      f"{D.loc[D['w3'] == 1, 'km_mean'].mean():.1f} km (pack size); riders with no valid km reading (median-filled): {n_km_missing}")

# %% [markdown]
# ## 3. Did new-rider retention fall? Cohort trend, same-season comparison and the censoring trap

# %%
coh = (df.groupby("coh").agg(n=("y", "size"), n_elig=("elig", "sum"),
                             naive=("any_after30", "mean"),
                             fail_per_att=("e_fail", "sum"), att=("e_att", "sum")).reset_index())
coh["fail_per_att"] = coh["fail_per_att"] / coh["att"]
ce = D.groupby("coh").agg(n=("y", "size"), k_ret=("r3060", "sum"), r14=("r14", "mean"),
                          fail_mean=("e_fail", "mean"), fail2p=("fail2p", "mean")).reset_index()
ce["ret"] = ce["k_ret"] / ce["n"]
ce["lo"], ce["hi"] = wilson(ce["k_ret"], ce["n"])
coh = coh.merge(ce[["coh", "k_ret", "ret", "lo", "hi", "r14", "fail_mean", "fail2p"]], on="coh", how="left")
# R90 among cohorts with >= 120 days of follow-up
r90 = df[df["t0"] < DATA_END - dt.timedelta(days=120)].groupby("coh")["r90"].mean().rename("r90")
coh = coh.merge(r90, on="coh", how="left")
print(coh[["coh", "n", "n_elig", "ret", "lo", "hi", "r14", "r90", "naive", "fail_per_att", "fail_mean"]]
      .round(3).to_string(index=False))


def two_prop(mask_a, mask_b, col="r3060", frame=D):
    a_, b_ = frame.loc[mask_a, col], frame.loc[mask_b, col]
    ka, na, kb, nb = a_.sum(), len(a_), b_.sum(), len(b_)
    p = (ka + kb) / (na + nb)
    z = (kb / nb - ka / na) / np.sqrt(p * (1 - p) * (1 / na + 1 / nb))
    return ka / na, na, kb / nb, nb, 2 * (1 - stats.norm.cdf(abs(z)))


per = D["coh"]
cmp_rows = []
for lab, ma, mb in [
        ("Same season: Jan-Mar 2024 vs Jan-Mar 2025", per.between("2024-01-01", "2024-03-01"), per.between("2025-01-01", "2025-03-01")),
        ("Same season: Apr 2024 vs Apr 2025", per == "2024-04-01", per == "2025-04-01"),
        ("Naive: Jan-Mar 2024 vs latest (Apr 2025)", per.between("2024-01-01", "2024-03-01"), per == "2025-04-01"),
        ("Summer dip 2024: Jan-Mar vs Apr-Jun 2024", per.between("2024-01-01", "2024-03-01"), per.between("2024-04-01", "2024-06-01")),
        ("Same season: Jan-Apr 2024 vs Jan-Apr 2025", per.between("2024-01-01", "2024-04-01"), per.between("2025-01-01", "2025-04-01")),
        ("Season-MIXED, not a valid trend test: H1-2024 vs Oct 2024-Apr 2025", per.between("2024-01-01", "2024-06-01"), per.between("2024-10-01", "2025-04-01"))]:
    pa, na, pb, nb, p = two_prop(ma, mb)
    cmp_rows.append(dict(comparison=lab, ret_a=pa, n_a=na, ret_b=pb, n_b=nb, diff_pp=100 * (pb - pa), p_value=p))
cmp_tab = pd.DataFrame(cmp_rows)
print("\nRetention (D31-60) comparisons:")
print(cmp_tab.round(4).to_string(index=False))
summer_m = per.dt.month.isin([4, 5, 6])
print(f"Why the season-mixed row is invalid: summer-entry (Apr-Jun) share of riders is "
      f"{pct(summer_m[per.between('2024-01-01', '2024-06-01')].mean())} in H1-2024 vs "
      f"{pct(summer_m[per.between('2024-10-01', '2025-04-01')].mean())} in Oct 2024-Apr 2025")

# May/Jun 2025 cohorts cannot be scored on D31-60 yet. Shorter outcomes give a same-season year-on-year check
# for them: R14 = completed swap in days 15-28 (needs 28 days of follow-up), D1-7 = completed swap in days 2-8.
ss_rows = []
for lab, col, cutoff in [("R14 (swap in days 15-28)", "r14", DATA_END - dt.timedelta(days=28)),
                         ("D1-7 (swap in days 2-8)", "r17", DATA_END - dt.timedelta(days=8))]:
    Dx_ = df[df["t0"] < cutoff]
    for mo in range(1, 7):
        a_ = Dx_[(Dx_["coh"].dt.year == 2024) & (Dx_["coh"].dt.month == mo)]
        b_ = Dx_[(Dx_["coh"].dt.year == 2025) & (Dx_["coh"].dt.month == mo)]
        if len(b_) < 50:
            continue
        ka, kb = a_[col].sum(), b_[col].sum()
        na, nb = len(a_), len(b_)
        pp = (ka + kb) / (na + nb)
        z = (kb / nb - ka / na) / np.sqrt(pp * (1 - pp) * (1 / na + 1 / nb))
        ss_rows.append(dict(outcome=lab, month=mo, ret_2024=ka / na, n_2024=na, ret_2025=kb / nb, n_2025=nb,
                            diff_pp=100 * (kb / nb - ka / na), p_value=2 * (1 - stats.norm.cdf(abs(z)))))
ss_tab = pd.DataFrame(ss_rows)
print("Same-calendar-month cohorts, 2024 vs 2025, on shorter outcomes (covers the May/Jun cohorts):")
print(ss_tab.round(4).to_string(index=False))
ss_tab.to_csv(TAB / "q6_same_season_short_outcomes.csv", index=False)
mix = D.groupby("coh_q").agg(n=("y", "size"), share_3w=("w3", "mean"), share_indep=("indep", "mean"),
                             share_zipdrop=("partner", lambda s: (s == ZIPDROP_ID).mean())).reset_index()
print("Cohort mix by quarter (eligible riders) - composition is stable, so mix shift cannot explain dips:")
print(mix.round(3).to_string(index=False))
r_corr = stats.pearsonr(ce["fail_mean"], ce["ret"])
print(f"Across {len(ce)} monthly cohorts, corr(mean early failures, D31-60 retention) = {r_corr[0]:.2f} (p={r_corr[1]:.4f})")
# ecological and mostly seasonal: repeat after removing quarter-of-year means from both series
ce["qoy"] = (ce["coh"].dt.month - 1) // 3
r_corr_w = stats.pearsonr(ce["fail_mean"] - ce.groupby("qoy")["fail_mean"].transform("mean"),
                          ce["ret"] - ce.groupby("qoy")["ret"].transform("mean"))
print(f"Within quarter-of-year (season demeaned) correlation = {r_corr_w[0]:.2f} (p={r_corr_w[1]:.3f})")
naive_last = coh.set_index("coh")["naive"]
print(f"Naive 'any swap after day 30' metric (ignores follow-up): Jan-24 {pct(naive_last.iloc[0])}, "
      f"May-25 {pct(naive_last.loc['2025-05-01'])}, Jun-25 {pct(naive_last.loc['2025-06-01'])} -> a censoring artefact")
coh.to_csv(TAB / "q6_cohort_retention.csv", index=False)
cmp_tab.to_csv(TAB / "q6_retention_comparisons.csv", index=False)

# %% [markdown]
# ### Survival by quarterly cohort (Kaplan-Meier, churn = 30+ idle days, administrative censoring)

# %%
from lifelines import KaplanMeierFitter, CoxPHFitter  # noqa: E402
from lifelines.statistics import multivariate_logrank_test  # noqa: E402

S = df[df["surv_T"] > 0].copy()
km_rows = []
fig, ax = plt.subplots(figsize=(10, 5.6))
qcols = dict(zip(sorted(S["coh_q"].unique()), SERIES))
for q, g in S.groupby("coh_q"):
    kmf = KaplanMeierFitter().fit(g["surv_T"], g["surv_E"], label=f"{q} (n={len(g):,})")
    tmax = g["surv_T"].max()
    kmf.plot_survival_function(ax=ax, ci_show=False, color=qcols[q], lw=2.2 if q in ("2024Q2",) else 1.6)
    row = {"cohort": q, "n": len(g)}
    for h in (30, 60, 90, 180, 270):
        row[f"S{h}"] = float(kmf.survival_function_at_times(h).iloc[0]) if tmax >= h else np.nan
    km_rows.append(row)
km_tab = pd.DataFrame(km_rows)
print(km_tab.round(3).to_string(index=False))
km_tab.to_csv(TAB / "q6_cohort_km.csv", index=False)
from lifelines.statistics import logrank_test  # noqa: E402
for qa, qb in [("2024Q1", "2025Q1"), ("2024Q1", "2024Q2"), ("2024Q2", "2025Q2")]:
    ga, gb = S[S["coh_q"] == qa], S[S["coh_q"] == qb]
    lt = logrank_test(ga["surv_T"], gb["surv_T"], ga["surv_E"], gb["surv_E"])
    print(f"Log-rank {qa} vs {qb}: chi2={lt.test_statistic:.2f}, p={lt.p_value:.4f}")
ax.set_xlim(0, 365)
ax.set_ylim(0.2, 1.0)
ax.set_xlabel("Days since first swap attempt")
ax.set_ylabel("Share of new riders not yet churned (30+ idle days)")
s2q2 = km_tab.set_index("cohort")
title_sub(ax, "Survival curves of later cohorts overlap early-2024 ones; summer-entry cohorts (Q2-24, Q2-25) churn faster",
          f"Kaplan-Meier by quarter of first attempt (churn = 30+ idle days). 90-day survival: 2024Q1 {pct(s2q2.loc['2024Q1','S90'])}, "
          f"2024Q2 {pct(s2q2.loc['2024Q2','S90'])}, 2024Q4 {pct(s2q2.loc['2024Q4','S90'])}, 2025Q1 {pct(s2q2.loc['2025Q1','S90'])}")
ax.legend(loc="lower left", ncol=2)
save_fig(fig, "q6_02_cohort_survival_curves")

# %% [markdown]
# ## 4. Univariate lift tables (eligible riders, primary outcome)

# %%
D["q_mean_bin"] = pd.qcut(D["q_mean"], 5, labels=["Q1 shortest", "Q2", "Q3", "Q4", "Q5 longest"]).astype(str)
D["kyron_bin"] = pd.cut(D["kyron_share"], [-0.01, 0, 0.25, 0.5, 1.01], labels=["0%", "1-25%", "26-50%", ">50%"]).astype(str)
D["km_bin"] = pd.qcut(D["km_mean"], 5, labels=["Q1 lowest km", "Q2", "Q3", "Q4", "Q5 highest km"]).astype(str)
D["peak_bin"] = np.where(D["peak_share"] == 0, "no PEAK swaps", np.where(D["peak_share"] < 0.4, "1-39% PEAK", "40%+ PEAK"))
D["comp_bin"] = np.where(D["comp_any"] == 1, "competitor active", "no competitor")
D["att_bin"] = pd.cut(D["e_att"], [0, 5, 9, 14, 999], labels=["1-5", "6-9", "10-14", "15+"]).astype(str)
lift_feats = ["fail_grp", "first_fail", "e_stockout", "e_abandon", "e_syserr", "e_cancel", "q_long_any", "q_mean_bin",
              "kyron_bin", "km_bin", "peak_bin", "comp_bin", "tk_any", "tk_unres_any", "segment", "partner",
              "vehicle_class", "vehicle_model", "signup_channel", "kyc_verified", "age_band", "declared_shift",
              "main_city", "main_wave", "main_gen", "main_conn", "main_loc", "att_bin"]
lrows = []
for f_ in lift_feats:
    col = D[f_].clip(upper=2) if f_ in ("e_stockout", "e_abandon", "e_syserr", "e_cancel") else D[f_]
    g = D.groupby(col)["y"].agg(["sum", "size"]).reset_index()
    for _, rr in g.iterrows():
        lo, hi = wilson(rr["sum"], rr["size"])
        lrows.append(dict(feature=f_, level=str(rr.iloc[0]) + ("+" if f_.startswith("e_") and rr.iloc[0] == 2 else ""),
                          n=int(rr["size"]), not_retained=rr["sum"] / rr["size"], ci_lo=float(lo), ci_hi=float(hi),
                          lift=(rr["sum"] / rr["size"]) / overall_nr))
lift = pd.DataFrame(lrows)
lift.to_csv(TAB / "q6_univariate_lift.csv", index=False)
show = lift[lift["n"] >= 100].copy()
show["spread"] = show.groupby("feature")["not_retained"].transform(lambda s: s.max() - s.min())
print("Univariate non-retention spread (max - min across levels with n>=100), top features:")
print(show.groupby("feature")["spread"].first().sort_values(ascending=False).round(3).head(16).to_string())
print(lift[lift["feature"].isin(["fail_grp", "e_stockout", "e_abandon", "e_syserr", "comp_bin", "kyron_bin", "segment", "vehicle_class", "att_bin"])]
      .round(3).to_string(index=False))

# declared shift vs actual activity hours
shift_tab = (early.join(base.select("rider_id", "declared_shift"), on="rider_id")
             .with_columns(band=pl.when(pl.col("hour").is_between(6, 11)).then(pl.lit("06-11"))
                           .when(pl.col("hour").is_between(12, 16)).then(pl.lit("12-16"))
                           .when(pl.col("hour").is_between(17, 21)).then(pl.lit("17-21")).otherwise(pl.lit("22-05")))
             .group_by(["declared_shift", "band"]).len()
             .with_columns(share=pl.col("len") / pl.col("len").sum().over("declared_shift"))
             .pivot(on="band", index="declared_shift", values="share").sort("declared_shift").to_pandas())
shift_tab = shift_tab[["declared_shift", "06-11", "12-16", "17-21", "22-05"]]
print("\nShare of early attempts by hour band, by DECLARED shift (identical rows => declaration is uninformative):")
print(shift_tab.round(3).to_string(index=False))
shift_tab.to_csv(TAB / "q6_declared_shift_vs_activity.csv", index=False)

# vehicle model is a proxy for fleet affiliation / vehicle class
vm = pd.crosstab(D["vehicle_model"], D["segment"], normalize="index").round(3)
vm["vehicle_class"] = D.groupby("vehicle_model")["vehicle_class"].agg(lambda s: s.mode()[0])
vm["not_retained"] = D.groupby("vehicle_model")["y"].mean().round(3)
vm["n"] = D.groupby("vehicle_model").size()
chi2, pchi, _, _ = stats.chi2_contingency(pd.crosstab(D["vehicle_model"], D["partner"]))
cramer_v = np.sqrt(chi2 / (len(D) * (min(D["vehicle_model"].nunique(), D["partner"].nunique()) - 1)))
print("\nVehicle model vs segment (row shares):")
print(vm.to_string())
print(f"Cramer's V(vehicle_model, partner) = {cramer_v:.2f}; Zipp E2 riders that are ZipDrop: "
      f"{pct((D.loc[D['vehicle_model']=='Zipp E2','partner']==ZIPDROP_ID).mean())}; "
      f"ZipDrop riders on Zipp E2: {pct((D.loc[D['partner']==ZIPDROP_ID,'vehicle_model']=='Zipp E2').mean())}")
vm.to_csv(TAB / "q6_vehicle_model_confounding.csv")

# %% [markdown]
# ## 5. Multivariable logistic regression (odds of NOT being retained)
# Fixed effects for cohort month and city absorb seasonality, network maturity and city mix. Standard errors are
# clustered by main station. Model A puts early failures in dose groups. Model B splits failures by type.
# Activity volume (early attempts) is left out because it is partly the outcome itself: early churners make few
# attempts. A sensitivity check adds it back.

# %%
D["kyron25"] = D["kyron_share"] / 0.25
D["peak25"] = D["peak_share"] / 0.25
D["km10"] = D["km_mean"] / 10          # raw km (old spec, kept only to document the 3W change)
D["soh10"] = D["soh_out"] / 10
D["e_stockout_c"] = D["e_stockout"].clip(upper=4)
D["e_abandon_c"] = D["e_abandon"].clip(upper=4)
D["e_syserr_c"] = D["e_syserr"].clip(upper=2)
D["e_cancel_c"] = D["e_cancel"].clip(upper=2)
D["station_code"] = D["main_station"].astype("category").cat.codes
COMMON = ("first_fail + q_mean + kyron25 + km10c + peak25 + comp_any + tk_any + tk_unres_any + w3 "
          "+ C(segment, Treatment('Other partner')) + C(signup_channel, Treatment('partner_onboarding')) + kyc_no "
          "+ C(age_band, Treatment('25-34')) + C(main_wave, Treatment('Launch')) + C(main_gen, Treatment('Gen2')) "
          "+ C(main_conn, Treatment('good'))")
FE = " + C(coh_s) + C(main_city)"
fA = "y ~ C(fail_grp, Treatment('0')) + " + COMMON + FE
fB = "y ~ e_stockout_c + e_abandon_c + e_syserr_c + e_cancel_c + " + COMMON + FE


def fit_logit(formula, data):
    return smf.logit(formula, data=data).fit(disp=0, maxiter=200, cov_type="cluster",
                                             cov_kwds={"groups": data["station_code"]})


mA = fit_logit(fA, D)
mB = fit_logit(fB, D)
mA_att = fit_logit(fA + " + log_att_c", D)


def or_table(m, model_name):
    ci = m.conf_int()
    t = pd.DataFrame({"term": m.params.index, "coef": m.params.values, "OR": np.exp(m.params.values),
                      "lo": np.exp(ci[0].values), "hi": np.exp(ci[1].values), "p": m.pvalues.values})
    t["model"] = model_name
    return t[~t["term"].str.startswith(("C(coh_s)", "C(main_city)", "Intercept"))]


orA, orB, orA2 = or_table(mA, "A_dose"), or_table(mB, "B_types"), or_table(mA_att, "A_plus_log_attempts")
ors = pd.concat([orA, orB, orA2])
ors.to_csv(TAB / "q6_logit_odds_ratios.csv", index=False)
print(f"Model A: n={int(mA.nobs):,}, pseudo-R2={mA.prsquared:.3f}; Model B pseudo-R2={mB.prsquared:.3f}")
print(orA.round(3).to_string(index=False))
print(orB[orB["term"].str.startswith("e_")].round(3).to_string(index=False))
print("Sensitivity (adds log early attempts, centred within vehicle class):")
print(orA2[orA2["term"].str.contains("fail_grp|log_att|comp_any|w3|first_fail|segment")].round(3).to_string(index=False))


def or_txt(m, term):
    ci_ = m.conf_int().loc[term]
    return f"OR {np.exp(m.params[term]):.2f} [{np.exp(ci_[0]):.2f}-{np.exp(ci_[1]):.2f}], p={m.pvalues[term]:.3g}"


# Headline exposure, stated consistently: 2+ vs 0-1 service failures (and 2+ vs 0-1 stockouts/abandoned queues)
m2p = fit_logit("y ~ fail2p + " + COMMON + FE, D)
msa = fit_logit("y ~ sa2p + " + COMMON + FE, D)
print(f"\nHEADLINE EXPOSURE. 2+ service failures in days 0-13: n={int(D['fail2p'].sum()):,}, not retained "
      f"{pct(D.loc[D['fail2p'] == 1, 'y'].mean())} vs {pct(D.loc[D['fail2p'] == 0, 'y'].mean())} for 0-1; "
      f"adjusted (2+ vs 0-1, Model A controls) {or_txt(m2p, 'fail2p')}")
print(f"2+ stockouts or abandoned queues only: n={int(D['sa2p'].sum()):,}, not retained "
      f"{pct(D.loc[D['sa2p'] == 1, 'y'].mean())} vs {pct(D.loc[D['sa2p'] == 0, 'y'].mean())}; adjusted {or_txt(msa, 'sa2p')}")

# 3W: the old spec used raw km per swap (km10), which read the 3W dummy at a 2W-typical km
mA_rawkm = fit_logit(fA.replace("km10c", "km10"), D)
mA_nokm = fit_logit(fA.replace("km10c + ", ""), D)
T2 = "C(fail_grp, Treatment('0'))[T.2]"
T3 = "C(fail_grp, Treatment('0'))[T.3+]"
print(f"3W raw non-retention {pct(D.loc[D['w3'] == 1, 'y'].mean())} (n={int(D['w3'].sum()):,}) vs 2W "
      f"{pct(D.loc[D['w3'] == 0, 'y'].mean())}. 3W in Model A: km centred within class {or_txt(mA, 'w3')}; "
      f"km dropped {or_txt(mA_nokm, 'w3')}; old spec with raw km {or_txt(mA_rawkm, 'w3')}; failure ORs identical "
      f"({np.exp(mA.params[T2]):.3f} vs {np.exp(mA_rawkm.params[T2]):.3f})")

# Is there a "tolerated first failure"? Pooled rates show 0 > 1 failures; check within early-activity bands
D["att_band"] = pd.cut(D["e_att"], [0, 5, 9, 14, 19, 999], labels=["1-5", "6-9", "10-14", "15-19", "20+"]).astype(str)
band_rate = pd.crosstab(D["att_band"], D["fail_grp"], values=D["y"], aggfunc="mean")
band_n = pd.crosstab(D["att_band"], D["fail_grp"])
print("\nNon-retention by early failures WITHIN early-activity bands (rows = attempts in days 0-13):")
print(band_rate.round(3).to_string())
print(band_n.to_string())
low = D[D["att_band"] == "1-5"]
print(f"Simpson's paradox: {len(low):,} riders made only 1-5 early attempts ({pct(low['y'].mean())} not retained) and "
      f"{int((low['fail_grp'] == '0').sum()):,} of them had 0 failures, which inflates the pooled 0-failure rate")
for g_ in ["1", "2", "3+"]:
    term_ = "C(fail_grp, Treatment('0'))[T." + g_ + "]"
    print(f"  {g_} failure(s) vs 0: activity-adjusted {or_txt(mA_att, term_)} | Model A {or_txt(mA, term_)}")
print(f"  First attempt failed: activity-adjusted {or_txt(mA_att, 'first_fail')} | Model A {or_txt(mA, 'first_fail')}")
print(f"  3W, activity-adjusted (km and attempts centred within class): {or_txt(mA_att, 'w3')}")
# exposure-free dose: failure RATE in days 0-13, with and without the activity control
fR = "y ~ C(frate_bin, Treatment('0')) + " + COMMON + FE
mR = fit_logit(fR + " + log_att_c", D)
mR0 = fit_logit(fR, D)
frate_tab = D.groupby("frate_bin")["y"].agg(["mean", "size"]).reindex(["0", "0-5%", "5-10%", "10-20%", ">20%"])
print("Non-retention by early failure RATE (share of attempts failed):")
print(frate_tab.round(3).to_string())
thr_rows = []
for b_ in ["0-5%", "5-10%", "10-20%", ">20%"]:
    term = f"C(frate_bin, Treatment('0'))[T.{b_}]"
    print(f"  failure rate {b_} vs 0: with log attempts {or_txt(mR, term)} | without {or_txt(mR0, term)}")
    for lab_, m_ in [("failure-rate model + log attempts", mR), ("failure-rate model", mR0)]:
        ci_ = m_.conf_int().loc[term]
        thr_rows.append(dict(model=lab_, level=b_, OR=np.exp(m_.params[term]), lo=np.exp(ci_[0]), hi=np.exp(ci_[1]), p=m_.pvalues[term]))
for g_ in ["1", "2", "3+"]:
    term = f"C(fail_grp, Treatment('0'))[T.{g_}]"
    for lab_, m_ in [("count model + log attempts", mA_att), ("count model (Model A)", mA)]:
        ci_ = m_.conf_int().loc[term]
        thr_rows.append(dict(model=lab_, level=f"{g_} failures", OR=np.exp(m_.params[term]), lo=np.exp(ci_[0]), hi=np.exp(ci_[1]), p=m_.pvalues[term]))
band_long = band_rate.stack().rename("not_retained").reset_index().merge(
    band_n.stack().rename("n").reset_index(), on=["att_band", "fail_grp"])
pd.DataFrame(thr_rows).to_csv(TAB / "q6_dose_gradient_models.csv", index=False)
band_long.to_csv(TAB / "q6_dose_within_activity_band.csv", index=False)

# Battery channel as a natural test: packs got worse for later cohorts (Kyron, degradation) but retention did not
mS = fit_logit(fA + " + soh10", D)
bat_coh = D.groupby("coh_s")[["soh_out", "km_mean", "kyron_share"]].mean()
print(f"\nBattery channel: SoH of packs received {or_txt(mS, 'soh10')} per +10 SoH points (Model A + SoH). "
      f"Mean SoH received Jan-24 cohort {bat_coh['soh_out'].iloc[0]:.1f} -> Apr-25 {bat_coh['soh_out'].iloc[-1]:.1f}; "
      f"km per swap {bat_coh['km_mean'].iloc[0]:.1f} -> {bat_coh['km_mean'].iloc[-1]:.1f}; Kyron share "
      f"{pct(bat_coh['kyron_share'].iloc[0])} -> {pct(bat_coh['kyron_share'].iloc[-1])}")

# robustness: full partner dummies and vehicle model (proxy) on top of segment controls
mP = fit_logit("y ~ C(fail_grp, Treatment('0')) + " + COMMON.replace("C(segment, Treatment('Other partner'))", "C(partner, Treatment('FP-01'))") + FE, D)
mV = fit_logit(fA + " + C(vehicle_model, Treatment('Zipp E2'))", D)
lr_vm = 2 * (mV.llf - mA.llf)
df_vm = mV.df_model - mA.df_model
p_vm = 1 - stats.chi2.cdf(lr_vm, df_vm)
print(f"Adding vehicle_model to Model A: LR chi2={lr_vm:.1f}, df={df_vm:.0f}, p={p_vm:.3f} (no independent effect if p>0.05)")
orP = or_table(mP, "partners")
print(orP[orP["term"].str.contains("partner")].round(3).to_string(index=False))
pd.concat([orP.assign(model="partners"), or_table(mV, "vehicle_model")]).to_csv(TAB / "q6_logit_robustness.csv", index=False)

# robustness on alternative outcomes (same Model A spec)
alt_rows = []
for lab, ycol, mask in [("R14 (no swap days 15-28)", "r14", df["t0"] < DATA_END - dt.timedelta(days=28)),
                        ("R90 (no swap days 91-120)", "r90", df["t0"] < DATA_END - dt.timedelta(days=120))]:
    Dx = df[mask].copy()
    Dx["y"] = (~Dx[ycol]).astype(int)
    Dx["kyron25"], Dx["peak25"] = Dx["kyron_share"] / 0.25, Dx["peak_share"] / 0.25
    Dx["station_code"] = Dx["main_station"].astype("category").cat.codes
    mx = fit_logit(fA, Dx)
    t = or_table(mx, lab)
    t = t[t["term"].str.contains("fail_grp|comp_any|w3|kyron25|first_fail")]
    t["n"] = int(mx.nobs)
    t["base_rate"] = Dx["y"].mean()
    alt_rows.append(t)
alt = pd.concat(alt_rows)
print(alt.round(3).to_string(index=False))
alt.to_csv(TAB / "q6_logit_alt_outcomes.csv", index=False)

# %% [markdown]
# ## 6. Trigger analysis: churn probability right after each attempt (all new-rider attempts)
# For each attempt, did the rider leave (no further attempt for 30 or more days)? Comparing what *just
# happened* to the rider tests whether failures **trigger** exits. The grouped binomial GLM is a **pooled**
# design (no rider fixed effect): it adjusts for tenure, calendar month, vehicle class, segment, competitor and
# pack supplier. A truly **within-rider** check follows: Mantel-Haenszel ORs with each rider (and each
# rider x month) as its own stratum, so every rider is compared only with themselves.

# %%
AT = (events(["rider_id", "event_ts", "event_type", "is_service_failure", "supplier_out", "competitor_active",
              "vehicle_class", "is_independent", "month"])
      .join(B, on="rider_id", how="inner").sort(["rider_id", "event_ts"])
      .with_columns(nxt=pl.col("event_ts").shift(-1).over("rider_id"),
                    f7=pl.col("is_service_failure").cast(pl.Int32).rolling_sum_by("event_ts", window_size="7d").over("rider_id"))
      .filter(pl.col("event_ts") < DATA_END - dt.timedelta(days=CHURN_GAP))
      .with_columns(churn=((pl.coalesce("nxt", pl.lit(DATA_END)) - pl.col("event_ts")).dt.total_seconds() / 86400 >= CHURN_GAP),
                    prior_f7=(pl.col("f7") - pl.col("is_service_failure").cast(pl.Int32)).clip(0, 3),
                    tenure=((pl.col("event_ts") - pl.col("t0")).dt.total_seconds() / 86400))
      .with_columns(ten_band=pl.when(pl.col("tenure") < 14).then(pl.lit("a_0-13d")).when(pl.col("tenure") < 60).then(pl.lit("b_14-59d"))
                    .when(pl.col("tenure") < 180).then(pl.lit("c_60-179d")).otherwise(pl.lit("d_180d+")),
                    kyron=(pl.col("supplier_out") == "Kyron").fill_null(False).cast(pl.Int8))
      .group_by(["event_type", "prior_f7", "ten_band", "month", "vehicle_class", "is_independent", "competitor_active", "kyron"])
      .agg(n=pl.len(), k=pl.col("churn").sum())
      .collect(engine="streaming").to_pandas())
AT["prop"] = AT["k"] / AT["n"]
AT["month_s"] = AT["month"].astype(str)
AT["prior_f7"] = AT["prior_f7"].astype(str).replace({"3": "3+"})
AT["etype"] = AT["event_type"].map({"swap_completed": "completed", "failed_no_charged_battery": "stockout",
                                    "abandoned_queue": "abandoned", "failed_system_error": "system_error",
                                    "cancelled_by_rider": "cancelled"})
raw_at = AT.groupby("etype")[["k", "n"]].sum()
raw_at["churn_rate"] = raw_at["k"] / raw_at["n"]
raw_pf = AT.groupby("prior_f7")[["k", "n"]].sum()
raw_pf["churn_rate"] = raw_pf["k"] / raw_pf["n"]
print(f"Attempts analysed: {int(AT['n'].sum()):,}; exits: {int(AT['k'].sum()):,}")
print(raw_at.round(5).to_string())
print(raw_pf.round(5).to_string())
mH = smf.glm("prop ~ C(etype, Treatment('completed')) + C(prior_f7, Treatment('0')) + C(ten_band) + C(month_s) "
             "+ C(vehicle_class) + C(is_independent) + C(competitor_active) + kyron",
             data=AT, family=sm.families.Binomial(), var_weights=AT["n"]).fit()
hz = or_table(mH, "attempt_exit_glm")
print(hz[~hz["term"].str.contains("month_s")].round(4).to_string(index=False))
hz.to_csv(TAB / "q6_attempt_exit_glm.csv", index=False)
raw_at.to_csv(TAB / "q6_attempt_exit_raw_by_outcome.csv")

# Within-rider Mantel-Haenszel OR: exit right after a stockout/abandoned queue vs after any other attempt
A2 = (events(["rider_id", "event_ts", "event_type", "month"]).join(B.select("rider_id"), on="rider_id", how="semi")
      .sort(["rider_id", "event_ts"])
      .with_columns(nxt=pl.col("event_ts").shift(-1).over("rider_id"))
      .filter(pl.col("event_ts") < DATA_END - dt.timedelta(days=CHURN_GAP))
      .with_columns(Y=((pl.coalesce("nxt", pl.lit(DATA_END)) - pl.col("event_ts")).dt.total_seconds() / 86400 >= CHURN_GAP),
                    X=pl.col("event_type").is_in(["failed_no_charged_battery", "abandoned_queue"])))
mh_rows = []
for lab, keys in [("rider", ["rider_id"]), ("rider x month", ["rider_id", "month"])]:
    s_ = (A2.group_by(keys).agg(a=(pl.col("X") & pl.col("Y")).sum(), b=(~pl.col("X") & pl.col("Y")).sum(),
                                c=(pl.col("X") & ~pl.col("Y")).sum(), d=(~pl.col("X") & ~pl.col("Y")).sum(), n=pl.len())
          .filter((pl.col("a") + pl.col("b")) > 0).select("a", "b", "c", "d", "n")
          .collect(engine="streaming").to_pandas().astype(float))
    Ri, Si = s_["a"] * s_["d"] / s_["n"], s_["b"] * s_["c"] / s_["n"]
    Pi, Qi = (s_["a"] + s_["d"]) / s_["n"], (s_["b"] + s_["c"]) / s_["n"]
    Rs, Ss = Ri.sum(), Si.sum()
    var_ = (Pi * Ri).sum() / (2 * Rs ** 2) + (Pi * Si + Qi * Ri).sum() / (2 * Rs * Ss) + (Qi * Si).sum() / (2 * Ss ** 2)
    or_mh = Rs / Ss
    mh_rows.append(dict(strata=lab, OR_MH=or_mh, lo=or_mh * np.exp(-1.96 * var_ ** 0.5), hi=or_mh * np.exp(1.96 * var_ ** 0.5),
                        n_strata_with_exit=len(s_)))
mh = pd.DataFrame(mh_rows)
print("Within-rider Mantel-Haenszel OR of exiting right after a stockout/abandoned queue (vs other attempts):")
print(mh.round(3).to_string(index=False))
mh.to_csv(TAB / "q6_attempt_exit_within_rider_mh.csv", index=False)
del A2

# %% [markdown]
# ## 7. Gradient boosting + SHAP (non-linear importance, interactions)

# %%
import lightgbm as lgb  # noqa: E402
import shap  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

GB_NUM = ["e_stockout", "e_abandon", "e_syserr", "e_cancel", "first_fail", "q_mean", "q_long_n", "kyron_share",
          "soh_out", "km_mean", "price_mean", "peak_share", "comp_share", "tk_n", "tk_unres", "w3", "kyc_no", "coh_idx"]
GB_CAT = ["segment", "signup_channel", "age_band", "main_city", "main_wave", "main_gen", "main_conn", "declared_shift"]
X = D[GB_NUM].astype(float).copy()
for c in GB_CAT:
    X[c] = D[c].astype("category")
yv = D["y"].values
params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=80, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, n_estimators=400, verbose=-1, random_state=1)
aucs = []
for tr, te in StratifiedKFold(5, shuffle=True, random_state=7).split(X, yv):
    gbm = lgb.LGBMClassifier(**params).fit(X.iloc[tr], yv[tr])
    aucs.append(roc_auc_score(yv[te], gbm.predict_proba(X.iloc[te])[:, 1]))
gbm = lgb.LGBMClassifier(**params).fit(X, yv)
sv = shap.TreeExplainer(gbm).shap_values(X)
sv = sv[1] if isinstance(sv, list) else sv
imp = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)
imp["share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
print(f"LightGBM (drivers only) 5-fold CV AUC = {np.mean(aucs):.3f} (+/- {np.std(aucs):.3f})")
print(imp.round(4).head(10).to_string(index=False))
# Shares such as Kyron share and mean queue wait are noisy for riders with only a few early swaps, and those
# riders are exactly the early leavers. So model 2 adds early activity volume as an exposure control (a
# symptom, not a driver), which keeps the trees from reading denominator noise as a "driver".
X2 = X.copy()
X2["e_att"] = D["e_att"].astype(float)
aucs2 = []
for tr, te in StratifiedKFold(5, shuffle=True, random_state=7).split(X2, yv):
    g2 = lgb.LGBMClassifier(**params).fit(X2.iloc[tr], yv[tr])
    aucs2.append(roc_auc_score(yv[te], g2.predict_proba(X2.iloc[te])[:, 1]))
gbm2 = lgb.LGBMClassifier(**params).fit(X2, yv)
sv2 = shap.TreeExplainer(gbm2).shap_values(X2)
sv2 = sv2[1] if isinstance(sv2, list) else sv2
imp2 = pd.DataFrame({"feature": X2.columns, "mean_abs_shap": np.abs(sv2).mean(0)}).sort_values("mean_abs_shap", ascending=False)
imp2["share_excl_activity"] = imp2["mean_abs_shap"] / imp2.loc[imp2["feature"] != "e_att", "mean_abs_shap"].sum()
imp2.loc[imp2["feature"] == "e_att", "share_excl_activity"] = np.nan
print(f"LightGBM (drivers + early activity control) 5-fold CV AUC = {np.mean(aucs2):.3f} (+/- {np.std(aucs2):.3f})")
print(imp2.round(4).to_string(index=False))
# direction of the SHAP effect for the failure counts: mean SHAP when count >= 2 vs 0
for c in ["e_stockout", "e_abandon"]:
    i = list(X2.columns).index(c)
    print(f"  mean SHAP of {c}: count>=2 {sv2[X2[c] >= 2, i].mean():+.3f} vs 0 {sv2[X2[c] == 0, i].mean():+.3f} (log-odds)")
imp.merge(imp2, on="feature", how="outer", suffixes=("_drivers_only", "_with_activity")).to_csv(TAB / "q6_shap_importance.csv", index=False)
# logistic (no FE) AUC for comparison
auc_logit = roc_auc_score(yv, mA.predict(D))
print(f"Logistic Model A in-sample AUC = {auc_logit:.3f}")

# %% [markdown]
# ## 8. Dose-response and stability across segments and cohorts

# %%
seg_defs = {"All": D.index == D.index, "Independent": D["indep"] == 1, "Partner": D["indep"] == 0,
            "2W": D["w3"] == 0, "3W": D["w3"] == 1}
for c in CITIES:
    seg_defs[c] = D["main_city"] == c
seg_defs["Cohorts H1-2024"] = D["coh"] < "2024-07-01"
seg_defs["Cohorts H2-2024"] = (D["coh"] >= "2024-07-01") & (D["coh"] < "2025-01-01")
seg_defs["Cohorts 2025"] = D["coh"] >= "2025-01-01"
seg_defs["Winter-entry cohorts (Nov-Feb)"] = D["coh"].dt.month.isin([11, 12, 1, 2])
seg_defs["ZipDrop"] = D["partner"] == ZIPDROP_ID
dose_rows, stab_rows = [], []
for s_, m_ in seg_defs.items():
    sub = D[m_]
    for g_ in ["0", "1", "2", "3+"]:
        gg = sub[sub["fail_grp"] == g_]
        lo, hi = wilson(gg["y"].sum(), len(gg))
        dose_rows.append(dict(segment=s_, fail_grp=g_, n=len(gg), not_retained=gg["y"].mean(), lo=float(lo), hi=float(hi)))
    # adjusted OR of 2+ early failures within the segment (cohort-quarter FE + city FE where they vary)
    covs = [c for c in ["first_fail", "w3", "indep", "comp_any"] if sub[c].nunique() > 1]
    fe = " + C(coh_q)" + (" + C(main_city)" if sub["main_city"].nunique() > 1 else "")
    try:
        ms = smf.logit("y ~ fail2p + " + " + ".join(covs) + fe, data=sub).fit(disp=0, maxiter=200)
        ci = ms.conf_int().loc["fail2p"]
        stab_rows.append(dict(segment=s_, n=len(sub), base_not_retained=sub["y"].mean(),
                              nr_0_1=sub.loc[sub["fail2p"] == 0, "y"].mean(), nr_2p=sub.loc[sub["fail2p"] == 1, "y"].mean(),
                              OR_2plus=np.exp(ms.params["fail2p"]), lo=np.exp(ci[0]), hi=np.exp(ci[1]), p=ms.pvalues["fail2p"]))
    except Exception as ex:  # noqa: BLE001
        print("segment model failed", s_, ex)
dose = pd.DataFrame(dose_rows)
stab = pd.DataFrame(stab_rows)
dose.to_csv(TAB / "q6_dose_response.csv", index=False)
stab.to_csv(TAB / "q6_segment_stability.csv", index=False)
print(dose[dose["segment"].isin(["All", "Independent", "Partner", "3W"])].round(3).to_string(index=False))
print(stab.round(3).to_string(index=False))
sub_stab = stab[stab["segment"] != "All"]          # the pooled "All" row is not a subgroup
n_sub = len(sub_stab)
n_stab_sig = int(((sub_stab["OR_2plus"] > 1) & (sub_stab["p"] < 0.05)).sum())
n_stab_pos = int((sub_stab["OR_2plus"] > 1).sum())
st_all = stab.set_index("segment").loc["All"]
print(f"2+ vs 0-1 early failures: OR>1 in {n_stab_pos}/{n_sub} subgroups, significant (p<0.05) in {n_stab_sig}/{n_sub}; "
      f"pooled (All) OR {st_all['OR_2plus']:.2f} [{st_all['lo']:.2f}-{st_all['hi']:.2f}]")
fail_city = D.groupby("main_city")["e_fail"].mean()
print(f"Mean early (day 0-13) service failures per rider by main-station city: {fail_city.round(2).to_dict()}")
# Kyron share dose-response within post-Sep-2024 cohorts only (Kyron packs did not exist before)
ky = D[D["coh"] >= "2024-09-01"].groupby("kyron_bin")["y"].agg(["mean", "size"])
print("Kyron share in first 14 days (cohorts Sep-24+), non-retention:")
print(ky.round(3).to_string())

# %% [markdown]
# ## 9. Kaplan-Meier and Cox PH by the top driver (landmark at day 14)
# Riders must still be active at day 14, and the clock starts there. Early failures are counted in days 0-13,
# so riders who stayed longer could not collect more failures simply by being around longer (no immortal-time bias).

# %%
L = df[(df["surv_T"] >= EARLY_DAYS) & (df["end_day"] >= EARLY_DAYS + CHURN_GAP + 1)].copy()
L["T"] = L["surv_T"] - EARLY_DAYS
L["E"] = L["surv_E"].astype(int)
L["kyron25"], L["peak25"] = L["kyron_share"] / 0.25, L["peak_share"] / 0.25
lr = multivariate_logrank_test(L["T"], L["fail_grp"], L["E"])
km_fail = {}
for g_ in ["0", "1", "2", "3+"]:
    gg = L[L["fail_grp"] == g_]
    kmf = KaplanMeierFitter().fit(gg["T"], gg["E"], label=g_)
    km_fail[g_] = (kmf, len(gg))
    print(f"fail_grp {g_}: n={len(gg):,}, S(60d after landmark)={float(kmf.survival_function_at_times(60).iloc[0]):.3f}, "
          f"S(180d)={float(kmf.survival_function_at_times(180).iloc[0]):.3f}")
print(f"Log-rank test across early-failure groups: chi2={lr.test_statistic:.1f}, p={lr.p_value:.2e}")
L["f1"] = (L["fail_grp"] == "1").astype(int)
L["f2"] = (L["fail_grp"] == "2").astype(int)
L["f3p"] = (L["fail_grp"] == "3+").astype(int)
cox_cols = ["f1", "f2", "f3p", "first_fail", "comp_any", "w3", "indep", "kyron25", "km10c", "peak25", "tk_unres_any", "q_mean"]
L["entry_season"] = L["coh"].dt.month.map(SEASON_OF_MONTH)
cph = CoxPHFitter(penalizer=0.001).fit(L[cox_cols + ["T", "E", "entry_season", "main_city"]], "T", "E",
                                        strata=["entry_season", "main_city"])
cox = cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].rename(
    columns={"exp(coef)": "HR", "exp(coef) lower 95%": "lo", "exp(coef) upper 95%": "hi"})
print(f"Cox PH (landmark day 14, strata = entry season x city), n={len(L):,}, events={int(L['E'].sum()):,}")
print(cox.round(3).to_string())
cox.to_csv(TAB / "q6_cox_hr.csv")
L["km10"] = L["km_mean"] / 10
cph_raw = CoxPHFitter(penalizer=0.001).fit(L[[c for c in cox_cols if c != "km10c"] + ["km10", "T", "E", "entry_season", "main_city"]],
                                           "T", "E", strata=["entry_season", "main_city"])
print(f"3W Cox HR, old spec with raw km per swap: {cph_raw.summary.loc['w3', 'exp(coef)']:.2f} "
      f"[{cph_raw.summary.loc['w3', 'exp(coef) lower 95%']:.2f}-{cph_raw.summary.loc['w3', 'exp(coef) upper 95%']:.2f}] "
      f"-> km centred within class: {cox.loc['w3', 'HR']:.2f} [{cox.loc['w3', 'lo']:.2f}-{cox.loc['w3', 'hi']:.2f}], p={cox.loc['w3', 'p']:.3f}")

# %% [markdown]
# ## 10. Policy events: base price (Jul-24), peak pilot (Oct-24, BLR/PUN), ZipDrop amendment (Nov-24)
# Difference-in-differences on D31-60 non-retention, with cohort-month and city fixed effects:
# * base price: independent riders (who pay) vs partner riders (employer pays), cohorts before vs after Jul-24;
# * pilot: independent riders in pilot cities vs elsewhere, cohorts before vs after Oct-24, with partner riders
#   (who never pay the peak surcharge) as a placebo;
# * ZipDrop: ZipDrop riders vs other partner riders, cohorts before vs after Nov-24.

# %%
D["post_price"] = (D["coh"] >= "2024-07-01").astype(int)
D["post_pilot"] = (D["coh"] >= "2024-10-01").astype(int)
D["post_zip"] = (D["coh"] >= "2024-11-01").astype(int)
D["zip"] = (D["partner"] == ZIPDROP_ID).astype(int)
did_rows = []
CTRL = " + w3 + C(fail_grp) + C(coh_s) + C(main_city)"
m1 = smf.logit("y ~ indep + indep:post_price" + CTRL, data=D[D["coh"] < "2024-10-01"]).fit(disp=0)
m2 = smf.logit("y ~ pilot_city:post_pilot" + CTRL, data=D[D["indep"] == 1]).fit(disp=0)
m2p = smf.logit("y ~ pilot_city:post_pilot" + CTRL, data=D[D["indep"] == 0]).fit(disp=0)
m3 = smf.logit("y ~ zip + zip:post_zip" + CTRL, data=D[D["indep"] == 0]).fit(disp=0)
for lab, m, term in [("Base price Jul-24: independent x post (cohorts Jan-Sep 2024)", m1, "indep:post_price"),
                     ("Peak pilot Oct-24: pilot city x post, INDEPENDENT riders", m2, "pilot_city:post_pilot"),
                     ("Peak pilot placebo: pilot city x post, PARTNER riders", m2p, "pilot_city:post_pilot"),
                     ("ZipDrop amendment Nov-24: ZipDrop x post (partner riders)", m3, "zip:post_zip")]:
    ci = m.conf_int().loc[term]
    did_rows.append(dict(test=lab, OR=np.exp(m.params[term]), lo=np.exp(ci[0]), hi=np.exp(ci[1]), p=m.pvalues[term], n=int(m.nobs)))
did = pd.DataFrame(did_rows)
print(did.round(3).to_string(index=False))
did.to_csv(TAB / "q6_policy_did.csv", index=False)
# competitor: riders whose main station got a competitor, cohorts 2025 only (competitor entry Jan-25)
c25 = D[D["coh"] >= "2025-01-01"]
pa, na, pb, nb, pcmp = two_prop(c25["comp_any"] == 0, c25["comp_any"] == 1, col="y", frame=c25)
print(f"2025 cohorts: non-retention without competitor exposure {pct(pa)} (n={na:,}) vs with {pct(pb)} (n={nb:,}), p={pcmp:.3f}")

# %% [markdown]
# ## 11. Decomposing the cohort-to-cohort changes (counterfactual swaps of feature blocks)
# A pooled logistic model is fitted **without** cohort fixed effects, so that measured experiences have to explain
# the trend. For each block of features, we give each comparison-cohort rider the block values of a randomly
# drawn reference-cohort rider (Jan-Mar 2024), keep everything else as it is, and read the change in predicted
# non-retention (averaged over 30 draws). The part of the gap that no block accounts for stays unexplained.

# %%
fPool = ("y ~ C(fail_grp, Treatment('0')) + first_fail + q_mean + kyron25 + km10c + peak25 + comp_any + tk_any "
         "+ tk_unres_any + w3 + C(segment) + C(signup_channel) + kyc_no + C(age_band) + C(main_wave) + C(main_gen) "
         "+ C(main_conn) + C(main_city)")
mPool = smf.logit(fPool, data=D).fit(disp=0, maxiter=200)
print("Pooled model (no cohort FE) battery terms:", {k: (round(float(np.exp(mPool.params[k])), 3), round(float(mPool.pvalues[k]), 4)) for k in ["kyron25", "km10c"]})
BLOCKS = {"Early service failures": ["fail_grp", "first_fail"],
          "Queue wait": ["q_mean"],
          "Battery (Kyron share, km range)": ["kyron25", "km10c"],
          "Peak-tariff exposure": ["peak25"],
          "Competitor nearby": ["comp_any"],
          "Support tickets": ["tk_any", "tk_unres_any"],
          "Segment & vehicle mix": ["segment", "w3"],
          "City & station mix": ["main_city", "main_wave", "main_gen", "main_conn"],
          "Rider profile (channel, KYC, age)": ["signup_channel", "kyc_no", "age_band"]}
pnames = list(mPool.params.index)
block_p = {}
for b, cols in BLOCKS.items():
    idx = [i for i, n in enumerate(pnames) if any(n == c or n.startswith(f"C({c}") for c in cols)]
    R = np.zeros((len(idx), len(pnames)))
    for r_i, i in enumerate(idx):
        R[r_i, i] = 1
    block_p[b] = float(mPool.wald_test(R, scalar=True).pvalue)
print("Joint Wald p-value per block (pooled model):", {k: round(v, 4) for k, v in block_p.items()})
REF =D["coh"].between("2024-01-01", "2024-03-01")
COMPS = {"Apr-Jun 2024 (summer)": D["coh"].between("2024-04-01", "2024-06-01"),
         "Jan-Mar 2025 (same season)": D["coh"].between("2025-01-01", "2025-03-01"),
         "Apr 2025 (latest)": D["coh"] == "2025-04-01",
         "Oct 2024-Apr 2025 (post-interventions)": D["coh"].between("2024-10-01", "2025-04-01")}


def decompose(model, ref, cmp_, draws=30):
    p_cmp = model.predict(cmp_).mean()
    out = {}
    for b, cols in BLOCKS.items():
        vals = []
        for _ in range(draws):
            cf = cmp_.copy()
            idx = RNG.integers(0, len(ref), len(cmp_))
            for c in cols:
                cf[c] = ref[c].values[idx]
            vals.append(model.predict(cf).mean())
        out[b] = p_cmp - np.mean(vals)
    return out


dec_rows = []
ref_df = D[REF]
for lab, msk in COMPS.items():
    cmp_df = D[msk]
    contrib = decompose(mPool, ref_df, cmp_df)
    obs_gap = cmp_df["y"].mean() - ref_df["y"].mean()
    pred_gap = mPool.predict(cmp_df).mean() - mPool.predict(ref_df).mean()
    for b, v in contrib.items():
        dec_rows.append(dict(comparison=lab, block=b, contribution_pp=100 * v, obs_gap_pp=100 * obs_gap,
                             pred_gap_pp=100 * pred_gap, n_ref=len(ref_df), n_cmp=len(cmp_df)))
dec = pd.DataFrame(dec_rows)
dec["unexplained_pp"] = dec["obs_gap_pp"] - dec.groupby("comparison")["contribution_pp"].transform("sum")
dec.to_csv(TAB / "q6_decomposition.csv", index=False)
piv = dec.pivot(index="block", columns="comparison", values="contribution_pp")
gap_line = dec.groupby("comparison")[["obs_gap_pp", "pred_gap_pp", "unexplained_pp"]].first()
print("Change in NON-retention vs Jan-Mar 2024 cohorts (percentage points); positive = worse:")
print(gap_line.round(2).to_string())
print(piv.round(2).to_string())

# failure-neutralised counterfactual per cohort (fail block drawn from Jan-Mar 2024 riders)
cf_rows = []
for c_, g in D.groupby("coh"):
    vals = []
    for _ in range(30):
        cf = g.copy()
        idx = RNG.integers(0, len(ref_df), len(g))
        for c in BLOCKS["Early service failures"]:
            cf[c] = ref_df[c].values[idx]
        vals.append(1 - mPool.predict(cf).mean())
    cf_rows.append(dict(coh=c_, ret_obs=1 - g["y"].mean(), ret_pred=1 - mPool.predict(g).mean(), ret_cf_fail=np.mean(vals)))
cfc = pd.DataFrame(cf_rows)
# The pure-model counterfactual (ret_cf_fail) must not be plotted against observed data: the pooled model itself
# reproduces only part of the cohort-to-cohort variation. The honest series removes the modelled early-failure
# effect from the OBSERVED rate: observed + (counterfactual - model prediction).
cfc["ret_fail_neutral"] = cfc["ret_obs"] + (cfc["ret_cf_fail"] - cfc["ret_pred"])
coh = coh.merge(cfc, on="coh", how="left")
print(cfc.round(4).to_string(index=False))
sd_ = {k: 100 * cfc[k].std() for k in ["ret_obs", "ret_pred", "ret_cf_fail", "ret_fail_neutral"]}
print(f"Cohort D31-60 retention, observed: {pct(cfc['ret_obs'].min())}-{pct(cfc['ret_obs'].max())} (sd {sd_['ret_obs']:.2f} pp); "
      f"observed with the early-failure effect removed: {pct(cfc['ret_fail_neutral'].min())}-{pct(cfc['ret_fail_neutral'].max())} "
      f"(sd {sd_['ret_fail_neutral']:.2f} pp)")
print(f"Model vs model: pooled-model prediction sd {sd_['ret_pred']:.2f} pp vs pure counterfactual sd {sd_['ret_cf_fail']:.2f} pp "
      f"(the model reproduces only part of the observed spread, so the pure counterfactual is not comparable with observed data)")
dip_closed = {}
for m_ in ["2024-04-01", "2024-05-01", "2024-06-01", "2025-04-01"]:
    r_ = cfc[cfc["coh"] == m_].iloc[0]
    dip_closed[m_[:7]] = (r_["ret_fail_neutral"] - r_["ret_obs"]) / ((1 - ref_df["y"].mean()) - r_["ret_obs"])
    print(f"  {m_[:7]}: observed {pct(r_['ret_obs'])} -> early-failure effect removed {pct(r_['ret_fail_neutral'])} "
          f"(Jan-Mar 2024 reference {pct(1 - ref_df['y'].mean())}; share of the dip closed {pct(dip_closed[m_[:7]], 0)})")
coh.to_csv(TAB / "q6_cohort_retention.csv", index=False)

# %% [markdown]
# ### 11b. How much of the summer dip do early failures explain? Four estimates
# The decomposition above is one model-based estimate. Three checks with different assumptions bracket it:
# (i) logit shrinkage: how much the summer coefficient (Apr-Jun vs Jan-Mar 2024 cohorts) shrinks when early
# failure groups are added; (ii) direct standardisation: re-weight non-retention rates to the other period's
# failure mix (no model); (iii) riders still active in days 14-29: add failures up to day 29 (still before the
# outcome window), which captures summer failures the 14-day window misses.

# %%
SA_ = D[D["coh"].between("2024-01-01", "2024-06-01")].assign(summer=lambda x: x["coh"].dt.month.isin([4, 5, 6]).astype(int))
c0 = smf.logit("y ~ summer", SA_).fit(disp=0).params["summer"]
c1 = smf.logit("y ~ summer + C(fail_grp)", SA_).fit(disp=0).params["summer"]
c2 = smf.logit("y ~ summer + C(fail_grp) + log_att", SA_).fit(disp=0).params["summer"]   # no w3 here, so uncentred
print(f"(i) Logit shrinkage, Jan-Jun 2024 cohorts (n={len(SA_):,}): summer log-odds {c0:.3f} -> {c1:.3f} with early "
      f"failure groups ({pct(1 - c1 / c0, 0)} explained) -> {c2:.3f} adding log attempts ({pct(1 - c2 / c0, 0)})")
std_rows = []
for lab, msk in {"Apr-Jun 2024 (summer)": D["coh"].between("2024-04-01", "2024-06-01"),
                 "Apr 2025 (latest)": D["coh"] == "2025-04-01"}.items():
    c_ = D[msk]
    w_ref, w_c = ref_df["fail_grp"].value_counts(normalize=True), c_["fail_grp"].value_counts(normalize=True)
    r_ref, r_c = ref_df.groupby("fail_grp")["y"].mean(), c_.groupby("fail_grp")["y"].mean()
    obs = c_["y"].mean() - ref_df["y"].mean()
    e1 = c_["y"].mean() - (w_ref * r_c).sum()          # comparison rates, reference failure mix
    e2 = (w_c * r_ref).sum() - ref_df["y"].mean()      # reference rates, comparison failure mix
    std_rows.append(dict(comparison=lab, obs_gap_pp=100 * obs, std_cmp_rates_pp=100 * e1, std_ref_rates_pp=100 * e2,
                         share_lo=min(e1, e2) / obs, share_hi=max(e1, e2) / obs,
                         decomposition_pp=piv.loc["Early service failures", lab], decomposition_share=piv.loc["Early service failures", lab] / (100 * obs)))
    print(f"(ii) Standardisation, {lab}: observed {100*obs:+.2f} pp; failure mix explains {100*e1:+.2f} / {100*e2:+.2f} pp "
          f"({pct(min(e1, e2) / obs, 0)}-{pct(max(e1, e2) / obs, 0)}); decomposition {piv.loc['Early service failures', lab]:+.2f} pp "
          f"({pct(piv.loc['Early service failures', lab] / (100 * obs), 0)})")
mid = (E.filter((pl.col("d") >= EARLY_DAYS) & (pl.col("d") < 30)).group_by("rider_id")
       .agg(m_att=pl.len(), m_fail=pl.col("is_service_failure").sum()).collect(engine="streaming").to_pandas())
SM_ = SA_.merge(mid, on="rider_id", how="inner")
SM_["rate029"] = (SM_["e_fail"] + SM_["m_fail"]) / (SM_["e_att"] + SM_["m_att"])
SM_["r029_bin"] = pd.cut(SM_["rate029"], [-0.01, 0, 0.05, 0.10, 0.20, 1.01], labels=["0", "a", "b", "c", "d"]).astype(str)
SM_["log_att029"] = np.log(SM_["e_att"] + SM_["m_att"])
b0 = smf.logit("y ~ summer", SM_).fit(disp=0).params["summer"]
b1 = smf.logit("y ~ summer + C(fail_grp) + log_att", SM_).fit(disp=0).params["summer"]
b2 = smf.logit("y ~ summer + C(r029_bin) + log_att029", SM_).fit(disp=0).params["summer"]
print(f"(iii) Riders active in days 14-29, Jan-Jun 2024 cohorts (n={len(SM_):,}): summer log-odds {b0:.3f}; "
      f"days 0-13 failures {b1:.3f} ({pct(1 - b1 / b0, 0)} explained); failure rate in days 0-29 {b2:.3f} ({pct(1 - b2 / b0, 0)} explained)")
expl_tab = pd.DataFrame(std_rows)
expl_tab = pd.concat([expl_tab, pd.DataFrame([
    dict(comparison="Logit shrinkage, Jan-Jun 2024 (+early failure groups)", obs_gap_pp=np.nan, share_lo=1 - c1 / c0, share_hi=1 - c1 / c0),
    dict(comparison="Logit shrinkage, Jan-Jun 2024 (+failure groups, log attempts)", obs_gap_pp=np.nan, share_lo=1 - c2 / c0, share_hi=1 - c2 / c0),
    dict(comparison="Active in days 14-29: days 0-13 failures + log attempts", obs_gap_pp=np.nan, share_lo=1 - b1 / b0, share_hi=1 - b1 / b0),
    dict(comparison="Active in days 14-29: days 0-29 failure rate + log attempts", obs_gap_pp=np.nan, share_lo=1 - b2 / b0, share_hi=1 - b2 / b0)])])
expl_tab.to_csv(TAB / "q6_summer_dip_explained.csv", index=False)


# %% [markdown]
# ## 12. Classify drivers: primary vs secondary
# Three independent designs: (i) rider-level logit (D31-60, cohort and city FE), (ii) landmark Cox PH (survival
# after day 14), (iii) attempt-level exit GLM (does the rider leave right after this experience?).
# **PRIMARY** = adverse and significant in all three designs, with a dose-response, the same direction in at
# least 75% of subgroups, and it explains at least 25% of the OBSERVED summer-2024 cohort dip (stricter than the
# share of the model-predicted dip).
# **SECONDARY** = adverse and significant in at least one design but fails a primary test (segment-specific,
# localised, not actionable on its own, or does not move the trend).
# **NOT SUPPORTED** = no adverse significant effect in any design.

# %%
def get(t, term, col="OR"):
    r_ = t[t["term"] == term]
    return float(r_[col].iloc[0]) if len(r_) else np.nan


summer_lab = "Apr-Jun 2024 (summer)"
summer_gap = gap_line.loc[summer_lab, "obs_gap_pp"]
summer_pred = gap_line.loc[summer_lab, "pred_gap_pp"]
fail_expl = piv.loc["Early service failures", summer_lab]
H = lambda term, col="OR": get(hz, term, col)  # noqa: E731
CX = lambda cov, col="HR": float(cox.loc[cov, col]) if cov in cox.index else np.nan  # noqa: E731
consistent_fail = n_stab_pos / n_sub >= 0.75
drivers = [
    dict(driver="Early service failures (2 vs 0 in days 0-13; attempt level: 2 in last 7 d)", logit_OR=get(orA, T2), logit_lo=get(orA, T2, "lo"), logit_hi=get(orA, T2, "hi"), logit_p=get(orA, T2, "p"),
         cox_HR=CX("f2"), cox_lo=CX("f2", "lo"), cox_hi=CX("f2", "hi"), cox_p=CX("f2", "p"),
         exit_OR=H("C(prior_f7, Treatment('0'))[T.2]"), exit_lo=H("C(prior_f7, Treatment('0'))[T.2]", "lo"), exit_hi=H("C(prior_f7, Treatment('0'))[T.2]", "hi"), exit_p=H("C(prior_f7, Treatment('0'))[T.2]", "p"),
         dose=True, consistent=consistent_fail, trend_share=fail_expl / summer_gap),
    dict(driver="Stockout (no charged pack)", logit_OR=get(orB, "e_stockout_c"), logit_lo=get(orB, "e_stockout_c", "lo"), logit_hi=get(orB, "e_stockout_c", "hi"), logit_p=get(orB, "e_stockout_c", "p"),
         cox_HR=np.nan, cox_lo=np.nan, cox_hi=np.nan, cox_p=np.nan,
         exit_OR=H("C(etype, Treatment('completed'))[T.stockout]"), exit_lo=H("C(etype, Treatment('completed'))[T.stockout]", "lo"), exit_hi=H("C(etype, Treatment('completed'))[T.stockout]", "hi"), exit_p=H("C(etype, Treatment('completed'))[T.stockout]", "p"),
         dose=True, consistent=consistent_fail, trend_share=fail_expl / summer_gap),
    dict(driver="Abandoned queue", logit_OR=get(orB, "e_abandon_c"), logit_lo=get(orB, "e_abandon_c", "lo"), logit_hi=get(orB, "e_abandon_c", "hi"), logit_p=get(orB, "e_abandon_c", "p"),
         cox_HR=np.nan, cox_lo=np.nan, cox_hi=np.nan, cox_p=np.nan,
         exit_OR=H("C(etype, Treatment('completed'))[T.abandoned]"), exit_lo=H("C(etype, Treatment('completed'))[T.abandoned]", "lo"), exit_hi=H("C(etype, Treatment('completed'))[T.abandoned]", "hi"), exit_p=H("C(etype, Treatment('completed'))[T.abandoned]", "p"),
         dose=True, consistent=consistent_fail, trend_share=fail_expl / summer_gap),
    dict(driver="First attempt failed", logit_OR=get(orA, "first_fail"), logit_lo=get(orA, "first_fail", "lo"), logit_hi=get(orA, "first_fail", "hi"), logit_p=get(orA, "first_fail", "p"),
         cox_HR=CX("first_fail"), cox_lo=CX("first_fail", "lo"), cox_hi=CX("first_fail", "hi"), cox_p=CX("first_fail", "p"),
         exit_OR=np.nan, exit_lo=np.nan, exit_hi=np.nan, exit_p=np.nan, dose=False, consistent=False, trend_share=np.nan),
    dict(driver="Competitor within 1.5 km (active)", logit_OR=get(orA, "comp_any"), logit_lo=get(orA, "comp_any", "lo"), logit_hi=get(orA, "comp_any", "hi"), logit_p=get(orA, "comp_any", "p"),
         cox_HR=CX("comp_any"), cox_lo=CX("comp_any", "lo"), cox_hi=CX("comp_any", "hi"), cox_p=CX("comp_any", "p"),
         exit_OR=H("C(competitor_active)[T.True]"), exit_lo=H("C(competitor_active)[T.True]", "lo"), exit_hi=H("C(competitor_active)[T.True]", "hi"), exit_p=H("C(competitor_active)[T.True]", "p"),
         dose=False, consistent=False, trend_share=piv.loc["Competitor nearby", "Jan-Mar 2025 (same season)"] / max(gap_line.loc["Jan-Mar 2025 (same season)", "pred_gap_pp"], 1e-9)),
    dict(driver="3W vehicle (vs 2W)", logit_OR=get(orA, "w3"), logit_lo=get(orA, "w3", "lo"), logit_hi=get(orA, "w3", "hi"), logit_p=get(orA, "w3", "p"),
         cox_HR=CX("w3"), cox_lo=CX("w3", "lo"), cox_hi=CX("w3", "hi"), cox_p=CX("w3", "p"),
         exit_OR=H("C(vehicle_class)[T.3W]"), exit_lo=H("C(vehicle_class)[T.3W]", "lo"), exit_hi=H("C(vehicle_class)[T.3W]", "hi"), exit_p=H("C(vehicle_class)[T.3W]", "p"),
         dose=False, consistent=False, trend_share=0.0),
    dict(driver="Independent rider (pays own swaps)", logit_OR=get(orA, "C(segment, Treatment('Other partner'))[T.Indep-PAYG]"), logit_lo=get(orA, "C(segment, Treatment('Other partner'))[T.Indep-PAYG]", "lo"),
         logit_hi=get(orA, "C(segment, Treatment('Other partner'))[T.Indep-PAYG]", "hi"), logit_p=get(orA, "C(segment, Treatment('Other partner'))[T.Indep-PAYG]", "p"),
         cox_HR=CX("indep"), cox_lo=CX("indep", "lo"), cox_hi=CX("indep", "hi"), cox_p=CX("indep", "p"),
         exit_OR=H("C(is_independent)[T.True]"), exit_lo=H("C(is_independent)[T.True]", "lo"), exit_hi=H("C(is_independent)[T.True]", "hi"), exit_p=H("C(is_independent)[T.True]", "p"),
         dose=False, consistent=False, trend_share=0.0),
    dict(driver="Kyron pack received (+25 pp share)", logit_OR=get(orA, "kyron25"), logit_lo=get(orA, "kyron25", "lo"), logit_hi=get(orA, "kyron25", "hi"), logit_p=get(orA, "kyron25", "p"),
         cox_HR=CX("kyron25"), cox_lo=CX("kyron25", "lo"), cox_hi=CX("kyron25", "hi"), cox_p=CX("kyron25", "p"),
         exit_OR=H("kyron"), exit_lo=H("kyron", "lo"), exit_hi=H("kyron", "hi"), exit_p=H("kyron", "p"), dose=False, consistent=False, trend_share=np.nan),
    dict(driver="Peak-tariff share (+25 pp, pilot)", logit_OR=get(orA, "peak25"), logit_lo=get(orA, "peak25", "lo"), logit_hi=get(orA, "peak25", "hi"), logit_p=get(orA, "peak25", "p"),
         cox_HR=CX("peak25"), cox_lo=CX("peak25", "lo"), cox_hi=CX("peak25", "hi"), cox_p=CX("peak25", "p"),
         exit_OR=np.nan, exit_lo=np.nan, exit_hi=np.nan, exit_p=np.nan, dose=False, consistent=False, trend_share=np.nan),
    dict(driver="Mean queue wait (+1 min, completed swaps)", logit_OR=get(orA, "q_mean"), logit_lo=get(orA, "q_mean", "lo"), logit_hi=get(orA, "q_mean", "hi"), logit_p=get(orA, "q_mean", "p"),
         cox_HR=CX("q_mean"), cox_lo=CX("q_mean", "lo"), cox_hi=CX("q_mean", "hi"), cox_p=CX("q_mean", "p"),
         exit_OR=np.nan, exit_lo=np.nan, exit_hi=np.nan, exit_p=np.nan, dose=False, consistent=False, trend_share=np.nan),
    dict(driver="Unresolved support ticket", logit_OR=get(orA, "tk_unres_any"), logit_lo=get(orA, "tk_unres_any", "lo"), logit_hi=get(orA, "tk_unres_any", "hi"), logit_p=get(orA, "tk_unres_any", "p"),
         cox_HR=CX("tk_unres_any"), cox_lo=CX("tk_unres_any", "lo"), cox_hi=CX("tk_unres_any", "hi"), cox_p=CX("tk_unres_any", "p"),
         exit_OR=np.nan, exit_lo=np.nan, exit_hi=np.nan, exit_p=np.nan, dose=False, consistent=False, trend_share=np.nan),
    dict(driver="System error at kiosk", logit_OR=get(orB, "e_syserr_c"), logit_lo=get(orB, "e_syserr_c", "lo"), logit_hi=get(orB, "e_syserr_c", "hi"), logit_p=get(orB, "e_syserr_c", "p"),
         cox_HR=np.nan, cox_lo=np.nan, cox_hi=np.nan, cox_p=np.nan,
         exit_OR=H("C(etype, Treatment('completed'))[T.system_error]"), exit_lo=H("C(etype, Treatment('completed'))[T.system_error]", "lo"), exit_hi=H("C(etype, Treatment('completed'))[T.system_error]", "hi"), exit_p=H("C(etype, Treatment('completed'))[T.system_error]", "p"),
         dose=False, consistent=False, trend_share=np.nan),
]
drv = pd.DataFrame(drivers)


def adverse(o, p):
    return (o > 1) and (p < 0.05)


SEGMENT_ATTRS = ("3W vehicle (vs 2W)", "Independent rider (pays own swaps)")


def classify(r_):
    flags = [adverse(r_["logit_OR"], r_["logit_p"]), adverse(r_["cox_HR"], r_["cox_p"]), adverse(r_["exit_OR"], r_["exit_p"])]
    if r_["driver"] in SEGMENT_ATTRS:  # per-attempt exit rates are confounded by attempt frequency for rider attributes
        return "SECONDARY" if (flags[0] or flags[1]) else "NOT SUPPORTED"
    tested = [not np.isnan(v) for v in (r_["logit_OR"], r_["cox_HR"], r_["exit_OR"])]
    n_adv, n_test = sum(flags), sum(tested)
    if n_adv == n_test and n_test >= 2 and r_["dose"] and r_["consistent"] and (r_["trend_share"] or 0) >= 0.25:
        return "PRIMARY"
    if n_adv >= 1:
        return "SECONDARY"
    return "NOT SUPPORTED"


drv["designs_adverse"] = drv.apply(lambda r_: f"{sum([adverse(r_['logit_OR'], r_['logit_p']), adverse(r_['cox_HR'], r_['cox_p']), adverse(r_['exit_OR'], r_['exit_p'])])}"
                                   f"/{sum([not np.isnan(v) for v in (r_['logit_OR'], r_['cox_HR'], r_['exit_OR'])])}", axis=1)
drv["class"] = drv.apply(classify, axis=1)
pd.set_option("display.max_colwidth", 45)
print(drv[["driver", "logit_OR", "logit_lo", "logit_hi", "logit_p", "cox_HR", "cox_p", "exit_OR", "exit_lo", "exit_hi", "exit_p",
           "trend_share", "designs_adverse", "class"]].round(3).to_string(index=False))
drv.to_csv(TAB / "q6_driver_classification.csv", index=False)
print(f"Early failures explain {fail_expl:.2f} of the {summer_pred:.2f} pp model-predicted summer-2024 dip "
      f"({pct(fail_expl / summer_pred, 0)}), i.e. {pct(fail_expl / summer_gap, 0)} of the {summer_gap:.2f} pp observed dip")
print("Weak secondary flags: 3W is borderline in the logit (" + or_txt(mA, "w3") + "); first attempt failed is not robust "
      "to the activity control (" + or_txt(mA_att, "first_fail") + ")")

# %% [markdown]
# ## 13. What would fixing early failures be worth? (model-based what-if)

# %%
cf = D.copy()
cf["fail_grp"] = np.where(cf["fail_grp"].isin(["2", "3+"]), "1", cf["fail_grp"])
p_now = mPool.predict(D).mean()
p_cf = mPool.predict(cf).mean()
n_2p = int(D["fail2p"].sum())
nr_2p = D.loc[D["fail2p"] == 1, "y"].mean()
nr_01 = D.loc[D["fail2p"] == 0, "y"].mean()
summer = D["coh"].dt.month.isin([4, 5, 6])
p_now_s, p_cf_s = mPool.predict(D[summer]).mean(), mPool.predict(cf[summer]).mean()
new_per_month = D.groupby("coh").size().loc["2024-10-01":"2025-04-01"].mean()
extra_ret_per_month = (p_now - p_cf) * new_per_month
ret_ids = base.filter(pl.col("r3060") & (pl.col("t0") < ELIG_CUTOFF)).select("rider_id")
sw_ret, rev_ret = (E.join(ret_ids.lazy(), on="rider_id", how="semi")
                   .filter(pl.col("is_completed") & (pl.col("d") >= OUT_LO) & (pl.col("d") < OUT_HI))
                   .select(pl.len(), pl.col("amount_charged_inr").sum()).collect(engine="streaming").row(0))
swaps_per_ret = sw_ret / ret_ids.height
rev_per_swap = rev_ret / sw_ret
print(f"Riders with 2+ early failures: {n_2p:,} ({pct(n_2p / len(D))}); non-retention {pct(nr_2p)} vs {pct(nr_01)} for 0-1 "
      f"({nr_2p / nr_01:.2f}x)")
print(f"What-if (cap early failures at 1): predicted non-retention {pct(p_now, 2)} -> {pct(p_cf, 2)} "
      f"({100 * (p_now - p_cf):.2f} pp); summer-entry cohorts (Apr-Jun) {pct(p_now_s, 2)} -> {pct(p_cf_s, 2)} "
      f"({100 * (p_now_s - p_cf_s):.2f} pp); at {new_per_month:.0f} new riders/month = {extra_ret_per_month:.0f} more retained/month")
print(f"A retained new rider makes {swaps_per_ret:.1f} completed swaps in days 31-60 -> "
      f"~{extra_ret_per_month * swaps_per_ret:,.0f} extra swaps per monthly cohort in its second month")
print(f"Revenue per completed swap of retained new riders in days 31-60: Rs {rev_per_swap:.1f} -> about Rs "
      f"{extra_ret_per_month * swaps_per_ret * rev_per_swap:,.0f} per monthly cohort in its second month, "
      f"Rs {12 * extra_ret_per_month * swaps_per_ret * rev_per_swap:,.0f} a year across 12 cohorts (second month only; "
      f"retained riders keep swapping for months, see the landmark KM)")
fail_events_month = (early.filter(pl.col("is_stockout") | pl.col("is_abandoned"))
                     .group_by(pl.col("date").dt.truncate("1mo")).len().sort("date").to_pandas())
fail_events_month["m"] = pd.to_datetime(fail_events_month["date"]).dt.month
fe_summer = fail_events_month.loc[fail_events_month["m"].isin([4, 5, 6]), "len"].mean()
fe_other = fail_events_month.loc[~fail_events_month["m"].isin([4, 5, 6]), "len"].mean()
print(f"Stockout+abandon events hitting riders in their first 14 days: mean {fe_summer:.0f}/month in Apr-Jun vs "
      f"{fe_other:.0f}/month in other months (Jan-24..Jun-25); May-25 = {int(fail_events_month.loc[fail_events_month['date'].astype(str).str.startswith('2025-05'), 'len'].iloc[0]):,}")
second_fail = (early.filter(pl.col("is_service_failure")).sort(["rider_id", "event_ts"])
               .with_columns(k=pl.col("event_ts").rank("ordinal").over("rider_id")).filter(pl.col("k") == 2))
sa_ev = (early.filter(pl.col("is_stockout") | pl.col("is_abandoned")).sort(["rider_id", "event_ts"])
         .with_columns(k=pl.col("event_ts").rank("ordinal").over("rider_id")))
second_sa, first_sa = sa_ev.filter(pl.col("k") == 2), sa_ev.filter(pl.col("k") == 1)
print(f"Riders reaching a 2nd service failure of any type inside their first 14 days (all cohorts): {second_fail.height:,} "
      f"(includes kiosk system errors)")
print(f"TRIGGER VOLUME (stockouts/abandoned queues only, all cohorts Jan-24..Jun-25): 1st event {first_sa.height:,} riders "
      f"(median day {first_sa['d'].median():.1f}); 2nd event {second_sa.height:,} riders (median day {second_sa['d'].median():.1f})")
fail_events_month.to_csv(TAB / "q6_newrider_failure_events_month.csv", index=False)

# %% [markdown]
# ## 14. Figures

# %%
# ---- Figure 1 (HERO): cohort retention trend with interventions
cp = coh.copy()
cp["coh"] = pd.to_datetime(cp["coh"])
ce_ = cp[cp["ret"].notna()]
fig, (ax, axf) = plt.subplots(2, 1, figsize=(12.5, 8.2), sharex=True, gridspec_kw={"height_ratios": [3.1, 1.15], "hspace": 0.08})
ax.fill_between(ce_["coh"], 100 * ce_["lo"], 100 * ce_["hi"], color=PALETTE["primary"], alpha=0.15, lw=0)
ax.plot(ce_["coh"], 100 * ce_["ret"], "-o", color=PALETTE["primary"], lw=2.6, ms=5.5, label="Observed, swap in days 31-60 (95% CI)")
ax.plot(ce_["coh"], 100 * ce_["ret_fail_neutral"], "--", color=PALETTE["good"], lw=2.2,
        label="Observed, early-failure effect removed")
ax.plot(cp["coh"], 100 * cp["naive"], ":", color=PALETTE["muted"], lw=1.6, label="Naive metric (censoring artefact)")
for i_ in ce_.index:
    if ce_.loc[i_, "coh"].month in (5,) or ce_.loc[i_, "coh"] == pd.Timestamp("2025-04-01"):
        ax.annotate(f"{100*ce_.loc[i_, 'ret']:.1f}%", (ce_.loc[i_, "coh"], 100 * ce_.loc[i_, "ret"]), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=10, color=PALETTE["primary"], fontweight="bold")
        is_may = ce_.loc[i_, "coh"].month == 5
        ax.annotate(f"{100*ce_.loc[i_, 'ret_fail_neutral']:.1f}%", (ce_.loc[i_, "coh"], 100 * ce_.loc[i_, "ret_fail_neutral"]),
                    textcoords="offset points", xytext=(30, -58) if is_may else (15, 3), ha="left", fontsize=10,
                    color="#2E7D4F", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color="#2E7D4F", lw=0.9) if is_may else None)
ax.set_ylim(60, 96)
ax.set_ylabel("New riders retained (%)")
events_v = [("2024-07-01", "Base price +Rs5/+Rs10"), ("2024-09-01", "Kyron packs arrive"), ("2024-10-01", "Peak pilot BLR/PUN"),
            ("2024-11-01", "ZipDrop discount 12->28%"), ("2025-01-15", "Competitor entry (10 stns)")]
for d_, lab in events_v:
    x_ = pd.Timestamp(d_)
    for a_ in (ax, axf):
        a_.axvline(x_, color="#555555", lw=0.9, ls=(0, (3, 3)))
    ax.text(x_ + pd.Timedelta(days=3), 69.3, lab, fontsize=9, color="#333333", rotation=90, va="bottom", ha="left",
            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
for s_, e_, lab in [("2024-04-01", "2024-06-30", "Summer stockout season 2024"), ("2025-04-01", "2025-06-30", "Summer 2025")]:
    for a_ in (ax, axf):
        a_.axvspan(pd.Timestamp(s_) - pd.Timedelta(days=14), pd.Timestamp(e_) - pd.Timedelta(days=14), color=PALETTE["warn"], alpha=0.12, lw=0)
    ax.text(pd.Timestamp(s_) - pd.Timedelta(days=10), 94.6, lab, fontsize=9, color="#9a6a00", va="top")
c1 = cmp_tab.iloc[0]
may24 = cfc[cfc["coh"] == "2024-05-01"].iloc[0]
ax.text(pd.Timestamp("2024-08-25"), 61.0,
        f"Same season, Jan-Mar 24 vs Jan-Mar 25: {pct(c1['ret_a'])} vs {pct(c1['ret_b'])} (p={c1['p_value']:.2f})\n"
        f"Failures removed: May-24 {pct(may24['ret_obs'])} -> {pct(may24['ret_fail_neutral'])} (dip shrinks, not gone)",
        fontsize=9.2, color="#222222", va="bottom", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
ax.legend(loc="lower left", fontsize=9.5, framealpha=0.9, frameon=True, edgecolor="#dddddd")
title_sub(ax, f"Retention did not fall structurally; summer-entry cohorts dip, and early failures explain "
              f"{100 * min(dip_closed.values()):.0f}-{100 * max(dip_closed.values()):.0f}% of each dip",
          "Cohort = month of first swap attempt; riders with >= 60 days of follow-up (May/Jun-25 cohorts appear only in the naive metric). "
          "Green = observed + (counterfactual - model prediction)")
axf.bar(cp["coh"], 100 * cp["fail_per_att"], width=21, color=[PALETTE["accent"] if v > 0.06 else "#F0A28C" for v in cp["fail_per_att"]])
for x_, v_ in zip(cp["coh"], cp["fail_per_att"]):
    if v_ > 0.06:
        axf.text(x_, 100 * v_ + 0.4, f"{100*v_:.1f}", ha="center", fontsize=8.5, color=PALETTE["accent"])
axf.set_ylim(0, 14.5)
axf.set_ylabel("Failure rate,\nfirst 14 days (%)")
for s_, e_, lab in [("2024-07-01", "2024-12-31", "Wave1 openings (H2-2024)"), ("2025-01-01", "2025-06-30", "Wave2 openings (H1-2025)")]:
    axf.annotate("", xy=(pd.Timestamp(e_), 13.2), xytext=(pd.Timestamp(s_), 13.2),
                 arrowprops=dict(arrowstyle="<->", color="#555555", lw=1))
    axf.text(pd.Timestamp(s_) + (pd.Timestamp(e_) - pd.Timestamp(s_)) / 2, 11.4, lab, ha="center", fontsize=9, color="#555555")
axf.set_xlabel("Cohort (month of first swap attempt)")
axf.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
save_fig(fig, "q6_01_cohort_retention_trend")

# ---- Figure 3 (HERO): driver effects across three designs
cls_col = {"PRIMARY": PALETTE["accent"], "SECONDARY": PALETTE["warn"], "NOT SUPPORTED": PALETTE["muted"]}
fd = drv.iloc[::-1].reset_index(drop=True)
WEAK = {"3W vehicle (vs 2W)", "First attempt failed"}
ylab = {"Early service failures (2 vs 0 in days 0-13; attempt level: 2 in last 7 d)":
        "Early service failures  [PRIMARY]\n(logit/Cox: 2 vs 0 in days 0-13; attempt: 2 in last 7 d)"}
fig, ax = plt.subplots(figsize=(12, 7.8))
for i, r_ in fd.iterrows():
    col = cls_col[r_["class"]]
    for off, (o, lo, hi, mk, lab) in zip([0.22, 0.0, -0.22], [(r_["logit_OR"], r_["logit_lo"], r_["logit_hi"], "o", "Rider-level logit: not retained D31-60 (OR)"),
                                                              (r_["cox_HR"], r_["cox_lo"], r_["cox_hi"], "^", "Landmark Cox: churn hazard after day 14 (HR)"),
                                                              (r_["exit_OR"], r_["exit_lo"], r_["exit_hi"], "D", "Attempt-level: leaves right after this experience (OR)")]):
        if np.isnan(o) or (mk == "D" and r_["driver"] in SEGMENT_ATTRS):
            continue   # attempt-level rates for rider attributes are confounded by swap frequency (not used to classify)
        ax.plot([lo, hi], [i + off] * 2, color=col, lw=2, alpha=0.9)
        ax.plot(o, i + off, mk, color=col, ms=7, mec="white", mew=0.8)
ax.axvline(1, color="#333333", lw=1)
ax.set_yticks(range(len(fd)))
ax.set_yticklabels([ylab.get(d_, f"{d_}  [{c_}{', weak' if d_ in WEAK else ''}]") for d_, c_ in zip(fd["driver"], fd["class"])], fontsize=10)
for t_, c_ in zip(ax.get_yticklabels(), fd["class"]):
    t_.set_color(cls_col[c_] if c_ != "NOT SUPPORTED" else "#555555")
    t_.set_fontweight("bold" if c_ == "PRIMARY" else "normal")
ax.set_xscale("log")
ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 3])
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlim(0.45, 3.3)
ax.set_xlabel("Effect on NOT returning (odds / hazard ratio, log scale; >1 = more churn), 95% CI")
from matplotlib.lines import Line2D  # noqa: E402
hand = [Line2D([], [], marker="o", ls="", color="#444444", label="Rider-level logit: no swap in days 31-60 (OR, cohort+city FE)"),
        Line2D([], [], marker="^", ls="", color="#444444", label="Landmark Cox: churn hazard after day 14 (HR)"),
        Line2D([], [], marker="D", ls="", color="#444444", label="Attempt-level: exits right after this experience (OR)"),
        Line2D([], [], color=PALETTE["accent"], lw=3, label="PRIMARY"), Line2D([], [], color=PALETTE["warn"], lw=3, label="SECONDARY"),
        Line2D([], [], color=PALETTE["muted"], lw=3, label="Not supported")]
ax.legend(handles=hand, loc="lower right", fontsize=9)
no_minor_labels(ax)
title_sub(ax, "Primary driver: service failures (stockouts, abandoned queues) early in a rider's life",
          "Secondary: competitor nearby; weak: 3W riders, failed first attempt. No adverse effect: Kyron packs, peak pricing, queue time, "
          "tickets.\nRider attributes (3W, independent) are judged on rider-level designs only")
save_fig(fig, "q6_03_driver_effects")

# ---- Figure 4: dose-response (rider level) + trigger (attempt level)
fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [1.35, 1]})
grp_cols = {"0": "#C9D6E3", "1": PALETTE["warn"], "2": PALETTE["accent"], "3+": "#A63A1C"}
x_groups = [("All riders\n(pooled)", None), ("6-9", "6-9"), ("10-14", "10-14"), ("15-19", "15-19"), ("20+", "20+")]
w = 0.2
for j, g_ in enumerate(["0", "1", "2", "3+"]):
    for k_, (xl, band) in enumerate(x_groups):
        sub_ = D if band is None else D[D["att_band"] == band]
        sub_ = sub_[sub_["fail_grp"] == g_]
        if len(sub_) < 40:
            axa.text(k_ + (j - 1.5) * w, 1.0, "n<40", rotation=90, fontsize=7.5, ha="center", color="#777777")
            continue
        r_, (lo_, hi_) = sub_["y"].mean(), wilson(sub_["y"].sum(), len(sub_))
        axa.bar(k_ + (j - 1.5) * w, 100 * r_, width=w, color=grp_cols[g_], label=g_ if k_ == 0 else None,
                edgecolor="#555555" if band is None else "none", lw=0.6)
        axa.errorbar(k_ + (j - 1.5) * w, 100 * r_, yerr=[[100 * (r_ - lo_)], [100 * (hi_ - r_)]], fmt="none", ecolor="#333333", lw=0.7, capsize=1.5)
        if band is None or band in ("10-14", "15-19"):
            axa.text(k_ + (j - 1.5) * w, 100 * hi_ + 0.6, f"{100*r_:.1f}", fontsize=8, ha="center", rotation=90, va="bottom", color="#333333")
axa.axvline(0.5, color="#999999", lw=0.8, ls=":")
axa.set_xticks(range(len(x_groups)))
axa.set_xticklabels([xl if band is None else f"{xl}\n(n={int((D['att_band'] == band).sum()):,})" for xl, band in x_groups])
axa.set_xlabel("Attempts in days 0-13 (early-activity band)")
axa.set_ylabel("Not retained: no swap in days 31-60 (%)")
axa.set_ylim(0, 64)
axa.legend(title="Failures, days 0-13", ncol=1, fontsize=9, title_fontsize=9, loc="upper left")
o1, o2, o3 = (np.exp(mA_att.params["C(fail_grp, Treatment('0'))[T." + g + "]"]) for g in ["1", "2", "3+"])
axa.text(0.985, 0.97, f"Activity-adjusted OR vs 0 failures:\n1 = {o1:.2f}, 2 = {o2:.2f}, 3+ = {o3:.2f}\n\n"
         f"Pooled 0 > 1 is a Simpson's paradox:\n{int((low['fail_grp'] == '0').sum())} of {len(low)} riders with only 1-5\n"
         f"attempts ({pct(low['y'].mean())} not retained)\nhad 0 failures", transform=axa.transAxes, fontsize=8.8,
         va="top", ha="right", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc"))
axa.set_title("Within each activity band, 2+ failures roughly double\nnon-retention; the 1st failure adds little")
rate_o = raw_at.loc[["completed", "stockout", "abandoned", "system_error", "cancelled"], "churn_rate"] * 1000
rate_p = raw_pf["churn_rate"] * 1000
labels_b = ["Completed", "Stockout", "Abandoned", "System error", "Cancelled"]
cb = [PALETTE["good"], PALETTE["accent"], PALETTE["accent"], PALETTE["muted"], PALETTE["muted"]]
axb.barh(np.arange(5) + 5.6, rate_o.values, color=cb, height=0.7)
axb.barh(np.arange(4), rate_p.values, color=[PALETTE["light"], PALETTE["warn"], "#E08A2E", PALETTE["accent"]], height=0.7)
for yv_, v_ in zip(list(np.arange(5) + 5.6) + list(np.arange(4)), list(rate_o.values) + list(rate_p.values)):
    axb.text(v_ + 0.08, yv_, f"{v_:.2f}", va="center", fontsize=9.5)
axb.set_yticks(list(np.arange(5) + 5.6) + list(range(4)))
axb.set_yticklabels(labels_b + [f"{k} prior failure{'s' if k != '1' else ''} in last 7 d" for k in ["0", "1", "2", "3+"]])
axb.invert_yaxis()
axb.set_xlabel("Exits (no attempt for 30+ days) per 1,000 attempts")
axb.set_title("Exits follow stockouts and abandons (not kiosk errors),\nand rise with each recent failure")
axb.set_xlim(0, 8.5)
fig.subplots_adjust(wspace=0.55)
fig.suptitle(f"Risk rises with early failures: the 1st adds a little (OR {o1:.2f}), the 2nd roughly doubles the odds "
             f"(OR {o2:.2f}, activity-adjusted)", x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.03)
save_fig(fig, "q6_04_dose_response")

# ---- Figure 5 (HERO): KM by early failures, landmark day 14
fig, ax = plt.subplots(figsize=(10.5, 5.8))
kcol = {"0": PALETTE["good"], "1": PALETTE["primary"], "2": PALETTE["warn"], "3+": PALETTE["accent"]}
for g_, (kmf, n_) in km_fail.items():
    sf = kmf.survival_function_
    ax.step(sf.index, 100 * sf.iloc[:, 0], where="post", color=kcol[g_], lw=2.3,
            label=f"{g_} early failure{'' if g_ == '1' else 's'} (n={n_:,})")
s180_txt = "Still active 180 days after the landmark:\n" + "\n".join(
    f"  {g_} early failure{'' if g_ == '1' else 's'}: {100*float(k_.survival_function_at_times(180).iloc[0]):.0f}%" for g_, (k_, _) in km_fail.items())
ax.text(0.70, 0.97, s180_txt, transform=ax.transAxes, va="top", fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
ax.axvline(180, color="#999999", lw=0.8, ls=":")
ax.set_xlim(0, 330)
ax.set_ylim(30, 101)
ax.set_xlabel("Days since day 14 (landmark; riders still active at day 14)")
ax.set_ylabel("Still active: no 30-day gap yet (%)")
title_sub(ax, "New riders with 2+ failures in their first two weeks churn faster for months afterwards",
          f"Kaplan-Meier, churn = 30+ idle days, administrative censoring. Log-rank p = {lr.p_value:.1e}; "
          f"Cox HR (2 vs 0 failures) = {CX('f2'):.2f} [{CX('f2','lo'):.2f}-{CX('f2','hi'):.2f}]")
ax.legend(loc="lower left")
save_fig(fig, "q6_05_km_by_early_failures")

# ---- Figure 6: decomposition
show_c = ["Apr-Jun 2024 (summer)", "Apr 2025 (latest)", "Jan-Mar 2025 (same season)"]
blocks_order = piv[show_c[0]].abs().sort_values(ascending=True).index.tolist()
fig, ax = plt.subplots(figsize=(12, 6))
hb = 0.26
for j, (c_, colr) in enumerate(zip(show_c, [PALETTE["accent"], SERIES[4], PALETTE["primary"]])):
    vals = piv.loc[blocks_order, c_]
    ys = np.arange(len(blocks_order)) + (1 - j) * hb
    ax.barh(ys, vals, height=hb, color=colr,
            label=f"{c_}: observed {gap_line.loc[c_, 'obs_gap_pp']:+.1f} pp, model {gap_line.loc[c_, 'pred_gap_pp']:+.1f} pp")
    for y_, v_ in zip(ys, vals):
        if abs(v_) >= 0.15:
            ax.text(v_ + (0.03 if v_ >= 0 else -0.03), y_, f"{v_:+.2f}", va="center", ha="left" if v_ >= 0 else "right", fontsize=9)
ax.axvline(0, color="#333333", lw=1)
ax.set_yticks(range(len(blocks_order)))
ax.set_yticklabels([b + ("" if block_p[b] < 0.05 else " (n.s.)") for b in blocks_order])
xl_ = ax.get_xlim()
ax.set_xlim(min(xl_[0], piv.loc[:, show_c].min().min() - 0.35), xl_[1])   # room for negative value labels
ax.set_xlabel("Contribution to change in NON-retention vs Jan-Mar 2024 cohorts (percentage points)")
title_sub(ax, f"Early failures are the largest measured contributor, but explain only {pct(fail_expl / summer_gap, 0)} "
              f"of the observed summer-2024 dip",
          f"Counterfactual block swaps on a pooled logit without cohort effects (30 draws); summer-2024 gap: observed "
          f"{summer_gap:+.2f} pp, modelled {summer_pred:+.2f} pp, failures {fail_expl:+.2f} pp. n.s. = joint Wald p>0.05")
ax.legend(loc="lower right", fontsize=9)
save_fig(fig, "q6_06_decline_decomposition")

# ---- Figure 7: SHAP importance (with activity control)
lab_map = {"e_att": "Early activity volume (control: symptom of early exit)", "e_stockout": "Stockouts (first 14 d)",
           "e_abandon": "Abandoned queues (first 14 d)", "e_syserr": "System errors", "e_cancel": "Cancellations",
           "first_fail": "First attempt failed", "q_mean": "Mean queue wait", "q_long_n": "Waits > 10 min",
           "kyron_share": "Kyron share of packs", "soh_out": "SoH of packs received",
           "km_mean": "km per swap (pack size + engagement proxy)",
           "price_mean": "Price paid per swap", "peak_share": "PEAK-tariff share", "comp_share": "Competitor-active share",
           "tk_n": "Tickets raised", "tk_unres": "Unresolved tickets", "w3": "3W vehicle", "kyc_no": "KYC not verified",
           "coh_idx": "Cohort month (trend)", "segment": "Segment (indep./partner/ZipDrop)", "signup_channel": "Signup channel",
           "age_band": "Age band", "main_city": "City", "main_wave": "Station wave", "main_gen": "Charger generation",
           "main_conn": "Connectivity tier", "declared_shift": "Declared shift"}
top = imp2.head(16).iloc[::-1]
fig, ax = plt.subplots(figsize=(10.5, 6.4))
cols_ = [PALETTE["muted"] if f == "e_att" else (PALETTE["accent"] if f in ("e_stockout", "e_abandon", "first_fail") else PALETTE["primary"]) for f in top["feature"]]
ax.barh([lab_map.get(f, f) for f in top["feature"]], top["mean_abs_shap"], color=cols_)
ax.set_xlabel("Mean |SHAP| (log-odds of not being retained)")
title_sub(ax, "Gradient boosting: stockouts are the top modifiable driver once early activity is controlled",
          f"LightGBM, 5-fold CV AUC {np.mean(aucs2):.2f} with activity control ({np.mean(aucs):.2f} drivers only). "
          "Continuous shares (km, SoH, queue) partly reflect few-swap noise")
save_fig(fig, "q6_07_shap_importance")

# ---- Figure 8: stability of the failure effect across segments
sb = stab.dropna(subset=["lo"]).iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(10, 6.4))
for i, r_ in sb.iterrows():
    c_ = PALETTE["accent"] if r_["p"] < 0.05 else PALETTE["muted"]
    ax.plot([r_["lo"], r_["hi"]], [i, i], color=c_, lw=2.2)
    ax.plot(r_["OR_2plus"], i, "o", color=c_, ms=7)
    ax.text(3.65, i, f"{100*r_['nr_0_1']:.0f}% -> {100*r_['nr_2p']:.0f}%  (n={int(r_['n']):,})", va="center", fontsize=9)
ax.axvline(1, color="#333333", lw=1)
ax.set_yticks(range(len(sb)))
ax.set_yticklabels(["All riders (pooled)" if s_ == "All" else s_ for s_ in sb["segment"]])
for t_ in ax.get_yticklabels():
    if t_.get_text().startswith("All riders"):
        t_.set_fontweight("bold")
ax.set_xscale("log")
ax.set_xticks([0.75, 1, 1.5, 2, 3])
ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlim(0.6, 6.5)
ax.set_xlabel("Adjusted odds ratio of not being retained, 2+ vs 0-1 early failures (95% CI, log scale)")
no_minor_labels(ax)
null_subs = sub_stab.loc[sub_stab["p"] >= 0.05, "segment"].tolist()
title_sub(ax, f"The failure effect holds in {n_stab_pos} of {n_sub} subgroups (significant in {n_stab_sig}"
              + (f"; {', '.join(null_subs)} null)" if null_subs else ")"),
          f"Logit within subgroup with cohort-quarter (+ city) fixed effects; pooled OR {st_all['OR_2plus']:.2f}. "
          "Right: non-retention 0-1 -> 2+ failures")
save_fig(fig, "q6_08_segment_stability")
print("Figures saved:", sorted(p.name for p in FIG.glob("q6_*.png")))
