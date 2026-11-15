# %% [markdown]
# # Q4 - Battery & equipment performance
# **Question.** How does battery health (SoH, charge cycles, supplier, manufacturing lot) relate to delivered range and
# swap frequency? Do any equipment or supplier cohorts stand out?
#
# **Approach.**
# 1. Cohort table supplier x pack type x lot (cost, initial SoH, cycles, degradation, BMS mix).
# 2. Degradation per pack, two ways: SoH lost per issue (the brief's per-cycle slope) and SoH lost per day in service
#    (each pack's own SoH-vs-calendar-time slope). We test which one drives fade (calendar ageing vs cycling), then
#    compare suppliers, lots and BMS firmware.
# 3. SoH trajectories by days in service (age measured from first appearance in the swap log, not the unreliable
#    commission date), plus an early-warning test: is the bad-lot fade visible in a pack's first 6-8 weeks?
# 4. Delivered range: km since last swap vs SoH of the returned pack. The main estimate uses each pack's de-noised SoH
#    (its own fitted SoH-vs-time line) within day x city, because single SoH readings are noisy and swap frequency moved.
# 5. Swap frequency: within-rider difference-in-differences around Kyron's arrival (Kyron-heavy vs Kyron-light cities),
#    placebos, a 3W negative control, a near-duplicate check, whether the gap tracks Kyron's fade, rider-week fixed
#    effects for the pack-level effect, and the network-wide extra swaps.
# 6. Kyron deep dive incl. the recorded 2025-06-15 retirement (did inventory actually fall?), with same-station checks.
# 7. Economics: wear cost per swap, excess wear vs a Cellora-like counterfactual under the brief's per-swap model AND
#    with depreciation capped at each pack's purchase price, purchase price, payback, lost-life value, replacement maths.
# 8. Thermal / quarantine check from station telemetry.
#
# Data: `data/clean/*` produced by `src/clean.py` (test stations removed, firmware timestamp bug fixed, km outliers
# nulled, SoC/SoH > 100 clipped, cost model attached).

# %%
import datetime as dt
import gc
import sys
import time
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # notebook
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from common import (EOL_SOH, FIG, PALETTE, SUPPLIER_COLORS, TAB, batteries, events, fmt_inr, hourly,  # noqa: E402
                    partners, save_fig, set_style, stations)

set_style()
T0 = time.time()
pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(24); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(40)
PTS = ["2W_2.1kWh", "3W_4.8kWh"]
PT_LABEL = {"2W_2.1kWh": "2W packs (2.1 kWh)", "3W_4.8kWh": "3W packs (4.8 kWh)"}
KYRON_ARRIVAL = dt.date(2024, 9, 10)
RECORDED_RETIREMENT = dt.date(2025, 6, 15)
PRE = (dt.date(2024, 8, 20), dt.date(2024, 9, 9))      # 3 weeks before Kyron arrival (Monsoon, after base-price change)
POST = (dt.date(2024, 9, 10), dt.date(2024, 9, 30))    # 3 weeks after (before the Oct-1 pricing pilot)
PLACEBO_PRE = (dt.date(2024, 7, 30), dt.date(2024, 8, 19))
PLACEBO_POST = (dt.date(2024, 8, 20), dt.date(2024, 9, 9))
POST_LONG = (dt.date(2024, 10, 1), dt.date(2025, 6, 14))  # Kyron fully in service
Q2_25 = (dt.date(2025, 4, 1), dt.date(2025, 6, 30))
D0 = dt.datetime(2024, 1, 1)                              # origin of calendar time t (days)
T_END = dt.datetime(2025, 7, 1)                           # end of the data window (for days in service)


def headline(fig, title, sub=None, top=0.86):
    """Insight title + grey subtitle at the top-left of the figure."""
    fig.subplots_adjust(top=top)
    fig.text(0.01, 0.995, title, fontsize=14.5, fontweight="bold", ha="left", va="top", color="#222222")
    if sub:
        fig.text(0.01, 0.945, sub, fontsize=10.5, ha="left", va="top", color="#555555")


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def fe_ols(df, y, xs, fes, cluster, iters=15):
    """OLS of y on xs after absorbing fixed effects `fes` (alternating projections); cluster-robust SEs."""
    cols = [y] + xs
    d = df.select(list(dict.fromkeys([*fes, cluster, *cols]))).drop_nulls()
    for _ in range(iters):
        for fe in fes:
            d = d.with_columns([(pl.col(c) - pl.col(c).mean().over(fe)) for c in cols])
    X = d.select(xs).to_numpy().astype(float)
    Y = d[y].to_numpy().astype(float)
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ Y
    e = Y - X @ beta
    sc = (d.select(cluster).with_columns([pl.Series(f"s{i}", X[:, i] * e) for i in range(len(xs))])
          .group_by(cluster).agg([pl.col(f"s{i}").sum() for i in range(len(xs))]))
    S = sc.select([f"s{i}" for i in range(len(xs))]).to_numpy()
    G, n, k = S.shape[0], len(Y), len(xs)
    V = xtx_inv @ (S.T @ S * (G / (G - 1)) * ((n - 1) / (n - k))) @ xtx_inv
    return beta, np.sqrt(np.diag(V)), n, G


# %% [markdown]
# ## 1. Cohorts: supplier x pack type x manufacturing lot
# Degradation is each pack's own OLS slope of issued SoH vs cycle index (from `src/clean.py`; every pack has >= 50
# cycles). A lot is flagged "fast-fade" if its median slope is > 1.5x the Cellora/Amptek median for the same pack type.

# %%
bt = batteries()
own = pl.col("soh_loss_per_cycle")
print("packs:", bt.height, "| packs with < 50 issued cycles:", bt.filter(pl.col("n_cycles") < 50).height, "| min issues per pack:", bt["n_cycles"].min())
ref = (bt.filter(pl.col("supplier").is_in(["Cellora", "Amptek"]))
       .group_by("pack_type").agg(own.median().alias("ref_loss")))
lot_med = (bt.group_by("supplier", "pack_type", "manufacturing_lot").agg(own.median().alias("loss_med"))
           .join(ref, on="pack_type").with_columns((pl.col("loss_med") / pl.col("ref_loss")).alias("ratio")))
FAST_LOTS = sorted(set(lot_med.filter(pl.col("ratio") > 1.5)["manufacturing_lot"].to_list()))
FAST = "Kyron " + "/".join(x[-4:] for x in FAST_LOTS)
LATE = "Kyron later lots"
print("Cellora/Amptek reference loss per cycle:", {r["pack_type"]: round(r["ref_loss"], 4) for r in ref.iter_rows(named=True)})
print("fast-fade lots (median slope > 1.5x reference):", FAST_LOTS)
print(lot_med.filter(pl.col("manufacturing_lot").is_in(FAST_LOTS) | (pl.col("supplier") == "Kyron"))
      .sort("pack_type", "manufacturing_lot"))

bt = bt.join(ref, on="pack_type").with_columns(
    pl.when(pl.col("supplier") != "Kyron").then(pl.col("supplier"))
    .when(pl.col("manufacturing_lot").is_in(FAST_LOTS)).then(pl.lit(FAST))
    .otherwise(pl.lit(LATE)).alias("cohort"))
COHORTS = ["Cellora", "Amptek", FAST, LATE]
COH_STYLE = {"Cellora": dict(color=SUPPLIER_COLORS["Cellora"], ls="-", lw=2.2, marker=None),
             "Amptek": dict(color=SUPPLIER_COLORS["Amptek"], ls="--", lw=2.2, marker=None),
             FAST: dict(color=SUPPLIER_COLORS["Kyron"], ls="-", lw=2.8, marker=None),
             LATE: dict(color=SUPPLIER_COLORS["Kyron"], ls=":", lw=2.4, marker=None)}

agg_cols = [
    pl.len().alias("packs"),
    pl.col("first_seen").min().dt.date().alias("first_seen_min"), pl.col("first_seen").max().dt.date().alias("first_seen_max"),
    pl.col("commission_date").min().alias("commission_min"), pl.col("commission_date").max().alias("commission_max"),
    pl.col("used_before_commission").mean().round(3).alias("share_used_before_commission"),
    pl.col("n_cycles").median().alias("cycles_med"),
    own.median().round(4).alias("loss_per_cycle_med"), own.quantile(0.25).round(4).alias("loss_p25"),
    own.quantile(0.75).round(4).alias("loss_p75"),
    pl.col("soh_first_obs").mean().round(2).alias("soh_at_first_issue"),
    pl.col("current_soh_pct").mean().round(2).alias("current_soh_mean"),
    pl.col("retired_date").is_not_null().sum().alias("retired_in_master"),
    pl.col("purchase_cost_inr").mean().round(0).alias("purchase_cost_mean"),
    pl.col("initial_soh_pct").mean().round(2).alias("initial_soh_mean"),
    pl.col("expected_cycles_to_eol").median().round(0).alias("cycles_to_eol_med"),
    pl.col("wear_cost_per_cycle").median().round(2).alias("wear_inr_per_cycle_med"),
    *[(pl.col("bms_firmware") == v).mean().round(3).alias(f"share_{v}") for v in ["BMS-4.0", "BMS-4.1", "BMS-4.2"]],
]
coh_lot = bt.group_by("supplier", "pack_type", "manufacturing_lot", "cohort").agg(agg_cols).sort("pack_type", "supplier", "manufacturing_lot")
coh_lot.write_csv(TAB / "q4_cohort_lots.csv")
coh_sup = bt.group_by("pack_type", "cohort").agg(agg_cols).sort("pack_type", "cohort")
coh_sup.write_csv(TAB / "q4_cohort_supplier.csv")
print(coh_sup.select("pack_type", "cohort", "packs", "first_seen_min", "first_seen_max", "cycles_med", "loss_per_cycle_med",
                     "soh_at_first_issue", "current_soh_mean", "retired_in_master", "purchase_cost_mean", "initial_soh_mean",
                     "cycles_to_eol_med", "wear_inr_per_cycle_med", "share_used_before_commission"))
print("packs used before their recorded commission date:", int(bt["used_before_commission"].fill_null(False).sum()), "of", bt.height)
k2 = bt.filter((pl.col("supplier") == "Kyron") & (pl.col("pack_type") == "2W_2.1kWh"))
print(f"Kyron 2W: {k2.height} packs, all first seen {k2['first_seen'].min().date()} .. {k2['first_seen'].max().date()}; "
      f"recorded retirement dates: {k2['retired_date'].unique().to_list()}; reason: {k2['retirement_reason'].unique().to_list()}")
n2w_other = bt.filter((pl.col("pack_type") == "2W_2.1kWh") & (pl.col("supplier") != "Kyron")).height
print(f"Kyron 2W added {k2.height:,} packs to a 2W fleet of {n2w_other:,} (+{pct(k2.height / n2w_other)}); "
      f"Kyron = {pct(k2.height / (k2.height + n2w_other))} of the 2W fleet after Sep-2024")
nk = bt.filter(~((pl.col("supplier") == "Kyron") & (pl.col("pack_type") == "2W_2.1kWh")))
print(f"all other {nk.height} packs first seen {nk['first_seen'].min().date()} .. {nk['first_seen'].max().date()} "
      f"(no other packs were added to the fleet during Jan-2024..Jun-2025)")
k3 = bt.filter((pl.col("supplier") == "Kyron") & (pl.col("pack_type") == "3W_4.8kWh"))
print("Kyron 3W retired in master by cohort:", k3.group_by("cohort").agg(pl.len().alias("packs"), pl.col("retired_date").is_not_null().sum().alias("retired"),
                                                                      pl.col("current_soh_pct").mean().round(1).alias("soh_now")).sort("cohort").to_dicts())
ky_all = bt.filter(pl.col("supplier") == "Kyron")
ky_bad = ky_all.filter(pl.col("cohort") == FAST)
print(f"lot-level defect coverage: {FAST} = {ky_bad.height:,} of {ky_all.height:,} Kyron packs ({pct(ky_bad.height / ky_all.height)}): "
      f"{ky_bad.filter(pl.col('pack_type') == '2W_2.1kWh').height:,} 2W (= every Kyron 2W pack) + {ky_bad.filter(pl.col('pack_type') == '3W_4.8kWh').height} 3W; "
      f"Kyron later lots = {ky_all.filter(pl.col('cohort') == LATE).height} packs, all 3W")

# %% [markdown]
# ### Tests: purchase cost, initial SoH, degradation by supplier, lot and BMS firmware

# %%
print("\n--- Was Kyron cheaper or healthier at purchase? (Welch t-test, Kyron vs Cellora+Amptek) ---")
for pt in PTS:
    k = bt.filter((pl.col("pack_type") == pt) & (pl.col("supplier") == "Kyron"))
    o = bt.filter((pl.col("pack_type") == pt) & (pl.col("supplier") != "Kyron"))
    for col in ["purchase_cost_inr", "initial_soh_pct"]:
        t = stats.ttest_ind(k[col].to_numpy(), o[col].to_numpy(), equal_var=False)
        print(f"{pt} {col}: Kyron {k[col].mean():,.2f} vs others {o[col].mean():,.2f} "
              f"(diff {k[col].mean() - o[col].mean():+,.2f}; Welch p={t.pvalue:.3f})")

print("\n--- Degradation per cycle: Kyron vs Cellora ---")
FADE_RATIO = {}
for pt in PTS:
    ce = bt.filter((pl.col("pack_type") == pt) & (pl.col("supplier") == "Cellora"))["soh_loss_per_cycle"].to_numpy()
    for coh in [FAST, LATE]:
        ky = bt.filter((pl.col("pack_type") == pt) & (pl.col("cohort") == coh))["soh_loss_per_cycle"].to_numpy()
        if len(ky) == 0:
            continue
        u = stats.mannwhitneyu(ky, ce)
        FADE_RATIO[(pt, coh)] = np.median(ky) / np.median(ce)
        print(f"{pt} {coh}: median {np.median(ky):.4f} vs Cellora {np.median(ce):.4f} pts/cycle -> "
              f"{np.median(ky) / np.median(ce):.2f}x (n={len(ky)} vs {len(ce)}; Mann-Whitney p={u.pvalue:.1e})")

print("\n--- Lot heterogeneity within each supplier x pack type (Kruskal-Wallis on per-pack slopes) ---")
kw_rows = []
for (sup, pt, coh), g in sorted(bt.group_by(["supplier", "pack_type", "cohort"]), key=lambda x: (x[0][1], x[0][0], x[0][2])):
    groups = [x["soh_loss_per_cycle"].to_numpy() for _, x in g.group_by("manufacturing_lot")]
    if len(groups) < 2:
        continue
    H, p = stats.kruskal(*groups)
    meds = [np.median(x) for x in groups]
    kw_rows.append(dict(supplier=sup, pack_type=pt, cohort=coh, lots=len(groups), H=round(H, 2), p=p,
                        lot_median_min=round(min(meds), 4), lot_median_max=round(max(meds), 4)))
    print(f"{pt} {coh:22s}: {len(groups):2d} lots, lot medians {min(meds):.4f}-{max(meds):.4f}, KW H={H:.1f}, p={p:.3f}")
k3_all = bt.filter((pl.col("supplier") == "Kyron") & (pl.col("pack_type") == "3W_4.8kWh"))
H, p = stats.kruskal(*[x["soh_loss_per_cycle"].to_numpy() for _, x in k3_all.group_by("manufacturing_lot")])
print(f"3W Kyron all 12 lots together: KW H={H:.1f}, p={p:.1e}  (the lot split, not the supplier, explains Kyron 3W)")

print("\n--- BMS firmware (expect null): OLS slope ~ lot FE + BMS version, type-II ANOVA ---")
bms_res = {}
pdf = bt.select("soh_loss_per_cycle", "manufacturing_lot", "pack_type", "bms_firmware").to_pandas()
for pt in PTS:
    m = smf.ols("soh_loss_per_cycle ~ C(manufacturing_lot) + C(bms_firmware)", data=pdf[pdf.pack_type == pt]).fit()
    an = sm.stats.anova_lm(m, typ=2)
    F, p = an.loc["C(bms_firmware)", "F"], an.loc["C(bms_firmware)", "PR(>F)"]
    b41 = m.params.get("C(bms_firmware)[T.BMS-4.1]", np.nan)
    b42 = m.params.get("C(bms_firmware)[T.BMS-4.2]", np.nan)
    base = pdf[pdf.pack_type == pt].soh_loss_per_cycle.median()
    bms_res[pt] = (F, p)
    print(f"{pt}: F={F:.2f}, p={p:.3f}; BMS-4.1 vs 4.0 {b41:+.5f}, BMS-4.2 vs 4.0 {b42:+.5f} pts/cycle "
          f"({100 * b41 / base:+.1f}% / {100 * b42 / base:+.1f}% of median slope)")

# %% [markdown]
# ## 2. SoH trajectories: per issue and per day in service
# Cycle index = n-th time the pack ID is issued in the swap log (age from first appearance). All non-Kyron-2W packs
# first appear in the first days of Jan-2024 at ~99% SoH, so there is no left-censoring of earlier wear and no
# survivorship (no pack disappears from the log before Jun-2025).
#
# **Calendar ageing or cycling?** If SoH falls per use, packs that happen to be issued more often should have lost more
# SoH by the same date. If it falls with time, how often a pack was issued should not matter, and the per-issue slope
# should shrink as issues rise. All Cellora 2W packs share the same service window, so they make a clean test.

# %%
cmap = bt.select("battery_id", "pack_type", "cohort")
iss = (events(["battery_out_id", "event_ts", "soh_out_pct", "is_completed"]).filter(pl.col("is_completed"))
       .select("battery_out_id", "event_ts", "soh_out_pct").sort("battery_out_id", "event_ts")
       .with_columns(pl.int_range(1, pl.len() + 1).over("battery_out_id").alias("cycle"),
                     pl.col("event_ts").dt.date().alias("date"),
                     ((pl.col("event_ts") - pl.lit(D0)).dt.total_seconds() / 86400.0).alias("t"),
                     pl.col("soh_out_pct").cast(pl.Float64).alias("soh_out_pct"))
       .with_columns((pl.col("t") - pl.col("t").min().over("battery_out_id")).alias("age_d"))
       .join(cmap.lazy(), left_on="battery_out_id", right_on="battery_id", how="left")
       .drop("event_ts")
       .collect(engine="streaming"))
print("issued swaps with a pack:", iss.height, "| unmatched pack IDs:", iss["cohort"].null_count())


def slope_expr(x, y, name):
    """Per-group OLS slope of y on x, sign flipped so that a positive value = SoH points lost per unit of x."""
    xm, ym = pl.col(x).mean(), pl.col(y).mean()
    return (-((pl.col(x) - xm) * (pl.col(y) - ym)).sum() / ((pl.col(x) - xm) ** 2).sum()).alias(name)


# each pack's own linear SoH-vs-calendar-time trajectory (also used to de-noise returned-pack SoH in section 3)
pk_t = (iss.group_by("battery_out_id")
        .agg(pl.len().alias("n_iss"), pl.col("t").mean().alias("tbar"), pl.col("soh_out_pct").mean().alias("ybar"),
             slope_expr("t", "soh_out_pct", "loss_per_day"), pl.col("t").min().alias("t_first"),
             pl.col("soh_out_pct").last().alias("soh_last_issue"))
        .rename({"battery_out_id": "battery_id"}))
bt = bt.join(pk_t, on="battery_id", how="left").with_columns(
    ((pl.col("initial_soh_pct") - EOL_SOH) / pl.col("loss_per_day")).alias("days_to_eol"),
    (pl.col("n_iss") / ((T_END - D0).total_seconds() / 86400 - pl.col("t_first"))).alias("issues_per_day"))
REF_T = {r["pack_type"]: r["v"] for r in bt.filter(pl.col("supplier").is_in(["Cellora", "Amptek"]))
         .group_by("pack_type").agg(pl.col("loss_per_day").median().alias("v")).iter_rows(named=True)}
print("Cellora/Amptek reference loss per DAY:", {k: round(v, 5) for k, v in REF_T.items()})
print("per-pack fade by cohort (medians): per issue, per day, days and issues to 70% SoH, issues per day in service:")
print(bt.group_by("pack_type", "cohort").agg(
    pl.len().alias("packs"), own.median().round(5).alias("loss_per_issue"), pl.col("loss_per_day").median().round(5).alias("loss_per_day"),
    pl.col("days_to_eol").median().round(0).alias("days_to_70"), pl.col("expected_cycles_to_eol").median().round(0).alias("issues_to_70"),
    pl.col("issues_per_day").median().round(3).alias("issues_per_day"), pl.col("n_iss").max().alias("max_issues"),
    pl.col("soh_last_issue").mean().round(1).alias("soh_at_last_issue")).sort("pack_type", "cohort"))
FADE_DAY = {}
for pt, coh in [("2W_2.1kWh", FAST), ("3W_4.8kWh", FAST), ("3W_4.8kWh", LATE)]:
    v = bt.filter((pl.col("pack_type") == pt) & (pl.col("cohort") == coh))["loss_per_day"].median()
    FADE_DAY[(pt, coh)] = v / REF_T[pt]
    print(f"per-DAY fade {pt} {coh}: {v:.4f} vs Cellora/Amptek {REF_T[pt]:.4f} pts/day -> {v / REF_T[pt]:.3f}x "
          f"(per issue: {FADE_RATIO[(pt, coh)]:.2f}x vs Cellora)")
d70 = {r["cohort"]: r["d"] for r in bt.filter(pl.col("pack_type") == "2W_2.1kWh").group_by("cohort")
       .agg(pl.col("days_to_eol").median().alias("d")).iter_rows(named=True)}
print(f"2W days to 70% SoH (median pack): Kyron {d70[FAST]:.0f} vs Cellora {d70['Cellora']:.0f}, Amptek {d70['Amptek']:.0f} "
      f"(Cellora/Amptek figure is a projection: none has reached 70%)")
c2w = bt.filter((pl.col("pack_type") == "2W_2.1kWh") & (pl.col("cohort") == "Cellora"))
print(f"Cellora 2W: most issues by any pack {c2w['n_iss'].max()} (issues per pack {c2w['n_iss'].min()}-{c2w['n_iss'].max()}); "
      f"mean SoH at each pack's last issue {c2w['soh_last_issue'].mean():.1f}% -> its ~{c2w['expected_cycles_to_eol'].median():,.0f}-cycle life is projected")
# calendar vs cycling test within Cellora 2W (identical service window)
t_end_d = (dt.datetime(2025, 6, 30) - D0).total_seconds() / 86400
c2w = c2w.with_columns((pl.col("initial_soh_pct") - (pl.col("ybar") - pl.col("loss_per_day") * (t_end_d - pl.col("tbar")))).alias("lost_by_end"))
ELAST = np.polyfit(np.log(c2w["n_iss"].to_numpy()), np.log(c2w["lost_by_end"].to_numpy()), 1)[0]
r_issue = stats.spearmanr(c2w["n_iss"].to_numpy(), c2w["soh_loss_per_cycle"].to_numpy()).statistic
r_day = stats.spearmanr(c2w["n_iss"].to_numpy(), c2w["loss_per_day"].to_numpy()).statistic
print(f"calendar vs cycling, Cellora 2W ({c2w.height:,} packs): elasticity of SoH lost by 30-Jun-25 w.r.t. issue count {ELAST:+.3f} "
      f"(1 = per-cycle wear, 0 = calendar ageing); Spearman(issues, loss per issue) {r_issue:+.2f} vs Spearman(issues, loss per day) {r_day:+.2f}")
for pt, coh in [("2W_2.1kWh", FAST), ("3W_4.8kWh", "Cellora")]:
    s = bt.filter((pl.col("pack_type") == pt) & (pl.col("cohort") == coh))
    print(f"   same test {pt} {coh}: Spearman(issues, loss per issue) {stats.spearmanr(s['n_iss'].to_numpy(), s['soh_loss_per_cycle'].to_numpy()).statistic:+.2f}, "
          f"Spearman(issues, loss per day) {stats.spearmanr(s['n_iss'].to_numpy(), s['loss_per_day'].to_numpy()).statistic:+.2f}")

# SoH by days in service (14-day bins) - the figure-1 curves
soh_day = (iss.with_columns((pl.col("age_d") // 14 * 14 + 7).alias("day_bin"))
           .group_by("pack_type", "cohort", "day_bin")
           .agg(pl.col("soh_out_pct").mean().alias("soh_mean"), pl.len().alias("n"), pl.col("battery_out_id").n_unique().alias("packs"))
           .filter(pl.col("n") >= 200).sort("pack_type", "cohort", "day_bin"))
soh_day.write_csv(TAB / "q4_soh_vs_days.csv")
traj_d = {}
for (pt, coh), g in sorted(soh_day.group_by(["pack_type", "cohort"]), key=lambda x: x[0]):
    g = g.filter(pl.col("packs") >= 0.5 * g["packs"].max())
    slope = -np.polyfit(g["day_bin"].to_numpy(), g["soh_mean"].to_numpy(), 1)[0]
    below = g.filter(pl.col("soh_mean") < EOL_SOH)
    traj_d[(pt, coh)] = dict(slope=slope, cross=below["day_bin"].min() if below.height else None, max_day=g["day_bin"].max(), last_soh=g["soh_mean"][-1])
    print(f"{pt} {coh:22s}: {slope * 100:.1f} SoH pts per 100 days (binned fit); mean SoH < 70% from day bin {traj_d[(pt, coh)]['cross']}; "
          f"last bin day {g['day_bin'].max():.0f} at {g['soh_mean'][-1]:.1f}%")

# early warning: is the bad-lot fade visible in a pack's first 6-8 weeks in service?
EARLY = {}
for wks in [6, 8]:
    e = (iss.filter(pl.col("age_d") < 7 * wks).group_by("battery_out_id").agg(slope_expr("t", "soh_out_pct", "b"), pl.len().alias("n"))
         .join(bt.select(pl.col("battery_id").alias("battery_out_id"), "supplier", "pack_type", "manufacturing_lot", "cohort"), on="battery_out_id"))
    eref = e.filter(pl.col("supplier").is_in(["Cellora", "Amptek"])).group_by("pack_type").agg(pl.col("b").median().alias("ref"))
    el = (e.group_by("pack_type", "cohort", "manufacturing_lot").agg(pl.col("b").median().alias("b_med"), pl.len().alias("packs"))
          .join(eref, on="pack_type").with_columns((pl.col("b_med") / pl.col("ref")).alias("ratio")))
    ep = e.join(eref, on="pack_type").with_columns((pl.col("b") > 1.5 * pl.col("ref")).alias("flag"))
    bad, oth = el.filter(pl.col("cohort") == FAST), el.filter(pl.col("cohort") != FAST)
    EARLY[wks] = dict(bad_lo=bad["ratio"].min(), bad_hi=bad["ratio"].max(), other_max=oth["ratio"].max(),
                      pack_tp=ep.filter(pl.col("cohort") == FAST)["flag"].mean(), pack_fp=ep.filter(pl.col("cohort") != FAST)["flag"].mean(),
                      n_med=e["n"].median())
    print(f"first {wks} weeks in service (median {e['n'].median():.0f} issues per pack): lot median per-day fade vs Cellora/Amptek in their own first "
          f"{wks} weeks -> KY-2407/08/09 lots {EARLY[wks]['bad_lo']:.2f}-{EARLY[wks]['bad_hi']:.2f}x, every other lot <= {EARLY[wks]['other_max']:.2f}x; "
          f"pack-level 1.5x rule flags {pct(EARLY[wks]['pack_tp'])} of bad-lot packs and {pct(EARLY[wks]['pack_fp'], 2)} of other packs")
    print(f"   first-{wks}-week reference (Cellora/Amptek median, pts/day): {dict(zip(eref['pack_type'].to_list(), [round(v, 4) for v in eref['ref'].to_list()]))}; "
          f"bad-lot ratios: {[(r['pack_type'][:2], r['manufacturing_lot'], round(r['ratio'], 2)) for r in bad.sort('pack_type', 'manufacturing_lot').iter_rows(named=True)]}")
# same calendar weeks for 2W (Kyron's first 8 weeks vs Cellora/Amptek over the same dates)
cw = (iss.filter((pl.col("pack_type") == "2W_2.1kWh") & pl.col("date").is_between(KYRON_ARRIVAL, KYRON_ARRIVAL + dt.timedelta(days=55)))
      .group_by("battery_out_id", "cohort").agg(slope_expr("t", "soh_out_pct", "b"))
      .group_by("cohort").agg(pl.col("b").median()).sort("cohort"))
cwv = {r["cohort"]: r["b"] for r in cw.iter_rows(named=True)}
EARLY_CONC = cwv[FAST] / np.median([cwv["Cellora"], cwv["Amptek"]])
print(f"2W over the same calendar weeks (10-Sep..4-Nov-2024): Kyron {cwv[FAST]:.4f} vs Cellora {cwv['Cellora']:.4f} / Amptek {cwv['Amptek']:.4f} pts/day "
      f"-> {EARLY_CONC:.2f}x")
soh_cyc = (iss.with_columns(((pl.col("cycle") - 1) // 20 * 20 + 10).alias("cycle_bin"))
           .group_by("pack_type", "cohort", "cycle_bin")
           .agg(pl.col("soh_out_pct").mean().alias("soh_mean"), pl.col("soh_out_pct").quantile(0.1).alias("soh_p10"),
                pl.col("soh_out_pct").quantile(0.9).alias("soh_p90"), pl.len().alias("n"),
                pl.col("battery_out_id").n_unique().alias("packs"))
           .filter(pl.col("n") >= 200).sort("pack_type", "cohort", "cycle_bin"))
soh_cyc.write_csv(TAB / "q4_soh_vs_cycles.csv")
traj = {}
for (pt, coh), g in sorted(soh_cyc.group_by(["pack_type", "cohort"]), key=lambda x: x[0]):
    g = g.filter(pl.col("packs") >= 0.5 * g["packs"].max())  # keep bins covered by most packs of the cohort
    slope = -np.polyfit(g["cycle_bin"].to_numpy(), g["soh_mean"].to_numpy(), 1)[0]
    below = g.filter(pl.col("soh_mean") < EOL_SOH)
    cross = below["cycle_bin"].min() if below.height else None
    traj[(pt, coh)] = dict(slope=slope, cross=cross, max_cycle=g["cycle_bin"].max(), last_soh=g["soh_mean"][-1])
    print(f"{pt} {coh:22s}: {slope * 100:.2f} SoH pts per 100 cycles (binned fit), mean SoH < 70% from cycle bin {cross}, "
          f"last bin {g['cycle_bin'].max()} at {g['soh_mean'][-1]:.1f}%")
r2 = FADE_RATIO[("2W_2.1kWh", FAST)]
print(f"Kyron 2W fade rate vs Cellora 2W: {r2:.2f}x per cycle (per-pack medians; binned-curve fit "
      f"{traj[('2W_2.1kWh', FAST)]['slope'] / traj[('2W_2.1kWh', 'Cellora')]['slope']:.2f}x)")

# measurement noise of SoH readings: residual SD of each pack's readings around its own linear trajectory
noise = (iss.select("battery_out_id", "pack_type", "cohort", pl.col("cycle").cast(pl.Float64).alias("x"), pl.col("soh_out_pct").cast(pl.Float64).alias("y"))
         .with_columns(pl.col("x").mean().over("battery_out_id").alias("mx"), pl.col("y").mean().over("battery_out_id").alias("my"))
         .with_columns((((pl.col("x") - pl.col("mx")) * (pl.col("y") - pl.col("my"))).sum().over("battery_out_id")
                        / ((pl.col("x") - pl.col("mx")) ** 2).sum().over("battery_out_id")).alias("b"))
         .with_columns((pl.col("y") - pl.col("my") - pl.col("b") * (pl.col("x") - pl.col("mx"))).alias("res"))
         .group_by("pack_type", "cohort").agg(pl.col("res").std().alias("res_sd")).sort("pack_type", "cohort"))
print("SD of SoH readings around each pack's own linear trajectory (reading noise):", {f"{r['pack_type']}|{r['cohort']}": round(r["res_sd"], 3) for r in noise.iter_rows(named=True)})
NOISE_SD_2W = noise.filter((pl.col("pack_type") == "2W_2.1kWh") & pl.col("cohort").is_in(["Cellora", "Amptek"]))["res_sd"].mean()

# sub-EOL issuance (packs issued below the 70% end-of-life threshold)
sub_eol = iss.filter(pl.col("soh_out_pct") < EOL_SOH)
print("swaps issuing a pack below 70% SoH:", sub_eol.height, "| by cohort:",
      sub_eol.group_by("pack_type", "cohort").len().sort("pack_type", "cohort").to_dicts())
print("first sub-EOL issue:", sub_eol["date"].min(), "| last:", sub_eol["date"].max())
eol_month = (sub_eol.with_columns(pl.col("date").dt.truncate("1mo").alias("month")).group_by("month").len().sort("month")
             .join(iss.with_columns(pl.col("date").dt.truncate("1mo").alias("month")).group_by("month").agg(pl.len().alias("all_swaps")), on="month")
             .with_columns((pl.col("len") / pl.col("all_swaps")).round(4).alias("share_of_all_swaps")).sort("month"))
print(eol_month)
q2_k2 = iss.filter(pl.col("date").is_between(*Q2_25) & (pl.col("cohort") == FAST) & (pl.col("pack_type") == "2W_2.1kWh"))
print(f"Q2-2025: {q2_k2.filter(pl.col('soh_out_pct') < EOL_SOH).height:,} of {q2_k2.height:,} Kyron 2W issues were below 70% SoH "
      f"({pct(q2_k2.filter(pl.col('soh_out_pct') < EOL_SOH).height / q2_k2.height)})")
jn_k2 = q2_k2.filter(pl.col("date") >= dt.date(2025, 6, 1))
print(f"Jun-2025: {jn_k2.filter(pl.col('soh_out_pct') < EOL_SOH).height:,} of {jn_k2.height:,} Kyron 2W issues were below 70% SoH "
      f"({pct(jn_k2.filter(pl.col('soh_out_pct') < EOL_SOH).height / jn_k2.height, 2)})")
daily_eol = sub_eol.group_by("date").len().sort("date")
kyr_daily = (iss.filter(pl.col("pack_type") == "2W_2.1kWh")
             .group_by("date").agg((pl.col("cohort") == FAST).sum().alias("kyron_issued"), pl.len().alias("swaps_2w"),
                                   pl.col("battery_out_id").filter(pl.col("cohort") == FAST).n_unique().alias("kyron_packs"),
                                   pl.col("soh_out_pct").filter(pl.col("cohort") == FAST).mean().alias("kyron_soh"),
                                   pl.col("soh_out_pct").filter(pl.col("cohort") != FAST).mean().alias("other_soh"))
             .sort("date"))
del iss, sub_eol
gc.collect()

# %% [markdown]
# ## 3. Delivered range: km since last swap vs SoH of the returned pack
# The returned pack's ID is *not* the pack issued to the rider at the previous swap (IDs are not physically tracked), but
# each row's SoH, SoC and km are internally consistent, so the dose-response is estimated on the returned pack in the
# same row. Only valid km (0-200 km). Three traps bite here:
# * **season** - range varies by month, and SoH falls steadily over calendar time -> compare within the same day (or
#   month) and city, or the same calendar month a year apart;
# * **swap frequency** - km *since last swap* shrinks mechanically when riders swap more often. Swap frequency jumped in
#   the Kyron-heavy cities (section 4) and also stepped up ~2% in the light cities when Kyron arrived, which inflates any
#   before/after comparison that spans Sep-2024 (incl. the year-on-year one) -> main estimate compares packs on the same
#   day in the same city;
# * **reading noise** - single SoH readings scatter ~0.5-0.7 pts around each pack's own trajectory, which attenuates
#   within-day/month regressions -> main estimate uses each pack's de-noised SoH (its own fitted SoH-vs-time line,
#   evaluated at the return time).

# %%
chain = (events(["is_completed", "rider_id", "event_ts", "battery_in_id", "battery_out_id"]).filter(pl.col("is_completed"))
         .select("rider_id", "event_ts", "battery_in_id", "battery_out_id").sort("rider_id", "event_ts")
         .with_columns(pl.col("battery_out_id").shift(1).over("rider_id").alias("prev_out"))
         .filter(pl.col("prev_out").is_not_null())
         .select((pl.col("battery_in_id") == pl.col("prev_out")).mean().alias("match"), pl.len().alias("n"))
         .collect(engine="streaming"))
print(f"returned pack == pack issued to the same rider at the previous swap in {pct(chain['match'][0], 2)} of {chain['n'][0]:,} "
      f"consecutive swap pairs -> pack IDs do not follow physical packs (per-pack SoH valid, per-pack location/rider not)")
city_ky = (events(["is_completed", "vehicle_class", "date", "city", "supplier_out"])
           .filter(pl.col("is_completed") & (pl.col("vehicle_class") == "2W") & pl.col("date").is_between(*POST_LONG))
           .group_by("city").agg((pl.col("supplier_out") == "Kyron").mean().alias("kyron_share_issued"), pl.len().alias("swaps"))
           .sort("city").collect())
HEAVY = sorted(city_ky.filter(pl.col("kyron_share_issued") > 0.25)["city"].to_list())
LIGHT = sorted(set(city_ky["city"].to_list()) - set(HEAVY))
print("Kyron share of issued 2W packs, Oct-24..14-Jun-25:", {r["city"]: round(r["kyron_share_issued"], 3) for r in city_ky.iter_rows(named=True)})
print("Kyron-heavy cities:", HEAVY, "| Kyron-light cities:", LIGHT)

# exposure: where was each pack ID issued? (hot Kyron-heavy cities vs the rest) - can lot, supplier and climate be separated?
expo = (events(["is_completed", "battery_out_id", "city"]).filter(pl.col("is_completed"))
        .group_by("battery_out_id").agg(pl.col("city").is_in(HEAVY).mean().alias("heavy_share"))
        .rename({"battery_out_id": "battery_id"}).collect(engine="streaming"))
bt = bt.join(expo, on="battery_id", how="left")
EXPO = {(r["pack_type"], r["cohort"]): r["heavy_share"] for r in bt.group_by("pack_type", "cohort").agg(pl.col("heavy_share").mean()).iter_rows(named=True)}
print("share of each cohort's issues made in the Kyron-heavy (hot) cities:", {f"{k[0][:2]} {k[1]}": round(v, 3) for k, v in sorted(EXPO.items())})
s_ = bt.filter((pl.col("pack_type") == "2W_2.1kWh") & (pl.col("cohort") == "Cellora"))
print(f"within Cellora 2W, Spearman(heavy-city share of issues, fade per issue) = {stats.spearmanr(s_['heavy_share'].to_numpy(), s_['soh_loss_per_cycle'].to_numpy()).statistic:+.3f} "
      f"(share sd {s_['heavy_share'].std():.3f}) -> exposure varies too little within a cohort to test climate; 3W (equal exposure) is the clean lot test")

rg = (events(["is_completed", "vehicle_class", "soh_in_pct", "soc_in_pct", "km_since_last_swap", "battery_in_id", "month", "city", "date", "event_ts"])
      .filter(pl.col("is_completed") & pl.col("soh_in_pct").is_not_null())
      .join(cmap.select("battery_id", pl.col("cohort").alias("cohort_in")).lazy(), left_on="battery_in_id", right_on="battery_id", how="left")
      .join(bt.select("battery_id", "tbar", "ybar", "loss_per_day").lazy(), left_on="battery_in_id", right_on="battery_id", how="left")
      .with_columns((pl.col("ybar") - pl.col("loss_per_day") * ((pl.col("event_ts") - pl.lit(D0)).dt.total_seconds() / 86400.0 - pl.col("tbar"))).alias("soh_fit"))
      .drop("battery_in_id", "is_completed", "event_ts", "tbar", "ybar", "loss_per_day").collect(engine="streaming"))
print(f"completed swaps with a returned pack: {rg.height:,}; km valid: {rg['km_since_last_swap'].is_not_null().sum():,} "
      f"({pct(rg['km_since_last_swap'].is_not_null().mean())})")
rg = rg.filter(pl.col("km_since_last_swap").is_not_null())
rng_bin = (rg.with_columns((pl.col("soh_in_pct") // 2 * 2 + 1).alias("soh_bin"), pl.col("city").is_in(LIGHT).alias("kyron_light_city"))
           .group_by("vehicle_class", "kyron_light_city", "cohort_in", "soh_bin")
           .agg(pl.col("km_since_last_swap").mean().alias("km_mean"), pl.col("km_since_last_swap").median().alias("km_median"),
                pl.col("soc_in_pct").mean().alias("soc_in_mean"), pl.len().alias("n"))
           .filter(pl.col("n") >= 300).sort("vehicle_class", "kyron_light_city", "cohort_in", "soh_bin"))
rng_bin.write_csv(TAB / "q4_range_vs_soh.csv")
for vc in ["2W", "3W"]:
    print(vc, "km by SoH bin, Kyron-light cities:", rng_bin.filter((pl.col("vehicle_class") == vc) & pl.col("kyron_light_city") & pl.col("soh_bin").is_in([61, 69, 79, 81, 89, 95, 97]))
          .select("cohort_in", "soh_bin", pl.col("km_mean").round(1), "n").to_dicts())
q2r = (rg.filter(pl.col("date").is_between(*Q2_25)).group_by("vehicle_class", "cohort_in")
       .agg(pl.col("km_since_last_swap").median().alias("km_median"), pl.col("km_since_last_swap").mean().alias("km_mean"),
            pl.col("soh_in_pct").mean().alias("soh_mean"), pl.len().alias("n")).sort("vehicle_class", "cohort_in"))
print("Q2-2025 km per swap by returned-pack cohort:\n", q2r)
q2v = {(r["vehicle_class"], r["cohort_in"]): r for r in q2r.iter_rows(named=True)}
print(f"Q2-2025 median km per swap, returned Kyron 2W pack {q2v[('2W', FAST)]['km_median']:.1f} vs Cellora {q2v[('2W', 'Cellora')]['km_median']:.1f} / "
      f"Amptek {q2v[('2W', 'Amptek')]['km_median']:.1f} -> {pct(q2v[('2W', FAST)]['km_median'] / q2v[('2W', 'Cellora')]['km_median'] - 1)} "
      f"(mean SoH {q2v[('2W', FAST)]['soh_mean']:.1f} vs {q2v[('2W', 'Cellora')]['soh_mean']:.1f})")
print("correlation km ~ SoC at return (2W):", round(rg.filter(pl.col("vehicle_class") == "2W").select(pl.corr("km_since_last_swap", "soc_in_pct")).item(), 4))

# (a) primary: same calendar month a year apart, Cellora/Amptek returned packs, Kyron-light cities
yoy, YOY = [], {}
for vc in ["2W", "3W"]:
    cc = (rg.filter((pl.col("vehicle_class") == vc) & pl.col("cohort_in").is_in(["Cellora", "Amptek"]) & pl.col("city").is_in(LIGHT))
          .group_by("month").agg(pl.col("km_since_last_swap").mean().alias("km"), pl.col("soh_in_pct").mean().alias("soh"), pl.len().alias("n"))
          .with_columns(pl.col("month").dt.month().alias("m"), pl.col("month").dt.year().alias("y")))
    a = (cc.filter(pl.col("y") == 2024).join(cc.filter(pl.col("y") == 2025), on="m", suffix="_25")
         .with_columns(((pl.col("km_25") - pl.col("km")) / (pl.col("soh_25") - pl.col("soh"))).alias("km_per_pt"),
                       pl.lit(vc).alias("vehicle_class")).sort("m"))
    YOY[vc] = dict(mean=a["km_per_pt"].mean(), lo=a["km_per_pt"].min(), hi=a["km_per_pt"].max(), km_new=a["km"][0], soh_new=a["soh"][0])
    yoy.append(a.select("vehicle_class", "m", pl.col("km").round(2), pl.col("km_25").round(2), pl.col("soh").round(2), pl.col("soh_25").round(2), pl.col("km_per_pt").round(3)))
    print(f"{vc} same-season (Jan-Jun 2024 vs 2025, Kyron-light cities, Cellora/Amptek): {YOY[vc]['mean']:.2f} km per SoH point "
          f"(monthly range {YOY[vc]['lo']:.2f}-{YOY[vc]['hi']:.2f}); = {100 * YOY[vc]['mean'] / YOY[vc]['km_new']:.2f}% of Jan-24 range "
          f"({YOY[vc]['km_new']:.1f} km at {YOY[vc]['soh_new']:.1f}% SoH) per point")
    print(a.select("m", "km", "km_25", "soh", "soh_25", "km_per_pt"))
pl.concat(yoy).write_csv(TAB / "q4_range_same_season.csv")

# why the year-on-year slope is too steep: light-city swap frequency stepped up when Kyron arrived
rdl = (events(["is_completed", "vehicle_class", "rider_id", "event_ts", "date", "city"]).filter(pl.col("is_completed"))
       .group_by("vehicle_class", "rider_id", "date").agg(pl.len().alias("n"), pl.col("city").sort_by("event_ts").first().alias("city"))
       .filter(pl.col("city").is_in(LIGHT)).with_columns(pl.col("vehicle_class").cast(pl.Utf8)))
lt_w = rdl.group_by("vehicle_class", pl.col("date").dt.truncate("1w").alias("week")).agg(pl.col("n").mean().alias("spd")).collect(engine="streaming")
lt_m = rdl.group_by("vehicle_class", pl.col("date").dt.truncate("1mo").alias("month")).agg(pl.col("n").mean().alias("spd")).collect(engine="streaming")
STEP = {}
for vc in ["2W", "3W"]:
    w_ = lt_w.filter(pl.col("vehicle_class") == vc)
    pre_, post_ = (w_.filter(pl.col("week").is_between(dt.date(2024, 8, 5), dt.date(2024, 9, 2)))["spd"].mean(),
                   w_.filter(pl.col("week").is_between(dt.date(2024, 9, 16), dt.date(2024, 10, 7)))["spd"].mean())
    STEP[vc] = (pre_, post_)
    print(f"light cities, {vc} swaps per active rider-day, 4 weeks before vs after Kyron's arrival: {pre_:.3f} -> {post_:.3f} ({pct(post_ / pre_ - 1)})")
lm_ = lt_m.with_columns(pl.col("month").dt.month().alias("m"), pl.col("month").dt.year().alias("y"))
spd_yoy = (lm_.filter(pl.col("y") == 2024).select("vehicle_class", "m", pl.col("spd").alias("spd24"))
           .join(lm_.filter(pl.col("y") == 2025).select("vehicle_class", "m", pl.col("spd").alias("spd25")), on=["vehicle_class", "m"]))
yy = (pl.concat(yoy).join(spd_yoy, on=["vehicle_class", "m"])
      .with_columns((pl.col("km_25") * pl.col("spd25") / pl.col("spd24")).alias("km_25_adj"))
      .with_columns(((pl.col("km_25_adj") - pl.col("km")) / (pl.col("soh_25") - pl.col("soh"))).alias("km_per_pt_adj")))
YOY_ADJ = {}
for vc in ["2W", "3W"]:
    g_ = yy.filter(pl.col("vehicle_class") == vc)
    YOY_ADJ[vc] = g_["km_per_pt_adj"].mean()
    print(f"{vc} year-on-year, light cities: swaps per active day Jan-Jun {g_['spd24'].mean():.3f} (2024) vs {g_['spd25'].mean():.3f} (2025), "
          f"{pct(g_['spd25'].mean() / g_['spd24'].mean() - 1)}; raw slope {YOY[vc]['mean']:.2f} km/pt -> holding km per active day fixed "
          f"(2025 km x swap-frequency ratio) {YOY_ADJ[vc]:.2f} km/pt")
del rdl


def within_slope(df, x, groups):
    """Fixed-effects slope of km on x within `groups` (demeaned OLS), plus the within-group variance of x."""
    d = df.select(x, "km_since_last_swap", *groups).drop_nulls().with_columns(
        (pl.col(x) - pl.col(x).mean().over(groups)).alias("xd"),
        (pl.col("km_since_last_swap") - pl.col("km_since_last_swap").mean().over(groups)).alias("yd"))
    return float((d["xd"] * d["yd"]).sum() / (d["xd"] ** 2).sum()), float((d["xd"] ** 2).mean()), d.height


# (a') main estimate: de-noised SoH (each pack's own fitted SoH-vs-time line at the return time), Cellora/Amptek
# returned packs, compared within the same day and city (and, as a check, the same month and city)
DN, dn_rows = {}, []
for vc in ["2W", "3W"]:
    base_ = rg.filter((pl.col("vehicle_class") == vc) & pl.col("cohort_in").is_in(["Cellora", "Amptek"]))
    for lab, sub_ in [("all cities", base_), ("light cities", base_.filter(pl.col("city").is_in(LIGHT)))]:
        s_noisy, v_noisy, n_ = within_slope(sub_, "soh_in_pct", ["month", "city"])
        s_fit_m, v_fit, _ = within_slope(sub_, "soh_fit", ["month", "city"])
        s_fit_d, _, _ = within_slope(sub_, "soh_fit", ["date", "city"])
        DN[(vc, lab)] = dict(day=s_fit_d, month=s_fit_m, noisy=s_noisy, rel=v_fit / v_noisy, n=n_)
        dn_rows.append(dict(vehicle_class=vc, cities=lab, km_per_pt_denoised_day_x_city=round(s_fit_d, 3),
                            km_per_pt_denoised_month_x_city=round(s_fit_m, 3), km_per_pt_raw_reading_month_x_city=round(s_noisy, 3),
                            reliability_raw_reading=round(v_fit / v_noisy, 3), swaps=n_))
        print(f"{vc} {lab}: de-noised SoH -> {s_fit_d:.2f} km per SoH point within day x city, {s_fit_m:.2f} within month x city "
              f"(raw single reading {s_noisy:.2f}: attenuated, reliability of raw readings {v_fit / v_noisy:.2f}); n={n_:,}")
pl.DataFrame(dn_rows).write_csv(TAB / "q4_range_denoised.csv")
PROP = {vc: YOY[vc]["km_new"] / YOY[vc]["soh_new"] for vc in ["2W", "3W"]}
for vc in ["2W", "3W"]:
    lo_, hi_ = min(DN[(vc, c)]["day"] for c in ["all cities", "light cities"]), max(DN[(vc, c)]["month"] for c in ["all cities", "light cities"])
    print(f"{vc}: range proportional to SoH predicts {PROP[vc]:.2f} km per point ({YOY[vc]['km_new']:.1f} km at {YOY[vc]['soh_new']:.1f}% SoH); "
          f"de-noised estimates {lo_:.2f}-{hi_:.2f} km = {100 * lo_ / YOY[vc]['km_new']:.2f}-{100 * hi_ / YOY[vc]['km_new']:.2f}% of new-pack range per point")

# (c) Kyron vs Cellora/Amptek at the same SoH: matched within city x month x 1-pt SoH bin (>= 30 swaps per side)
cells = (rg.with_columns(pl.col("soh_in_pct").floor().alias("sb"))
         .with_columns(pl.when(pl.col("cohort_in").is_in(["Cellora", "Amptek"])).then(pl.lit("ref")).otherwise(pl.col("cohort_in")).alias("g"))
         .group_by("vehicle_class", "city", "month", "sb", "g").agg(pl.col("km_since_last_swap").mean().alias("km"), pl.len().alias("n")))
MATCH = {}
for vc, coh in [("2W", FAST), ("3W", FAST), ("3W", LATE)]:
    mt = (cells.filter((pl.col("vehicle_class") == vc) & (pl.col("g") == coh))
          .join(cells.filter((pl.col("vehicle_class") == vc) & (pl.col("g") == "ref")), on=["city", "month", "sb"], suffix="_ref")
          .filter((pl.col("n") >= 30) & (pl.col("n_ref") >= 30)))
    pen = float(((mt["km"] - mt["km_ref"]) * mt["n"]).sum() / mt["n"].sum())
    rel = pen / float((mt["km_ref"] * mt["n"]).sum() / mt["n"].sum())
    MATCH[(vc, coh)] = (pen, rel, int(mt["n"].sum()), mt.height)
    pooled = (cells.filter((pl.col("vehicle_class") == vc) & (pl.col("g") == coh)).group_by("month", "sb").agg((pl.col("km") * pl.col("n")).sum().alias("kn"), pl.col("n").sum())
              .join(cells.filter((pl.col("vehicle_class") == vc) & (pl.col("g") == "ref")).group_by("month", "sb").agg((pl.col("km") * pl.col("n")).sum().alias("kn"), pl.col("n").sum()), on=["month", "sb"], suffix="_ref")
              .filter((pl.col("n") >= 100) & (pl.col("n_ref") >= 100)))
    pen_pooled = float(((pooled["kn"] / pooled["n"] - pooled["kn_ref"] / pooled["n_ref"]) * pooled["n"]).sum() / pooled["n"].sum())
    print(f"{vc} {coh} vs Cellora/Amptek at equal SoH: matched within city x month x SoH {pen:+.2f} km ({pct(rel)}; {MATCH[(vc, coh)][2]:,} swaps, "
          f"{mt.height} cells) | NOT matching on city: {pen_pooled:+.2f} km (city-mix confounding)")
# same matching on the de-noised SoH (is the Kyron gap an artefact of noisier Kyron readings?)
cells_f = (rg.with_columns(pl.col("soh_fit").floor().alias("sb"))
           .with_columns(pl.when(pl.col("cohort_in").is_in(["Cellora", "Amptek"])).then(pl.lit("ref")).otherwise(pl.col("cohort_in")).alias("g"))
           .filter(pl.col("vehicle_class") == "2W")
           .group_by("city", "month", "sb", "g").agg(pl.col("km_since_last_swap").mean().alias("km"), pl.len().alias("n")))
mt_f = (cells_f.filter(pl.col("g") == FAST).join(cells_f.filter(pl.col("g") == "ref"), on=["city", "month", "sb"], suffix="_ref")
        .filter((pl.col("n") >= 30) & (pl.col("n_ref") >= 30)))
pen_f = float(((mt_f["km"] - mt_f["km_ref"]) * mt_f["n"]).sum() / mt_f["n"].sum())
print(f"2W {FAST} vs Cellora/Amptek matched on DE-NOISED SoH within city x month: {pen_f:+.2f} km "
      f"({pct(pen_f / float((mt_f['km_ref'] * mt_f['n']).sum() / mt_f['n'].sum()))}; {int(mt_f['n'].sum()):,} swaps)")
del cells_f

# (b) robustness: OLS with month + city FE (attenuated by SoH reading noise) and a reliability correction
reg_rows, reg = [], {}
for vc in ["2W", "3W"]:
    s = rg.filter(pl.col("vehicle_class") == vc)
    s = s.sort("month", "city", "soh_in_pct", "soc_in_pct", "km_since_last_swap", "cohort_in", nulls_last=True).sample(n=min(250_000, s.height), seed=42).with_columns(
        (pl.col("cohort_in") == FAST).cast(pl.Int8).alias("ky_fast"), (pl.col("cohort_in") == LATE).cast(pl.Int8).alias("ky_late"),
        (pl.col("soh_in_pct") - 95).alias("soh_c"), pl.col("month").cast(pl.Utf8).alias("mo"))
    d = s.select("km_since_last_swap", "soh_c", "soc_in_pct", "ky_fast", "ky_late", "mo", "city").to_pandas().rename(columns={"km_since_last_swap": "km"})
    f = "km ~ soh_c * ky_fast + soc_in_pct + C(mo) + C(city)" + (" + ky_late" if d.ky_late.sum() > 0 else "")
    m = smf.ols(f, data=d).fit(cov_type="HC1")
    ci = m.conf_int()
    b_soh, b_ky, b_int = m.params["soh_c"], m.params["ky_fast"], m.params["soh_c:ky_fast"]
    V = m.cov_params()
    pen80 = b_ky + (80 - 95) * b_int
    se80 = np.sqrt(V.loc["ky_fast", "ky_fast"] + 15 ** 2 * V.loc["soh_c:ky_fast", "soh_c:ky_fast"] - 2 * 15 * V.loc["ky_fast", "soh_c:ky_fast"])
    km100 = float(m.predict(d.assign(soh_c=5.0, ky_fast=0, ky_late=0, soc_in_pct=d.soc_in_pct.mean())).mean())
    reg[vc] = dict(b_soh=b_soh, b_soh_lo=ci.loc["soh_c", 0], b_soh_hi=ci.loc["soh_c", 1], ky95=b_ky, ky95_lo=ci.loc["ky_fast", 0],
                   ky95_hi=ci.loc["ky_fast", 1], ky_slope=b_soh + b_int, pen80=pen80, pen80_se=se80, soc=m.params["soc_in_pct"],
                   soc_lo=ci.loc["soc_in_pct", 0], soc_hi=ci.loc["soc_in_pct", 1], r2=m.rsquared, n=int(m.nobs), km100=km100,
                   ky_late=m.params.get("ky_late", np.nan), ky_late_lo=ci.loc["ky_late", 0] if "ky_late" in ci.index else np.nan,
                   ky_late_hi=ci.loc["ky_late", 1] if "ky_late" in ci.index else np.nan)
    r = reg[vc]
    print(f"\n{vc} OLS (n={r['n']:,}, R2={r['r2']:.3f}, month+city FE, HC1): Cellora/Amptek {r['b_soh']:.3f} km per SoH point "
          f"[{r['b_soh_lo']:.3f}, {r['b_soh_hi']:.3f}] = {100 * r['b_soh'] / r['km100']:.2f}% of new-pack range ({r['km100']:.1f} km at 100% SoH) per point; "
          f"Kyron {FAST} slope {r['ky_slope']:.3f} km/pt; Kyron penalty at equal SoH 95%: {r['ky95']:+.2f} km [{r['ky95_lo']:+.2f}, {r['ky95_hi']:+.2f}], "
          f"at SoH 80%: {r['pen80']:+.2f} km (se {r['pen80_se']:.2f}); SoC-at-return coefficient {r['soc']:+.4f} km per pt [{r['soc_lo']:+.4f}, {r['soc_hi']:+.4f}]"
          + (f"; Kyron later lots vs Cellora/Amptek at equal SoH: {r['ky_late']:+.2f} km [{r['ky_late_lo']:+.2f}, {r['ky_late_hi']:+.2f}]" if vc == "3W" else ""))
    reg_rows.append(dict(vehicle_class=vc, **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
pl.DataFrame(reg_rows).write_csv(TAB / "q4_range_regression.csv")
print(f"2W month+city FE OLS on the raw reading ({reg['2W']['b_soh']:.3f}) is attenuated by reading noise (issue-reading noise SD {NOISE_SD_2W:.2f} pts; "
      f"reliability of raw readings within month x city {DN[('2W', 'all cities')]['rel']:.2f}); the de-noised estimate within month x city is "
      f"{DN[('2W', 'all cities')]['month']:.2f}, within day x city {DN[('2W', 'all cities')]['day']:.2f}")
del rg, cells
gc.collect()

# %% [markdown]
# ## 4. Swap frequency: do riders on Kyron / degraded packs swap more often?
# Kyron was not spread evenly: the share of Kyron among issued 2W packs differs sharply by city. This gives a natural
# experiment around the overnight arrival of 1,385 Kyron packs on 2024-09-10: **within-rider difference-in-differences**
# (Kyron-heavy vs Kyron-light cities, 3 weeks before vs after, same Monsoon season, no pricing change in either window),
# a placebo on the 3 earlier weeks, a 6-city dose-response, and a rider-week FE panel for the pack-level effect.

# %%
sw = (events(["is_completed", "vehicle_class", "rider_id", "event_ts", "date", "city", "supplier_out", "supplier_in",
              "soh_out_pct", "soh_in_pct", "km_since_last_swap", "revenue"])
      .filter(pl.col("is_completed") & (pl.col("vehicle_class") == "2W")).drop("is_completed", "vehicle_class")
      .sort("rider_id", "event_ts").collect(engine="streaming"))
grp_share = (sw.filter(pl.col("date").is_between(*POST_LONG)).with_columns(pl.col("city").is_in(HEAVY).alias("heavy"))
             .group_by("heavy").agg((pl.col("supplier_out") == "Kyron").mean().alias("ky")).sort("heavy"))
KY_SHARE = {r["heavy"]: r["ky"] for r in grp_share.iter_rows(named=True)}
print(f"Kyron share of issued 2W packs: heavy {pct(KY_SHARE[True])}, light {pct(KY_SHARE[False])}")

rd = (sw.group_by("rider_id", "date")
      .agg(pl.len().alias("n"), pl.col("city").first().alias("city"), (pl.col("supplier_out").first() == "Kyron").alias("first_kyron"),
           (pl.col("supplier_out") == "Kyron").sum().alias("n_ky"), pl.col("soh_out_pct").mean().alias("soh_out"),
           pl.col("km_since_last_swap").mean().alias("km_mean"), pl.col("km_since_last_swap").sum().alias("km_day"),
           pl.col("revenue").sum().alias("rev"))
      .with_columns(pl.col("city").is_in(HEAVY).alias("heavy")))
print(f"2W active rider-days: {rd.height:,}")

wk = (rd.with_columns(pl.col("date").dt.truncate("1w").alias("week")).group_by("week", "heavy")
      .agg(pl.col("n").mean().alias("swaps_per_active_day"), pl.col("km_mean").mean().alias("km_per_swap"), pl.len().alias("rider_days"))
      .sort("week", "heavy"))
wk.write_csv(TAB / "q4_swapfreq_weekly.csv")


def rider_did(pre, post, label, frame=None):
    frame = rd if frame is None else frame
    a = (frame.filter(pl.col("date").is_between(*pre)).group_by("rider_id")
         .agg(pl.col("n").mean().alias("pre"), pl.len().alias("d_pre"), pl.col("city").mode().sort().first().alias("city"),
              pl.col("km_mean").mean().alias("km_pre"), pl.col("km_day").mean().alias("kmd_pre")))
    b = (frame.filter(pl.col("date").is_between(*post)).group_by("rider_id")
         .agg(pl.col("n").mean().alias("post"), pl.len().alias("d_post"), pl.col("km_mean").mean().alias("km_post"),
              pl.col("km_day").mean().alias("kmd_post")))
    ab = (a.join(b, on="rider_id").filter((pl.col("d_pre") >= 3) & (pl.col("d_post") >= 3))
          .with_columns((pl.col("post") - pl.col("pre")).alias("delta"), (pl.col("km_post") - pl.col("km_pre")).alias("dkm"),
                        (pl.col("kmd_post") - pl.col("kmd_pre")).alias("dkmd"), pl.col("city").is_in(HEAVY).cast(pl.Int8).alias("heavy")))
    m = smf.ols("delta ~ heavy", data=ab.select("delta", "heavy").to_pandas()).fit(cov_type="HC1")
    ci = m.conf_int().loc["heavy"]
    g = ab.group_by("heavy").agg(pl.col("pre").mean(), pl.col("post").mean(), pl.col("delta").mean(), pl.col("km_pre").mean(), pl.col("dkm").mean(),
                                 pl.col("kmd_pre").mean(), pl.col("dkmd").mean(), pl.len().alias("riders")).sort("heavy")
    print(f"{label}: within-rider DiD (heavy - light) = {m.params['heavy']:+.4f} swaps per active rider-day "
          f"[{ci[0]:+.4f}, {ci[1]:+.4f}], p={m.pvalues['heavy']:.1e}, riders={ab.height:,}")
    print(g)
    return m.params["heavy"], ci[0], ci[1], m.pvalues["heavy"], ab, g


did, did_lo, did_hi, did_p, ab_main, g_main = rider_did(PRE, POST, "Kyron arrival (20 Aug-9 Sep vs 10-30 Sep 2024)")
pl_did, pl_lo, pl_hi, pl_p, _, _ = rider_did(PLACEBO_PRE, PLACEBO_POST, "Placebo (30 Jul-19 Aug vs 20 Aug-9 Sep 2024)")
re_did, re_lo, re_hi, re_p, _, g_ret = rider_did((dt.date(2025, 5, 25), dt.date(2025, 6, 14)), (dt.date(2025, 6, 15), dt.date(2025, 6, 30)),
                                                 "Recorded retirement (25 May-14 Jun vs 15-30 Jun 2025)")
pl2 = rider_did((dt.date(2024, 4, 1), dt.date(2024, 4, 21)), (dt.date(2024, 4, 22), dt.date(2024, 5, 12)), "Extra placebo (Apr/May-2024)")
pl3 = rider_did((dt.date(2024, 6, 1), dt.date(2024, 6, 21)), (dt.date(2024, 6, 22), dt.date(2024, 7, 12)), "Extra placebo (Jun/Jul-2024, spans price change)")
# negative control: 3W riders in the same cities and windows (no 3W packs arrived on 10-Sep-2024)
rd3 = (events(["is_completed", "vehicle_class", "rider_id", "event_ts", "date", "city", "km_since_last_swap"])
       .filter(pl.col("is_completed") & (pl.col("vehicle_class") == "3W") & pl.col("date").is_between(PRE[0], POST[1]))
       .group_by("rider_id", "date").agg(pl.len().alias("n"), pl.col("city").sort_by("event_ts").first().alias("city"),
                                         pl.col("km_since_last_swap").mean().alias("km_mean"), pl.col("km_since_last_swap").sum().alias("km_day"))
       .collect(engine="streaming"))
nc_did, nc_lo, nc_hi, nc_p, _, _ = rider_did(PRE, POST, "NEGATIVE CONTROL: 3W riders, same cities and windows", frame=rd3)
del rd3
# are the extra heavy-city swaps real visits or near-duplicate records (offline-sync retries)?
dup = (events(["is_completed", "vehicle_class", "rider_id", "event_ts", "date", "city", "km_since_last_swap", "station_id"])
       .filter(pl.col("is_completed") & (pl.col("vehicle_class") == "2W") & pl.col("date").is_between(PRE[0] - dt.timedelta(days=7), POST[1]))
       .sort("rider_id", "event_ts")
       .with_columns(((pl.col("event_ts") - pl.col("event_ts").shift(1).over("rider_id")).dt.total_seconds() / 60).alias("gap_min"),
                     (pl.col("station_id") == pl.col("station_id").shift(1).over("rider_id")).alias("same_stn"))
       .filter(pl.col("date") >= PRE[0])
       .with_columns(pl.col("city").is_in(HEAVY).alias("heavy"), (pl.col("date") >= KYRON_ARRIVAL).alias("post"))
       .group_by("heavy", "post").agg(pl.len().alias("swaps"), (pl.col("km_since_last_swap") < 5).mean().round(4).alias("km_lt5"),
                                      pl.col("km_since_last_swap").median().alias("km_p50"), pl.col("km_since_last_swap").quantile(0.1).alias("km_p10"),
                                      pl.col("gap_min").median().alias("gap_med_min"),
                                      ((pl.col("gap_min") < 30) & pl.col("same_stn")).mean().round(4).alias("same_stn_lt30m"))
       .sort("heavy", "post").collect(engine="streaming"))
print("near-duplicate check, 2W swaps 20-Aug..30-Sep-2024:\n", dup)
dv = {(r["heavy"], r["post"]): r for r in dup.iter_rows(named=True)}
print(f"heavy cities before -> after Kyron: km < 5 share {dv[(True, False)]['km_lt5']:.4f} -> {dv[(True, True)]['km_lt5']:.4f}; median km "
      f"{dv[(True, False)]['km_p50']:.1f} -> {dv[(True, True)]['km_p50']:.1f}; median gap to previous swap {dv[(True, False)]['gap_med_min']:,.0f} -> "
      f"{dv[(True, True)]['gap_med_min']:,.0f} min; same-station re-swaps < 30 min {pct(dv[(True, False)]['same_stn_lt30m'], 2)} -> "
      f"{pct(dv[(True, True)]['same_stn_lt30m'], 2)} -> real, earlier visits, not duplicate records")
base_heavy = g_main.filter(pl.col("heavy") == 1)["pre"].item()
print(f"Kyron-arrival DiD as % of heavy-city pre level ({base_heavy:.3f}): {pct(did / base_heavy)}")
gh, gl = g_main.filter(pl.col("heavy") == 1).row(0, named=True), g_main.filter(pl.col("heavy") == 0).row(0, named=True)
print(f"km per swap change (within rider): heavy {gh['dkm']:+.2f} km (from {gh['km_pre']:.1f}), light {gl['dkm']:+.2f} km (from {gl['km_pre']:.1f}) "
      f"-> DiD {gh['dkm'] - gl['dkm']:+.2f} km per swap")
print(f"km per active day (within rider): heavy {gh['kmd_pre']:.1f} {gh['dkmd']:+.2f} ({pct(gh['dkmd'] / gh['kmd_pre'])}), "
      f"light {gl['kmd_pre']:.1f} {gl['dkmd']:+.2f} ({pct(gl['dkmd'] / gl['kmd_pre'])}) -> same daily distance, more swaps")
ca = (sw.filter(pl.col("date").is_between(PRE[0], POST[1]) & (pl.col("supplier_in") != "Kyron") & pl.col("km_since_last_swap").is_not_null())
      .with_columns(pl.col("city").is_in(HEAVY).alias("heavy"), (pl.col("date") >= KYRON_ARRIVAL).alias("post"))
      .group_by("heavy", "post").agg(pl.col("km_since_last_swap").mean().alias("km"), pl.col("soh_in_pct").mean().alias("soh")).sort("heavy", "post"))
cv = {(r["heavy"], r["post"]): r for r in ca.iter_rows(named=True)}
print(f"Cellora/Amptek returned packs only: km per swap heavy {cv[(True, False)]['km']:.1f} -> {cv[(True, True)]['km']:.1f} "
      f"(SoH {cv[(True, False)]['soh']:.1f} -> {cv[(True, True)]['soh']:.1f}), light {cv[(False, False)]['km']:.1f} -> {cv[(False, True)]['km']:.1f} "
      f"-> km falls for every supplier where swapping became more frequent")
# anything else that changed on/near 10-Sep-2024 in the heavy cities?
fp = partners()
near = lambda c: pl.col(c).is_between(PRE[0], POST[1])
print("fleet-partner contract starts/amendments 20-Aug..30-Sep-2024:",
      fp.filter(near("contract_start_date") | near("amendment_date")).select("partner_id", "contract_start_date", "amendment_date").to_dicts())
stn = stations().filter(~pl.col("is_test"))
print("stations commissioned 20-Aug..30-Sep-2024 by city group:",
      stn.filter(near("commissioned_date")).with_columns(pl.col("city").is_in(HEAVY).alias("heavy")).group_by("heavy").len().sort("heavy").to_dicts())

# 6-city dose-response: within-rider change vs Kyron share of issued packs in the POST window
post_share = (sw.filter(pl.col("date").is_between(*POST)).group_by("city")
              .agg((pl.col("supplier_out") == "Kyron").mean().alias("kyron_share_post")))
dose = (ab_main.group_by("city").agg(pl.col("delta").mean().alias("delta"), pl.len().alias("riders"))
        .join(post_share, on="city").join(city_ky.select("city", "kyron_share_issued"), on="city").sort("kyron_share_post"))
wls = sm.WLS(dose["delta"].to_numpy(), sm.add_constant(dose["kyron_share_post"].to_numpy()), weights=dose["riders"].to_numpy()).fit()
DOSE_SLOPE, DOSE_INT = wls.params[1], wls.params[0]
print(dose)
print(f"city line: +{DOSE_SLOPE:.3f} swaps per active rider-day per 100% Kyron share = +{DOSE_SLOPE / 10:.3f} per 10 pts "
      f"(intercept {DOSE_INT:+.4f}, R2={wls.rsquared:.3f}, 6 cities); placebo DiD = {pct(pl_did / did, 0)} of the main DiD")
for hv in [True, False]:
    s_ = dose.filter(pl.col("city").is_in(HEAVY) == hv)
    print(f"   {'heavy' if hv else 'light'} cluster: Kyron share {100 * s_['kyron_share_post'].min():.1f}-{100 * s_['kyron_share_post'].max():.1f}% (10-30 Sep), "
          f"{100 * s_['kyron_share_issued'].min():.1f}-{100 * s_['kyron_share_issued'].max():.1f}% (Oct-Jun); within-rider change "
          f"{s_['delta'].min():+.3f} to {s_['delta'].max():+.3f}")
print("   -> the 6 cities form two tight clusters, so the R2 only restates the two-group contrast; it is not dose-response evidence")
dose.write_csv(TAB / "q4_swapfreq_city_dose.csv")

# does the heavy-light gap track Kyron's fade? (if the fast fade caused the extra swaps, the gap should grow as Kyron SoH falls)
kyw = (kyr_daily.with_columns(pl.col("date").dt.truncate("1w").alias("week"))
       .group_by("week").agg(((pl.col("kyron_soh") * pl.col("kyron_issued")).sum() / pl.col("kyron_issued").sum()).alias("ky_soh"),
                             ((pl.col("other_soh") * (pl.col("swaps_2w") - pl.col("kyron_issued"))).sum()
                              / (pl.col("swaps_2w") - pl.col("kyron_issued")).sum()).alias("other_soh")))
gapw = (wk.pivot(on="heavy", index="week", values="swaps_per_active_day").rename({"true": "heavy", "false": "light"})
        .with_columns((pl.col("heavy") - pl.col("light")).alias("gap")).join(kyw, on="week", how="left").sort("week"))
gapw.write_csv(TAB / "q4_swapfreq_gap_vs_kyron_soh.csv")
gpost = gapw.filter(pl.col("week").is_between(dt.date(2024, 9, 16), dt.date(2025, 6, 23)))
gpre = gapw.filter(pl.col("week") < dt.date(2024, 9, 2))
GAP_R = stats.pearsonr(gpost["gap"].to_numpy(), gpost["ky_soh"].to_numpy())
g1 = gpost.row(0, named=True)
GAP = dict(pre=gpre["gap"].mean(), post=gpost["gap"].mean(), w1=g1["gap"], ky1=g1["ky_soh"], oth1=g1["other_soh"],
           ky_last=gpost["ky_soh"][-1], gap_last=gpost["gap"][-1], r=GAP_R.statistic, p=GAP_R.pvalue)
print(f"weekly heavy-light gap: pre-arrival mean {GAP['pre']:+.3f}; first full Kyron week (w/c {g1['week']}) {GAP['w1']:+.3f} while Kyron packs were at "
      f"{GAP['ky1']:.1f}% SoH vs {GAP['oth1']:.1f}% for the rest of the 2W fleet; mean over 16-Sep-24..23-Jun-25 {GAP['post']:+.3f}; last week {GAP['gap_last']:+.3f} "
      f"with Kyron at {GAP['ky_last']:.1f}%; corr(weekly gap, Kyron mean SoH) r = {GAP['r']:+.2f} (p = {GAP['p']:.2f}, {gpost.height} weeks) "
      f"-> the gap was full-size when Kyron packs were new and never tracked their fade")

# pack-level effect: rider-week panel, rider FE + city-week FE (Kyron share within a city-week is as-good-as random)
rw = (rd.filter(pl.col("date").is_between(*POST_LONG)).with_columns(pl.col("date").dt.truncate("1w").alias("week"))
      .group_by("rider_id", "week")
      .agg(pl.col("n").sum().alias("swaps"), pl.len().alias("active_days"), pl.col("n_ky").sum().alias("ky"),
           pl.col("city").mode().sort().first().alias("city"), (pl.col("soh_out") * pl.col("n")).sum().alias("soh_sum"))
      .with_columns((pl.col("swaps") / pl.col("active_days")).alias("y"), (pl.col("ky") / pl.col("swaps")).alias("ky_share"),
                    (pl.col("soh_sum") / pl.col("swaps") / 10).alias("soh10"),
                    (pl.col("city") + "_" + pl.col("week").cast(pl.Utf8)).alias("cw")))
b1, se1, n1, G1 = fe_ols(rw, "y", ["ky_share"], ["rider_id", "cw"], "rider_id")
b2, se2, n2, G2 = fe_ols(rw, "y", ["ky_share", "soh10"], ["rider_id", "cw"], "rider_id")
print(f"rider-week FE panel ({n1:,} rider-weeks, {G1:,} riders): Kyron share of packs issued that week -> "
      f"{b1[0]:+.4f} swaps per active day (95% CI {b1[0] - 1.96 * se1[0]:+.4f}, {b1[0] + 1.96 * se1[0]:+.4f})")
print(f"  with mean SoH of issued packs: Kyron share {b2[0]:+.4f} (se {se2[0]:.4f}); per +10 SoH pts {b2[1]:+.4f} (se {se2[1]:.4f})")
rdp = rd.filter(pl.col("date").is_between(*POST_LONG)).with_columns(
    pl.col("first_kyron").cast(pl.Float64).alias("fk"), (pl.col("n") >= 2).cast(pl.Float64).alias("two_plus"),
    (pl.col("city") + "_" + pl.col("date").dt.truncate("1mo").cast(pl.Utf8)).alias("cm"))
b3, se3, n3, G3 = fe_ols(rdp, "two_plus", ["fk"], ["rider_id", "cm"], "rider_id")
print(f"rider-day FE ({n3:,} rider-days): first pack of the day is Kyron -> P(2+ swaps that day) {100 * b3[0]:+.2f} pp "
      f"(95% CI {100 * (b3[0] - 1.96 * se3[0]):+.2f}, {100 * (b3[0] + 1.96 * se3[0]):+.2f})")
print("raw rider-day means by first issued pack (post period):",
      rdp.group_by("heavy", "first_kyron").agg(pl.col("n").mean().round(3), pl.len()).sort("heavy", "first_kyron").to_dicts())

# gradual SoH decline: does the gap between same-day swaps shrink as the returned pack's SoH falls?
gap = (sw.with_columns(((pl.col("event_ts") - pl.col("event_ts").shift(1).over("rider_id")).dt.total_seconds() / 3600).alias("gap_h"),
                       (pl.col("date") == pl.col("date").shift(1).over("rider_id")).alias("same_day"))
       .filter(pl.col("same_day") & pl.col("date").is_between(*POST_LONG)))
gap_bin = (gap.with_columns((pl.col("soh_in_pct") // 5 * 5).alias("soh_bin")).group_by("soh_bin")
           .agg(pl.col("gap_h").median().alias("gap_median_h"), pl.len().alias("n")).filter(pl.col("n") >= 1000).sort("soh_bin"))
print("median hours between same-day swaps by SoH of the returned pack:", gap_bin.to_dicts())
print("corr(gap, SoH of returned pack):", round(gap.select(pl.corr("gap_h", "soh_in_pct")).item(), 4))
light_m = (rd.filter(~pl.col("heavy")).with_columns(pl.col("date").dt.truncate("1mo").alias("month")).group_by("month")
           .agg(pl.col("n").mean().round(3).alias("swaps_per_active_day"), pl.col("km_mean").mean().round(1).alias("km_per_swap"),
                pl.col("soh_out").mean().round(1).alias("soh_issued")).sort("month"))
print("Kyron-light cities by month (gradual fade, little Kyron):\n", light_m)
lm = light_m.filter(pl.col("month") >= dt.date(2024, 10, 1))
print(f"light cities Oct-24..Jun-25: swaps/active day {lm['swaps_per_active_day'].min():.3f}-{lm['swaps_per_active_day'].max():.3f} while issued SoH "
      f"fell {lm['soh_issued'][0]:.1f} -> {lm['soh_issued'][-1]:.1f} and km per swap {lm['km_per_swap'][0]:.1f} -> {lm['km_per_swap'][-1]:.1f} "
      f"(km per active day {lm['swaps_per_active_day'][0] * lm['km_per_swap'][0]:.1f} -> {lm['swaps_per_active_day'][-1] * lm['km_per_swap'][-1]:.1f})")
lmu = (rd.filter(~pl.col("heavy") & (pl.col("date") >= dt.date(2024, 10, 1))).with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
       .group_by("month").agg(pl.col("km_mean").mean().alias("km"), pl.col("soh_out").mean().alias("soh")).sort("month"))
LIGHT_FLAT = (lmu["km"][0] - lmu["km"][-1]) / (lmu["soh"][0] - lmu["soh"][-1])
LIGHT_FLAT_OLS = np.polyfit(lmu["soh"].to_numpy(), lmu["km"].to_numpy(), 1)[0]
print(f"range check while swap frequency was flat (light cities, Oct-24 -> Jun-25): km per swap fell {LIGHT_FLAT:.2f} km per SoH point "
      f"(endpoints; OLS across the 9 months {LIGHT_FLAT_OLS:.2f}) - spans seasons, so a corroborating check only")

# network-wide extra swaps attributable to Kyron in Q2-2025
q2 = rd.filter(pl.col("date").is_between(*Q2_25))
q2_heavy_days = q2.filter(pl.col("heavy")).height
q2_heavy_swaps = q2.filter(pl.col("heavy"))["n"].sum()
q2_2w_swaps = q2["n"].sum()
all_q2 = events(["is_completed", "date"]).filter(pl.col("is_completed") & pl.col("date").is_between(*Q2_25)).select(pl.len()).collect().item()
EXTRA_DID = did * q2_heavy_days
q2_city = (q2.group_by("city").agg(pl.len().alias("rider_days"), pl.col("n").sum().alias("swaps"))
           .join(sw.filter(pl.col("date").is_between(*Q2_25)).group_by("city").agg((pl.col("supplier_out") == "Kyron").mean().alias("ky")), on="city"))
EXTRA_DOSE = float((q2_city["rider_days"] * q2_city["ky"] * DOSE_SLOPE).sum())
rev_per_2w_swap_heavy = q2.filter(pl.col("heavy"))["rev"].sum() / q2_heavy_swaps
print(f"Q2-2025: heavy-city 2W rider-days {q2_heavy_days:,}, heavy-city 2W swaps {q2_heavy_swaps:,}, all 2W swaps {q2_2w_swaps:,}, all swaps {all_q2:,}")
print(f"extra swaps attributable to Kyron, Q2-2025 (DiD x heavy rider-days): {EXTRA_DID:,.0f} = {pct(EXTRA_DID / q2_heavy_swaps)} of heavy-city 2W swaps, "
      f"{pct(EXTRA_DID / q2_2w_swaps)} of all 2W swaps, {pct(EXTRA_DID / all_q2)} of all swaps; {EXTRA_DID / 91:,.0f} per day")
print(f"upper bound (6-city dose-response incl. light cities): {EXTRA_DOSE:,.0f} = {pct(EXTRA_DOSE / q2_2w_swaps)} of 2W swaps, {pct(EXTRA_DOSE / all_q2)} of all swaps")
print(f"riders/fleets paid for these extra swaps: ~{fmt_inr(EXTRA_DID * rev_per_2w_swap_heavy)} in Q2-2025 (avg {rev_per_2w_swap_heavy:.1f} INR per heavy-city 2W swap)")

# weekly Kyron series for the timeline figure (needs sw)
wk_k = (sw.with_columns(pl.col("date").dt.truncate("1w").alias("week"), pl.col("city").is_in(HEAVY).alias("heavy"))
        .group_by("week", "heavy").agg((pl.col("supplier_out") == "Kyron").mean().alias("ky_share")).sort("week"))
del sw, gap, rw, rdp
gc.collect()

# %% [markdown]
# ## 5. Kyron deep dive: arrival, decline, and the recorded 2025-06-15 retirement
# batteries.csv says every Kyron 2W pack (and 53 Kyron 3W) was retired on 2025-06-15. If that happened, Kyron-heavy
# cities lost ~40% of their 2W inventory overnight. Test: were Kyron packs still issued after 15 Jun, and did stockouts /
# queues / charged inventory move in Kyron-heavy vs light cities (late vs early June 2025, net of the same windows in 2024)?

# %%
kd = kyr_daily.filter(pl.col("date") >= dt.date(2025, 6, 1)).with_columns((pl.col("date") >= RECORDED_RETIREMENT).alias("after"))
kd_s = kd.group_by("after").agg(pl.col("kyron_issued").mean().round(0).alias("kyron_swaps_per_day"), pl.col("swaps_2w").mean().round(0),
                                pl.col("kyron_packs").mean().round(0).alias("distinct_kyron_packs_per_day"),
                                pl.col("kyron_soh").mean().round(1).alias("kyron_soh"), (pl.col("kyron_issued").sum() / pl.col("swaps_2w").sum()).round(3).alias("share")).sort("after")
print("Kyron 2W issuance, 1-14 Jun vs 15-30 Jun 2025:\n", kd_s)
post_ret = (events(["is_completed", "date", "battery_out_id", "supplier_out", "vehicle_class"])
            .filter(pl.col("is_completed") & (pl.col("date") >= RECORDED_RETIREMENT) & (pl.col("supplier_out") == "Kyron"))
            .group_by("vehicle_class").agg(pl.len().alias("swaps"), pl.col("battery_out_id").n_unique().alias("distinct_packs")).collect())
print("Kyron packs issued on/after the recorded retirement date (15-30 Jun 2025):", post_ret.sort("vehicle_class").to_dicts())
km = (kyr_daily.with_columns(pl.col("date").dt.truncate("1mo").alias("month")).group_by("month")
      .agg((pl.col("kyron_issued").sum() / pl.col("swaps_2w").sum()).round(3).alias("kyron_share_2w"),
           pl.col("kyron_soh").mean().round(1).alias("kyron_soh"), pl.col("other_soh").mean().round(1).alias("cellora_amptek_soh")).sort("month"))
km = km.join(eol_month.select("month", pl.col("len").alias("sub_eol_swaps")), on="month", how="left").fill_null(0)
km.write_csv(TAB / "q4_kyron_monthly.csv")
print(km)

att = (events(["vehicle_class", "date", "city", "station_id", "is_completed", "is_stockout", "is_abandoned", "is_service_failure", "queue_wait_sec"])
       .filter((pl.col("vehicle_class") == "2W") & (
           pl.col("date").is_between(dt.date(2024, 6, 1), dt.date(2024, 6, 30)) | pl.col("date").is_between(dt.date(2025, 6, 1), dt.date(2025, 6, 30))
           | pl.col("date").is_between(PRE[0], POST[1])))
       .with_columns(pl.col("city").is_in(HEAVY).alias("heavy")).collect(engine="streaming"))


def svc(df):
    return df.agg(pl.len().alias("attempts"), pl.col("is_stockout").mean().alias("stockout"), pl.col("is_abandoned").mean().alias("abandon"),
                  pl.col("is_service_failure").mean().alias("failure"), pl.col("queue_wait_sec").filter(pl.col("is_completed")).mean().alias("qwait_s"))


june = svc(att.filter(pl.col("date").dt.month() == 6).with_columns(pl.col("date").dt.year().alias("year"), (pl.col("date") >= pl.date(pl.col("date").dt.year(), 6, 15)).alias("late"))
           .group_by("year", "heavy", "late")).sort("year", "heavy", "late")
print("June service metrics (2W):\n", june)
june.write_csv(TAB / "q4_june_retirement_check.csv")


def triple(metric):
    v = {(r["year"], r["heavy"], r["late"]): r[metric] for r in june.iter_rows(named=True)}
    d25 = (v[(2025, True, True)] - v[(2025, True, False)]) - (v[(2025, False, True)] - v[(2025, False, False)])
    d24 = (v[(2024, True, True)] - v[(2024, True, False)]) - (v[(2024, False, True)] - v[(2024, False, False)])
    return d25, d24, d25 - d24


for mtr, scale, unit in [("stockout", 100, "pp"), ("abandon", 100, "pp"), ("failure", 100, "pp"), ("qwait_s", 1, "s")]:
    d25, d24, ddd = triple(mtr)
    print(f"June late-vs-early, heavy-minus-light {mtr}: 2025 {d25 * scale:+.2f}{unit}, 2024 {d24 * scale:+.2f}{unit}, triple difference {ddd * scale:+.2f}{unit}")
arr = svc(att.filter(pl.col("date").is_between(PRE[0], POST[1])).with_columns((pl.col("date") >= KYRON_ARRIVAL).alias("post")).group_by("heavy", "post")).sort("heavy", "post")
print("Kyron arrival window service metrics (2W):\n", arr)
v = {(r["heavy"], r["post"]): r for r in arr.iter_rows(named=True)}
for mtr, scale, unit in [("stockout", 100, "pp"), ("failure", 100, "pp"), ("qwait_s", 1, "s")]:
    dd = (v[(True, True)][mtr] - v[(True, False)][mtr]) - (v[(False, True)][mtr] - v[(False, False)][mtr])
    print(f"Kyron arrival DiD heavy-minus-light {mtr}: {dd * scale:+.2f}{unit} (heavy {v[(True, False)][mtr] * scale:.2f} -> {v[(True, True)][mtr] * scale:.2f})")
print("June 2024 vs June 2025 stockout rate, heavy cities:", {r["year"]: round(100 * r["stockout"], 2) for r in svc(att.filter((pl.col("date").dt.month() == 6) & pl.col("heavy")).with_columns(pl.col("date").dt.year().alias("year")).group_by("year")).iter_rows(named=True)},
      "| light cities:", {r["year"]: round(100 * r["stockout"], 2) for r in svc(att.filter((pl.col("date").dt.month() == 6) & ~pl.col("heavy")).with_columns(pl.col("date").dt.year().alias("year")).group_by("year")).iter_rows(named=True)},
      "(all stations - includes ~60 stations opened after Jun-24)")
OLD_STN = stn.filter(pl.col("commissioned_date") < dt.date(2024, 6, 1))["station_id"].to_list()
SAME_SO = {(r["year"], r["heavy"]): 100 * r["stockout"] for r in svc(att.filter((pl.col("date").dt.month() == 6) & pl.col("station_id").is_in(OLD_STN))
                                                                     .with_columns(pl.col("date").dt.year().alias("year")).group_by("year", "heavy")).iter_rows(named=True)}
print(f"June 2W stockout rate at the SAME {len(OLD_STN)} stations (commissioned before Jun-24): heavy {SAME_SO[(2024, True)]:.2f}% -> {SAME_SO[(2025, True)]:.2f}%, "
      f"light {SAME_SO[(2024, False)]:.2f}% -> {SAME_SO[(2025, False)]:.2f}% -> the all-station fall in the heavy cities is new-station composition")
del att
gc.collect()

# station telemetry: charged 2W inventory, quarantined packs, cabinet temperature, charge time
hr = (hourly(["station_id", "date", "city", "telemetry_status", "charged_2w_avg", "packs_quarantined", "cabinet_temp_c", "avg_charge_minutes", "inventory_target_2w"])
      .filter((pl.col("telemetry_status") == "ok") & (
          pl.col("date").is_between(PRE[0], POST[1]) | (pl.col("date").dt.month() == 6)))
      .with_columns(pl.col("city").is_in(HEAVY).alias("heavy")).collect(engine="streaming"))
tele = lambda g: g.agg(pl.col("charged_2w_avg").mean().round(3).alias("charged_2w_avg"), pl.col("packs_quarantined").mean().round(4).alias("quarantined"),
                       pl.col("cabinet_temp_c").mean().round(2).alias("cabinet_c"), pl.col("avg_charge_minutes").mean().round(1).alias("charge_min"), pl.len().alias("station_hours"))
t_arr = tele(hr.filter(pl.col("date").is_between(PRE[0], POST[1])).with_columns((pl.col("date") >= KYRON_ARRIVAL).alias("post")).group_by("heavy", "post")).sort("heavy", "post")
print("telemetry around Kyron arrival:\n", t_arr)
tv = {(r["heavy"], r["post"]): r for r in t_arr.iter_rows(named=True)}
for mtr in ["charged_2w_avg", "quarantined", "cabinet_c", "charge_min"]:
    dd = (tv[(True, True)][mtr] - tv[(True, False)][mtr]) - (tv[(False, True)][mtr] - tv[(False, False)][mtr])
    print(f"telemetry DiD at Kyron arrival (heavy-light) {mtr}: {dd:+.3f}")
t_jun = tele(hr.filter(pl.col("date").dt.month() == 6).with_columns(pl.col("date").dt.year().alias("year"), (pl.col("date").dt.day() >= 15).alias("late"))
             .group_by("year", "heavy", "late")).sort("year", "heavy", "late")
print("telemetry June halves:\n", t_jun)
jv = {(r["year"], r["heavy"], r["late"]): r for r in t_jun.iter_rows(named=True)}
for mtr in ["charged_2w_avg", "quarantined"]:
    d25 = (jv[(2025, True, True)][mtr] - jv[(2025, True, False)][mtr]) - (jv[(2025, False, True)][mtr] - jv[(2025, False, False)][mtr])
    d24 = (jv[(2024, True, True)][mtr] - jv[(2024, True, False)][mtr]) - (jv[(2024, False, True)][mtr] - jv[(2024, False, False)][mtr])
    print(f"telemetry June triple difference {mtr}: {d25 - d24:+.3f} (2025 {d25:+.3f}, 2024 {d24:+.3f})")
SAME_CH = {(r["year"], r["heavy"]): r["charged_2w_avg"] for r in tele(hr.filter((pl.col("date").dt.month() == 6) & pl.col("station_id").is_in(OLD_STN))
                                                                       .with_columns(pl.col("date").dt.year().alias("year")).group_by("year", "heavy")).iter_rows(named=True)}
print(f"June charged-2W packs per station-hour at the same stations: heavy {SAME_CH[(2024, True)]:.2f} -> {SAME_CH[(2025, True)]:.2f}, "
      f"light {SAME_CH[(2024, False)]:.2f} -> {SAME_CH[(2025, False)]:.2f} (despite +{pct(k2.height / n2w_other)} 2W pack IDs since Sep-24)")
del hr
qt = (hourly(["telemetry_status", "cabinet_temp_c", "packs_quarantined", "month"]).filter(pl.col("telemetry_status") == "ok")
      .with_columns((pl.col("cabinet_temp_c") // 3 * 3).alias("cabinet_bin_c"))
      .group_by("cabinet_bin_c").agg(pl.col("packs_quarantined").mean().round(4).alias("quarantined_per_station_hour"),
                                     (pl.col("packs_quarantined") > 0).mean().round(4).alias("share_hours_any"), pl.len().alias("station_hours"))
      .filter(pl.col("station_hours") >= 1000).sort("cabinet_bin_c").collect(engine="streaming"))
print("thermal quarantine vs cabinet temperature (all ok station-hours):\n", qt)
qt.write_csv(TAB / "q4_quarantine_vs_cabinet_temp.csv")
gc.collect()

# %% [markdown]
# ## 6. Economics: wear cost, excess wear vs a Cellora-like counterfactual, purchase price and payback
# Cost model from the brief: wear per swap = issued pack's purchase cost x its own SoH loss per cycle / (initial SoH - 70).
# Counterfactual: the same pack (same price, same initial SoH) fading at the Cellora/Amptek median rate of its pack type.
# All levels are "under our usage-based battery-depreciation assumption" (no salvage value, 70% EOL).
#
# **Cap at purchase price.** The per-swap model keeps charging ~₹104 per Kyron swap after a pack has passed its own 70%
# point, i.e. it depreciates a pack by more than it cost. We therefore also report a capped version: wear stops once a
# pack's cumulative charge reaches its purchase price (the fractional last cycle is pro-rated). Capped excess wear =
# value of battery life actually destroyed relative to a Cellora-like pack delivering the same swaps.

# %%
ec = (events(["is_completed", "vehicle_class", "battery_out_id", "event_ts", "date", "revenue", "energy_cost", "wear_cost", "cm1"])
      .filter(pl.col("is_completed"))
      .sort("battery_out_id", "event_ts")
      .with_columns(pl.int_range(1, pl.len() + 1).over("battery_out_id").alias("k"))
      .join(bt.select("battery_id", "cohort", "purchase_cost_inr", "initial_soh_pct", "ref_loss", "loss_per_cycle_used").lazy(),
            left_on="battery_out_id", right_on="battery_id", how="left")
      .with_columns((pl.col("purchase_cost_inr") * pl.col("ref_loss") / (pl.col("initial_soh_pct") - EOL_SOH)).alias("wear_cf"),
                    ((pl.col("initial_soh_pct") - EOL_SOH) / pl.col("loss_per_cycle_used")).alias("k_eol"))
      .with_columns((pl.col("wear_cost") * (pl.col("k_eol") - (pl.col("k") - 1)).clip(0, 1)).alias("wear_capped"),
                    (pl.col("k") > pl.col("k_eol")).alias("past_eol"))
      .with_columns(pl.when(pl.col("cohort").str.starts_with("Kyron")).then(pl.col("wear_cost") - pl.col("wear_cf")).otherwise(0.0).alias("excess_wear"),
                    pl.when(pl.col("cohort").str.starts_with("Kyron")).then(pl.col("wear_capped") - pl.col("wear_cf")).otherwise(0.0).alias("excess_capped"),
                    pl.col("date").dt.truncate("1mo").alias("month"))
      .drop("battery_out_id", "event_ts", "purchase_cost_inr", "initial_soh_pct", "ref_loss", "loss_per_cycle_used", "k_eol")
      .collect(engine="streaming"))
by_sup = (ec.filter(pl.col("date").is_between(*POST_LONG)).group_by("vehicle_class", "cohort")
          .agg(pl.len().alias("swaps"), pl.col("revenue").mean().round(2), pl.col("energy_cost").mean().round(2), pl.col("wear_cost").mean().round(2),
               pl.col("cm1").mean().round(2), (pl.col("revenue") - pl.col("energy_cost")).mean().round(2).alias("margin_before_wear"))
          .sort("vehicle_class", "cohort"))
print("per-swap economics by issued-pack cohort, Oct-24..14-Jun-25:\n", by_sup)
by_sup.write_csv(TAB / "q4_economics_by_supplier.csv")
exc = ec.group_by("vehicle_class", "cohort").agg(pl.len().alias("swaps"), pl.col("wear_cost").sum().alias("wear_actual"), pl.col("wear_cf").sum().alias("wear_cf"),
                                                  pl.col("excess_wear").sum().alias("excess"), pl.col("wear_capped").sum().alias("wear_capped"),
                                                  pl.col("excess_capped").sum().alias("excess_capped"),
                                                  pl.col("wear_cost").filter(pl.col("past_eol")).sum().alias("wear_past_eol")
                                                  ).filter(pl.col("cohort").str.starts_with("Kyron")).sort("vehicle_class", "cohort")
print("Kyron wear: actual vs Cellora-like counterfactual (whole window), per-swap model and capped at purchase price:\n", exc)
exc.write_csv(TAB / "q4_excess_wear_by_cohort.csv")
xv = {(r["vehicle_class"], r["cohort"]): r for r in exc.iter_rows(named=True)}
EXC_2W = xv[("2W", FAST)]["excess"]
EXC_3W_FAST = xv[("3W", FAST)]["excess"]
EXC_3W_LATE = xv[("3W", LATE)]["excess"]
EXC_TOT = EXC_2W + EXC_3W_FAST + EXC_3W_LATE
CAP_2W, CAP_3W = xv[("2W", FAST)]["excess_capped"], xv[("3W", FAST)]["excess_capped"] + xv[("3W", LATE)]["excess_capped"]
EXC_CAP_TOT = CAP_2W + CAP_3W
print(f"excess wear, PER-SWAP MODEL: Kyron 2W {fmt_inr(EXC_2W)} ({EXC_2W:,.0f} INR) | Kyron 3W fast-fade lots {fmt_inr(EXC_3W_FAST)} | "
      f"Kyron 3W later lots {fmt_inr(EXC_3W_LATE)} | total {fmt_inr(EXC_TOT)}")
print(f"excess wear, CAPPED AT PURCHASE PRICE (value destroyed to date): 2W {fmt_inr(CAP_2W)} + 3W {fmt_inr(CAP_3W)} = {fmt_inr(EXC_CAP_TOT)}; "
      f"the other {fmt_inr(EXC_TOT - EXC_CAP_TOT)} of the per-swap total is wear charged after packs passed their own 70% point "
      f"(wear charged on past-EOL Kyron swaps: {fmt_inr(sum(r['wear_past_eol'] for r in xv.values()))})")
print(f"Kyron 2W swaps were charged {fmt_inr(xv[('2W', FAST)]['wear_actual'])} of wear against a fleet purchase value of {fmt_inr(k2['purchase_cost_inr'].sum())} "
      f"(capped: {fmt_inr(xv[('2W', FAST)]['wear_capped'])})")
mon = (ec.group_by("month").agg(pl.len().alias("swaps"), pl.col("wear_cost").sum().alias("wear"), pl.col("excess_wear").sum().alias("excess"),
                                 pl.col("wear_capped").sum().alias("wear_capped"), pl.col("excess_capped").sum().alias("excess_capped"),
                                 pl.col("cm1").sum().alias("cm1"), pl.col("revenue").sum().alias("revenue"))
       .sort("month").with_columns((pl.col("wear") / pl.col("swaps")).round(2).alias("wear_per_swap"),
                                   ((pl.col("wear") - pl.col("excess")) / pl.col("swaps")).round(2).alias("wear_per_swap_cf"),
                                   (pl.col("cm1") / pl.col("swaps")).round(2).alias("cm1_per_swap"),
                                   ((pl.col("cm1") + pl.col("excess")) / pl.col("swaps")).round(2).alias("cm1_per_swap_cf"),
                                   ((pl.col("cm1") + pl.col("wear") - pl.col("wear_capped")) / pl.col("swaps")).round(2).alias("cm1_per_swap_capped"),
                                   pl.col("excess").cum_sum().alias("excess_cum"), pl.col("excess_capped").cum_sum().alias("excess_capped_cum")))
mon.write_csv(TAB / "q4_excess_wear_monthly.csv")
print(mon.select("month", "swaps", "wear_per_swap", "wear_per_swap_cf", "cm1_per_swap", "cm1_per_swap_capped", "cm1_per_swap_cf",
                 pl.col("excess").round(0), pl.col("excess_capped").round(0), pl.col("excess_cum").round(0), pl.col("excess_capped_cum").round(0)))
h1 = mon.filter(pl.col("month") >= dt.date(2025, 1, 1))
H1 = dict(swaps=h1["swaps"].sum(), exc=h1["excess"].sum(), exc_cap=h1["excess_capped"].sum(), cm1=h1["cm1"].sum(), wear=h1["wear"].sum(), wear_cap=h1["wear_capped"].sum())
print(f"H1-2025 ({H1['swaps']:,} swaps): per-swap model - Kyron excess {fmt_inr(H1['exc'])} = {H1['exc'] / H1['swaps']:.2f} INR per network swap, "
      f"CM1/swap actual {H1['cm1'] / H1['swaps']:+.2f} vs Cellora-like counterfactual {(H1['cm1'] + H1['exc']) / H1['swaps']:+.2f}; "
      f"excess = {pct(H1['exc'] / h1['revenue'].sum())} of revenue")
print(f"H1-2025 capped at purchase price: excess {fmt_inr(H1['exc_cap'])} = {H1['exc_cap'] / H1['swaps']:.2f} INR per swap; CM1/swap actual "
      f"{(H1['cm1'] + H1['wear'] - H1['wear_cap']) / H1['swaps']:+.2f} vs counterfactual {(H1['cm1'] + H1['exc']) / H1['swaps']:+.2f}")
jun25 = mon.filter(pl.col("month") == dt.date(2025, 6, 1)).row(0, named=True)
print(f"Jun-2025 (per-swap model): wear/swap {jun25['wear_per_swap']} actual vs {jun25['wear_per_swap_cf']} counterfactual; cm1/swap {jun25['cm1_per_swap']} vs "
      f"{jun25['cm1_per_swap_cf']}; excess wear in Jun-2025 alone {fmt_inr(jun25['excess'])} (capped: {fmt_inr(jun25['excess_capped'])})")
avoid = mon.filter(pl.col("month") >= dt.date(2024, 11, 1))["excess"].sum()
avoid_cap = mon.filter(pl.col("month") >= dt.date(2024, 11, 1))["excess_capped"].sum()
pre_sep = mon.filter(pl.col("month") < dt.date(2024, 9, 1))["excess"].sum()
print(f"excess accrued Nov-2024..Jun-2025 (after the fade was visible in Sep-Oct 2024): per-swap model {fmt_inr(avoid)} of {fmt_inr(EXC_TOT)}; "
      f"capped {fmt_inr(avoid_cap)} of {fmt_inr(EXC_CAP_TOT)} (recoverable only through a warranty/replacement claim)")
print(f"excess before Sep-2024 (Kyron 3W fast-fade packs, in service since Jan-24): {fmt_inr(pre_sep)}; Sep-24 bar {fmt_inr(mon.filter(pl.col('month') == dt.date(2024, 9, 1))['excess'].item())}")

# June 2025: what would pulling and replacing the fast-fade lots do to depreciation?
j = ec.filter(pl.col("month") == dt.date(2025, 6, 1))
jall, jbad = j.filter(pl.col("cohort").str.starts_with("Kyron")), j.filter(pl.col("cohort") == FAST)
print(f"Jun-2025: {jall.height:,} Kyron issues, {pct(jall['past_eol'].mean())} past their own end-of-life cycle (fast-fade lots: {pct(jbad['past_eol'].mean())}); "
      f"of June's {fmt_inr(j['excess_wear'].sum())} excess, {fmt_inr(jall.filter(pl.col('past_eol'))['excess_wear'].sum())} was charged on past-EOL Kyron swaps")
REPL = dict(new=jbad["wear_cf"].sum(), old=jbad["wear_capped"].sum(), new_all=jall["wear_cf"].sum(), old_all=jall["wear_capped"].sum())
print(f"1:1 replacement of the fast-fade lots at June volumes: new packs depreciate {fmt_inr(REPL['new'])}/month vs {fmt_inr(REPL['old'])} still being "
      f"depreciated on those packs (capped) -> adds {fmt_inr(REPL['new'] - REPL['old'])}/month = {(REPL['new'] - REPL['old']) / jun25['swaps']:.2f} INR per June swap "
      f"(all Kyron incl. later 3W lots: {fmt_inr(REPL['new_all'])} vs {fmt_inr(REPL['old_all'])} -> {fmt_inr(REPL['new_all'] - REPL['old_all'])})")
del j, jall, jbad

# per-DAY reference (fade is calendar-driven): value of SoH consumed beyond a Cellora-like per-day fade, to 30-Jun-2025
kyp = (bt.filter(pl.col("supplier") == "Kyron")
       .with_columns(((T_END - D0).total_seconds() / 86400 - pl.col("t_first")).alias("days_in"),
                     pl.col("pack_type").replace_strict(REF_T).alias("ref_t"), (pl.col("initial_soh_pct") - EOL_SOH).alias("head"))
       .with_columns((pl.col("purchase_cost_inr") * (pl.col("loss_per_day") - pl.col("ref_t")) * pl.col("days_in") / pl.col("head")).alias("x"),
                     (pl.col("purchase_cost_inr") * ((pl.col("loss_per_day") * pl.col("days_in")).clip(upper_bound=pl.col("head"))
                                                     - pl.col("ref_t") * pl.col("days_in")) / pl.col("head")).alias("x_cap")))
xd = {(r["pack_type"], r["cohort"]): r for r in kyp.group_by("pack_type", "cohort").agg(pl.col("x").sum(), pl.col("x_cap").sum()).iter_rows(named=True)}
XDAY, XDAY_CAP = sum(r["x"] for r in xd.values()), sum(r["x_cap"] for r in xd.values())
print(f"excess with a per-DAY reference: uncapped {fmt_inr(XDAY)}, capped {fmt_inr(XDAY_CAP)} "
      f"(2W {fmt_inr(xd[('2W_2.1kWh', FAST)]['x_cap'])}, 3W fast-fade {fmt_inr(xd[('3W_4.8kWh', FAST)]['x_cap'])} capped)")
# lost-life value of the fast-fade lots = purchase value x (1 - life ratio): the basis for a warranty claim
LOST = {}
for pt in PTS:
    val = bt.filter((pl.col("pack_type") == pt) & (pl.col("cohort") == FAST))["purchase_cost_inr"].sum()
    LOST[pt] = (val, val * (1 - 1 / FADE_RATIO[(pt, FAST)]), val * (1 - 1 / FADE_DAY[(pt, FAST)]))
    print(f"lost-life value {pt} fast-fade lots: purchase value {fmt_inr(val)} x (1 - 1/fade ratio) = {fmt_inr(LOST[pt][1])} (per-issue ratio "
          f"{FADE_RATIO[(pt, FAST)]:.2f}x) to {fmt_inr(LOST[pt][2])} (per-day ratio {FADE_DAY[(pt, FAST)]:.2f}x)")
print(f"lost-life value, all fast-fade packs: {fmt_inr(LOST['2W_2.1kWh'][1] + LOST['3W_4.8kWh'][1])} - {fmt_inr(LOST['2W_2.1kWh'][2] + LOST['3W_4.8kWh'][2])}")

# payback per pack: lifetime contribution before wear (cycles to EOL x (revenue - energy) per swap) vs purchase cost
pay = (by_sup.filter(pl.col("vehicle_class") == "2W").select("cohort", "margin_before_wear")
       .join(coh_sup.filter(pl.col("pack_type") == "2W_2.1kWh").select("cohort", "cycles_to_eol_med", "purchase_cost_mean", "cycles_med"), on="cohort")
       .with_columns((pl.col("cycles_to_eol_med") * pl.col("margin_before_wear")).round(0).alias("lifetime_margin_to_eol"),
                     (pl.col("purchase_cost_mean") / pl.col("margin_before_wear")).round(0).alias("cycles_to_pay_back"))
       .with_columns((pl.col("lifetime_margin_to_eol") / pl.col("purchase_cost_mean")).round(3).alias("cost_recovered_by_eol"))
       .sort("cohort"))
print("2W pack payback:\n", pay)
pay.write_csv(TAB / "q4_payback_2w.csv")
pv = {r["cohort"]: r for r in pay.iter_rows(named=True)}
print(f"Kyron 2W: pays back after {pv[FAST]['cycles_to_pay_back']:.0f} cycles but reaches 70% SoH after {pv[FAST]['cycles_to_eol_med']:.0f} -> recovers "
      f"{pct(pv[FAST]['cost_recovered_by_eol'])} of its price; Cellora: {pv['Cellora']['cycles_to_pay_back']:.0f} vs {pv['Cellora']['cycles_to_eol_med']:.0f} "
      f"(projected: no Cellora pack has been issued more than {c2w['n_iss'].max()} times) -> {pct(pv['Cellora']['cost_recovered_by_eol'])}")
kfleet_cost = coh_sup.filter((pl.col("pack_type") == "2W_2.1kWh") & (pl.col("cohort") == FAST)).select(pl.col("packs") * pl.col("purchase_cost_mean")).item()
print(f"Kyron 2W fleet purchase value: {fmt_inr(kfleet_cost)} for {k2.height} packs")
del ec
gc.collect()

# %% [markdown]
# ## 7. Figures

# %%
# Figure 1 (hero): SoH vs days in service by cohort (fade is calendar-driven)
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)
for ax, pt in zip(axes, PTS):
    sub = soh_day.filter(pl.col("pack_type") == pt)
    for coh in COHORTS:
        s = sub.filter(pl.col("cohort") == coh)
        if s.height == 0:
            continue
        stl = COH_STYLE[coh]
        lab = "Kyron (lots 2407-09)" if (pt == "2W_2.1kWh" and coh == FAST) else coh
        ax.plot(s["day_bin"], s["soh_mean"], color=stl["color"], ls=stl["ls"], lw=stl["lw"], label=f"{lab}  ({traj_d[(pt, coh)]['slope'] * 100:.1f} pts/100 days)")
    ax.axhline(EOL_SOH, color=PALETTE["muted"], ls="--", lw=1)
    ax.text(5, EOL_SOH - 1.6, "70% end-of-life threshold", color="#555555", fontsize=10)
    ax.set_title(PT_LABEL[pt], fontsize=12)
    ax.set_xlabel("Days since the pack first appears in the swap log")
    ax.legend(loc="upper right" if pt == "2W_2.1kWh" else "lower left", fontsize=10)
    ax.set_ylim(55, 101)
    ax.set_xlim(-5, 560)
axes[0].set_ylabel("Mean SoH of issued packs (%)")
tk = traj_d[("2W_2.1kWh", FAST)]
last_ky_soh = kyr_daily.filter(pl.col("date") == kyr_daily["date"].max())["kyron_soh"].item()
print(f"Kyron 2W mean SoH at issue on {kyr_daily['date'].max()}: {last_ky_soh:.1f}%")
axes[0].annotate(f"Kyron below 70% after ~{tk['cross']:.0f} days\n(median pack: {d70[FAST]:.0f} days); still\nissued at {last_ky_soh:.0f}% on 30-Jun-25",
                 xy=(tk["cross"], 70), xytext=(300, 66.5), va="top", fontsize=10, arrowprops=dict(arrowstyle="->", color="#444444"), color="#222222")
tc = traj_d[("2W_2.1kWh", "Cellora")]
axes[0].annotate(f"Cellora/Amptek ~{tc['last_soh']:.0f}% after {tc['max_day']:.0f} days;\n70% projected at ~{d70['Cellora']:.0f} days",
                 xy=(tc["max_day"], tc["last_soh"]), xytext=(320, 77.5), va="top", fontsize=10, arrowprops=dict(arrowstyle="->", color="#444444"), color="#222222")
headline(fig, f"Kyron lots KY-2407/08/09 lose health {FADE_DAY[('2W_2.1kWh', FAST)]:.1f}x faster per day than Cellora/Amptek ({r2:.1f}x per issue)",
         "Mean state of health (SoH) of packs at issue by days in service (14-day bins). Fade follows calendar time, not issue count; "
         "later Kyron 3W lots track Cellora/Amptek", top=0.84)
save_fig(fig, "q4_01_soh_vs_cycles")

# Figure 2: per-pack degradation by lot
fig, axes = plt.subplots(2, 1, figsize=(14, 8.2))
order = {"Cellora": 0, "Amptek": 1, "Kyron": 2}
for ax, pt in zip(axes, PTS):
    d = coh_lot.filter(pl.col("pack_type") == pt).with_columns(pl.col("supplier").replace_strict(order).alias("o")).sort("o", "manufacturing_lot")
    x = np.arange(d.height)
    for i, r in enumerate(d.iter_rows(named=True)):
        c = SUPPLIER_COLORS[r["supplier"]]
        mk = "o" if r["supplier"] == "Cellora" else ("s" if r["supplier"] == "Amptek" else "D")
        ax.vlines(i, r["loss_p25"] * 100, r["loss_p75"] * 100, color=c, lw=2.5, alpha=0.8)
        ax.plot(i, r["loss_per_cycle_med"] * 100, marker=mk, color=c, ms=6.5, mec="white", mew=0.8, ls="none")
    rl = ref.filter(pl.col("pack_type") == pt)["ref_loss"].item() * 100
    ax.axhline(rl, color=PALETTE["muted"], ls="--", lw=1)
    ax.text(-0.5, rl + 0.3, f"Cellora/Amptek median {rl:.2f}", ha="left", fontsize=10, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels(d["manufacturing_lot"].to_list(), rotation=90, fontsize=8.5)
    ax.set_xlim(-1, d.height)
    ax.set_ylabel("SoH pts lost per 100 cycles")
    F, p = bms_res[pt]
    ax.set_title(f"{PT_LABEL[pt]}  |  BMS firmware effect within lot: F={F:.2f}, p={p:.2f} (none)", fontsize=11.5)
    for sup, mk in [("Cellora", "o"), ("Amptek", "s"), ("Kyron", "D")]:
        ax.plot([], [], marker=mk, color=SUPPLIER_COLORS[sup], ls="none", label=sup)
    ax.legend(loc="upper left", ncol=3, fontsize=10)
headline(fig, "Fast fade is confined to Kyron lots KY-2407/08/09 (2W and 3W); every other lot - incl. later Kyron 3W - is normal",
         "Per-pack SoH loss per 100 cycles by manufacturing lot: dot = lot median, bar = interquartile range", top=0.88)
fig.tight_layout(rect=[0, 0, 1, 0.9])
save_fig(fig, "q4_02_degradation_by_lot")

# Figure 3 (hero): delivered range vs SoH
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
for ax, vc in zip(axes, ["2W", "3W"]):
    sub = rng_bin.filter((pl.col("vehicle_class") == vc) & pl.col("kyron_light_city") & pl.col("soh_bin").is_between(55, 99))
    for coh in COHORTS:
        s = sub.filter(pl.col("cohort_in") == coh)
        if s.height == 0:
            continue
        stl = COH_STYLE[coh]
        lab = "Kyron" if (vc == "2W" and coh == FAST) else coh
        ax.plot(s["soh_bin"], s["km_mean"], color=stl["color"], ls=stl["ls"], lw=stl["lw"], marker="o", ms=3.5, label=lab)
    xs_ = np.array([55.0, 100.0])
    ax.plot(xs_, PROP[vc] * xs_, color=PALETTE["muted"], ls="--", lw=1.2, label="range proportional to SoH")
    pen, d_ = MATCH[(vc, FAST)], DN[(vc, "all cities")]
    txt = (f"De-noised SoH, same day and city: {d_['day']:.2f} km per point\n(same month and city {d_['month']:.2f}); range proportional\n"
           f"to SoH predicts {PROP[vc]:.2f} = 1% of new-pack range per point\n"
           f"Year-on-year design gave {YOY[vc]['mean']:.2f}: too steep, swaps per\nrider-day also rose {100 * (yy.filter(pl.col('vehicle_class') == vc)['spd25'].mean() / yy.filter(pl.col('vehicle_class') == vc)['spd24'].mean() - 1):+.1f}% in these cities\n"
           f"Kyron{' lots 2407-09' if vc == '3W' else ''} vs others at equal SoH,\nsame city and month: {pen[0]:+.1f} km ({pct(pen[1])})")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=9.5, color="#222222")
    ax.set_title(f"{vc} swaps", fontsize=12)
    ax.set_xlabel("SoH of the returned pack (%)")
    ax.legend(loc="lower right", fontsize=9.5)
axes[0].set_ylabel("Mean km since last swap (valid readings)")
headline(fig, f"Each SoH point lost costs ~1% of range: ~{DN[('2W', 'all cities')]['month']:.1f} km per 2W swap, ~{DN[('3W', 'all cities')]['month']:.1f} km per 3W swap",
         f"Mean km ridden before the swap by SoH of the returned pack (2-pt bins, >= 300 swaps), Kyron-light cities ({', '.join(LIGHT)}). "
         f"Kyron's range per SoH point is near-normal - its SoH just falls faster", top=0.84)
save_fig(fig, "q4_03_range_vs_soh")

# Figure 4 (hero): swap frequency event study + dose-response
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), gridspec_kw=dict(width_ratios=[2.3, 1]))
ax = axes[0]
for hv, c, lab in [(True, PALETTE["accent"], f"Kyron-heavy: {', '.join(HEAVY)} (~{100 * KY_SHARE[True]:.0f}% Kyron)"),
                   (False, PALETTE["primary"], f"Kyron-light: {', '.join(LIGHT)} (~{100 * KY_SHARE[False]:.0f}%)")]:
    s = wk.filter((pl.col("heavy") == hv) & (pl.col("week") >= dt.date(2024, 1, 8)) & (pl.col("week") <= dt.date(2025, 6, 23)))
    ax.plot(s["week"], s["swaps_per_active_day"], color=c, lw=2.2, label=lab)
ax.axvline(KYRON_ARRIVAL, color="#444444", lw=1, ls="--")
ax.text(KYRON_ARRIVAL, 1.478, "  1,385 Kyron 2W packs enter service 10-Sep-24", fontsize=10, va="top")
ax.axvline(RECORDED_RETIREMENT, color="#444444", lw=1, ls=":")
ax.text(RECORDED_RETIREMENT, 1.255, "recorded Kyron  \nretirement 15-Jun-25  \n(packs still issued)  ", fontsize=9.5, ha="right", va="bottom")
# the gap at the start and at the end of Kyron's service: full-size when Kyron was new, unchanged as it faded
w_first = gapw.filter(pl.col("week") == g1["week"]).row(0, named=True)
w_last = gpost.row(gpost.height - 1, named=True)
for w_, ha_, dx_ in [(w_first, "left", 6), (w_last, "right", -6)]:
    ax.annotate("", xy=(w_["week"], w_["heavy"]), xytext=(w_["week"], w_["light"]), arrowprops=dict(arrowstyle="<->", color="#222222", lw=1.2))
    ax.text(w_["week"] + dt.timedelta(days=dx_), w_["heavy"] - 0.022,
            f"gap {w_['gap']:+.2f}\nKyron {w_['ky_soh']:.0f}% SoH\nfleet {w_['other_soh']:.0f}%", fontsize=9.5, ha=ha_, va="top")
ax.text(dt.date(2025, 2, 1), 1.349, f"Gap does not track Kyron's fade:\nweekly r = {GAP['r']:+.2f} (p = {GAP['p']:.2f})",
        fontsize=9.5, ha="center", va="center", color="#222222")
ax.set_ylim(1.22, 1.48)
ax.set_ylabel("Completed 2W swaps per active rider-day")
ax.legend(loc="lower left", fontsize=9.5, bbox_to_anchor=(0.0, 0.0))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
ax.set_title(f"Within-rider DiD {did:+.3f} swaps/day [{did_lo:+.3f}, {did_hi:+.3f}] = {100 * did / base_heavy:+.1f}%; placebo {pl_did:+.3f}; "
             f"3W control {nc_did:+.3f}", fontsize=10.5)
ax = axes[1]
ax.scatter(dose["kyron_share_post"] * 100, dose["delta"], s=dose["riders"] / dose["riders"].max() * 250 + 30, color=PALETTE["primary"], alpha=0.85, zorder=3)
offs = {"Jaipur": (-50, -4), "Hyderabad": (8, 2), "Delhi NCR": (8, -14), "Bengaluru": (10, 0), "Pune": (10, -4), "Mumbai": (10, -10)}
for r in dose.iter_rows(named=True):
    ax.annotate(r["city"], (r["kyron_share_post"] * 100, r["delta"]), xytext=offs.get(r["city"], (6, -4)), textcoords="offset points", fontsize=10)
hv_, lt_ = dose.filter(pl.col("city").is_in(HEAVY)), dose.filter(~pl.col("city").is_in(HEAVY))
spread_ = 100 * max(hv_["kyron_share_post"].max() - hv_["kyron_share_post"].min(), lt_["kyron_share_post"].max() - lt_["kyron_share_post"].min())
between_ = 100 * (hv_["kyron_share_post"].mean() - lt_["kyron_share_post"].mean())
ax.text(0.03, 0.62, f"Kyron-heavy: {hv_['delta'].min():+.3f} to {hv_['delta'].max():+.3f}\nKyron-light: {lt_['delta'].min():+.3f} to {lt_['delta'].max():+.3f}\n"
        f"Kyron share spread: <= {spread_:.1f} pts within\na cluster, ~{between_:.0f} pts between: a two-\ngroup contrast, not a dose-response", transform=ax.transAxes, fontsize=9.5, va="top")
print(f"q4_04 right panel: Kyron-share spread within a cluster <= {spread_:.1f} pts, between clusters ~{between_:.0f} pts (10-30 Sep 2024)")
ax.set_xlim(-2, 47)
ax.set_xlabel("Kyron share of issued 2W packs, 10-30 Sep 2024 (%)")
ax.set_ylabel("Within-rider change in swaps per active day")
ax.set_title("Cities fall into two clusters", fontsize=11)
headline(fig, f"Riders in the Kyron-heavy cities swapped {100 * did / base_heavy:.0f}% more often from the week Kyron arrived - but the gap never tracked its fade",
         f"Weekly completed swaps per active 2W rider-day. Same riders, same daily km ({pct(gh['dkmd'] / gh['kmd_pre'])}), more visits: "
         f"~{EXTRA_DID / 91:,.0f} extra swaps/day in Q2-2025 ({pct(EXTRA_DID / q2_heavy_swaps)} of those cities' 2W swaps)",
         top=0.84)
save_fig(fig, "q4_04_swap_frequency_did")

# Figure 5: Kyron timeline (small multiples, shared time axis)
fig, axes = plt.subplots(3, 1, figsize=(13.5, 9.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.2, 1]))
x0, x1 = dt.date(2024, 8, 1), dt.date(2025, 6, 30)
ax = axes[0]
for hv, c, lab in [(True, PALETTE["accent"], "Kyron-heavy cities"), (False, PALETTE["primary"], "Kyron-light cities")]:
    s = wk_k.filter((pl.col("heavy") == hv) & pl.col("week").is_between(x0, x1))
    ax.plot(s["week"], s["ky_share"] * 100, color=c, lw=2.2, label=lab)
ax.set_ylabel("Kyron share of\nissued 2W packs (%)")
ax.legend(loc="center", bbox_to_anchor=(0.5, 0.6), fontsize=10, ncol=2)
ax.text(KYRON_ARRIVAL + dt.timedelta(days=3), 22, "Kyron arrives 10-Sep-24", fontsize=10, va="center")
ax.text(RECORDED_RETIREMENT - dt.timedelta(days=3), 22, "recorded retirement 15-Jun-25", fontsize=10, va="center", ha="right")
ax = axes[1]
s = kyr_daily.filter(pl.col("date").is_between(x0, x1))
ax.plot(s["date"], s["other_soh"], color=SUPPLIER_COLORS["Cellora"], lw=2, label="Cellora + Amptek 2W")
ax.plot(s["date"], s["kyron_soh"], color=SUPPLIER_COLORS["Kyron"], lw=2.6, label="Kyron 2W")
ax.axhline(EOL_SOH, color=PALETTE["muted"], ls="--", lw=1)
ax.text(x0, EOL_SOH + 0.6, "70% end-of-life", fontsize=10, color="#555555")
ax.set_ylabel("Mean SoH of\nissued packs (%)")
ax.legend(loc="lower left", fontsize=10)
ax = axes[2]
s = daily_eol.filter(pl.col("date").is_between(x0, x1))
ax.bar(s["date"], s["len"], width=1.0, color=SUPPLIER_COLORS["Kyron"], alpha=0.85)
ax.set_ylabel("Swaps issuing a pack\nbelow 70% SoH (per day)")
for a in axes:
    a.axvline(KYRON_ARRIVAL, color="#444444", lw=1, ls="--")
    a.axvline(RECORDED_RETIREMENT, color="#444444", lw=1, ls=":")
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
headline(fig, f"Kyron packs were never pulled: all {post_ret.filter(pl.col('vehicle_class') == '2W')['distinct_packs'].item():,} were still issued after their recorded 15-Jun retirement, below 70% SoH",
         f"{km['sub_eol_swaps'].sum():,} swaps handed riders a pack under the end-of-life threshold; Kyron share stayed ~{100 * kd_s.filter(pl.col('after'))['share'].item():.0f}% of 2W swaps after 15-Jun-25",
         top=0.9)
fig.tight_layout(rect=[0, 0, 1, 0.91])
save_fig(fig, "q4_05_kyron_timeline")

# Figure 6 (hero): economics
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw=dict(width_ratios=[1, 1.3]))
ax = axes[0]
cohs = ["Cellora", "Amptek", FAST]
labs = ["Cellora", "Amptek", "Kyron"]
xx = np.arange(len(cohs))
cost = [pv[c]["purchase_cost_mean"] / 1000 for c in cohs]
life = [pv[c]["lifetime_margin_to_eol"] / 1000 for c in cohs]
ax.bar(xx - 0.2, cost, width=0.38, color=PALETTE["muted"], label="Purchase price", edgecolor="white", linewidth=2)
bars = ax.bar(xx + 0.2, life, width=0.38, color=[SUPPLIER_COLORS[l] for l in labs], label="Contribution before wear, life to 70% SoH", edgecolor="white", linewidth=2)
bars[2].set_hatch("//")
for i, c in enumerate(cohs):
    ax.text(xx[i] + 0.2, life[i] + 0.6, f"{life[i]:.0f}K\n({pct(pv[c]['cost_recovered_by_eol'], 0)})", ha="center", fontsize=10)
    ax.text(xx[i] - 0.2, cost[i] + 0.6, f"{cost[i]:.0f}K", ha="center", fontsize=10)
ax.set_xticks(xx)
ax.set_xticklabels([f"{l}\n{pv[c]['cycles_to_eol_med']:,.0f} cycles" + ("\n(projected)" if c != FAST else "\n(observed)") for l, c in zip(labs, cohs)])
ax.set_ylabel("INR thousand per 2W pack")
ax.set_ylim(0, max(life) * 1.35)
ax.legend(loc="upper right", fontsize=9.5)
ax.set_title("Same price, one-third the life", fontsize=11.5)
ax = axes[1]
ax.plot(mon["month"], mon["excess_cum"] / 1e5, color=SUPPLIER_COLORS["Kyron"], lw=2.4, marker="o", ms=4, label="Per-swap model (brief)")
ax.plot(mon["month"], mon["excess_capped_cum"] / 1e5, color="#222222", lw=2.4, marker="o", ms=4, label="Capped at each pack's purchase price")
ax.fill_between(mon["month"], mon["excess_capped_cum"] / 1e5, mon["excess_cum"] / 1e5, color=SUPPLIER_COLORS["Kyron"], alpha=0.15, lw=0)
ax.text(mon["month"][-1], mon["excess_cum"][-1] / 1e5, f"  {fmt_inr(mon['excess_cum'][-1])}", va="center", fontsize=10.5, fontweight="bold", color=SUPPLIER_COLORS["Kyron"])
ax.text(mon["month"][-1], mon["excess_capped_cum"][-1] / 1e5, f"  {fmt_inr(mon['excess_capped_cum'][-1])}", va="center", fontsize=10.5, fontweight="bold")
ax.annotate(f"{fmt_inr(EXC_TOT - EXC_CAP_TOT)} of the per-swap total is\nwear charged after packs passed\ntheir own 70% SoH point (from Apr-25)",
            xy=(dt.date(2025, 6, 1), (mon["excess_cum"][-2] + mon["excess_capped_cum"][-2]) / 2e5),
            xytext=(dt.date(2024, 1, 1), 330), va="top", fontsize=10, arrowprops=dict(arrowstyle="->", color="#444444"))
ax.text(dt.date(2024, 1, 1), 60, f"Jan-Aug 2024: {fmt_inr(pre_sep)} on 3W\nfast-fade packs (in service\nsince Jan-24)", fontsize=9.5, color="#555555")
ax.set_ylabel("Cumulative excess wear, INR lakh")
ax.set_xlim(dt.date(2023, 12, 1), dt.date(2025, 9, 15))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
ax.legend(loc="upper left", fontsize=10)
ax.set_title("Excess wear vs a Cellora-like fade rate, cumulative", fontsize=11.5)
headline(fig, f"Kyron's fast fade destroyed ₹{EXC_CAP_TOT / 1e7:.1f}-{EXC_TOT / 1e7:.1f} Cr of battery value: same price, one-third the life",
         f"{fmt_inr(EXC_CAP_TOT)} with depreciation capped at purchase price; {fmt_inr(EXC_TOT)} under the per-swap model, which keeps charging wear "
         f"past 70% SoH. Usage-based depreciation, no salvage value", top=0.84)
save_fig(fig, "q4_06_kyron_economics")
print("figures saved:", sorted(p.name for p in FIG.glob("q4_*.png")))
print(f"runtime {time.time() - T0:.1f}s")
