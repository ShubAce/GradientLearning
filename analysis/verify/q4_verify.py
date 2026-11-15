# %% [markdown]
# # Q4 verification - independent re-derivation of the battery & equipment findings
# Adversarial check of `outputs/findings/q4.md`. Fresh code: every number is rebuilt from `data/clean` through the
# `src/common.py` loaders (the analyst's script is not imported). Sections:
# A. per-pack fade slopes (own OLS), supplier / lot / BMS tests, price & initial SoH, EOL cycles, like-for-like window,
#    linearity and heat-exposure checks;
# B. sub-EOL issuing and the recorded 15-Jun-2025 retirement;
# C. delivered range vs SoH (same-season, FE with a de-noised SoH, Kyron matched comparison with noisy vs de-noised SoH);
# D. swap-frequency difference-in-differences, placebo, persistence vs Kyron's SoH decline, extra swaps;
# E. economics: wear-cost formula, excess wear, H1-2025 CM1, payback, and a sensitivity that stops wear at end-of-life;
# F. telemetry: charged inventory and stockouts around Kyron's arrival and June 2024 vs June 2025; thermal quarantine.

# %%
import datetime as dt
import gc
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

try:
    ROOT = Path(__file__).resolve().parents[2]
except NameError:
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from common import EOL_SOH, TAB, batteries, city_daily, events, hourly, stations  # noqa: E402

T0 = time.time()
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(200)
BAD_LOTS = ["KY-2407", "KY-2408", "KY-2409"]
ARRIVAL = dt.date(2024, 9, 10)
RETIRE = dt.date(2025, 6, 15)
D0 = dt.datetime(2024, 1, 1)


def say(*a):
    print(*a, flush=True)


# %% [markdown]
# ## 0. Kyron-heavy vs Kyron-light cities (Kyron share of issued 2W packs, Oct-24 to 14-Jun-25)

# %%
city_ky = (events(["is_completed", "vehicle_class", "date", "city", "supplier_out"])
           .filter(pl.col("is_completed") & (pl.col("vehicle_class") == "2W")
                   & pl.col("date").is_between(dt.date(2024, 10, 1), dt.date(2025, 6, 14)))
           .group_by("city").agg((pl.col("supplier_out") == "Kyron").mean().alias("ky_share"), pl.len().alias("swaps"))
           .sort("ky_share").collect(engine="streaming"))
say(city_ky)
HEAVY = sorted(city_ky.filter(pl.col("ky_share") > 0.25)["city"].to_list())
LIGHT = sorted(city_ky.filter(pl.col("ky_share") <= 0.25)["city"].to_list())
grp = city_ky.with_columns(pl.col("city").is_in(HEAVY).alias("heavy")).group_by("heavy").agg(
    ((pl.col("ky_share") * pl.col("swaps")).sum() / pl.col("swaps").sum()).alias("share")).sort("heavy")
say("HEAVY", HEAVY, "LIGHT", LIGHT, "| pooled Kyron share light/heavy:", [round(x, 4) for x in grp["share"].to_list()])

# %% [markdown]
# ## A. Per-pack fade slopes from the issue log (own OLS of issued SoH on issue index)

# %%
bt = batteries().with_columns(
    pl.when(pl.col("supplier") != "Kyron").then(pl.col("supplier"))
    .when(pl.col("manufacturing_lot").is_in(BAD_LOTS)).then(pl.lit("Kyron bad lots"))
    .otherwise(pl.lit("Kyron later lots")).alias("cohort"))
say("packs:", bt.height, "| used before commission:", int(bt["used_before_commission"].sum()))
say(bt.group_by("pack_type", "cohort").agg(pl.len().alias("packs"), pl.col("manufacturing_lot").unique().sort().alias("lots"),
                                          pl.col("retired_date").drop_nulls().unique().alias("retired_dates"),
                                          pl.col("retired_date").is_not_null().sum().alias("retired")).sort("pack_type", "cohort"))

iss = (events(["is_completed", "battery_out_id", "event_ts", "soh_out_pct", "city"])
       .filter(pl.col("is_completed")).drop("is_completed")
       .sort(["battery_out_id", "event_ts"])
       .with_columns((pl.int_range(pl.len()).over("battery_out_id") + 1).cast(pl.Float64).alias("k"),
                     ((pl.col("event_ts") - pl.lit(D0)).dt.total_seconds() / 86400.0).alias("t"),
                     pl.col("soh_out_pct").cast(pl.Float64).alias("y"),
                     pl.col("event_ts").dt.date().alias("date"))
       .drop("soh_out_pct")
       .collect(engine="streaming"))
say("issued rows:", iss.height, "| null battery_out_id:", iss["battery_out_id"].null_count())


def slope_agg(x, y, name):
    xm, ym = pl.col(x).mean(), pl.col(y).mean()
    return [(-((pl.col(x) - xm) * (pl.col(y) - ym)).sum() / ((pl.col(x) - xm) ** 2).sum()).alias(name)]


pk = (iss.group_by("battery_out_id").agg(
    pl.len().alias("n"), pl.col("event_ts").min().alias("first_issue"),
    pl.col("t").mean().alias("tbar"), pl.col("y").mean().alias("ybar"), pl.col("k").mean().alias("kbar"),
    *slope_agg("k", "y", "b_k"), *slope_agg("t", "y", "b_t"),
    pl.col("city").is_in(HEAVY).mean().alias("heavy_share"),
    pl.col("k").filter(pl.col("y") < EOL_SOH).min().alias("first_k_below70_raw"))
    .rename({"battery_out_id": "battery_id"}))
# slope on issues from Kyron's arrival onward (same calendar window for every supplier)
pk_win = (iss.filter(pl.col("event_ts") >= dt.datetime(2024, 9, 10)).group_by("battery_out_id")
          .agg(*slope_agg("k", "y", "b_win"), pl.len().alias("n_win")).rename({"battery_out_id": "battery_id"}))
# linearity: slope on the first 200 issues vs issues 201+
pk_early = (iss.filter(pl.col("k") <= 200).group_by("battery_out_id").agg(*slope_agg("k", "y", "b_early"))
            .rename({"battery_out_id": "battery_id"}))
pk_late = (iss.filter(pl.col("k") > 200).group_by("battery_out_id").agg(*slope_agg("k", "y", "b_late"), pl.len().alias("n_late"))
           .rename({"battery_out_id": "battery_id"}))
bt = (bt.join(pk, on="battery_id", how="left").join(pk_win, on="battery_id", how="left")
      .join(pk_early, on="battery_id", how="left").join(pk_late, on="battery_id", how="left"))
say("max |own slope - clean.py slope|:", float((bt["b_k"] - bt["soh_loss_per_cycle"]).abs().max()),
    "| min issues per pack:", int(bt["n"].min()))
# fitted-trajectory EOL cycle (own intercept) and the master-based one
bt = bt.with_columns(((pl.col("ybar") + pl.col("b_k") * pl.col("kbar") - EOL_SOH) / pl.col("b_k")).alias("k_eol_fit"),
                     ((pl.col("initial_soh_pct") - EOL_SOH) / pl.col("b_k")).alias("k_eol_master"))

ref = (bt.filter(pl.col("supplier").is_in(["Cellora", "Amptek"])).group_by("pack_type")
       .agg(pl.col("b_k").median().alias("ref")))
REF = {r["pack_type"]: r["ref"] for r in ref.iter_rows(named=True)}
say("Cellora+Amptek reference median slope:", {k: round(v, 5) for k, v in REF.items()})
lots = (bt.group_by("supplier", "pack_type", "manufacturing_lot").agg(pl.col("b_k").median().alias("med"), pl.len().alias("packs"))
        .with_columns((pl.col("med") / pl.col("pack_type").replace_strict(REF)).alias("ratio")))
say("lots with median slope > 1.5x reference:", lots.filter(pl.col("ratio") > 1.5).sort("pack_type", "manufacturing_lot").to_dicts())
say("max ratio among all other lots:", float(lots.filter(pl.col("ratio") <= 1.5)["ratio"].max()))

coh = (bt.group_by("pack_type", "cohort").agg(
    pl.len().alias("packs"), pl.col("b_k").median().round(5).alias("slope_med"), pl.col("b_win").median().round(5).alias("slope_from_10sep24"),
    pl.col("b_early").median().round(5).alias("slope_k1_200"), pl.col("b_late").median().round(5).alias("slope_k201plus"),
    pl.col("n").median().alias("issues_med"), pl.col("k_eol_master").median().round(1).alias("eol_cycles_master"),
    pl.col("k_eol_fit").median().round(1).alias("eol_cycles_fit"), pl.col("first_k_below70_raw").median().alias("first_k_below70_raw_med"),
    pl.col("first_k_below70_raw").is_not_null().mean().round(3).alias("share_ever_below70"),
    pl.col("heavy_share").mean().round(3).alias("heavy_city_share_of_issues"),
    pl.col("first_issue").min().dt.date().alias("first_issue_min"), pl.col("first_issue").max().dt.date().alias("first_issue_max"),
    pl.col("current_soh_pct").mean().round(2).alias("current_soh"),
    pl.col("purchase_cost_inr").mean().round(2).alias("price"), pl.col("initial_soh_pct").mean().round(3).alias("init_soh"))
       .sort("pack_type", "cohort"))
say(coh)
coh.write_csv(TAB / "q4v_cohorts.csv")


def col(pt, c, field="b_k"):
    return bt.filter((pl.col("pack_type") == pt) & (pl.col("cohort") == c))[field].drop_nulls().to_numpy()


for pt, c in [("2W_2.1kWh", "Kyron bad lots"), ("3W_4.8kWh", "Kyron bad lots"), ("3W_4.8kWh", "Kyron later lots")]:
    a, b = col(pt, c), col(pt, "Cellora")
    u = stats.mannwhitneyu(a, b)
    say(f"{pt} {c}: median {np.median(a):.5f} vs Cellora {np.median(b):.5f} -> {np.median(a) / np.median(b):.3f}x "
        f"(n={len(a)} vs {len(b)}, MW p={u.pvalue:.2e}); vs Cellora+Amptek ref {np.median(a) / REF[pt]:.3f}x")
    w1, w2 = col(pt, c, "b_win"), col(pt, "Cellora", "b_win")
    say(f"   same calendar window (issues from 10-Sep-24): {np.median(w1):.5f} vs {np.median(w2):.5f} -> {np.median(w1) / np.median(w2):.3f}x")

for (pt, c), g in sorted(bt.group_by(["pack_type", "cohort"]), key=lambda z: z[0]):
    groups = [x["b_k"].to_numpy() for _, x in g.group_by("manufacturing_lot")]
    if len(groups) > 1:
        H, p = stats.kruskal(*groups)
        say(f"KW within {pt} {c}: lots={len(groups)} H={H:.2f} p={p:.3f}")
k3 = bt.filter((pl.col("supplier") == "Kyron") & (pl.col("pack_type") == "3W_4.8kWh"))
H, p = stats.kruskal(*[x["b_k"].to_numpy() for _, x in k3.group_by("manufacturing_lot")])
say(f"KW all 12 Kyron 3W lots: H={H:.1f} p={p:.1e}")

pdf = bt.select("b_k", "manufacturing_lot", "bms_firmware", "pack_type").to_pandas()
for pt in ["2W_2.1kWh", "3W_4.8kWh"]:
    m = smf.ols("b_k ~ C(manufacturing_lot) + C(bms_firmware)", data=pdf[pdf.pack_type == pt]).fit()
    an = sm.stats.anova_lm(m, typ=2)
    eff = {k: round(100 * v / pdf[pdf.pack_type == pt].b_k.median(), 2) for k, v in m.params.items() if "bms" in k}
    say(f"BMS {pt}: F={an.loc['C(bms_firmware)', 'F']:.2f} p={an.loc['C(bms_firmware)', 'PR(>F)']:.3f}; effects % of median slope {eff}")

for pt in ["2W_2.1kWh", "3W_4.8kWh"]:
    k = bt.filter((pl.col("pack_type") == pt) & (pl.col("supplier") == "Kyron"))
    o = bt.filter((pl.col("pack_type") == pt) & (pl.col("supplier") != "Kyron"))
    for c in ["purchase_cost_inr", "initial_soh_pct"]:
        t = stats.ttest_ind(k[c].to_numpy(), o[c].to_numpy(), equal_var=False)
        say(f"Welch {pt} {c}: Kyron {k[c].mean():,.2f} vs others {o[c].mean():,.2f} p={t.pvalue:.3f}")

# heat / city exposure: does the share of a pack's issues in hot Kyron-heavy cities predict its fade?
for pt, c in [("2W_2.1kWh", "Cellora"), ("2W_2.1kWh", "Amptek"), ("2W_2.1kWh", "Kyron bad lots"), ("3W_4.8kWh", "Cellora")]:
    s = bt.filter((pl.col("pack_type") == pt) & (pl.col("cohort") == c))
    r = stats.spearmanr(s["heavy_share"].to_numpy(), s["b_k"].to_numpy())
    say(f"exposure {pt} {c}: heavy-city share of issues mean {s['heavy_share'].mean():.3f} (sd {s['heavy_share'].std():.3f}); "
        f"Spearman(share, slope) = {r.statistic:+.3f} p={r.pvalue:.2f}")

# binned trajectories (20-issue bins)
traj = (iss.join(bt.select(pl.col("battery_id").alias("battery_out_id"), "pack_type", "cohort"), on="battery_out_id")
        .with_columns(((pl.col("k") - 1) // 20 * 20 + 10).alias("kbin"))
        .group_by("pack_type", "cohort", "kbin").agg(pl.col("y").mean().alias("soh"), pl.col("battery_out_id").n_unique().alias("packs"), pl.len().alias("n"))
        .filter(pl.col("n") >= 200).sort("pack_type", "cohort", "kbin"))
for (pt, c), g in sorted(traj.group_by(["pack_type", "cohort"]), key=lambda z: z[0]):
    gg = g.filter(pl.col("packs") >= 0.5 * g["packs"].max())
    sl = -np.polyfit(gg["kbin"].to_numpy(), gg["soh"].to_numpy(), 1)[0]
    below = g.filter(pl.col("soh") < EOL_SOH)
    say(f"binned {pt} {c}: {100 * sl:.2f} pts/100 issues; first bin with mean < 70: {below['kbin'].min() if below.height else None}; "
        f"last bin {g['kbin'].max()} at {g['soh'][-1]:.1f}%")

# %% [markdown]
# ## B. Sub-EOL issuing and the recorded 15-Jun-2025 retirement

# %%
iss = iss.join(bt.select(pl.col("battery_id").alias("battery_out_id"), "pack_type", "cohort", "supplier",
                         pl.col("retired_date").is_not_null().alias("retired_in_master")), on="battery_out_id")
sub = iss.filter(pl.col("y") < EOL_SOH)
say("issues below 70% SoH:", sub.height, "| by cohort:", sub.group_by("pack_type", "cohort").len().sort("pack_type").to_dicts(),
    "| first/last date:", sub["date"].min(), sub["date"].max())
mon_all = iss.group_by(pl.col("date").dt.truncate("1mo").alias("m")).agg(pl.len().alias("all"), (pl.col("y") < EOL_SOH).sum().alias("sub70")).sort("m")
say(mon_all.filter(pl.col("m") >= dt.date(2025, 3, 1)).with_columns((pl.col("sub70") / pl.col("all")).round(4).alias("share")))
q2k = iss.filter(pl.col("date").is_between(dt.date(2025, 4, 1), dt.date(2025, 6, 30)) & (pl.col("pack_type") == "2W_2.1kWh") & (pl.col("supplier") == "Kyron"))
say(f"Q2-2025 Kyron 2W issues {q2k.height:,}, below 70: {(q2k['y'] < 70).sum():,} ({(q2k['y'] < 70).mean():.4f})")
jk = iss.filter((pl.col("pack_type") == "2W_2.1kWh") & (pl.col("supplier") == "Kyron") & (pl.col("date") >= dt.date(2025, 6, 1)))
say(f"June-2025 Kyron 2W issues {jk.height:,}, of which below 70: {(jk['y'] < 70).mean():.4f}")
post = iss.filter((pl.col("date") >= RETIRE) & (pl.col("supplier") == "Kyron"))
say("after 15-Jun-2025, Kyron issues by pack type:", post.group_by("pack_type").agg(pl.len().alias("swaps"), pl.col("battery_out_id").n_unique().alias("packs")).sort("pack_type").to_dicts())
say("after 15-Jun-2025, Kyron 3W packs the master lists as retired:",
    post.filter((pl.col("pack_type") == "3W_4.8kWh") & pl.col("retired_in_master")).select(pl.len().alias("swaps"), pl.col("battery_out_id").n_unique().alias("packs")).to_dicts())
day = (iss.filter((pl.col("pack_type") == "2W_2.1kWh") & (pl.col("date") >= dt.date(2025, 6, 1)))
       .group_by("date").agg((pl.col("supplier") == "Kyron").sum().alias("ky"), pl.len().alias("all2w"),
                             pl.col("battery_out_id").filter(pl.col("supplier") == "Kyron").n_unique().alias("ky_packs"),
                             pl.col("y").filter(pl.col("supplier") == "Kyron").mean().alias("ky_soh")))
say(day.group_by(pl.col("date") >= RETIRE).agg(pl.col("ky").mean().round(1), pl.col("ky_packs").mean().round(1),
                                               (pl.col("ky").sum() / pl.col("all2w").sum()).round(4).alias("share")).sort("date"))
say("Kyron 2W mean issued SoH on 30-Jun-2025:", round(float(day.filter(pl.col("date") == dt.date(2025, 6, 30))["ky_soh"].item()), 2))
msoh = (iss.filter((pl.col("pack_type") == "2W_2.1kWh") & pl.col("date").is_between(dt.date(2024, 9, 1), dt.date(2024, 11, 30)))
        .group_by(pl.col("date").dt.truncate("1mo").alias("m"), (pl.col("supplier") == "Kyron").alias("kyron")).agg(pl.col("y").mean().round(2)).sort("kyron", "m"))
say("monthly mean issued 2W SoH Sep-Nov 2024:", msoh.to_dicts())
# fleet structure: first appearance
say("first issue dates by cohort:", bt.group_by("pack_type", "cohort").agg(pl.col("first_issue").min().dt.date().alias("first_min"),
                                                                          pl.col("first_issue").max().dt.date().alias("first_max")).sort("pack_type").to_dicts())
# is fade driven by cycles or by calendar time? Within a cohort (same service window), a per-cycle process gives per-cycle
# slopes unrelated to usage intensity, while a calendar process gives per-cycle slopes ~ 1 / issues.
for pt, c in [("2W_2.1kWh", "Cellora"), ("2W_2.1kWh", "Kyron bad lots"), ("3W_4.8kWh", "Cellora")]:
    s = bt.filter((pl.col("pack_type") == pt) & (pl.col("cohort") == c))
    rk = stats.spearmanr(s["n"].to_numpy(), s["b_k"].to_numpy()).statistic
    rt = stats.spearmanr(s["n"].to_numpy(), s["b_t"].to_numpy()).statistic
    say(f"usage check {pt} {c}: issues per pack {s['n'].min()}-{s['n'].max()}; Spearman(issues, loss per cycle) {rk:+.3f}, "
        f"Spearman(issues, loss per day) {rt:+.3f}")
n2w_other = bt.filter((pl.col("pack_type") == "2W_2.1kWh") & (pl.col("supplier") != "Kyron")).height
say(f"Kyron 2W = +{1385 / n2w_other:.4f} on a 2W fleet of {n2w_other}")

# Kyron 2W weekly mean issued SoH (used in section D)
ky_week = (iss.filter((pl.col("pack_type") == "2W_2.1kWh") & (pl.col("supplier") == "Kyron"))
           .group_by(pl.col("date").dt.truncate("1w").alias("week")).agg(pl.col("y").mean().alias("ky_soh")).sort("week"))
# keep per-pack trajectory for de-noising returned-pack SoH in section C (linear and quadratic in calendar time)
qs = (iss.join(bt.select(pl.col("battery_id").alias("battery_out_id"), "tbar"), on="battery_out_id")
      .with_columns((pl.col("t") - pl.col("tbar")).alias("tc"))
      .group_by("battery_out_id").agg(pl.len().alias("S0"), *[(pl.col("tc") ** p).sum().alias(f"S{p}") for p in (2, 3, 4)],
                                      pl.col("y").sum().alias("Sy"), (pl.col("tc") * pl.col("y")).sum().alias("S1y"),
                                      (pl.col("tc") ** 2 * pl.col("y")).sum().alias("S2y")).sort("battery_out_id"))
A = np.stack([np.stack([qs["S0"].to_numpy(), np.zeros(qs.height), qs["S2"].to_numpy()], 1),
              np.stack([np.zeros(qs.height), qs["S2"].to_numpy(), qs["S3"].to_numpy()], 1),
              np.stack([qs["S2"].to_numpy(), qs["S3"].to_numpy(), qs["S4"].to_numpy()], 1)], 1).astype(float)
rhs = np.stack([qs["Sy"].to_numpy(), qs["S1y"].to_numpy(), qs["S2y"].to_numpy()], 1).astype(float)
coef = np.linalg.solve(A, rhs[..., None])[..., 0]
quad = pl.DataFrame({"battery_id": qs["battery_out_id"], "qa": coef[:, 0], "qb": coef[:, 1], "qc": coef[:, 2]})
traj_par = bt.select("battery_id", "cohort", "pack_type", "tbar", "ybar", "b_t").join(quad, on="battery_id")
say("per-day fade (median SoH pts/day) and implied days to 70%:",
    bt.group_by("pack_type", "cohort").agg(pl.col("b_t").median().round(5).alias("loss_per_day"),
                                          ((pl.col("initial_soh_pct") - EOL_SOH) / pl.col("b_t")).median().round(0).alias("days_to_eol")).sort("pack_type", "cohort").to_dicts())
REF_T = {r["pack_type"]: r["v"] for r in bt.filter(pl.col("supplier").is_in(["Cellora", "Amptek"])).group_by("pack_type").agg(pl.col("b_t").median().alias("v")).iter_rows(named=True)}
for pt, c in [("2W_2.1kWh", "Kyron bad lots"), ("3W_4.8kWh", "Kyron bad lots"), ("3W_4.8kWh", "Kyron later lots")]:
    say(f"per-day fade {pt} {c}: {np.median(col(pt, c, 'b_t')):.5f} vs Cellora+Amptek {REF_T[pt]:.5f} -> {np.median(col(pt, c, 'b_t')) / REF_T[pt]:.3f}x per day")
# issues per pack-day in service (first issue to 30-Jun-2025)
say("issues per pack per day in service:", bt.with_columns((pl.col("n") / ((pl.lit(dt.datetime(2025, 7, 1)) - pl.col("first_issue")).dt.total_seconds() / 86400)).alias("ipd"))
    .group_by("pack_type", "cohort").agg(pl.col("ipd").median().round(3)).sort("pack_type", "cohort").to_dicts())
# within Cellora 2W (identical service window): does a pack's SoH on 30-Jun-25 depend on how often it was issued?
c2 = bt.filter((pl.col("pack_type") == "2W_2.1kWh") & (pl.col("cohort") == "Cellora")).with_columns(
    (pl.col("ybar") - pl.col("b_t") * ((dt.datetime(2025, 6, 30) - D0).total_seconds() / 86400 - pl.col("tbar"))).alias("soh_end"))
el = np.polyfit(np.log(c2["n"].to_numpy()), np.log(c2["initial_soh_pct"].to_numpy() - c2["soh_end"].to_numpy()), 1)[0]
say(f"Cellora 2W: Spearman(issues, fitted SoH lost by 30-Jun-25) {stats.spearmanr(c2['n'].to_numpy(), (c2['initial_soh_pct'] - c2['soh_end']).to_numpy()).statistic:+.3f}; "
    f"elasticity of SoH lost w.r.t. issues {el:+.3f} (1 = per-cycle wear, 0 = calendar ageing)")
# per-day counterfactual excess wear (value of SoH consumed beyond the reference per-day fade, to 30-Jun-25)
ky = bt.filter(pl.col("supplier") == "Kyron").with_columns(
    ((dt.datetime(2025, 7, 1) - D0).total_seconds() / 86400 - (pl.col("first_issue") - pl.lit(D0)).dt.total_seconds() / 86400).alias("days"),
    pl.col("pack_type").replace_strict(REF_T).alias("ref_t"))
ex_day = ky.with_columns((pl.col("purchase_cost_inr") * (pl.col("b_t") - pl.col("ref_t")) * pl.col("days") / (pl.col("initial_soh_pct") - EOL_SOH)).alias("x"),
                         (pl.col("purchase_cost_inr") * (pl.col("b_t") * pl.col("days")).clip(upper_bound=pl.col("initial_soh_pct") - EOL_SOH) / (pl.col("initial_soh_pct") - EOL_SOH)
                          - pl.col("purchase_cost_inr") * pl.col("ref_t") * pl.col("days") / (pl.col("initial_soh_pct") - EOL_SOH)).alias("x_cap"))
say("excess wear with a per-DAY reference (uncapped / capped at purchase value), Cr:",
    ex_day.group_by("pack_type", "cohort").agg((pl.col("x").sum() / 1e7).round(3), (pl.col("x_cap").sum() / 1e7).round(3)).sort("pack_type", "cohort").to_dicts())
noise = (iss.join(bt.select(pl.col("battery_id").alias("battery_out_id"), "kbar", "ybar", "b_k"), on="battery_out_id")
         .with_columns((pl.col("y") - (pl.col("ybar") - pl.col("b_k") * (pl.col("k") - pl.col("kbar")))).alias("res"))
         .group_by("pack_type", "cohort").agg(pl.col("res").std().round(3).alias("noise_sd")).sort("pack_type", "cohort"))
say("reading noise SD around own trajectory (issue readings):", noise.to_dicts())
del iss, sub, post, q2k, jk
gc.collect()

# %% [markdown]
# ## C. Delivered range vs SoH of the returned pack

# %%
ret = (events(["is_completed", "battery_in_id", "event_ts", "soh_in_pct", "soc_in_pct", "km_since_last_swap", "city", "month", "date", "vehicle_class"])
       .filter(pl.col("is_completed") & pl.col("km_since_last_swap").is_not_null() & pl.col("soh_in_pct").is_not_null())
       .drop("is_completed")
       .join(traj_par.lazy(), left_on="battery_in_id", right_on="battery_id", how="left")
       .with_columns(((pl.col("event_ts") - pl.lit(D0)).dt.total_seconds() / 86400.0).alias("t"),
                     pl.col("soh_in_pct").cast(pl.Float64).alias("soh"), pl.col("km_since_last_swap").cast(pl.Float64).alias("km"))
       .with_columns((pl.col("ybar") - pl.col("b_t") * (pl.col("t") - pl.col("tbar"))).alias("soh_fit"),
                     (pl.col("qa") + pl.col("qb") * (pl.col("t") - pl.col("tbar")) + pl.col("qc") * (pl.col("t") - pl.col("tbar")) ** 2).alias("soh_fitq"))
       .select("vehicle_class", "city", "month", "date", "cohort", "soh", "soh_fit", "soh_fitq", "km", pl.col("soc_in_pct").cast(pl.Float64).alias("soc"))
       .collect(engine="streaming"))
say("returned-pack rows with valid km:", ret.height, "| unmatched cohort:", ret["cohort"].null_count())
say("returned readings: SD of (reading - pack's fitted trajectory) by cohort:",
    ret.group_by("vehicle_class", "cohort").agg((pl.col("soh") - pl.col("soh_fit")).std().round(3).alias("sd"), (pl.col("soh") - pl.col("soh_fit")).mean().round(3).alias("mean")).sort("vehicle_class", "cohort").to_dicts())
say("corr(km, SoC at return) 2W:", round(ret.filter(pl.col("vehicle_class") == "2W").select(pl.corr("km", "soc")).item(), 4))
say("Q2-2025 km by returned cohort:", ret.filter(pl.col("date").is_between(dt.date(2025, 4, 1), dt.date(2025, 6, 30)))
    .group_by("vehicle_class", "cohort").agg(pl.col("km").median().round(2).alias("km_med"), pl.col("soh").mean().round(2).alias("soh_mean"), pl.len()).sort("vehicle_class", "cohort").to_dicts())

REFC = ["Cellora", "Amptek"]
for vc in ["2W", "3W"]:
    cc = (ret.filter((pl.col("vehicle_class") == vc) & pl.col("cohort").is_in(REFC) & pl.col("city").is_in(LIGHT))
          .group_by("month").agg(pl.col("km").mean(), pl.col("soh").mean()).with_columns(pl.col("month").dt.month().alias("mm"), pl.col("month").dt.year().alias("yy")))
    a = cc.filter(pl.col("yy") == 2024).join(cc.filter(pl.col("yy") == 2025), on="mm", suffix="_25").sort("mm").with_columns(
        ((pl.col("km_25") - pl.col("km")) / (pl.col("soh_25") - pl.col("soh"))).alias("km_per_pt"))
    say(f"{vc} same-month YoY (light cities, Cellora/Amptek): mean {a['km_per_pt'].mean():.3f} km/pt, range {a['km_per_pt'].min():.3f}-{a['km_per_pt'].max():.3f}; "
        f"Jan-24 {a['km'][0]:.2f} km at {a['soh'][0]:.2f}% -> {100 * a['km_per_pt'].mean() / a['km'][0]:.2f}% per pt")
    # same YoY design in the heavy cities (swap frequency jumped there)
    hh = (ret.filter((pl.col("vehicle_class") == vc) & pl.col("cohort").is_in(REFC) & pl.col("city").is_in(HEAVY))
          .group_by("month").agg(pl.col("km").mean(), pl.col("soh").mean()).with_columns(pl.col("month").dt.month().alias("mm"), pl.col("month").dt.year().alias("yy")))
    b = hh.filter(pl.col("yy") == 2024).join(hh.filter(pl.col("yy") == 2025), on="mm", suffix="_25").with_columns(
        ((pl.col("km_25") - pl.col("km")) / (pl.col("soh_25") - pl.col("soh"))).alias("km_per_pt"))
    say(f"   same design in HEAVY cities: {b['km_per_pt'].mean():.3f} km/pt")


def within_slope(df, x, groups):
    d = df.select(x, "km", *groups).drop_nulls().with_columns(
        (pl.col(x) - pl.col(x).mean().over(groups)).alias("xd"), (pl.col("km") - pl.col("km").mean().over(groups)).alias("yd"))
    return float((d["xd"] * d["yd"]).sum() / (d["xd"] ** 2).sum()), float((d["xd"] ** 2).mean()), d.height


for vc in ["2W", "3W"]:
    base = ret.filter((pl.col("vehicle_class") == vc) & pl.col("cohort").is_in(REFC))
    for lab, sub_ in [("all cities", base), ("light cities", base.filter(pl.col("city").is_in(LIGHT)))]:
        s_noisy, v_noisy, n = within_slope(sub_, "soh", ["month", "city"])
        s_fit, v_fit, _ = within_slope(sub_, "soh_fit", ["month", "city"])
        s_q, v_q, _ = within_slope(sub_, "soh_fitq", ["month", "city"])
        s_d, _, _ = within_slope(sub_, "soh_fit", ["date", "city"])
        say(f"{vc} {lab}: within month x city slope on noisy SoH {s_noisy:.3f} km/pt (within var {v_noisy:.3f}); "
            f"on de-noised SoH {s_fit:.3f} km/pt (within var {v_fit:.3f}); quadratic de-noise {s_q:.3f}; within DAY x city {s_d:.3f}; "
            f"reliability ~{v_fit / v_noisy:.2f}; n={n:,}")


def matched(df, vc, coh, x, by_city=True, nmin=30):
    keys = ["city", "month", "sb"] if by_city else ["month", "sb"]
    d = df.filter(pl.col("vehicle_class") == vc).with_columns(pl.col(x).floor().alias("sb"),
                                                              pl.when(pl.col("cohort").is_in(REFC)).then(pl.lit("ref")).otherwise(pl.col("cohort")).alias("g"))
    cells = d.group_by(*keys, "g").agg(pl.col("km").mean().alias("km"), pl.len().alias("n"))
    m = cells.filter(pl.col("g") == coh).join(cells.filter(pl.col("g") == "ref"), on=keys, suffix="_r").filter((pl.col("n") >= nmin) & (pl.col("n_r") >= nmin))
    pen = float(((m["km"] - m["km_r"]) * m["n"]).sum() / m["n"].sum())
    rel = pen / float((m["km_r"] * m["n"]).sum() / m["n"].sum())
    return pen, rel, int(m["n"].sum()), m.height


for vc, coh_ in [("2W", "Kyron bad lots"), ("3W", "Kyron bad lots"), ("3W", "Kyron later lots")]:
    a1 = matched(ret, vc, coh_, "soh", True)
    a2 = matched(ret, vc, coh_, "soh", False, 100)
    a3 = matched(ret, vc, coh_, "soh_fit", True)
    a4 = matched(ret, vc, coh_, "soh_fitq", True)
    say(f"{vc} {coh_} vs Cellora/Amptek at equal SoH: within city x month x 1-pt bin {a1[0]:+.2f} km ({100 * a1[1]:+.1f}%, {a1[2]:,} swaps, {a1[3]} cells); "
        f"pooled over cities {a2[0]:+.2f} km; matched on DE-NOISED SoH within city x month {a3[0]:+.2f} km ({100 * a3[1]:+.1f}%, {a3[2]:,} swaps); "
        f"quadratic de-noise {a4[0]:+.2f} km ({100 * a4[1]:+.1f}%)")
del ret
gc.collect()

# %% [markdown]
# ## D. Swap frequency: within-rider difference-in-differences around Kyron's arrival

# %%
rd = (events(["is_completed", "vehicle_class", "rider_id", "event_ts", "date", "city", "supplier_out", "revenue"])
      .filter(pl.col("is_completed") & (pl.col("vehicle_class") == "2W"))
      .group_by("rider_id", "date").agg(pl.len().alias("n"), pl.col("city").sort_by("event_ts").first().alias("city"),
                                         pl.col("revenue").sum().alias("rev"))
      .with_columns(pl.col("city").is_in(HEAVY).alias("heavy"))
      .collect(engine="streaming"))
say("2W active rider-days:", rd.height)


def did(pre, post, label):
    a = rd.filter(pl.col("date").is_between(*pre)).group_by("rider_id").agg(pl.col("n").mean().alias("pre"), pl.len().alias("dpre"),
                                                                          pl.col("city").mode().sort().first().alias("city"))
    b = rd.filter(pl.col("date").is_between(*post)).group_by("rider_id").agg(pl.col("n").mean().alias("post"), pl.len().alias("dpost"))
    ab = a.join(b, on="rider_id").filter((pl.col("dpre") >= 3) & (pl.col("dpost") >= 3)).with_columns(
        (pl.col("post") - pl.col("pre")).alias("delta"), pl.col("city").is_in(HEAVY).cast(pl.Int8).alias("h"))
    m = smf.ols("delta ~ h", data=ab.select("delta", "h").to_pandas()).fit(cov_type="HC1")
    ci = m.conf_int().loc["h"]
    base = ab.filter(pl.col("h") == 1)["pre"].mean()
    say(f"{label}: DiD {m.params['h']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] p={m.pvalues['h']:.1e}, riders={ab.height:,}, heavy pre {base:.3f} "
        f"-> {100 * m.params['h'] / base:+.1f}%")
    return m.params["h"], ab


did_main, ab = did((dt.date(2024, 8, 20), dt.date(2024, 9, 9)), (dt.date(2024, 9, 10), dt.date(2024, 9, 30)), "Kyron arrival")
did_pl, _ = did((dt.date(2024, 7, 30), dt.date(2024, 8, 19)), (dt.date(2024, 8, 20), dt.date(2024, 9, 9)), "Placebo")
did_ret, _ = did((dt.date(2025, 5, 25), dt.date(2025, 6, 14)), (dt.date(2025, 6, 15), dt.date(2025, 6, 30)), "Recorded retirement")
# alternative placebo windows a year earlier is impossible (data start Jan-24); use two more pre-period placebos
did((dt.date(2024, 6, 1), dt.date(2024, 6, 21)), (dt.date(2024, 6, 22), dt.date(2024, 7, 12)), "Placebo Jun/Jul-24 (spans price change)")
did((dt.date(2024, 4, 1), dt.date(2024, 4, 21)), (dt.date(2024, 4, 22), dt.date(2024, 5, 12)), "Placebo Apr/May-24")
dose = ab.group_by("city").agg(pl.col("delta").mean(), pl.len().alias("riders")).join(city_ky.select("city", "ky_share"), on="city").sort("ky_share")
w = sm.WLS(dose["delta"].to_numpy(), sm.add_constant(dose["ky_share"].to_numpy()), weights=dose["riders"].to_numpy()).fit()
say("dose-response (city delta vs Kyron share Oct-Jun):", dose.to_dicts(), f"slope per 10 pts {w.params[1] / 10:+.4f}, intercept {w.params[0]:+.4f}, R2 {w.rsquared:.3f}")
say("   within-cluster spread of city deltas: heavy", dose.filter(pl.col("ky_share") > 0.25)["delta"].round(4).to_list(),
    "light", dose.filter(pl.col("ky_share") <= 0.25)["delta"].round(4).to_list())

# persistence: weekly heavy-minus-light gap vs Kyron's mean issued SoH
wk = (rd.with_columns(pl.col("date").dt.truncate("1w").alias("week")).group_by("week", "heavy").agg(pl.col("n").mean().alias("spd"))
      .pivot(on="heavy", index="week", values="spd").rename({"true": "heavy", "false": "light"}).sort("week")
      .with_columns((pl.col("heavy") - pl.col("light")).alias("gap")).join(ky_week, on="week", how="left"))
pre_gap = wk.filter(pl.col("week") < dt.date(2024, 9, 2))["gap"]
post_wk = wk.filter(pl.col("week").is_between(dt.date(2024, 9, 16), dt.date(2025, 6, 23)))
r = stats.pearsonr(post_wk["gap"].to_numpy(), post_wk["ky_soh"].to_numpy())
say(f"weekly gap heavy-light: pre-arrival mean {pre_gap.mean():+.4f} (sd {pre_gap.std():.4f}); post mean {post_wk['gap'].mean():+.4f} (sd {post_wk['gap'].std():.4f}); "
    f"first post weeks {post_wk['gap'].head(3).round(3).to_list()}, last weeks {post_wk['gap'].tail(3).round(3).to_list()}; "
    f"corr(gap, Kyron mean SoH) = {r.statistic:+.3f} (p={r.pvalue:.2f}) while Kyron SoH fell {post_wk['ky_soh'][0]:.1f} -> {post_wk['ky_soh'][-1]:.1f}")
wk.write_csv(TAB / "q4v_weekly_gap.csv")

say("light-city weekly swaps per active day, 4 weeks before vs 4 weeks after arrival:",
    round(wk.filter(pl.col("week").is_between(dt.date(2024, 8, 5), dt.date(2024, 9, 2)))["light"].mean(), 4),
    round(wk.filter(pl.col("week").is_between(dt.date(2024, 9, 16), dt.date(2024, 10, 7)))["light"].mean(), 4))

# are the extra heavy-city swaps real visits or near-duplicate records? gap and km to the rider's previous swap
dup = (events(["is_completed", "vehicle_class", "rider_id", "event_ts", "date", "city", "km_since_last_swap", "station_id", "sync_mode"])
       .filter(pl.col("is_completed") & (pl.col("vehicle_class") == "2W") & pl.col("date").is_between(dt.date(2024, 8, 13), dt.date(2024, 9, 30)))
       .sort("rider_id", "event_ts")
       .with_columns(((pl.col("event_ts") - pl.col("event_ts").shift(1).over("rider_id")).dt.total_seconds() / 60).alias("gap_min"),
                     (pl.col("station_id") == pl.col("station_id").shift(1).over("rider_id")).alias("same_stn"))
       .filter(pl.col("date") >= dt.date(2024, 8, 20))
       .with_columns(pl.col("city").is_in(HEAVY).alias("heavy"), (pl.col("date") >= ARRIVAL).alias("post"))
       .group_by("heavy", "post").agg(pl.len().alias("swaps"), (pl.col("gap_min") < 30).mean().round(4).alias("gap_lt30m"),
                                      (pl.col("gap_min") < 120).mean().round(4).alias("gap_lt2h"),
                                      ((pl.col("gap_min") < 30) & pl.col("same_stn")).mean().round(4).alias("same_stn_lt30m"),
                                      (pl.col("km_since_last_swap") < 5).mean().round(4).alias("km_lt5"),
                                      pl.col("km_since_last_swap").quantile(0.1).alias("km_p10"), pl.col("km_since_last_swap").median().alias("km_p50"),
                                      pl.col("gap_min").median().round(0).alias("gap_med_min"), (pl.col("sync_mode") == "offline_batch").mean().round(4).alias("offline"))
       .sort("heavy", "post").collect(engine="streaming"))
say("near-duplicate check, 2W swaps 20-Aug..30-Sep-24:", dup.to_dicts())

# 3W riders (Kyron 3W was in the fleet all along; no 3W arrival): same windows
rd3 = (events(["is_completed", "vehicle_class", "rider_id", "event_ts", "date", "city"])
       .filter(pl.col("is_completed") & (pl.col("vehicle_class") == "3W") & pl.col("date").is_between(dt.date(2024, 8, 20), dt.date(2024, 9, 30)))
       .group_by("rider_id", "date").agg(pl.len().alias("n"), pl.col("city").sort_by("event_ts").first().alias("city")).collect(engine="streaming"))
rd_backup, rd = rd, rd3
did((dt.date(2024, 8, 20), dt.date(2024, 9, 9)), (dt.date(2024, 9, 10), dt.date(2024, 9, 30)), "3W riders, same windows")
rd = rd_backup
del rd3, rd_backup

# finding 8: light cities, monthly swaps per active day vs issued SoH and km; same-day gap vs returned-pack SoH
lm = (events(["is_completed", "vehicle_class", "rider_id", "date", "city", "soh_out_pct", "km_since_last_swap"])
      .filter(pl.col("is_completed") & (pl.col("vehicle_class") == "2W") & pl.col("city").is_in(LIGHT) & (pl.col("date") >= dt.date(2024, 10, 1)))
      .group_by("rider_id", "date").agg(pl.len().alias("n"), pl.col("km_since_last_swap").mean().alias("km"), pl.col("soh_out_pct").mean().alias("soh"))
      .group_by(pl.col("date").dt.truncate("1mo").alias("m")).agg(pl.col("n").mean().round(3), pl.col("km").mean().round(1), pl.col("soh").mean().round(1))
      .sort("m").collect(engine="streaming"))
say("light cities monthly (Oct-24..Jun-25):", lm.to_dicts())
gp = (events(["is_completed", "vehicle_class", "rider_id", "event_ts", "date", "soh_in_pct"])
      .filter(pl.col("is_completed") & (pl.col("vehicle_class") == "2W"))
      .sort("rider_id", "event_ts")
      .with_columns(((pl.col("event_ts") - pl.col("event_ts").shift(1).over("rider_id")).dt.total_seconds() / 3600).alias("gap_h"),
                    (pl.col("date") == pl.col("date").shift(1).over("rider_id")).alias("same_day"))
      .filter(pl.col("same_day") & pl.col("date").is_between(dt.date(2024, 10, 1), dt.date(2025, 6, 14)))
      .select(pl.corr("gap_h", "soh_in_pct").alias("r"), pl.len().alias("n")).collect(engine="streaming"))
say("same-day gap vs returned SoH:", gp.to_dicts())

q2 = rd.filter(pl.col("date").is_between(dt.date(2025, 4, 1), dt.date(2025, 6, 30)))
qh = q2.filter(pl.col("heavy"))
all_q2 = events(["is_completed", "date"]).filter(pl.col("is_completed") & pl.col("date").is_between(dt.date(2025, 4, 1), dt.date(2025, 6, 30))).select(pl.len()).collect().item()
extra = did_main * qh.height
say(f"Q2-2025: heavy 2W rider-days {qh.height:,}, heavy 2W swaps {qh['n'].sum():,}, all 2W swaps {q2['n'].sum():,}, all swaps {all_q2:,}; "
    f"extra swaps {extra:,.0f} = {extra / qh['n'].sum():.4f} of heavy 2W, {extra / q2['n'].sum():.4f} of 2W, {extra / all_q2:.4f} of all; per day {extra / 91:.0f}; "
    f"paid {extra * qh['rev'].sum() / qh['n'].sum() / 1e5:.2f} L at {qh['rev'].sum() / qh['n'].sum():.2f}/swap")
# other things that changed at stations in the window
st = stations().filter(~pl.col("is_test"))
win = lambda c: pl.col(c).is_between(dt.date(2024, 8, 20), dt.date(2024, 9, 30))
say("stations commissioned 20-Aug..30-Sep-24:", st.filter(win("commissioned_date")).group_by(pl.col("city").is_in(HEAVY).alias("heavy")).len().sort("heavy").to_dicts(),
    "| firmware updates in window:", st.filter(win("firmware_updated_date")).group_by(pl.col("city").is_in(HEAVY).alias("heavy")).len().sort("heavy").to_dicts())
cdy = city_daily().filter(pl.col("date").is_between(dt.date(2024, 8, 20), dt.date(2024, 9, 30))).with_columns(
    (pl.col("date") >= ARRIVAL).alias("post"), pl.col("city").is_in(HEAVY).alias("heavy"))
say("city-day context in the DiD window:", cdy.group_by("heavy", "post").agg(
    pl.col("max_temp_c").mean().round(1), pl.col("rainfall_mm").mean().round(1), pl.col("flood_disruption").mean().round(3),
    pl.col("heat_alert").mean().round(3), pl.col("festival_or_event").is_not_null().mean().round(3).alias("festival"),
    pl.col("ecommerce_sale_event").mean().round(3), pl.col("grid_outage_hours").mean().round(3)).sort("heavy", "post").to_dicts())
del rd, ab, q2, qh
gc.collect()

# %% [markdown]
# ## E. Economics: wear formula, excess wear, H1-2025 CM1, payback, and an end-of-life cap sensitivity

# %%
ec = (events(["is_completed", "battery_out_id", "event_ts", "date", "vehicle_class", "revenue", "energy_cost", "wear_cost", "cm1"])
      .filter(pl.col("is_completed")).drop("is_completed")
      .sort(["battery_out_id", "event_ts"])
      .with_columns((pl.int_range(pl.len()).over("battery_out_id") + 1).alias("k"))
      .join(bt.select(pl.col("battery_id").alias("battery_out_id"), "pack_type", "cohort", "purchase_cost_inr", "initial_soh_pct",
                      "loss_per_cycle_used", "b_k").lazy(), on="battery_out_id", how="left")
      .with_columns((pl.col("purchase_cost_inr") * pl.col("loss_per_cycle_used") / (pl.col("initial_soh_pct") - EOL_SOH)).alias("wear_chk"),
                    (pl.col("purchase_cost_inr") * pl.col("pack_type").replace_strict(REF) / (pl.col("initial_soh_pct") - EOL_SOH)).alias("wear_cf"),
                    ((pl.col("initial_soh_pct") - EOL_SOH) / pl.col("loss_per_cycle_used")).alias("k_eol"),
                    pl.col("date").dt.truncate("1mo").alias("m"),
                    pl.col("cohort").str.starts_with("Kyron").alias("is_ky"))
      .with_columns(pl.when(pl.col("is_ky")).then(pl.col("wear_cost") - pl.col("wear_cf")).otherwise(0.0).alias("excess"),
                    (pl.col("wear_cost") * (pl.col("k_eol") - (pl.col("k") - 1)).clip(0, 1)).alias("wear_capped"))
      .with_columns(pl.when(pl.col("is_ky")).then(pl.col("wear_capped") - pl.col("wear_cf")).otherwise(0.0).alias("excess_capped"))
      .drop("battery_out_id", "event_ts")
      .collect(engine="streaming"))
say("completed swaps:", ec.height, "| max |wear_cost - price*loss/(init-70)|:", float((ec["wear_cost"] - ec["wear_chk"]).abs().max()),
    "| wear_cost on swaps where the pack is past its own EOL cycle:", round(float(ec.filter(pl.col("k") > pl.col("k_eol"))["wear_cost"].sum()) / 1e7, 3), "Cr")
by = ec.group_by("vehicle_class", "cohort").agg(pl.len().alias("swaps"), (pl.col("wear_cost").sum() / 1e7).round(4).alias("wear_cr"),
                                                (pl.col("wear_cf").sum() / 1e7).round(4).alias("cf_cr"), (pl.col("excess").sum() / 1e7).round(4).alias("excess_cr"),
                                                (pl.col("wear_capped").sum() / 1e7).round(4).alias("wear_capped_cr"),
                                                (pl.col("excess_capped").sum() / 1e7).round(4).alias("excess_capped_cr")).sort("vehicle_class", "cohort")
say(by)
EXC = float(ec["excess"].sum()); EXC_CAP = float(ec["excess_capped"].sum())
say(f"total excess wear (brief's per-swap model) {EXC / 1e7:.3f} Cr; with wear stopped at each pack's EOL {EXC_CAP / 1e7:.3f} Cr")
k2 = bt.filter((pl.col("pack_type") == "2W_2.1kWh") & (pl.col("supplier") == "Kyron"))
say(f"Kyron 2W fleet purchase value {k2['purchase_cost_inr'].sum() / 1e7:.3f} Cr vs wear charged to Kyron 2W swaps "
    f"{ec.filter((pl.col('pack_type') == '2W_2.1kWh') & pl.col('is_ky'))['wear_cost'].sum() / 1e7:.3f} Cr")
mon = (ec.group_by("m").agg(pl.len().alias("swaps"), pl.col("wear_cost").sum(), pl.col("excess").sum(), pl.col("cm1").sum(), pl.col("revenue").sum(),
                             pl.col("wear_capped").sum(), pl.col("excess_capped").sum(), pl.col("energy_cost").sum())
       .sort("m").with_columns((pl.col("wear_cost") / pl.col("swaps")).round(2).alias("wear_ps"),
                               ((pl.col("wear_cost") - pl.col("excess")) / pl.col("swaps")).round(2).alias("wear_cf_ps"),
                               (pl.col("cm1") / pl.col("swaps")).round(2).alias("cm1_ps"),
                               ((pl.col("cm1") + pl.col("excess")) / pl.col("swaps")).round(2).alias("cm1_cf_ps"),
                               ((pl.col("cm1") + pl.col("wear_cost") - pl.col("wear_capped")) / pl.col("swaps")).round(2).alias("cm1_capped_ps"),
                               (pl.col("excess") / 1e5).round(2).alias("excess_L"), (pl.col("excess_capped") / 1e5).round(2).alias("excess_capped_L")))
say(mon.select("m", "swaps", "wear_ps", "wear_cf_ps", "cm1_ps", "cm1_cf_ps", "cm1_capped_ps", "excess_L", "excess_capped_L"))
mon.write_csv(TAB / "q4v_monthly_economics.csv")
h1 = mon.filter(pl.col("m") >= dt.date(2025, 1, 1))
hs = h1["swaps"].sum()
say(f"H1-2025: swaps {hs:,}; excess {h1['excess'].sum() / 1e7:.3f} Cr = {h1['excess'].sum() / hs:.2f}/swap; CM1 actual {h1['cm1'].sum() / hs:+.2f}, "
    f"counterfactual {(h1['cm1'].sum() + h1['excess'].sum()) / hs:+.2f}; CM1 with wear stopped at EOL {(h1['cm1'].sum() + h1['wear_cost'].sum() - h1['wear_capped'].sum()) / hs:+.2f}; "
    f"capped excess {h1['excess_capped'].sum() / hs:.2f}/swap")
say(f"Nov-24..Jun-25 excess {mon.filter(pl.col('m') >= dt.date(2024, 11, 1))['excess'].sum() / 1e7:.3f} Cr; pre-Sep-24 excess (3W bad lots only) "
    f"{mon.filter(pl.col('m') < dt.date(2024, 9, 1))['excess'].sum() / 1e5:.2f} L")
j = ec.filter(pl.col("m") == dt.date(2025, 6, 1))
jk2 = j.filter(pl.col("is_ky"))
say(f"Jun-2025: Kyron issues {jk2.height:,} of which past own EOL cycle {(jk2['k'] > jk2['k_eol']).mean():.3f}; excess {j['excess'].sum() / 1e5:.2f} L; "
    f"excess on past-EOL Kyron swaps {jk2.filter(pl.col('k') > pl.col('k_eol'))['excess'].sum() / 1e5:.2f} L; "
    f"replacing them 1:1 with new reference-grade packs adds depreciation of {jk2['wear_cf'].sum() / 1e5:.2f} L/month vs {jk2['wear_capped'].sum() / 1e5:.2f} L still being depreciated on Kyron")
pay = (ec.filter(pl.col("date").is_between(dt.date(2024, 10, 1), dt.date(2025, 6, 14)) & (pl.col("vehicle_class") == "2W"))
       .group_by("cohort").agg((pl.col("revenue") - pl.col("energy_cost")).mean().alias("mbw"), pl.col("wear_cost").mean().round(2).alias("wear"), pl.col("cm1").mean().round(2).alias("cm1"))
       .join(bt.filter(pl.col("pack_type") == "2W_2.1kWh").group_by("cohort").agg(pl.col("expected_cycles_to_eol").median().alias("eol"), pl.col("purchase_cost_inr").mean().alias("price")), on="cohort")
       .with_columns((pl.col("mbw") * pl.col("eol") / pl.col("price")).round(4).alias("recovered"), (pl.col("price") / pl.col("mbw")).round(0).alias("breakeven_cycles")).sort("cohort"))
say("2W payback (Oct-24..14-Jun-25):", pay.to_dicts())
del ec, j, jk2
gc.collect()

# %% [markdown]
# ## F. Telemetry and stockouts: Kyron arrival and June 2024 vs June 2025

# %%
hr = (hourly(["station_id", "date", "city", "telemetry_status", "charged_2w_avg", "packs_quarantined", "cabinet_temp_c", "avg_charge_minutes"])
      .filter((pl.col("telemetry_status") == "ok") & (pl.col("date").is_between(dt.date(2024, 8, 20), dt.date(2024, 9, 30)) | (pl.col("date").dt.month() == 6)))
      .with_columns(pl.col("city").is_in(HEAVY).alias("heavy")).collect(engine="streaming"))
old = set(st.filter(pl.col("commissioned_date") < dt.date(2024, 6, 1))["station_id"].to_list())
a = hr.filter(pl.col("date").is_between(dt.date(2024, 8, 20), dt.date(2024, 9, 30))).with_columns((pl.col("date") >= ARRIVAL).alias("post")).group_by("heavy", "post").agg(
    pl.col("charged_2w_avg").mean(), pl.col("packs_quarantined").mean(), pl.col("cabinet_temp_c").mean()).sort("heavy", "post")
v = {(r["heavy"], r["post"]): r for r in a.iter_rows(named=True)}
for mtr in ["charged_2w_avg", "packs_quarantined", "cabinet_temp_c"]:
    say(f"Kyron-arrival DiD {mtr}: {(v[(True, True)][mtr] - v[(True, False)][mtr]) - (v[(False, True)][mtr] - v[(False, False)][mtr]):+.4f}")
jn = hr.filter((pl.col("date").dt.month() == 6) & pl.col("station_id").is_in(list(old))).group_by(pl.col("date").dt.year().alias("yr"), "heavy").agg(
    pl.col("charged_2w_avg").mean().round(3), pl.col("avg_charge_minutes").mean().round(1), pl.col("cabinet_temp_c").mean().round(2), pl.len().alias("station_hours")).sort("heavy", "yr")
say("June telemetry, same stations (commissioned before Jun-24):", jn.to_dicts())
att = (events(["vehicle_class", "date", "city", "station_id", "is_stockout"]).filter((pl.col("vehicle_class") == "2W") & (pl.col("date").dt.month() == 6))
       .with_columns(pl.col("city").is_in(HEAVY).alias("heavy"), pl.col("station_id").is_in(list(old)).alias("old"))
       .group_by(pl.col("date").dt.year().alias("yr"), "heavy", "old").agg(pl.len().alias("attempts"), pl.col("is_stockout").sum().alias("so")).collect())
for lab, f in [("all stations", pl.lit(True)), ("same stations", pl.col("old"))]:
    t = att.filter(f).group_by("yr", "heavy").agg((pl.col("so").sum() / pl.col("attempts").sum() * 100).round(2).alias("stockout_pct")).sort("heavy", "yr")
    d = {(r["yr"], r["heavy"]): r["stockout_pct"] for r in t.iter_rows(named=True)}
    say(f"June 2W stockout %, {lab}: {t.to_dicts()} -> heavy {d[(2025, True)] - d[(2024, True)]:+.2f} pp vs light {d[(2025, False)] - d[(2024, False)]:+.2f} pp")
qt = (hourly(["telemetry_status", "cabinet_temp_c", "packs_quarantined"]).filter(pl.col("telemetry_status") == "ok")
      .group_by((pl.col("cabinet_temp_c") >= 39).alias("hot39")).agg((pl.col("packs_quarantined") > 0).mean().round(4).alias("share_any"), pl.len()).collect())
say("quarantine share of station-hours by cabinet >= 39C:", qt.sort("hot39").to_dicts())
say(f"runtime {time.time() - T0:.1f}s")
