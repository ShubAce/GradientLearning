# %% [markdown]
# # Q6 verification: independent re-derivation of the new-rider retention findings
# Verifier script (not the analyst's code). It re-derives every headline number of `outputs/findings/q6.md`
# from `data/clean` through the `src/common.py` loaders, then runs extra checks on the method:
# survival handling of riders with no completed swap, seasonal-mix bias in pooled comparisons, a
# non-parametric decomposition of the summer dip, same-season checks for May/Jun cohorts on shorter outcomes,
# and an exposure-free failure-rate specification.

# %%
import sys
import datetime as dt
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from common import *  # noqa: E402,F401,F403

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
from scipy import stats  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from statsmodels.stats.proportion import proportions_ztest  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
END = dt.datetime(2025, 7, 1)      # exclusive end of the event log
CUT = dt.datetime(2025, 5, 1)      # first attempt before this -> >= 60 days of follow-up


def pc(x, d=1):
    return f"{100 * x:.{d}f}%"


def ztest(a, b):
    """Two-proportion z-test on boolean/0-1 Series a, b. Returns (pa, na, pb, nb, p)."""
    k = np.array([a.sum(), b.sum()])
    n = np.array([len(a), len(b)])
    _, p = proportions_ztest(k, n)
    return a.mean(), len(a), b.mean(), len(b), p


# %% [markdown]
# ## 1. New-rider base, day index and outcomes (re-derived)

# %%
R = riders().filter(pl.col("signup_date") >= dt.date(2024, 1, 1))
print(f"Signups >= 2024-01-01: {R.height:,}")
EVC = ["rider_id", "event_ts", "date", "station_id", "city", "event_type", "is_completed", "is_stockout",
       "is_abandoned", "is_system_error", "is_cancelled", "is_service_failure", "vehicle_class",
       "competitor_active", "supplier_out", "is_independent", "month"]
E = (events(EVC).join(R.select("rider_id").lazy(), on="rider_id", how="semi")
     .collect(engine="streaming").sort(["rider_id", "event_ts"]))
t0 = E.group_by("rider_id").agg(t0=pl.col("event_ts").min())
E = E.join(t0, on="rider_id").with_columns(dd=(pl.col("event_ts") - pl.col("t0")).dt.total_seconds() / 86400.0)
E = E.with_columns(day=pl.col("dd").floor().cast(pl.Int32))
print(f"New-rider attempts: {E.height:,}; riders with >= 1 attempt: {t0.height:,}")
# cross-check t0 against riders.first_attempt_ts
chk = t0.join(R.select("rider_id", "first_attempt_ts"), on="rider_id")
print("t0 == riders.first_attempt_ts for", int((chk["t0"] == chk["first_attempt_ts"]).sum()), "of", chk.height)

cd = E.filter(pl.col("is_completed")).select("rider_id", "day").unique()
OUT = cd.group_by("rider_id").agg(
    ret=pl.col("day").is_between(30, 59).any(),
    any30=(pl.col("day") >= 30).any(),
    r_d1_7=pl.col("day").is_between(1, 7).any(),
    r14=pl.col("day").is_between(14, 27).any(),
    r90=pl.col("day").is_between(90, 119).any())
early = E.filter(pl.col("dd") < 14)
F = early.group_by("rider_id").agg(
    e_att=pl.len(), e_comp=pl.col("is_completed").sum(), e_fail=pl.col("is_service_failure").sum(),
    e_so=pl.col("is_stockout").sum(), e_ab=pl.col("is_abandoned").sum(),
    e_se=pl.col("is_system_error").sum(), e_ca=pl.col("is_cancelled").sum(),
    first_fail=pl.col("is_service_failure").sort_by("event_ts").first(),
    comp_any=pl.col("competitor_active").any(),
    kyron_share=(pl.col("supplier_out") == "Kyron").filter(pl.col("is_completed")).mean(),
    main_station=pl.col("station_id").mode().sort().first(),
    first_city=pl.col("city").sort_by("event_ts").first())
st = stations().select("station_id", main_city="city")
D0 = (R.join(t0, on="rider_id", how="inner").join(OUT, on="rider_id", how="left").join(F, on="rider_id", how="left")
      .join(st, left_on="main_station", right_on="station_id", how="left")
      .with_columns(pl.col("ret", "any30", "r_d1_7", "r14", "r90").fill_null(False),
                    coh=pl.col("t0").dt.truncate("1mo").dt.date(),
                    end_day=(pl.lit(END) - pl.col("t0")).dt.total_seconds() / 86400.0)
      .to_pandas())
D0["coh"] = pd.to_datetime(D0["coh"])
D0["y"] = (~D0["ret"]).astype(int)
D0["fail_grp"] = pd.cut(D0["e_fail"], [-1, 0, 1, 2, 999], labels=["0", "1", "2", "3+"]).astype(str)
D0["fail2p"] = (D0["e_fail"] >= 2).astype(int)
D0["sa"] = D0["e_so"] + D0["e_ab"]                         # stockouts + abandoned queues only
D0["sa2p"] = (D0["sa"] >= 2).astype(int)
D0["first_fail"] = D0["first_fail"].astype(int)
D0["comp_any"] = D0["comp_any"].astype(int)
D0["w3"] = (D0["vehicle_class"] == "3W").astype(int)
D0["indep"] = D0["partner_id"].isna().astype(int)
D0["segment"] = np.where(D0["partner_id"].isna(), np.where(D0["plan_type"] == "prepaid_pack", "IndepPrepaid", "IndepPAYG"),
                         np.where(D0["partner_id"] == ZIPDROP_ID, "ZipDrop", "OtherPartner"))
D0["coh_s"] = D0["coh"].dt.strftime("%Y-%m")
D0["coh_q"] = D0["coh"].dt.to_period("Q").astype(str)
D0["kyron25"] = D0["kyron_share"].fillna(0) / 0.25
D0["log_att"] = np.log(D0["e_att"])
D0["fail_rate"] = D0["e_fail"] / D0["e_att"]
D0["stn"] = D0["main_station"].astype("category").cat.codes
D = D0[D0["t0"] < CUT].copy()
print(f"Eligible (first attempt < 2025-05-01): {len(D):,}; retained D31-60 {pc(1 - D['y'].mean())}; "
      f"not retained {int(D['y'].sum()):,} = {pc(D['y'].mean(), 2)}")
print("Riders with zero completed swaps ever:", int((D0["e_comp"].fillna(0) == 0).sum()), "(zero in first 14 d)")

# %% [markdown]
# ## 2. Claim 1: cohort trend, same-season tests, naive metric

# %%
ct = D.groupby("coh").agg(n=("y", "size"), ret=("ret", "mean"), fail_mean=("e_fail", "mean")).reset_index()
fr = D0.groupby("coh").agg(att=("e_att", "sum"), fail=("e_fail", "sum"), naive=("any30", "mean")).reset_index()
fr["fail_rate_att"] = fr["fail"] / fr["att"]
ct = ct.merge(fr[["coh", "fail_rate_att"]], on="coh")
print(ct.round(4).to_string(index=False))
print("Naive any-swap-after-day-30 by cohort:", {str(c.date())[:7]: round(v, 3) for c, v in zip(fr["coh"], fr["naive"])})
m = D["coh"]
cmp_def = {
    "JanMar24 vs JanMar25": (m.between("2024-01-01", "2024-03-01"), m.between("2025-01-01", "2025-03-01")),
    "Apr24 vs Apr25": (m == "2024-04-01", m == "2025-04-01"),
    "JanMar24 vs Apr25": (m.between("2024-01-01", "2024-03-01"), m == "2025-04-01"),
    "JanMar24 vs AprJun24": (m.between("2024-01-01", "2024-03-01"), m.between("2024-04-01", "2024-06-01")),
    "H1-24 vs Oct24-Apr25 (MIXED seasons)": (m.between("2024-01-01", "2024-06-01"), m.between("2024-10-01", "2025-04-01")),
    "JanApr24 vs JanApr25 (season-matched)": (m.between("2024-01-01", "2024-04-01"), m.between("2025-01-01", "2025-04-01")),
}
for k, (a, b) in cmp_def.items():
    pa, na, pb, nb, p = ztest(D.loc[a, "ret"].astype(int), D.loc[b, "ret"].astype(int))
    print(f"{k:42s} {pc(pa)} (n={na:,}) -> {pc(pb)} (n={nb:,})  diff {100*(pb-pa):+.2f} pp  p={p:.4f}")
# month-mix-standardised H1-24 vs Oct24-Apr25: compare calendar-month-matched only
print("Share of summer (Apr-Jun) cohorts: H1-24", pc(m[m.between('2024-01-01', '2024-06-01')].dt.month.isin([4, 5, 6]).mean()),
      "| Oct24-Apr25", pc(m[m.between('2024-10-01', '2025-04-01')].dt.month.isin([4, 5, 6]).mean()))
r = stats.pearsonr(ct["fail_mean"], ct["ret"])
r2 = stats.pearsonr(ct["fail_rate_att"], ct["ret"])
print(f"corr(mean early failures, retention) over {len(ct)} cohorts r={r[0]:.3f} p={r[1]:.2e}; "
      f"with attempt-level failure rate r={r2[0]:.3f}")
# season-adjusted (partial) correlation: remove calendar-quarter means from both series
ct["q"] = ct["coh"].dt.month.map(lambda x: (x - 1) // 3)
ct["ret_dm"] = ct["ret"] - ct.groupby("q")["ret"].transform("mean")
ct["fail_dm"] = ct["fail_mean"] - ct.groupby("q")["fail_mean"].transform("mean")
r3 = stats.pearsonr(ct["fail_dm"], ct["ret_dm"])
print(f"Within-quarter-of-year correlation (season demeaned): r={r3[0]:.3f} p={r3[1]:.3f}")

# %% [markdown]
# ## 3. Survival by quarterly cohort (KM, churn = 30+ idle days) and the zero-swap riders

# %%
from lifelines import KaplanMeierFitter, CoxPHFitter  # noqa: E402
from lifelines.statistics import logrank_test, multivariate_logrank_test  # noqa: E402

endd = D0[["rider_id", "end_day"]]
a = (cd.sort(["rider_id", "day"]).join(pl.from_pandas(endd), on="rider_id")
     .with_columns(nxt=pl.col("day").shift(-1).over("rider_id")))
a = a.with_columns(gap=pl.coalesce(pl.col("nxt").cast(pl.Float64), pl.col("end_day")) - pl.col("day"))
ch = (a.filter((pl.col("gap") >= 30) & (pl.col("day") <= pl.col("end_day") - 30))
      .group_by("rider_id").agg(churn_day=pl.col("day").min()))
S0 = D0.merge(ch.to_pandas(), on="rider_id", how="left")
S0["E_an"] = S0["churn_day"].notna()                               # analyst's convention
S0["T_an"] = S0["churn_day"].fillna(S0["end_day"] - 30)
zero = S0["e_comp"].fillna(0).eq(0) & ~S0["rider_id"].isin(cd["rider_id"].unique().to_list())
print(f"Riders with NO completed swap at all: {int(zero.sum())}; under the analyst's convention they are "
      f"censored survivors (E=0, T=end_day-30). Corrected: churn at day 0.")
S0["E_fx"] = S0["E_an"] | (zero & (S0["end_day"] >= 30))
S0["T_fx"] = np.where(zero, 0.0, S0["T_an"])
Sq = S0[S0["T_an"] > 0]
for qa, qb in [("2024Q1", "2025Q1"), ("2024Q1", "2024Q2"), ("2024Q2", "2025Q2")]:
    ga, gb = Sq[Sq["coh_q"] == qa], Sq[Sq["coh_q"] == qb]
    lt = logrank_test(ga["T_an"], gb["T_an"], ga["E_an"], gb["E_an"])
    print(f"log-rank {qa} vs {qb}: p={lt.p_value:.4f}")
for q in ["2024Q1", "2024Q2", "2024Q4", "2025Q1", "2025Q2"]:
    g = Sq[Sq["coh_q"] == q]
    k = KaplanMeierFitter().fit(g["T_an"], g["E_an"])
    s = {h: float(k.survival_function_at_times(h).iloc[0]) for h in (60, 90)}
    print(f"KM {q}: n={len(g):,} S60={pc(s[60])} S90={pc(s[90])}")
print("Churn events (analyst convention):", int(S0["E_an"].sum()))

# %% [markdown]
# ## 4. Claim 3: dose-response and the rider-level logit

# %%
dose = D.groupby("fail_grp")["y"].agg(["mean", "size"])
print(dose.round(4))
print(f"2+ failures: n={int(D['fail2p'].sum()):,} ({pc(D['fail2p'].mean())}) non-ret {pc(D.loc[D.fail2p == 1, 'y'].mean())} "
      f"vs 0-1 {pc(D.loc[D.fail2p == 0, 'y'].mean())}")
print("2+ stockouts:", pc(D.loc[D.e_so >= 2, "y"].mean()), int((D.e_so >= 2).sum()), "| 0 stockouts", pc(D.loc[D.e_so == 0, "y"].mean()))
print("2+ abandons:", pc(D.loc[D.e_ab >= 2, "y"].mean()), int((D.e_ab >= 2).sum()), "| 0 abandons", pc(D.loc[D.e_ab == 0, "y"].mean()))
print("1 system error:", pc(D.loc[D.e_se == 1, "y"].mean()), int((D.e_se == 1).sum()))
# dose-response within early-activity strata (exposure confound check)
D["att_bin"] = pd.cut(D["e_att"], [0, 5, 9, 14, 19, 999], labels=["1-5", "6-9", "10-14", "15-19", "20+"]).astype(str)
print("Non-retention by activity band:", D.groupby("att_bin")["y"].agg(["mean", "size"]).round(3).to_dict("index"))
print("Non-retention by failures within activity band (rows=band):")
print(pd.crosstab(D["att_bin"], D["fail_grp"], values=D["y"], aggfunc="mean").round(3))
print(pd.crosstab(D["att_bin"], D["fail_grp"]))


def logit(f, data, cluster=True):
    if cluster:
        return smf.logit(f, data=data).fit(disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": data["stn"]})
    return smf.logit(f, data=data).fit(disp=0, maxiter=300)


def orow(mdl, term):
    ci = mdl.conf_int().loc[term]
    return f"OR {np.exp(mdl.params[term]):.2f} [{np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f}] p={mdl.pvalues[term]:.3g}"


D["main_city"] = D["main_city"].fillna("unknown")
CTL = " + first_fail + w3 + comp_any + kyron25 + C(segment, Treatment('OtherPartner')) + C(coh_s) + C(main_city)"
mA = logit("y ~ C(fail_grp, Treatment('0'))" + CTL, D)
for t in ["1", "2", "3+"]:
    print(f"Lean logit fail_grp {t}: {orow(mA, f'C(fail_grp, Treatment(\'0\'))[T.{t}]')}")
for t in ["first_fail", "w3", "comp_any", "kyron25"]:
    print(f"Lean logit {t}: {orow(mA, t)}")
mAa = logit("y ~ C(fail_grp, Treatment('0'))" + CTL + " + log_att", D)
for t in ["1", "2", "3+"]:
    print(f"+log attempts fail_grp {t}: {orow(mAa, f'C(fail_grp, Treatment(\'0\'))[T.{t}]')}")
D["e_so4"], D["e_ab4"], D["e_se2"], D["e_ca2"] = D.e_so.clip(upper=4), D.e_ab.clip(upper=4), D.e_se.clip(upper=2), D.e_ca.clip(upper=2)
mB = logit("y ~ e_so4 + e_ab4 + e_se2 + e_ca2" + CTL, D)
for t in ["e_so4", "e_ab4", "e_se2", "e_ca2"]:
    print(f"Per-type logit {t}: {orow(mB, t)}")
# exposure-free specification: failure RATE in first 14 days (share of attempts failed)
D["frate_bin"] = pd.cut(D["fail_rate"], [-0.01, 0, 0.05, 0.10, 0.20, 1.01], labels=["0", "0-5%", "5-10%", "10-20%", ">20%"]).astype(str)
print("Non-retention by early failure RATE:", D.groupby("frate_bin")["y"].agg(["mean", "size"]).round(3).to_dict("index"))
mR = logit("y ~ C(frate_bin, Treatment('0'))" + CTL + " + log_att", D)
for t in ["0-5%", "5-10%", "10-20%", ">20%"]:
    print(f"Failure-rate logit (+log att) {t}: {orow(mR, f'C(frate_bin, Treatment(\'0\'))[T.{t}]')}")

# %% [markdown]
# ## 5. Claim 4: attempt-level exits (own construction of prior failures via as-of join)

# %%
A = (E.select("rider_id", "event_ts", "event_type", "is_service_failure", "vehicle_class", "is_independent",
              "competitor_active", "supplier_out", "dd", "month")
     .with_columns(nxt=pl.col("event_ts").shift(-1).over("rider_id"),
                   cf=pl.col("is_service_failure").cast(pl.Int32).cum_sum().over("rider_id"),
                   t7=pl.col("event_ts") - pl.duration(days=7)))
look = A.select("rider_id", ts_l="event_ts", cf_l="cf").sort("ts_l")
A = A.sort("t7").join_asof(look, left_on="t7", right_on="ts_l", by="rider_id", strategy="backward")
A = A.with_columns(prior7=((pl.col("cf") - pl.col("is_service_failure").cast(pl.Int32)) - pl.col("cf_l").fill_null(0)).clip(0, 3))
A = A.filter(pl.col("event_ts") < END - dt.timedelta(days=30)).with_columns(
    exit=((pl.coalesce("nxt", pl.lit(END)) - pl.col("event_ts")).dt.total_seconds() / 86400.0 >= 30),
    ten=pl.when(pl.col("dd") < 14).then(pl.lit("a")).when(pl.col("dd") < 60).then(pl.lit("b"))
    .when(pl.col("dd") < 180).then(pl.lit("c")).otherwise(pl.lit("d")),
    kyron=(pl.col("supplier_out") == "Kyron").fill_null(False))
print(f"Attempts analysed: {A.height:,}; exits: {int(A['exit'].sum()):,}")
raw = A.group_by("event_type").agg(n=pl.len(), k=pl.col("exit").sum()).with_columns(per1000=1000 * pl.col("k") / pl.col("n"))
print(raw.sort("event_type"))
rawp = A.group_by("prior7").agg(n=pl.len(), k=pl.col("exit").sum()).with_columns(per1000=1000 * pl.col("k") / pl.col("n"))
print(rawp.sort("prior7"))
G = (A.group_by(["event_type", "prior7", "ten", "month", "vehicle_class", "is_independent", "competitor_active", "kyron"])
     .agg(n=pl.len(), k=pl.col("exit").sum()).to_pandas())
G["month"] = G["month"].astype(str)
G["prior7"] = G["prior7"].astype(str)
G["et"] = G["event_type"].map({"swap_completed": "a_completed", "failed_no_charged_battery": "stockout",
                               "abandoned_queue": "abandoned", "failed_system_error": "syserr", "cancelled_by_rider": "cancel"})
X = pd.get_dummies(G[["et", "prior7", "ten", "month", "vehicle_class"]], drop_first=True).astype(float)
X["indep"] = G["is_independent"].astype(float)
X["comp"] = G["competitor_active"].astype(float)
X["kyron"] = G["kyron"].astype(float)
X = sm.add_constant(X)
glm = sm.GLM(np.column_stack([G["k"], G["n"] - G["k"]]), X, family=sm.families.Binomial()).fit()
ci = glm.conf_int()
for t in ["et_stockout", "et_abandoned", "et_syserr", "et_cancel", "prior7_1", "prior7_2", "prior7_3", "comp", "kyron",
          "vehicle_class_3W", "ten_b", "ten_c"]:
    print(f"exit GLM {t}: OR {np.exp(glm.params[t]):.3f} [{np.exp(ci.loc[t, 0]):.2f}-{np.exp(ci.loc[t, 1]):.2f}] p={glm.pvalues[t]:.3g}")
del A, look

# %% [markdown]
# ## 6. Claim 5: landmark KM and Cox from day 14 (analyst convention vs zero-swap fix)

# %%
for conv in ["an", "fx"]:
    L = S0[(S0[f"T_{conv}"] >= 14) & (S0["end_day"] >= 45)].copy()
    L["T"] = L[f"T_{conv}"] - 14
    L["E"] = L[f"E_{conv}"].astype(int)
    lr = multivariate_logrank_test(L["T"], L["fail_grp"], L["E"])
    s180 = {}
    for g in ["0", "1", "2", "3+"]:
        gg = L[L["fail_grp"] == g]
        s180[g] = float(KaplanMeierFitter().fit(gg["T"], gg["E"]).survival_function_at_times(180).iloc[0])
    L["f1"], L["f2"], L["f3"] = (L.fail_grp == "1").astype(int), (L.fail_grp == "2").astype(int), (L.fail_grp == "3+").astype(int)
    L["season"] = L["coh"].dt.month.map(SEASON_OF_MONTH)
    L["main_city"] = L["main_city"].fillna("unknown")
    cph = CoxPHFitter(penalizer=0.001).fit(L[["T", "E", "f1", "f2", "f3", "first_fail", "w3", "indep", "comp_any", "season", "main_city"]],
                                           "T", "E", strata=["season", "main_city"])
    sm_ = cph.summary
    print(f"[{conv}] landmark n={len(L):,} events={int(L['E'].sum()):,} log-rank p={lr.p_value:.2e} "
          f"S180: " + ", ".join(f"{g}={pc(v)}" for g, v in s180.items()))
    for t in ["f1", "f2", "f3", "comp_any", "w3", "first_fail"]:
        print(f"   Cox {t}: HR {sm_.loc[t, 'exp(coef)']:.3f} [{sm_.loc[t, 'exp(coef) lower 95%']:.2f}-{sm_.loc[t, 'exp(coef) upper 95%']:.2f}] p={sm_.loc[t, 'p']:.3g}")
    zero_in_L = int((L["e_comp"].fillna(0) == 0).sum()) if conv == "an" else None
    if conv == "an":
        print(f"   zero-completed riders inside the analyst's landmark set: {zero_in_L}")

# %% [markdown]
# ## 7. Claim 5b: within-segment stability of the 2+ failure effect

# %%
segs = {"All": D.index == D.index, "Independent": D.indep == 1, "Partner": D.indep == 0, "2W": D.w3 == 0, "3W": D.w3 == 1,
        "H1-2024": D.coh < "2024-07-01", "H2-2024": (D.coh >= "2024-07-01") & (D.coh < "2025-01-01"), "2025": D.coh >= "2025-01-01",
        "Winter-entry": D.coh.dt.month.isin([11, 12, 1, 2]), "ZipDrop": D.partner_id == ZIPDROP_ID}
for c in CITIES:
    segs[c] = D.main_city == c
rows = []
for s, msk in segs.items():
    sub = D[msk]
    covs = [c for c in ["first_fail", "w3", "indep", "comp_any"] if sub[c].nunique() > 1]
    f = "y ~ fail2p + " + " + ".join(covs) + " + C(coh_q)" + (" + C(main_city)" if sub.main_city.nunique() > 1 else "")
    mm = smf.logit(f, data=sub).fit(disp=0, maxiter=300)
    ci = mm.conf_int().loc["fail2p"]
    rows.append((s, len(sub), np.exp(mm.params["fail2p"]), np.exp(ci[0]), np.exp(ci[1]), mm.pvalues["fail2p"]))
stab = pd.DataFrame(rows, columns=["segment", "n", "OR", "lo", "hi", "p"])
print(stab.round(3).to_string(index=False))
print(f"OR>1: {(stab.OR > 1).sum()}/{len(stab)}; significant: {((stab.OR > 1) & (stab.p < 0.05)).sum()}/{len(stab)} "
      f"(excluding pooled 'All': {((stab.OR > 1) & (stab.p < 0.05) & (stab.segment != 'All')).sum()}/{len(stab) - 1})")

# %% [markdown]
# ## 8. Claim 6: how much of the summer dip do early failures explain? (non-parametric + model)

# %%
ref = D[D.coh.between("2024-01-01", "2024-03-01")]
for lab, msk in {"AprJun24": D.coh.between("2024-04-01", "2024-06-01"), "Apr25": D.coh == "2025-04-01",
                 "JanMar25": D.coh.between("2025-01-01", "2025-03-01")}.items():
    c = D[msk]
    w_ref, w_c = ref.fail_grp.value_counts(normalize=True), c.fail_grp.value_counts(normalize=True)
    r_ref, r_c = ref.groupby("fail_grp").y.mean(), c.groupby("fail_grp").y.mean()
    obs = c.y.mean() - ref.y.mean()
    expl1 = (c.y.mean() - (w_ref * r_c).sum())          # comparison-group rates, reference failure mix
    expl2 = ((w_c * r_ref).sum() - ref.y.mean())       # reference rates, comparison failure mix
    print(f"{lab}: observed gap {100*obs:+.2f} pp; failure-mix effect {100*expl1:+.2f} pp (cmp rates) / "
          f"{100*expl2:+.2f} pp (ref rates) -> {pc(expl1/obs, 0)} / {pc(expl2/obs, 0)} of observed")
    print(f"     non-retention within fail groups ref vs cmp: " +
          ", ".join(f"{g}: {pc(r_ref.get(g, np.nan))}->{pc(r_c.get(g, np.nan))}" for g in ["0", "1", "2", "3+"]))
# pooled logit without cohort effects: model-predicted gap vs failure counterfactual (cap at 1)
fP = "y ~ C(fail_grp, Treatment('0')) + first_fail + w3 + comp_any + kyron25 + C(segment) + C(main_city)"
mP = smf.logit(fP, data=D).fit(disp=0, maxiter=300)
cf = D.copy()
cf["fail_grp"] = np.where(cf.fail_grp.isin(["2", "3+"]), "1", cf.fail_grp)
summer = D.coh.dt.month.isin([4, 5, 6])
print(f"What-if cap at 1 failure: overall {pc(mP.predict(D).mean(), 2)} -> {pc(mP.predict(cf).mean(), 2)}; "
      f"summer-entry {pc(mP.predict(D[summer]).mean(), 2)} -> {pc(mP.predict(cf[summer]).mean(), 2)}")
pr = D.assign(p=mP.predict(D)).groupby("coh").agg(obs=("y", "mean"), pred=("p", "mean"))
print(f"Cohort NON-retention: observed range {pc(pr.obs.min())}-{pc(pr.obs.max())} (sd {100*pr.obs.std():.2f} pp); "
      f"model-predicted (no cohort FE) {pc(pr.pred.min())}-{pc(pr.pred.max())} (sd {100*pr.pred.std():.2f} pp)")
npm = D.groupby("coh").size().loc["2024-10-01":"2025-04-01"].mean()
print(f"Eligible new riders per month (Oct-24..Apr-25 cohorts): {npm:.0f}")

# %% [markdown]
# ## 9. Missed angle: same-season comparisons for May/Jun cohorts on shorter outcomes

# %%
for lab, col, cutoff in [("R14 (swap in days 14-27)", "r14", END - dt.timedelta(days=28)),
                         ("D1-7 (swap in days 1-7)", "r_d1_7", END - dt.timedelta(days=8))]:
    Dx = D0[D0.t0 < cutoff]
    print(lab)
    for mo in [1, 2, 3, 4, 5, 6]:
        a_ = Dx[(Dx.coh.dt.year == 2024) & (Dx.coh.dt.month == mo)][col].astype(int)
        b_ = Dx[(Dx.coh.dt.year == 2025) & (Dx.coh.dt.month == mo)][col].astype(int)
        if len(b_) < 50:
            continue
        pa, na, pb, nb, p = ztest(a_, b_)
        print(f"   month {mo}: 2024 {pc(pa)} (n={na:,}) vs 2025 {pc(pb)} (n={nb:,}) diff {100*(pb-pa):+.2f} pp p={p:.3f}")
    # mean early failures by cohort month for the same comparison
print("Mean early failures per rider by cohort (all riders):",
      {str(k.date())[:7]: round(v, 2) for k, v in D0.groupby("coh").e_fail.mean().items()})

# %% [markdown]
# ## 10. Secondary / not-supported claims and recommendation numbers

# %%
print("3W share by quarter:", D.groupby("coh_q").w3.mean().round(3).to_dict())
print("Raw non-retention 3W vs 2W:", pc(D.loc[D.w3 == 1, "y"].mean()), pc(D.loc[D.w3 == 0, "y"].mean()))
print("3W by segment:", pd.crosstab(D.w3, D.segment).to_dict())
c25 = D[D.coh >= "2025-01-01"]
print(f"2025 cohorts competitor-exposed: n={int(c25.comp_any.sum())}, non-ret {pc(c25.loc[c25.comp_any == 1, 'y'].mean())} vs "
      f"{pc(c25.loc[c25.comp_any == 0, 'y'].mean())}; all eligible exposed: {int(D.comp_any.sum())}")
# base-price DiD (independent vs partner, cohorts Jan-Sep 2024)
Dp = D[D.coh < "2024-10-01"].assign(post=lambda x: (x.coh >= "2024-07-01").astype(int))
mdid = smf.logit("y ~ indep + indep:post + w3 + C(fail_grp) + C(coh_s) + C(main_city)", data=Dp).fit(disp=0)
print("Base-price DiD indep:post:", orow(mdid, "indep:post"))
# activity bins and early-warning list
print("Activity bins (1-5, 6-9):", D.groupby(pd.cut(D.e_att, [0, 5, 9, 999]))["y"].agg(["mean", "size"]).round(3).to_dict("index"))
# 2nd failure riders (all cohorts), any failure vs stockout/abandon only
for lab, flag in [("any service failure", "is_service_failure"), ("stockout/abandon only", None)]:
    ee = early.filter(pl.col(flag)) if flag else early.filter(pl.col("is_stockout") | pl.col("is_abandoned"))
    sec = ee.sort(["rider_id", "event_ts"]).with_columns(k=pl.cum_count("event_ts").over("rider_id")).filter(pl.col("k") == 2)
    print(f"Riders reaching a 2nd {lab} within 14 days: {sec.height:,}; median day {sec['dd'].median():.1f}")
sa = (early.filter(pl.col("is_stockout") | pl.col("is_abandoned")).group_by(pl.col("date").dt.truncate("1mo").alias("m"))
      .len().sort("m").to_pandas())
sa["mo"] = pd.to_datetime(sa["m"]).dt.month
print(f"Stockout+abandon events in first 14 days: Apr-Jun mean {sa.loc[sa.mo.isin([4,5,6]), 'len'].mean():.0f}/month, "
      f"other {sa.loc[~sa.mo.isin([4,5,6]), 'len'].mean():.0f}/month; by month {dict(zip(sa.m.astype(str).str[:7], sa['len']))}")
retd = D[D.ret]["rider_id"]
sw = E.filter(pl.col("rider_id").is_in(retd.tolist()) & pl.col("is_completed") & pl.col("day").is_between(30, 59)).height
print(f"Completed swaps per retained rider in days 31-60: {sw / len(retd):.1f}")
print("done")

# %% [markdown]
# ## 11. Deeper checks: headline label, within-rider trigger test, honest counterfactual, 3W fragility,
# post-day-14 failures, battery-range channel, win-back

# %%
# (a) headline says "2+ stockouts or abandoned queues"; the 13.1% -> 22.6% numbers are for 2+ service failures
print(f"2+ stockout/abandon (excl. system errors): n={int(D.sa2p.sum()):,} non-ret {pc(D.loc[D.sa2p == 1, 'y'].mean())} "
      f"vs 0-1 {pc(D.loc[D.sa2p == 0, 'y'].mean())}")
mH2 = logit("y ~ fail2p" + CTL, D)
print("2+ vs 0-1 service failures, cohort-month + city FE:", orow(mH2, "fail2p"))

# (b) truly within-rider trigger test: Mantel-Haenszel OR of exit after stockout/abandon, strata = rider (x month)
A2 = (E.select("rider_id", "event_ts", "event_type", "month")
      .with_columns(nxt=pl.col("event_ts").shift(-1).over("rider_id"))
      .filter(pl.col("event_ts") < END - dt.timedelta(days=30))
      .with_columns(Y=((pl.coalesce("nxt", pl.lit(END)) - pl.col("event_ts")).dt.total_seconds() / 86400.0 >= 30),
                    X=pl.col("event_type").is_in(["failed_no_charged_battery", "abandoned_queue"])))
for lab, keys in [("rider", ["rider_id"]), ("rider x month", ["rider_id", "month"])]:
    s = (A2.group_by(keys).agg(a=(pl.col("X") & pl.col("Y")).sum(), b=(~pl.col("X") & pl.col("Y")).sum(),
                               c=(pl.col("X") & ~pl.col("Y")).sum(), d=(~pl.col("X") & ~pl.col("Y")).sum(), n=pl.len())
         .filter((pl.col("a") + pl.col("b")) > 0))
    sp = s.select("a", "b", "c", "d", "n").to_pandas().astype(float)
    Ri, Si = sp.a * sp.d / sp.n, sp.b * sp.c / sp.n
    Pi, Qi = (sp.a + sp.d) / sp.n, (sp.b + sp.c) / sp.n
    Rs, Ss = Ri.sum(), Si.sum()
    var = (Pi * Ri).sum() / (2 * Rs ** 2) + (Pi * Si + Qi * Ri).sum() / (2 * Rs * Ss) + (Qi * Si).sum() / (2 * Ss ** 2)
    orm = Rs / Ss
    print(f"Within-{lab} Mantel-Haenszel OR (exit after stockout/abandon vs other attempts): {orm:.2f} "
          f"[{orm * np.exp(-1.96 * var ** 0.5):.2f}-{orm * np.exp(1.96 * var ** 0.5):.2f}] ({s.height:,} strata with an exit)")
del A2

# (c) honest counterfactual for Fig q6_01: observed + (counterfactual - model prediction), lean pooled model
RNG = np.random.default_rng(1)
rows = []
for c_, g in D.groupby("coh"):
    pred = mP.predict(g).mean()
    vals = []
    for _ in range(30):
        cfg = g.copy()
        idx = RNG.integers(0, len(ref), len(g))
        cfg["fail_grp"], cfg["first_fail"] = ref["fail_grp"].values[idx], ref["first_fail"].values[idx]
        vals.append(mP.predict(cfg).mean())
    rows.append(dict(coh=str(c_.date())[:7], obs_ret=1 - g.y.mean(), model_ret=1 - pred, cf_ret=1 - np.mean(vals),
                     obs_fail_neutral=1 - g.y.mean() + (pred - np.mean(vals))))
cfc = pd.DataFrame(rows)
print(cfc.round(3).to_string(index=False))
print(f"Observed retention sd {100*cfc.obs_ret.std():.2f} pp (range {pc(cfc.obs_ret.min())}-{pc(cfc.obs_ret.max())}); "
      f"model-predicted sd {100*cfc.model_ret.std():.2f} pp; pure-model counterfactual sd {100*cfc.cf_ret.std():.2f} pp; "
      f"observed with early failures neutralised: sd {100*cfc.obs_fail_neutral.std():.2f} pp, "
      f"range {pc(cfc.obs_fail_neutral.min())}-{pc(cfc.obs_fail_neutral.max())}")

# (d) analyst-style full covariate set: is the 3W OR of 1.52 robust?
EX = (events(["rider_id", "event_ts", "is_completed", "queue_wait_sec", "km_since_last_swap", "tariff_code", "soh_out_pct"])
      .join(t0.lazy(), on="rider_id", how="inner")
      .filter((pl.col("event_ts") - pl.col("t0")) < pl.duration(days=14))
      .group_by("rider_id").agg(
          q_mean=pl.col("queue_wait_sec").filter(pl.col("is_completed")).mean() / 60.0,
          km_mean=pl.col("km_since_last_swap").mean(),
          peak_share=(pl.col("tariff_code") == "PEAK").filter(pl.col("is_completed")).mean(),
          soh_mean=pl.col("soh_out_pct").filter(pl.col("is_completed")).mean())
      .collect(engine="streaming").to_pandas())
TK = (tickets().filter(~pl.col("is_test")).join(t0, on="rider_id", how="inner")
      .filter((pl.col("created_ts") >= pl.col("t0")) & (pl.col("created_ts") < pl.col("t0") + pl.duration(days=14)))
      .group_by("rider_id").agg(tk_any=pl.len() > 0, tk_unres_any=(pl.col("resolution_status") == "unresolved").any())
      .to_pandas())
ST = stations().select("station_id", "expansion_wave", "charger_generation", "connectivity_tier").to_pandas()
DD = D.merge(EX, on="rider_id", how="left").merge(TK, on="rider_id", how="left").merge(
    ST, left_on="main_station", right_on="station_id", how="left")
for c in ["q_mean", "km_mean", "soh_mean"]:
    DD[c] = DD[c].fillna(DD[c].median())
DD["peak25"] = DD["peak_share"].fillna(0) / 0.25
DD["km10"] = DD["km_mean"] / 10
DD["soh10"] = DD["soh_mean"] / 10
DD["tk_any"] = DD["tk_any"].fillna(False).astype(int)
DD["tk_unres_any"] = DD["tk_unres_any"].fillna(False).astype(int)
DD["kyc_no"] = (~DD["kyc_verified"].astype(bool)).astype(int)
DD["age_band"] = DD["age_band"].fillna("unknown")
FULL = ("y ~ C(fail_grp, Treatment('0')) + first_fail + q_mean + kyron25 + km10 + peak25 + comp_any + tk_any + tk_unres_any + w3"
        " + C(segment, Treatment('OtherPartner')) + C(signup_channel) + kyc_no + C(age_band) + C(expansion_wave)"
        " + C(charger_generation) + C(connectivity_tier) + C(coh_s) + C(main_city)")
mF = logit(FULL, DD)
print("Analyst-style full spec:", "2 fails", orow(mF, "C(fail_grp, Treatment('0'))[T.2]"), "| w3", orow(mF, "w3"), "| km10", orow(mF, "km10"))
mF2 = logit(FULL.replace(" + km10", ""), DD)
print("   same without km10: w3", orow(mF2, "w3"))
mF3 = logit(FULL.replace(" + C(signup_channel)", ""), DD)
print("   same without signup_channel: w3", orow(mF3, "w3"))
print("   3W x signup channel:", pd.crosstab(DD.w3, DD.signup_channel).to_dict("index"))
print("   mean km per swap 2W vs 3W:", DD.groupby("w3").km_mean.mean().round(1).to_dict())

# (g) battery-range channel: SoH of packs received and km per swap, with and without the activity control
for spec in ["", " + log_att"]:
    mS = logit("y ~ C(fail_grp, Treatment('0')) + soh10 + km10 + kyron25 + first_fail + w3 + comp_any"
               " + C(segment, Treatment('OtherPartner')) + C(coh_s) + C(main_city)" + spec, DD)
    print(f"Battery channel{spec or ' (no activity control)'}: soh10 {orow(mS, 'soh10')} | km10 {orow(mS, 'km10')} | kyron25 {orow(mS, 'kyron25')}")
DD["soh_q"] = DD.groupby("coh_s")["soh_mean"].transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=False))
print("Non-retention by within-cohort quartile of SoH received (0=lowest):", DD.groupby("soh_q").y.mean().round(3).to_dict())
DD["km_q"] = DD.groupby("coh_s")["km_mean"].transform(lambda s: pd.qcut(s.rank(method="first"), 4, labels=False))
print("Non-retention by within-cohort quartile of km per swap (0=lowest):", DD.groupby("km_q").y.mean().round(3).to_dict())
print("Mean early activity by km quartile:", DD.groupby("km_q").e_att.mean().round(1).to_dict())
print("Mean SoH received / km per swap by cohort:")
print(DD.groupby("coh_s")[["soh_mean", "km_mean", "kyron_share"]].mean().round(2).T.to_string())

# (f) how much of the summer-2024 gap do failures in days 0-13 vs days 0-29 explain? (riders active in days 14-29)
mid = (E.filter((pl.col("dd") >= 14) & (pl.col("dd") < 30)).group_by("rider_id")
       .agg(m_att=pl.len(), m_fail=pl.col("is_service_failure").sum()).to_pandas())
SM = D[D.coh.between("2024-01-01", "2024-06-01")].merge(mid, on="rider_id", how="inner")
SM["summer"] = SM.coh.dt.month.isin([4, 5, 6]).astype(int)
SM["rate029"] = (SM.e_fail + SM.m_fail) / (SM.e_att + SM.m_att)
SM["r029_bin"] = pd.cut(SM.rate029, [-0.01, 0, 0.05, 0.10, 0.20, 1.01], labels=["0", "a", "b", "c", "d"]).astype(str)
SM["log_att029"] = np.log(SM.e_att + SM.m_att)
b0 = smf.logit("y ~ summer", SM).fit(disp=0).params["summer"]
b1 = smf.logit("y ~ summer + C(fail_grp) + log_att", SM).fit(disp=0).params["summer"]
b2 = smf.logit("y ~ summer + C(r029_bin) + log_att029", SM).fit(disp=0).params["summer"]
print(f"Riders active in days 14-29, cohorts Jan-Jun 2024 (n={len(SM):,}): summer log-odds gap {b0:.3f}; "
      f"after days 0-13 failures {b1:.3f} ({pc(1 - b1 / b0, 0)} explained); after days 0-29 failure rate {b2:.3f} ({pc(1 - b2 / b0, 0)} explained)")
SA = D[D.coh.between("2024-01-01", "2024-06-01")].assign(summer=lambda x: x.coh.dt.month.isin([4, 5, 6]).astype(int))
c0 = smf.logit("y ~ summer", SA).fit(disp=0).params["summer"]
c1 = smf.logit("y ~ summer + C(fail_grp)", SA).fit(disp=0).params["summer"]
c2 = smf.logit("y ~ summer + C(fail_grp) + log_att", SA).fit(disp=0).params["summer"]
print(f"All eligible Jan-Jun 2024 riders: summer log-odds gap {c0:.3f}; + early failure groups {c1:.3f} ({pc(1 - c1 / c0, 0)}); "
      f"+ log attempts {c2:.3f} ({pc(1 - c2 / c0, 0)})")

# (h) win-back among churners
nx = (a.filter((pl.col("gap") >= 30) & (pl.col("day") <= pl.col("end_day") - 30)).sort(["rider_id", "day"])
      .group_by("rider_id").agg(came_back=pl.col("nxt").first().is_not_null()))
print(f"Churners: {nx.height:,}; came back later: {int(nx['came_back'].sum()):,} ({pc(nx['came_back'].mean())})")
print("done2")

# %% [markdown]
# ## 12. Remaining claims: policy DiDs, declared shift, GBM AUC, why Jaipur is null

# %%
DD["pilot_city"] = DD["main_city"].isin(PILOT_CITIES).astype(int)
DD["post_pilot"] = (DD["coh"] >= "2024-10-01").astype(int)
DD["zip"] = (DD["partner_id"] == ZIPDROP_ID).astype(int)
DD["post_zip"] = (DD["coh"] >= "2024-11-01").astype(int)
CT2 = " + w3 + C(fail_grp) + C(coh_s) + C(main_city)"
m_pil = smf.logit("y ~ pilot_city:post_pilot" + CT2, data=DD[DD.indep == 1]).fit(disp=0)
m_zip = smf.logit("y ~ zip + zip:post_zip" + CT2, data=DD[DD.indep == 0]).fit(disp=0)
print("Pilot DiD (independent riders):", orow(m_pil, "pilot_city:post_pilot"))
print("ZipDrop DiD (partner riders):", orow(m_zip, "zip:post_zip"))
sh = (early.join(R.select("rider_id", "declared_shift"), on="rider_id")
      .group_by("declared_shift").agg(eve=pl.col("event_ts").dt.hour().is_between(17, 21).mean(), n=pl.len()).sort("declared_shift"))
print("Share of early attempts at 17-21h by declared shift:", dict(zip(sh["declared_shift"].to_list(), [round(v, 3) for v in sh["eve"].to_list()])))
import lightgbm as lgb  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
num = ["e_so", "e_ab", "e_se", "e_ca", "first_fail", "q_mean", "kyron_share", "soh_mean", "km_mean", "peak_share", "comp_any",
       "tk_any", "tk_unres_any", "w3", "kyc_no"]
cat = ["segment", "signup_channel", "age_band", "main_city", "expansion_wave", "charger_generation", "connectivity_tier"]
XG = DD[num].astype(float).copy()
XG["coh_idx"] = (DD.coh.dt.year - 2024) * 12 + DD.coh.dt.month
for c in cat:
    XG[c] = DD[c].fillna("unknown").astype("category")
yv = DD["y"].values
prm = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=80, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=5.0, n_estimators=400, verbose=-1, random_state=3)
for lab, Xm in [("drivers only", XG), ("+ early attempts", XG.assign(e_att=DD["e_att"].astype(float)))]:
    au = [roc_auc_score(yv[te], lgb.LGBMClassifier(**prm).fit(Xm.iloc[tr], yv[tr]).predict_proba(Xm.iloc[te])[:, 1])
          for tr, te in StratifiedKFold(5, shuffle=True, random_state=11).split(Xm, yv)]
    print(f"LightGBM 5-fold CV AUC ({lab}): {np.mean(au):.3f}")
# Jaipur: failure-type mix of early failures, and the 2+ stockout/abandon effect
mix = D.groupby("main_city")[["e_so", "e_ab", "e_se", "e_fail"]].sum()
mix["sa_share"] = (mix.e_so + mix.e_ab) / mix.e_fail
mix["fail_per_rider"] = D.groupby("main_city").e_fail.mean()
print(mix[["sa_share", "fail_per_rider"]].round(3).to_string())
jp = D[D.main_city == "Jaipur"]
jcov = [c for c in ["first_fail", "w3", "indep", "comp_any"] if jp[c].nunique() > 1]
print("Jaipur covariates with variation:", jcov, "| competitor-exposed riders in Jaipur:", int(jp.comp_any.sum()))
mj = smf.logit("y ~ sa2p + " + " + ".join(jcov) + " + C(coh_q)", data=jp).fit(disp=0)
print("Jaipur, 2+ stockout/abandon:", orow(mj, "sa2p"), "| non-ret", pc(jp.loc[jp.sa2p == 1, "y"].mean()), "vs", pc(jp.loc[jp.sa2p == 0, "y"].mean()))
print("Jaipur non-retention by fail group:", jp.groupby("fail_grp").y.agg(["mean", "size"]).round(3).to_dict("index"))
print("done3")
