# %% [markdown]
# # Q5 verification - independent re-derivation of the pricing & partner-economics claims
# Written from scratch (does not import or copy `analysis/q5_pricing_partners.py`). Re-derives every number from
# `data/clean` through the shared loaders, with deliberately *different* estimators where possible:
# * two-way fixed effects by weighted **alternating projections** (not dummy demeaning), cluster-robust SEs;
# * simple swap/attempt-weighted **2x2 DiDs** on city aggregates and on a **balanced station panel**;
# * mechanical (price x mix) checks of every revenue number.
#
# Run: `PYTHONIOENCODING=utf-8 python analysis/verify/q5_verify.py`

# %%
import sys
import datetime as dt
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import *  # noqa: E402,F401

pl.Config.set_tbl_rows(60); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

D = lambda y, m: dt.date(y, m, 1)  # noqa: E731
PRE = [D(2024, 7), D(2024, 8), D(2024, 9)]
POST = [D(2024, 10), D(2024, 11), D(2024, 12), D(2025, 1), D(2025, 2), D(2025, 3)]
CUR = (D(2024, 11), D(2025, 6))
CTRL = [c for c in CITIES if c not in PILOT_CITIES]


def out(*a):
    print(*a, flush=True)


# %% [markdown]
# ## 1. Pilot tariff schedule read from the data (independents, pilot cities, after 2024-10-01)
# Also validates the firmware timestamp correction: inside the v3.2.0 bug window the tariff code should line up with
# the *corrected* hour, not the raw logged hour.

# %%
base = events(["rider_id", "station_id", "month", "hour", "event_ts_raw", "ts_corrected", "is_completed", "is_stockout",
               "is_abandoned", "is_service_failure", "queue_wait_sec", "vehicle_class", "city", "tariff_code",
               "list_price_inr", "discount_inr", "amount_charged_inr", "partner_id", "is_independent", "plan_type",
               "revenue", "energy_cost", "wear_cost", "fixed_cost", "cm1", "cm2", "supplier_out", "season"])
sched = (base.filter(pl.col("is_completed") & pl.col("is_independent") & pl.col("city").is_in(PILOT_CITIES)
                     & (pl.col("month") >= PILOT_START))
         .group_by("hour").agg(pl.len().alias("n"),
                               (pl.col("tariff_code") == "PEAK").mean().alias("peak"),
                               (pl.col("tariff_code") == "OFFPEAK").mean().alias("off"),
                               (pl.col("amount_charged_inr") - pl.col("list_price_inr")).mean().alias("chg_minus_list"),
                               (pl.col("amount_charged_inr") - pl.col("list_price_inr")).filter(pl.col("vehicle_class") == "3W").mean().alias("chg_minus_list_3W"))
         .sort("hour").collect())
out(sched)
PEAK_H = sched.filter(pl.col("peak") > 0.5)["hour"].to_list()
OFF_H = sched.filter(pl.col("off") > 0.5)["hour"].to_list()
out("PEAK hours:", PEAK_H, " OFFPEAK hours:", OFF_H)
bug = (base.filter(pl.col("ts_corrected") & pl.col("is_completed") & pl.col("is_independent") & pl.col("city").is_in(PILOT_CITIES))
       .with_columns(pl.col("event_ts_raw").dt.hour().alias("raw_hour"))
       .select((pl.col("hour").is_in(PEAK_H) == (pl.col("tariff_code") == "PEAK")).mean().alias("match_corrected"),
               (pl.col("raw_hour").is_in(PEAK_H) == (pl.col("tariff_code") == "PEAK")).mean().alias("match_raw"),
               pl.len().alias("n")).collect())
out("Firmware-bug window, PEAK code vs hour band agreement:", bug.to_dicts())

MID, EVE, AFT = [12, 13], [19, 20, 21, 22], [14, 15, 16]
NIGHT = [h for h in OFF_H if h not in AFT]
LF = base.with_columns(
    pl.when(pl.col("hour").is_in(PEAK_H)).then(pl.lit("peak")).when(pl.col("hour").is_in(OFF_H)).then(pl.lit("off"))
      .otherwise(pl.lit("std")).alias("band"),
    pl.when(pl.col("hour").is_in(MID)).then(pl.lit("a_mid12-13")).when(pl.col("hour").is_in(EVE)).then(pl.lit("b_eve19-22"))
      .when(pl.col("hour").is_in(AFT)).then(pl.lit("c_off14-16")).when(pl.col("hour").is_in(NIGHT)).then(pl.lit("d_off23-05"))
      .otherwise(pl.lit("e_std")).alias("sub"),
    pl.col("partner_id").fill_null("IND").alias("pid"),
    (pl.col("supplier_out") == "Kyron").alias("ky"))
P = partners()
LF = LF.join(P.lazy().select(pl.col("partner_id").alias("pid"), "partner_name", "partner_segment", "peak_surcharge_billable",
                             "payment_terms_days"), on="pid", how="left").with_columns(
    pl.when(pl.col("pid") == "IND").then(pl.lit("IND")).when(pl.col("peak_surcharge_billable") == "Y").then(pl.lit("PY"))
      .otherwise(pl.lit("PN")).alias("grp"))

# %% [markdown]
# ## 2. Partner surcharge billing in pilot cities (amount - (list - discount))

# %%
sur = (LF.filter(pl.col("is_completed") & (pl.col("grp") != "IND") & pl.col("city").is_in(PILOT_CITIES) & (pl.col("month") >= PILOT_START))
       .with_columns((pl.col("amount_charged_inr") - pl.col("list_price_inr") + pl.col("discount_inr")).alias("s"))
       .group_by("grp", "band").agg(pl.len(), pl.col("s").mean().round(3).alias("mean"), pl.col("s").min().alias("min"),
                                    pl.col("s").max().alias("max")).sort("grp", "band").collect())
out(sur)

# %% [markdown]
# ## 3. Congestion by hour band (network, all attempts Jan-24..Jun-25), and by season

# %%
cb = (LF.group_by("sub").agg(pl.len().alias("att"), (100 * pl.col("is_service_failure").mean()).alias("fail%"),
                             (100 * pl.col("is_stockout").mean()).alias("stockout%"), (100 * pl.col("is_abandoned").mean()).alias("aband%"),
                             pl.col("queue_wait_sec").filter(pl.col("is_completed")).mean().alias("qwait_s"))
      .with_columns((100 * pl.col("att") / pl.col("att").sum()).alias("att_share%")).sort("sub").collect())
out(cb, "total attempts", cb["att"].sum())
cbs = (LF.group_by("season", "sub").agg((100 * pl.col("is_service_failure").mean()).alias("fail%"),
                                        (100 * pl.col("is_stockout").mean()).alias("so%"))
       .collect().pivot(on="sub", index="season", values="so%").sort("season"))
out("Stockout % by season x band:\n", cbs)
cbc = (LF.group_by("city", "sub").agg((100 * pl.col("is_stockout").mean()).alias("so%"))
       .collect().pivot(on="sub", index="city", values="so%").sort("city"))
out("Stockout % by city x band:\n", cbc)

# %% [markdown]
# ## 4. Rider-city-month panel (completed swaps) and FE estimator (alternating projections)

# %%
RM = (LF.filter(pl.col("is_completed")).group_by("rider_id", "city", "month")
      .agg(pl.len().alias("sw"), (pl.col("band") == "peak").sum().alias("n_pk"), (pl.col("band") == "off").sum().alias("n_off"),
           *[(pl.col("sub") == s).sum().alias(f"n_{s[:5]}") for s in ("a_mid12-13", "b_eve19-22", "c_off14-16", "d_off23-05")],
           pl.col("revenue").sum().alias("rev"), pl.col("cm1").sum().alias("cm1"), pl.col("ky").sum().alias("n_ky"),
           pl.col("grp").first(), pl.col("pid").first(), pl.col("vehicle_class").mode().first().alias("vc"))
      .collect(engine="streaming").to_pandas())
RM["unit"] = RM["rider_id"] + "|" + RM["city"]
RM["tr"] = RM["city"].isin(PILOT_CITIES)
for c in ["n_pk", "n_off", "n_a_mid", "n_b_eve", "n_c_off", "n_d_off"]:
    RM["sh_" + c[2:]] = RM[c] / RM["sw"]
RM["rps"] = RM["rev"] / RM["sw"]; RM["cps"] = RM["cm1"] / RM["sw"]; RM["kysh"] = RM["n_ky"] / RM["sw"]
RM["lsw"] = np.log(RM["sw"])
RM["month"] = pd.to_datetime(RM["month"]).dt.date
out(f"panel rows {len(RM):,}")


def within2(M, g1, g2, w, tol=1e-10, maxit=3000):
    M = M.astype(float).copy()
    s1 = np.bincount(g1, weights=w); s2 = np.bincount(g2, weights=w)
    for _ in range(maxit):
        m1 = np.column_stack([np.bincount(g1, weights=w * M[:, j], minlength=len(s1)) for j in range(M.shape[1])]) / s1[:, None]
        M -= m1[g1]
        m2 = np.column_stack([np.bincount(g2, weights=w * M[:, j], minlength=len(s2)) for j in range(M.shape[1])]) / s2[:, None]
        M -= m2[g2]
        if max(np.abs(m1).max(), np.abs(m2).max()) < tol:
            break
    return M


def fe(df, y, xs, unit="unit", time="month", w=None):
    d = df.dropna(subset=[y] + xs)
    g1 = pd.factorize(d[unit])[0]; g2 = pd.factorize(d[time])[0]
    ww = d[w].to_numpy(float) if w else np.ones(len(d))
    M = within2(d[[y] + xs].to_numpy(float), g1, g2, ww)
    Y, X = M[:, 0], M[:, 1:]
    XtW = X.T * ww
    A = np.linalg.inv(XtW @ X)
    b = A @ (XtW @ Y)
    e = Y - X @ b
    S = np.column_stack([np.bincount(g1, weights=ww * e * X[:, j]) for j in range(X.shape[1])])
    G = S.shape[0]
    se = np.sqrt(np.diag(A @ (S.T @ S) @ A) * G / (G - 1))
    return {x: (b[i], b[i] - 1.96 * se[i], b[i] + 1.96 * se[i]) for i, x in enumerate(xs)}


def fmt(t, sc=1.0):
    return f"{t[0]*sc:+.3f} [{t[1]*sc:+.3f}, {t[2]*sc:+.3f}]"


def city2x2(df, y, w, pre=PRE, post=POST):
    s = df[df["month"].isin(pre + post)].copy()
    s["post"] = s["month"].isin(post)
    s["wy"] = s[y] * s[w]
    g = s.groupby(["tr", "post"])[["wy", w]].sum()
    m = g["wy"] / g[w]
    return (m[(True, True)] - m[(True, False)]) - (m[(False, True)] - m[(False, False)]), m


def perm(df, y, w, pre=PRE, post=POST):
    s = df[df["month"].isin(pre + post)].copy()
    s["post"] = s["month"].isin(post); s["wy"] = s[y] * s[w]
    g = s.groupby(["city", "post"])[["wy", w]].sum()
    cm = (g["wy"] / g[w]).unstack("post")
    dlt = cm[True] - cm[False]
    eff = lambda tr: dlt[list(tr)].mean() - dlt.drop(list(tr)).mean()  # noqa: E731
    a = eff(PILOT_CITIES)
    return a, np.mean([abs(eff(c)) >= abs(a) - 1e-12 for c in combinations(CITIES, 2)])


# %% [markdown]
# ## 5. Pilot DiD - independents (pre Jul-Sep 2024, post Oct-24..Mar-25)

# %%
ind = RM[(RM["grp"] == "IND") & RM["month"].isin(PRE + POST)].copy()
ind["tp"] = (ind["tr"] & ind["month"].isin(POST)).astype(float)
R = {}
for y, w, sc in (("sh_pk", "sw", 100), ("sh_off", "sw", 100), ("rps", "sw", 1), ("cps", "sw", 1),
                 ("sh_a_mid", "sw", 100), ("sh_b_eve", "sw", 100), ("sh_c_off", "sw", 100), ("sh_d_off", "sw", 100)):
    R[y] = fe(ind, y, ["tp"], w=w)["tp"]
    c2, m = city2x2(ind, y, w)
    pv = perm(ind, y, w)
    pre_t = m[(True, False)]
    out(f"IND {y:9s} FE {fmt(R[y], sc)} | city 2x2 {c2*sc:+.3f} | perm p {pv[1]:.3f} | treated pre mean {pre_t*sc:.3f}")
ky = fe(ind, "cps", ["tp", "kysh"], w="sw")
out("IND cm1/swap with Kyron-share control:", fmt(ky["tp"]), " naive/adjusted =", round(R["cps"][0] / ky["tp"][0], 2))
mum = ind[ind["city"].isin(PILOT_CITIES + ["Mumbai"])]
out("IND cm1/swap Mumbai-only control:", fmt(fe(mum, "cps", ["tp"], w="sw")["tp"]),
    "| rev/swap Mumbai-only:", fmt(fe(mum, "rps", ["tp"], w="sw")["tp"]))
# mechanical check of revenue
pk_post = ind[ind["tr"] & ind["month"].isin(POST)]
pkp = (pk_post["n_pk"].sum() / pk_post["sw"].sum()); ofp = pk_post["n_off"].sum() / pk_post["sw"].sum()
pre_t = ind[ind["tr"] & ind["month"].isin(PRE)]
out(f"Mechanical: treated post peak share {100*pkp:.1f}%, off {100*ofp:.1f}% -> 15*pk-8*off = Rs{15*pkp-8*ofp:.2f}; "
    f"at pre timing ({100*pre_t['n_pk'].sum()/pre_t['sw'].sum():.1f}% / {100*pre_t['n_off'].sum()/pre_t['sw'].sum():.1f}%) "
    f"= Rs{15*pre_t['n_pk'].sum()/pre_t['sw'].sum()-8*pre_t['n_off'].sum()/pre_t['sw'].sum():.2f}")

# event-study style monthly gaps (treated - control, swap-weighted, city aggregate), relative to Sep-24
allm = RM[(RM["grp"] == "IND")].copy()
allm["w_pk"] = allm["n_pk"]; gm = allm.groupby(["month", "tr"])[["n_pk", "sw"]].sum()
gap = (gm["n_pk"] / gm["sw"]).unstack("tr"); gap = 100 * (gap[True] - gap[False]); gap = gap - gap.loc[D(2024, 9)]
out("IND peak-share gap (pp, vs Sep-24):", ", ".join(f"{m:%y-%m} {v:+.2f}" for m, v in gap.items()))

# %% [markdown]
# ## 6. Segment heterogeneity (Partner Y / Partner N / independent sub-groups)

# %%
for g in ("PY", "PN"):
    s = RM[(RM["grp"] == g) & RM["month"].isin(PRE + POST)].copy()
    s["tp"] = (s["tr"] & s["month"].isin(POST)).astype(float)
    r1 = fe(s, "sh_pk", ["tp"], w="sw")["tp"]; r2 = fe(s, "sh_off", ["tp"], w="sw")["tp"]; r3 = fe(s, "rps", ["tp"], w="sw")["tp"]
    pre_t = s[s["tr"] & s["month"].isin(PRE)]
    out(f"{g}: peak {fmt(r1,100)} | off {fmt(r2,100)} | rev {fmt(r3)} | pre rev {np.average(pre_t['rps'], weights=pre_t['sw']):.2f} "
        f"| post peak/off share treated {100*s[s['tr'] & s['month'].isin(POST)]['n_pk'].sum()/s[s['tr'] & s['month'].isin(POST)]['sw'].sum():.1f}%/"
        f"{100*s[s['tr'] & s['month'].isin(POST)]['n_off'].sum()/s[s['tr'] & s['month'].isin(POST)]['sw'].sum():.1f}%")
for vc in ("2W", "3W"):
    s = ind[ind["vc"] == vc]
    out(f"IND {vc}: peak {fmt(fe(s, 'sh_pk', ['tp'], w='sw')['tp'], 100)}")

# %% [markdown]
# ## 7. Congestion DiD - station x month panel, PEAK-hour attempts (the headline "+0.05 pp")

# %%
SM = (LF.filter(pl.col("band") == "peak").group_by("station_id", "city", "month", "sub")
      .agg(pl.len().alias("att"), pl.col("is_stockout").sum().alias("so"), pl.col("is_service_failure").sum().alias("fail"),
           pl.col("is_abandoned").sum().alias("ab"))
      .collect().to_pandas())
SM["month"] = pd.to_datetime(SM["month"]).dt.date
SM["tr"] = SM["city"].isin(PILOT_CITIES)
S = SM.groupby(["station_id", "city", "month", "tr"], as_index=False)[["att", "so", "fail", "ab"]].sum()
for k in ("so", "fail", "ab"):
    S[k + "_r"] = S[k] / S["att"]
s9 = S[S["month"].isin(PRE + POST)].copy()
s9["tp"] = (s9["tr"] & s9["month"].isin(POST)).astype(float)
for k in ("so_r", "fail_r", "ab_r"):
    r = fe(s9, k, ["tp"], unit="station_id", w="att")["tp"]
    c2, m = city2x2(s9, k, "att")
    out(f"PEAK {k}: station+month FE {fmt(r, 100)} | city-aggregate 2x2 {100*c2:+.3f} | perm p {perm(s9, k, 'att')[1]:.3f} | treated pre {100*m[(True, False)]:.3f}")
# balanced panel: stations observed in every month Jul-24..Mar-25
nm = s9.groupby("station_id")["month"].nunique()
bal = s9[s9["station_id"].isin(nm[nm == len(PRE + POST)].index)]
out(f"balanced stations: {bal['station_id'].nunique()} of {s9['station_id'].nunique()}")
for k in ("so_r", "fail_r"):
    c2, m = city2x2(bal, k, "att")
    r = fe(bal, k, ["tp"], unit="station_id", w="att")["tp"]
    out(f"  balanced {k}: 2x2 {100*c2:+.3f} | FE {fmt(r, 100)}")
# unweighted station FE (each station-month equal)
out("  unweighted station FE so_r:", fmt(fe(s9, "so_r", ["tp"], unit="station_id")["tp"], 100))
# monthly gaps vs Sep-24 (balanced Launch stations present all 18 months)
nm18 = S.groupby("station_id")["month"].nunique()
L18 = S[S["station_id"].isin(nm18[nm18 == 18].index)]
g18 = L18.groupby(["month", "tr"])[["so", "att"]].sum(); r18 = (g18["so"] / g18["att"]).unstack("tr")
gg = 100 * (r18[True] - r18[False]); gg = gg - gg.loc[D(2024, 9)]
out(f"Stockout gap treated-control (pp vs Sep-24), {L18['station_id'].nunique()} stations present all 18 months: "
    + ", ".join(f"{m:%y-%m} {v:+.2f}" for m, v in gg.items()))
# evening sub-band only (where congestion is)
SE = SM[SM["sub"] == "b_eve19-22"].copy(); SE = SE[SE["month"].isin(PRE + POST)]
SE["so_r"] = SE["so"] / SE["att"]; SE["tp"] = (SE["tr"] & SE["month"].isin(POST)).astype(float)
c2, _ = city2x2(SE, "so_r", "att")
out(f"Evening 19-22h stockout: FE {fmt(fe(SE, 'so_r', ['tp'], unit='station_id', w='att')['tp'], 100)} | 2x2 {100*c2:+.3f}")
# load change: peak-hour attempts per station-month
s9["latt"] = np.log(s9["att"])
out("log peak attempts per station-month DiD:", fmt(fe(s9, "latt", ["tp"], unit="station_id")["tp"], 100))

# %% [markdown]
# ## 8. Kyron share of issued packs by city, Oct-24..Mar-25

# %%
kyc = (LF.filter(pl.col("is_completed") & pl.col("month").is_in(POST)).group_by("city")
       .agg((100 * pl.col("ky").mean()).round(1).alias("kyron_%")).sort("kyron_%").collect())
out(kyc)

# %% [markdown]
# ## 9. Partner P&L - current terms (Nov-24..Jun-25) and full window

# %%
def pnl(m0, m1):
    t = (LF.filter(pl.col("is_completed") & pl.col("month").is_between(m0, m1)).group_by("pid")
         .agg(pl.len().alias("swaps"), pl.col("rider_id").n_unique().alias("riders"), pl.col("revenue").sum().alias("rev"),
              pl.col("list_price_inr").sum().alias("list"), pl.col("discount_inr").sum().alias("disc"),
              pl.col("energy_cost").sum().alias("en"), pl.col("wear_cost").sum().alias("wear"), pl.col("fixed_cost").sum().alias("fix"),
              pl.col("cm1").sum(), pl.col("cm2").sum(), pl.col("ky").sum().alias("ky"),
              (pl.col("energy_cost") + pl.col("wear_cost")).filter(~pl.col("ky")).sum().alias("var_nonky"),
              (pl.col("band") == "peak").sum().alias("pk"), (pl.col("band") == "off").sum().alias("off"),
              (pl.col("vehicle_class") == "3W").sum().alias("n3w"), pl.col("partner_name").first(),
              pl.col("payment_terms_days").first().fill_null(0))
         .collect().to_pandas())
    t["partner_name"] = t["partner_name"].fillna("Independents")
    t["share_net"] = t["swaps"] / t["swaps"].sum()
    t["share_partner"] = np.where(t["pid"] == "IND", np.nan, t["swaps"] / t.loc[t["pid"] != "IND", "swaps"].sum())
    for k in ("rev", "en", "wear", "fix", "cm1", "cm2"):
        t[k + "_ps"] = t[k] / t["swaps"]
    t["disc_pct"] = t["disc"] / t["list"]
    t["wc"] = t["rev"] * 0.12 * t["payment_terms_days"] / 365
    t["cm2wc"] = t["cm2"] - t["wc"]; t["cm2wc_ps"] = t["cm2wc"] / t["swaps"]
    t["cm1_ss_ps"] = t["rev_ps"] - t["var_nonky"] / (t["swaps"] - t["ky"])
    t["pk_sh"] = t["pk"] / t["swaps"]; t["sh3w"] = t["n3w"] / t["swaps"]
    pt = t[t["pid"] != "IND"].copy()
    for k in ("cm1", "cm2wc", "cm1_ps", "cm2wc_ps", "rev", "swaps", "riders"):
        t["rk_" + k] = pt[k].rank(ascending=False).reindex(t.index)
    return t.sort_values("swaps", ascending=False)


cols = ["pid", "partner_name", "swaps", "riders", "share_net", "share_partner", "rev", "disc_pct", "rev_ps", "en_ps", "wear_ps",
        "fix_ps", "cm1_ps", "cm2wc_ps", "cm1", "cm2wc", "cm1_ss_ps", "pk_sh", "sh3w", "rk_cm1", "rk_cm2wc", "rk_cm1_ps", "rk_cm2wc_ps"]
PT = pnl(*CUR)
out("CURRENT TERMS Nov-24..Jun-25\n", PT[cols].round(3).to_string(index=False))
PF = pnl(D(2024, 1), D(2025, 6))
out("FULL WINDOW\n", PF[["pid", "partner_name", "swaps", "rev", "cm1", "cm2wc", "rk_cm1", "rk_cm2wc", "rk_rev"]].round(0).to_string(index=False))
PRE_AM = pnl(D(2024, 1), D(2024, 10))
out("PRE-AMENDMENT Jan-Oct 2024 (rank by revenue / swaps):\n",
    PRE_AM[["pid", "partner_name", "swaps", "rev", "riders", "rk_rev", "rk_swaps"]].head(4).to_string(index=False))

# %% [markdown]
# ## 10. Within-cell cost-to-serve check (ZipDrop vs other partners, same city x month x vehicle class x band)

# %%
cell = (LF.filter(pl.col("is_completed") & (pl.col("pid") != "IND") & pl.col("month").is_between(*CUR))
        .with_columns((pl.col("pid") == ZIPDROP_ID).alias("zd"))
        .group_by("zd", "city", "month", "vehicle_class", "band")
        .agg(pl.len().alias("n"), pl.col("revenue").sum().alias("rev"), (pl.col("energy_cost") + pl.col("wear_cost")).sum().alias("var"),
             pl.col("fixed_cost").sum().alias("fix"), pl.col("cm1").sum().alias("cm1")).collect().to_pandas())
z = cell[cell["zd"]].merge(cell[~cell["zd"]], on=["city", "month", "vehicle_class", "band"], suffixes=("_z", "_o"))
for k in ("rev", "var", "fix", "cm1"):
    gapk = (z[k + "_z"] - z["n_z"] * z[k + "_o"] / z["n_o"]).sum() / z["n_z"].sum()
    out(f"ZipDrop within-cell gap vs other partners, {k}: {gapk:+.2f} Rs/swap")

# %% [markdown]
# ## 11. ZipDrop amendment (2W): DiD vs other partners - all cities and pilot-free control cities

# %%
zm = (LF.filter(pl.col("is_completed") & (pl.col("pid") != "IND") & (pl.col("vehicle_class") == "2W") & (pl.col("month") >= D(2024, 1)))
      .with_columns((pl.col("pid") == ZIPDROP_ID).alias("zd"), pl.col("city").is_in(PILOT_CITIES).alias("pc"))
      .group_by("zd", "pc", "month").agg(pl.len().alias("n"), pl.col("revenue").sum().alias("rev"), pl.col("cm1").sum().alias("cm1"),
                                         pl.col("list_price_inr").sum().alias("list"), pl.col("discount_inr").sum().alias("disc"))
      .collect().to_pandas())
zm["month"] = pd.to_datetime(zm["month"]).dt.date


def zdid(d, pre_m, post_m):
    r = {}
    for zd in (True, False):
        for nm_, ms in (("pre", pre_m), ("post", post_m)):
            s = d[(d["zd"] == zd) & d["month"].isin(ms)]
            r[(zd, nm_)] = (s["rev"].sum() / s["n"].sum(), s["cm1"].sum() / s["n"].sum(), s["disc"].sum() / s["list"].sum())
    dr = (r[(True, "post")][0] - r[(True, "pre")][0]) - (r[(False, "post")][0] - r[(False, "pre")][0])
    dc = (r[(True, "post")][1] - r[(True, "pre")][1]) - (r[(False, "post")][1] - r[(False, "pre")][1])
    return dr, dc, r


PRE_Z = [D(2024, m) for m in (7, 8, 9, 10)]
POST_Z = [D(2024, 11), D(2024, 12)] + [D(2025, m) for m in range(1, 7)]
dr, dc, r = zdid(zm, PRE_Z, POST_Z)
out(f"All cities: ZipDrop rev {r[(True,'pre')][0]:.2f}->{r[(True,'post')][0]:.2f}, disc {100*r[(True,'pre')][2]:.1f}%->{100*r[(True,'post')][2]:.1f}%, "
    f"cm1 {r[(True,'pre')][1]:.2f}->{r[(True,'post')][1]:.2f}; others rev {r[(False,'pre')][0]:.2f}->{r[(False,'post')][0]:.2f}, "
    f"cm1 {r[(False,'pre')][1]:.2f}->{r[(False,'post')][1]:.2f}; DiD rev {dr:+.2f}, cm1 {dc:+.2f}")
dr2, dc2, r2 = zdid(zm[~zm["pc"]], PRE_Z, POST_Z)
out(f"Control cities only (no pilot surcharge): others rev {r2[(False,'pre')][0]:.2f}->{r2[(False,'post')][0]:.2f}; DiD rev {dr2:+.2f}, cm1 {dc2:+.2f}")
dr3, dc3, _ = zdid(zm, [D(2024, m) for m in (7, 8, 9)], POST_Z)
out(f"All cities, pre = Jul-Sep only: DiD rev {dr3:+.2f}, cm1 {dc3:+.2f}")
vol = zm.groupby(["zd", "month"])["n"].sum().unstack("zd")
g_pre = vol.loc[PRE_Z].mean(); g_post = vol.loc[POST_Z].mean(); g_early = vol.loc[[D(2024, m) for m in range(1, 7)]].mean()
out(f"2W volume growth Nov-Jun vs Jul-Oct avg: ZipDrop {100*(g_post[True]/g_pre[True]-1):+.1f}%, others {100*(g_post[False]/g_pre[False]-1):+.1f}%; "
    f"pre-trend Jul-Oct vs Jan-Jun: ZipDrop {100*(g_pre[True]/g_early[True]-1):+.1f}%, others {100*(g_pre[False]/g_early[False]-1):+.1f}%; "
    f"share {100*g_pre[True]/(g_pre.sum()):.1f}% -> {100*g_post[True]/g_post.sum():.1f}%")
zall = PT[PT["pid"] == ZIPDROP_ID].iloc[0]
lst_pm = zall["rev"] / (1 - zall["disc_pct"]) / 8
out(f"Amendment cost: list revenue/month Rs{lst_pm/1e5:.2f} L x 16 pp = Rs{0.16*lst_pm/1e5:.2f} L/month = Rs{12*0.16*lst_pm/1e5:.1f} L/yr")

# %% [markdown]
# ## 12. 3W economics (Nov-24..Jun-25)

# %%
v3 = (LF.filter(pl.col("is_completed") & pl.col("month").is_between(*CUR)).group_by("vehicle_class")
      .agg(pl.len().alias("n"), pl.col("revenue").mean().alias("rev"), pl.col("list_price_inr").mean().alias("list"),
           pl.col("energy_cost").mean().alias("en"), pl.col("wear_cost").mean().alias("wear"), pl.col("cm1").mean().alias("cm1"),
           pl.col("energy_cost").filter(~pl.col("ky")).mean().alias("en_nk"), pl.col("wear_cost").filter(~pl.col("ky")).mean().alias("wear_nk"))
      .with_columns((pl.col("rev") - pl.col("en_nk") - pl.col("wear_nk")).alias("cm1_ss"),
                    ((pl.col("en_nk") + pl.col("wear_nk")) / (pl.col("rev") / pl.col("list"))).alias("breakeven_list"))
      .sort("vehicle_class").collect())
out(v3)

# %% [markdown]
# ## 13. Budget (c) rollout in the four control cities (Jan-Jun 2025) - DiD-based vs mechanical; (d) ZipDrop exclusive

# %%
bc = (LF.filter(pl.col("is_completed") & pl.col("city").is_in(CTRL) & pl.col("month").is_between(D(2025, 1), D(2025, 6)))
      .group_by("grp").agg((pl.len() / 6).alias("swaps_pm"), ((pl.col("band") == "peak").sum() / 6).alias("pk_pm"),
                           ((pl.col("band") == "off").sum() / 6).alias("off_pm"))
      .sort("grp").collect().to_pandas().set_index("grp"))
bc["pk_sh"] = bc["pk_pm"] / bc["swaps_pm"]; bc["off_sh"] = bc["off_pm"] / bc["swaps_pm"]
out(bc.round(3))
did_i, did_y = R["rps"][0], None
s = RM[(RM["grp"] == "PY") & RM["month"].isin(PRE + POST)].copy(); s["tp"] = (s["tr"] & s["month"].isin(POST)).astype(float)
did_y = fe(s, "rps", ["tp"], w="sw")["tp"][0]
dd = bc.loc["IND", "swaps_pm"] * did_i + bc.loc["PY", "swaps_pm"] * did_y
mech = sum(bc.loc[g, "swaps_pm"] * (15 * (bc.loc[g, "pk_sh"] - 0.025) - 8 * (bc.loc[g, "off_sh"] + 0.025)) for g in ("IND", "PY"))
out(f"(c) DiD-based uplift Rs{dd/1e5:.2f} L/month; mechanical on control-city timing (after a 2.5 pp shift) Rs{mech/1e5:.2f} L/month")
pn = bc.loc["PN"]
out(f"(c) ZipDrop+Swiggle Go peak swaps/month {pn['pk_pm']:,.0f} x 15 = Rs{15*pn['pk_pm']/1e5:.2f} L; "
    f"net of -8 off-peak ({pn['off_pm']:,.0f}/month): Rs{(15*pn['pk_pm']-8*pn['off_pm'])/1e5:.2f} L")
zz = PT[PT["pid"] == ZIPDROP_ID].iloc[0]
out(f"(d) ZipDrop annualised swaps {zz['swaps']*1.5:,.0f}; cm1 Rs{zz['cm1']*1.5/1e7:.2f} Cr/yr; cm2 after WC Rs{zz['cm2wc']*1.5/1e7:.2f} Cr/yr; "
    f"peak share {100*zz['pk_sh']:.1f}%")
z2 = (LF.filter(pl.col("is_completed") & (pl.col("pid") == ZIPDROP_ID) & (pl.col("vehicle_class") == "2W") & pl.col("month").is_between(*CUR))
      .select((pl.col("energy_cost") + pl.col("wear_cost")).filter(~pl.col("ky")).mean().alias("var_ss"),
              pl.col("fixed_cost").mean().alias("fix"), pl.col("list_price_inr").mean().alias("list")).collect().to_dicts()[0])
out(f"(d) ZipDrop 2W steady-state variable cost Rs{z2['var_ss']:.2f}; break-even discount cm1 {100*(1-z2['var_ss']/z2['list']):.1f}%, "
    f"cm2 after WC {100*(1-(z2['var_ss']+z2['fix'])/(z2['list']*(1-0.12*45/365))):.1f}%")
ff = PT[PT["pid"] == "FP-01"].iloc[0]
out(f"FeastFly (largest by revenue/riders): cm1/swap {ff['cm1_ps']:.2f}, steady-state cm1/swap {ff['cm1_ss_ps']:.2f}, "
    f"steady-state cm1/yr Rs{ff['cm1_ss_ps']*ff['swaps']*1.5/1e5:.1f} L")
pkall = (LF.filter(pl.col("is_completed") & pl.col("month").is_between(*CUR) & (pl.col("band") == "peak"))
         .select((pl.col("pid") == ZIPDROP_ID).mean()).collect().item())
out(f"ZipDrop share of all PEAK-hour swaps: {100*pkall:.1f}%")
wc_ff = ff["rev_ps"] * 0.12 * 30 / 365
out(f"FeastFly steady-state cm2 after WC per swap: {ff['cm1_ss_ps'] - ff['fix_ps'] - wc_ff:+.2f}")

# %% [markdown]
# ### Extra robustness checks the findings depend on
# * ZipDrop amendment DiD at rider level (rider-city + month FE) restricted to the four non-pilot cities, where other
#   partners' revenue is not moved by the pilot surcharge.
# * Congestion DiD with Mumbai (same ~10% Kyron share as the pilot cities) as the only control.
# * Is "evening = real congestion" network-wide, or concentrated in some cities / seasons?

# %%
zr = RM[(RM["grp"] != "IND") & (RM["vc"] == "2W") & (RM["month"] >= D(2024, 7))].copy()
zr["zp"] = ((zr["pid"] == ZIPDROP_ID) & (zr["month"] >= ZIPDROP_AMEND)).astype(float)
for lab, s in (("all cities", zr), ("non-pilot cities", zr[~zr["tr"]])):
    out(f"ZipDrop amendment rider-FE DiD ({lab}): rev/swap {fmt(fe(s, 'rps', ['zp'], w='sw')['zp'])} | cm1/swap {fmt(fe(s, 'cps', ['zp'], w='sw')['zp'])}")
sm_m = s9[s9["city"].isin(PILOT_CITIES + ["Mumbai"])]
out("PEAK stockout DiD, Mumbai-only control:", fmt(fe(sm_m, "so_r", ["tp"], unit="station_id", w="att")["tp"], 100),
    "| 2x2", f"{100*city2x2(sm_m, 'so_r', 'att')[0]:+.3f}")
fb = (LF.filter(pl.col("sub").is_in(["a_mid12-13", "b_eve19-22"]))
      .group_by("city", "season", "sub").agg((100 * pl.col("is_service_failure").mean()).alias("fail"))
      .collect().pivot(on="sub", index=["city", "season"], values="fail")
      .with_columns((pl.col("b_eve19-22") - pl.col("a_mid12-13")).round(2).alias("eve_minus_mid_pp")).sort("city", "season"))
out("Failure % evening vs midday by city x season:\n", fb)
fc = (LF.filter(pl.col("sub").is_in(["a_mid12-13", "b_eve19-22"]))
      .group_by("city", "sub").agg((100 * pl.col("is_service_failure").mean()).alias("fail"))
      .collect().pivot(on="sub", index="city", values="fail")
      .with_columns((pl.col("b_eve19-22") - pl.col("a_mid12-13")).round(2).alias("eve_minus_mid_pp")).sort("city"))
out("Failure % evening vs midday by city (all months):\n", fc)

# %% [markdown]
# ### Revenue at stake in the 12-13h band (recommendation to drop it) and volume / continuation claims

# %%
mid = (LF.filter(pl.col("is_completed") & pl.col("city").is_in(CTRL) & pl.col("month").is_between(D(2025, 1), D(2025, 6)) & pl.col("grp").is_in(["IND", "PY"]))
       .group_by("grp").agg((pl.len() / 6).alias("swaps_pm"), (pl.col("sub") == "a_mid12-13").mean().alias("mid_sh")).sort("grp").collect().to_pandas().set_index("grp"))
lost = (15 * mid["swaps_pm"] * (mid["mid_sh"] - 0.0098)).sum()
out(mid.round(3), f"\nDropping 12-13h from the surcharge forgoes ~Rs{lost/1e5:.2f} L/month of the projected rollout uplift "
    f"(IND treated post midday share {100*(0.2128-0.0098):.1f}% x Rs15 = Rs{15*(0.2128-0.0098):.2f} of the Rs7.90/swap)")
for g in ("PY", "PN"):
    s = RM[(RM["grp"] == g) & RM["month"].isin(PRE + POST)].copy()
    s["tp"] = (s["tr"] & s["month"].isin(POST)).astype(float)
    out(f"{g} log swaps per active rider-month DiD (+Kyron share): {fmt(fe(s, 'lsw', ['tp', 'kysh'])['tp'], 100)}")
out(f"IND log swaps DiD (no Kyron control): {fmt(fe(ind, 'lsw', ['tp'])['tp'], 100)}")
dd3 = RM[RM["grp"].isin(["IND", "PN"]) & RM["month"].isin(PRE + POST)].copy()
dd3["post"] = dd3["month"].isin(POST).astype(float); dd3["i"] = (dd3["grp"] == "IND").astype(float)
dd3["tp"] = (dd3["tr"] & dd3["month"].isin(POST)).astype(float); dd3["ip"] = dd3["i"] * dd3["post"]; dd3["tpi"] = dd3["tp"] * dd3["i"]
r3 = fe(dd3, "lsw", ["tp", "ip", "tpi", "kysh"])
out(f"DDD log swaps (IND vs PN, +Kyron): {fmt(r3['tpi'], 100)}; common city shock {fmt(r3['tp'], 100)}")
# continuation: independent rider active (>=1 completed swap) in month m+1, transitions Jul->Aug, Aug->Sep (pre) vs Oct..Mar (post)
act = RM.groupby(["rider_id", "month"], as_index=False).agg(grp=("grp", "first"), tr=("tr", "max"))
act["nxt"] = [(pd.Timestamp(m) + pd.DateOffset(months=1)).date() for m in act["month"]]
keyset = set(zip(act["rider_id"], act["month"]))
act["on"] = [float((r, n) in keyset) for r, n in zip(act["rider_id"], act["nxt"])]
ai = act[(act["grp"] == "IND") & act["month"].isin([D(2024, 7), D(2024, 8)] + POST)].copy()
ai["post"] = ai["month"].isin(POST)
gi = ai.groupby(["tr", "post"])["on"].mean()
out(f"Pilot continuation 2x2 (IND): {100*((gi[(True, True)]-gi[(True, False)])-(gi[(False, True)]-gi[(False, False)])):+.2f} pp")
ab = act[act["nxt"].between(D(2024, 2), D(2024, 8))].copy()
ab["post"] = ab["nxt"] >= BASE_PRICE_CHANGE; ab["i"] = ab["grp"] == "IND"
gb = ab.groupby(["i", "post"])["on"].mean()
out(f"Base-price continuation 2x2 (IND vs partner): {100*((gb[(True, True)]-gb[(True, False)])-(gb[(False, True)]-gb[(False, False)])):+.2f} pp; "
    f"IND pre {100*gb[(True, False)]:.1f}%")

# %% [markdown]
# ## 14. Base price change (2024-07-01): revenue per swap and 2W usage DiD (independents vs partner-billed, Jan-Aug 2024)

# %%
bpr = (LF.filter(pl.col("is_completed") & pl.col("month").is_between(D(2024, 4), D(2024, 8)))
       .with_columns((pl.col("grp") == "IND").alias("ind"), (pl.col("month") >= BASE_PRICE_CHANGE).alias("post"))
       .group_by("ind", "vehicle_class", "post").agg(pl.col("revenue").mean().round(2), pl.col("cm1").mean().round(2))
       .sort("ind", "vehicle_class", "post").collect())
out(bpr)
b = RM[RM["month"] <= D(2024, 8)].copy()
b["ip"] = ((b["grp"] == "IND") & (b["month"] >= BASE_PRICE_CHANGE)).astype(float)
b["swf"] = b["sw"].astype(float)
for vc in ("2W", "3W"):
    s = b[b["vc"] == vc]
    rl = fe(s, "lsw", ["ip"])["ip"]; rv = fe(s, "swf", ["ip"])["ip"]
    pre_mean = s[(s["grp"] == "IND") & (s["month"] < BASE_PRICE_CHANGE)]["sw"].mean()
    dp = 5 / 60 if vc == "2W" else 10 / 100
    out(f"{vc}: log-swaps DiD {fmt(rl, 100)} (elasticity {rl[0]/np.log(1+dp):+.2f}); levels {fmt(rv)} = {100*rv[0]/pre_mean:+.1f}% "
        f"(elasticity {rv[0]/pre_mean/dp:+.2f}, CI {rv[1]/pre_mean/dp:+.2f} to {rv[2]/pre_mean/dp:+.2f}); n={len(s):,}")
out("done.")
