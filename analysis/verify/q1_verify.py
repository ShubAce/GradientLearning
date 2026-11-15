# %% [markdown]
# # Q1 verification - independent re-derivation (adversarial check of analysis/q1_network_performance.py)
#
# Fresh code: every number is rebuilt from `data/clean/events.parquet` via the `src/common.py` loaders.
# Flags are rebuilt from `event_type` (not the precomputed `is_*` columns) and station aggregates are rebuilt from
# events (not `station_month.parquet`), so a bug in either shared artefact would show up as a mismatch.
# Sections: A monthly KPIs + headline; B failure claim (naive vs same season); C Launch panel / mix decomposition;
# D charger generation, utilisation, demand deciles; E queue wait (survivorship test); F cm1 bridge (class-level,
# different segmentation from the analyst); G Kyron counterfactual; H wear sensitivity; I pack-health trend;
# J extra checks (Kyron after retirement date, failure -> later completion, Gen1 load confound).

# %%
import sys
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

try:
    ROOT = Path(__file__).resolve().parents[2]
except NameError:
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from common import *  # noqa: F401,F403

T0 = time.time()
RNG = np.random.default_rng(20261115)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
OUT = []   # (key, value) pairs collected for the verification table


def rec(key, val, fmt="{:.4f}"):
    OUT.append((key, val))
    print(f"  {key:<62} {fmt.format(val) if isinstance(val, (int, float, np.floating, np.integer)) else val}")


def M(y, m):
    return dt.date(y, m, 1)


PER = {"Jan-24": (M(2024, 1), M(2024, 1)), "May-24": (M(2024, 5), M(2024, 5)), "Jun-24": (M(2024, 6), M(2024, 6)),
       "May-25": (M(2025, 5), M(2025, 5)), "Jun-25": (M(2025, 6), M(2025, 6)),
       "Q1-24": (M(2024, 1), M(2024, 3)), "Q2-24": (M(2024, 4), M(2024, 6)),
       "Q1-25": (M(2025, 1), M(2025, 3)), "Q2-25": (M(2025, 4), M(2025, 6)),
       "H1-24": (M(2024, 1), M(2024, 6)), "Jul-Aug-24": (M(2024, 7), M(2024, 8)), "H1-25": (M(2025, 1), M(2025, 6))}


def nmonths(p):
    a, b = PER[p]
    return (b.year - a.year) * 12 + b.month - a.month + 1


ET = pl.col("event_type")
IS_C = ET == "swap_completed"
IS_F = ET.is_in(["failed_no_charged_battery", "abandoned_queue", "failed_system_error"])

# %% [markdown]
# ## A. Monthly KPIs and headline numbers

# %%
mon = (events(["month", "event_type", "amount_charged_inr", "energy_cost", "wear_cost", "fixed_cost"])
       .group_by("month").agg(
           pl.len().alias("att"), IS_C.sum().alias("sw"), IS_F.sum().alias("fail"),
           (ET == "failed_no_charged_battery").sum().alias("stock"), (ET == "abandoned_queue").sum().alias("aband"),
           (ET == "failed_system_error").sum().alias("syserr"), (ET == "cancelled_by_rider").sum().alias("canc"),
           # revenue rebuilt from amount_charged_inr on completed swaps only (not the precomputed revenue column)
           pl.col("amount_charged_inr").filter(IS_C).cast(pl.Float64).sum().alias("rev"),
           pl.col("energy_cost").filter(IS_C).sum().alias("en"), pl.col("wear_cost").filter(IS_C).sum().alias("wr"),
           pl.col("fixed_cost").filter(IS_C).sum().alias("fx"))
       .sort("month").collect(engine="streaming"))
act = (events(["month", "event_type", "rider_id", "station_id"]).filter(IS_C).group_by("month")
       .agg(pl.col("rider_id").n_unique().alias("riders"), pl.col("station_id").n_unique().alias("stations"))
       .collect(engine="streaming"))
mon = mon.join(act, on="month").sort("month").with_columns((pl.col("rev") - pl.col("en") - pl.col("wr")).alias("cm1"))
mon = mon.with_columns((pl.col("cm1") - pl.col("fx")).alias("cm2"))
print("[A] monthly panel")
print(mon.with_columns((pl.col("fail") / pl.col("att") * 100).round(2).alias("fail%"),
                       (pl.col("cm1") / pl.col("sw")).round(2).alias("cm1ps"), (pl.col("cm2") / pl.col("sw")).round(2).alias("cm2ps"),
                       (pl.col("rev") / 1e6).round(3).alias("revM")).select("month", "att", "sw", "fail%", "revM", "cm1ps", "cm2ps", "riders", "stations"))
print(f"  total attempts {mon['att'].sum():,}  completed {mon['sw'].sum():,}")


def P(p):
    a, b = PER[p]
    d = mon.filter(pl.col("month").is_between(a, b))
    s = {k: d[k].sum() for k in ["att", "sw", "fail", "stock", "aband", "syserr", "canc", "rev", "en", "wr", "fx", "cm1", "cm2"]}
    s["riders_avg"] = d["riders"].mean()
    s["n"] = d.height
    return s


PP = {p: P(p) for p in PER}
a, b = PP["H1-24"], PP["H1-25"]
print("\n[A] Headline / finding 1")
rec("H1 attempts x", b["att"] / a["att"])
rec("H1 swaps x", b["sw"] / a["sw"])
rec("H1 revenue x", b["rev"] / a["rev"])
rec("H1 failures x", b["fail"] / a["fail"])
rec("H1 avg monthly active riders x", b["riders_avg"] / a["riders_avg"])
rec("H1-24 failure rate %", a["fail"] / a["att"] * 100)
rec("H1-25 failure rate %", b["fail"] / b["att"] * 100)
for p in ["H1-24", "Jul-Aug-24", "H1-25", "Q2-24", "Q2-25"]:
    s = PP[p]
    rec(f"{p} cm1/swap", s["cm1"] / s["sw"])
    rec(f"{p} cm2/swap", s["cm2"] / s["sw"])
    rec(f"{p} wear/swap", s["wr"] / s["sw"])
    rec(f"{p} fixed/swap", s["fx"] / s["sw"])
    rec(f"{p} total cm2 per month (M)", s["cm2"] / s["n"] / 1e6)
j, u = PP["Jan-24"], PP["Jun-25"]
rec("Jan-24 -> Jun-25 swaps x", u["sw"] / j["sw"])
rec("Jan-24 -> Jun-25 revenue x", u["rev"] / j["rev"])
rec("Jan-24 revenue (M)", j["rev"] / 1e6)
rec("Jun-25 revenue (M)", u["rev"] / 1e6)
rec("Jan-24 active stations", float(mon["stations"][0]))
rec("Jun-25 active stations", float(mon["stations"][-1]))
base = {k: mon.filter(pl.col("month").is_between(*PER["H1-24"]))[k].mean() for k in ["sw", "rev", "riders"]}
last = mon.row(-1, named=True)
rec("Index Jun-25 swaps (H1-24 avg=100)", last["sw"] / base["sw"] * 100)
rec("Index Jun-25 revenue", last["rev"] / base["rev"] * 100)
rec("Index Jun-25 riders", last["riders"] / base["riders"] * 100)

# %% [markdown]
# ## B. "Failures rose faster than swaps": naive vs same season

# %%
print("\n[B] growth multiples (failures vs swaps) and failure rates")
for p0, p1 in [("Jan-24", "Jun-25"), ("Q1-24", "Q1-25"), ("Q2-24", "Q2-25"), ("H1-24", "H1-25"), ("Jun-24", "Jun-25")]:
    s0, s1 = PP[p0], PP[p1]
    print(f"  {p0}->{p1}: failures x{s1['fail'] / s0['fail']:.3f}  swaps x{s1['sw'] / s0['sw']:.3f}  "
          f"rate {s0['fail'] / s0['att'] * 100:.2f}% -> {s1['fail'] / s1['att'] * 100:.2f}%")
    rec(f"{p0}->{p1} failures x", s1["fail"] / s0["fail"])
    rec(f"{p0}->{p1} swaps x", s1["sw"] / s0["sw"])
    rec(f"{p0} fail rate %", s0["fail"] / s0["att"] * 100)
    rec(f"{p1} fail rate %", s1["fail"] / s1["att"] * 100)
for p in ["May-24", "May-25"]:
    s = PP[p]
    rec(f"{p} fail rate %", s["fail"] / s["att"] * 100)
rec("May-24 stockout % of attempts", PP["May-24"]["stock"] / PP["May-24"]["att"] * 100)
rec("May-24 abandoned % of attempts", PP["May-24"]["aband"] / PP["May-24"]["att"] * 100)
jm = mon.filter(~pl.col("month").dt.month().is_in([4, 5, 6]))
rec("Jul-Mar months fail rate range min %", float((jm["fail"] / jm["att"]).min() * 100))
rec("Jul-Mar months fail rate range max %", float((jm["fail"] / jm["att"]).max() * 100))
rec("System error rate, all months min %", float((mon["syserr"] / mon["att"]).min() * 100))
rec("System error rate, all months max %", float((mon["syserr"] / mon["att"]).max() * 100))
rec("Cancel rate, all months min %", float((mon["canc"] / mon["att"]).min() * 100))
rec("Cancel rate, all months max %", float((mon["canc"] / mon["att"]).max() * 100))

# %% [markdown]
# ## C. Launch panel and mix decomposition (station aggregates rebuilt from events)

# %%
st = (stations().filter(~pl.col("is_test"))
      .select("station_id", "city", "expansion_wave", "charger_generation", "slots_2w", "slots_3w",
              "inventory_target_2w", "inventory_target_3w", "commissioned_date"))
sq = (events(["station_id", "month", "date", "event_type"])
      .with_columns((pl.col("month").dt.year().cast(pl.Utf8) + "Q" + pl.col("month").dt.quarter().cast(pl.Utf8)).alias("q"))
      .group_by("station_id", "q").agg(pl.len().alias("att"), IS_F.sum().alias("fail"), IS_C.sum().alias("sw"),
                                       (ET == "failed_no_charged_battery").sum().alias("stock"),
                                       pl.col("date").n_unique().alias("days"))
      .collect(engine="streaming").join(st, on="station_id", how="left")
      .with_columns(pl.when(pl.col("expansion_wave") == "Launch").then(pl.lit("Launch")).otherwise(pl.lit("New")).alias("grp"),
                    (pl.col("slots_2w") + pl.col("slots_3w")).alias("slots")))
QS = {"Q1-24": "2024Q1", "Q2-24": "2024Q2", "Q1-25": "2025Q1", "Q2-25": "2025Q2"}


def sqp(p):
    return sq.filter(pl.col("q") == QS[p]).to_pandas().sort_values("station_id").reset_index(drop=True)


print("\n[C] decomposition and Launch panel")
for qq in ["Q1", "Q2"]:
    s0, s1 = sqp(f"{qq}-24"), sqp(f"{qq}-25")
    assert (s0.grp == "Launch").all(), "2024 H1 should contain only Launch stations"
    r0 = s0.fail.sum() / s0.att.sum()
    r1 = s1.fail.sum() / s1.att.sum()
    L1 = s1[s1.grp == "Launch"]
    N1 = s1[s1.grp == "New"]
    wN = N1.att.sum() / s1.att.sum()
    rL1 = L1.fail.sum() / L1.att.sum()
    rN1 = N1.fail.sum() / N1.att.sum()
    within = (1 - wN) * (rL1 - r0)
    mix = wN * (rN1 - r0)
    rec(f"{qq} network rate 2024 %", r0 * 100)
    rec(f"{qq} network rate 2025 %", r1 * 100)
    rec(f"{qq} within-Launch term pp", within * 100)
    rec(f"{qq} new-station mix term pp", mix * 100)
    rec(f"{qq}-25 new-station share of attempts %", wN * 100)
    rec(f"{qq}-25 new-station fail rate %", rN1 * 100)
    rec(f"{qq}-25 Launch fail rate %", rL1 * 100)
    # paired station-cluster bootstrap: resample Launch stations jointly across years; new stations resampled separately
    pan = s0.merge(L1, on="station_id", suffixes=("_24", "_25"))
    B = 4000
    iL = RNG.integers(0, len(pan), (B, len(pan)))
    iN = RNG.integers(0, len(N1), (B, len(N1)))
    f24, a24 = pan.fail_24.values[iL].sum(1), pan.att_24.values[iL].sum(1)
    f25L, a25L = pan.fail_25.values[iL].sum(1), pan.att_25.values[iL].sum(1)
    f25N, a25N = N1.fail.values[iN].sum(1), N1.att.values[iN].sum(1)
    dnet = (f25L + f25N) / (a25L + a25N) - f24 / a24
    dlaunch = f25L / a25L - f24 / a24
    # analyst-style unpaired bootstrap (2024 and 2025 resampled independently) for comparison
    i0 = RNG.integers(0, len(s0), (B, len(s0)))
    i1 = RNG.integers(0, len(s1), (B, len(s1)))
    dun = s1.fail.values[i1].sum(1) / s1.att.values[i1].sum(1) - s0.fail.values[i0].sum(1) / s0.att.values[i0].sum(1)
    rec(f"{qq} network change pp, paired-cluster CI lo", np.percentile(dnet, 2.5) * 100)
    rec(f"{qq} network change pp, paired-cluster CI hi", np.percentile(dnet, 97.5) * 100)
    rec(f"{qq} network change pp, unpaired CI lo", np.percentile(dun, 2.5) * 100)
    rec(f"{qq} network change pp, unpaired CI hi", np.percentile(dun, 97.5) * 100)
    rec(f"{qq} Launch pooled change pp", (pan.fail_25.sum() / pan.att_25.sum() - pan.fail_24.sum() / pan.att_24.sum()) * 100)
    rec(f"{qq} Launch pooled change CI lo", np.percentile(dlaunch, 2.5) * 100)
    rec(f"{qq} Launch pooled change CI hi", np.percentile(dlaunch, 97.5) * 100)
    dd = pan.fail_25 / pan.att_25 - pan.fail_24 / pan.att_24
    rec(f"{qq} Launch median station change pp", float(np.median(dd) * 100))
    rec(f"{qq} Launch stations worse (n of 90)", float((dd > 0).sum()), "{:.0f}")
    rec(f"{qq} Launch Wilcoxon p", float(stats.wilcoxon(dd).pvalue))
    rec(f"{qq} Launch attempts growth %", (pan.att_25.sum() / pan.att_24.sum() - 1) * 100)
    g = (pan.att_25 / pan.days_25) / (pan.att_24 / pan.days_24) - 1
    rho = stats.spearmanr(g, dd)
    rec(f"{qq} Spearman(demand growth, fail change) rho", float(rho.statistic))
    rec(f"{qq} Spearman p", float(rho.pvalue))
    if qq == "Q2":
        PAN_Q2, S1_Q2, S0_Q2 = pan, s1, s0

# May at Launch stations, both years
smay = (events(["station_id", "month", "event_type", "expansion_wave"]).filter(pl.col("month").dt.month() == 5)
        .group_by(pl.col("month").dt.year().alias("yr"), (pl.col("expansion_wave") == "Launch").alias("launch"))
        .agg(pl.len().alias("att"), IS_F.sum().alias("fail")).collect(engine="streaming").sort("yr", "launch"))
for r in smay.iter_rows(named=True):
    rec(f"May-{r['yr'] % 100} {'Launch' if r['launch'] else 'New'} fail rate %", r["fail"] / r["att"] * 100)

# %% [markdown]
# ## D. Charger generation, utilisation, like-for-like cells and demand deciles

# %%
print("\n[D] Q2 failure by group x generation")
for p in ["Q1-24", "Q1-25", "Q2-24", "Q2-25"]:
    d = sqp(p)
    g = d.groupby(["grp", "charger_generation"]).agg(n=("station_id", "size"), att=("att", "sum"), fail=("fail", "sum"))
    for (gr, gen), r in g.iterrows():
        rec(f"{p} {gr} {gen} fail rate % (n={r.n})", r.fail / r.att * 100)
q2 = sqp("Q2-25")
q2["util_slot"] = q2.att / (q2.days * q2.slots)
for gr in ["Launch", "New"]:
    d = q2[q2.grp == gr]
    rec(f"Q2-25 {gr} attempts per slot-day (days with activity)", d.att.sum() / (d.days * d.slots).sum())
    rec(f"Q2-25 {gr} attempts per station-day", d.att.sum() / d.days.sum())
# station_month-style active days (calendar days since commissioning) for comparison
smo = station_month().filter(pl.col("month").is_between(*PER["Q2-25"])).group_by("station_id").agg(pl.col("active_days").sum().alias("cal_days")).to_pandas()
q2 = q2.merge(smo, on="station_id", how="left")
for gr in ["Launch", "New"]:
    d = q2[q2.grp == gr]
    rec(f"Q2-25 {gr} attempts per slot-day (calendar active days)", d.att.sum() / (d.cal_days * d.slots).sum())
g1 = q2[q2.charger_generation == "Gen1"]
rec("Q2-25 Gen1 stations", float(g1.station_id.nunique()), "{:.0f}")
rec("Q2-25 Gen1 share of stations %", g1.station_id.nunique() / q2.station_id.nunique() * 100)
rec("Q2-25 Gen1 share of attempts %", g1.att.sum() / q2.att.sum() * 100)
rec("Q2-25 Gen1 failures", float(g1.fail.sum()), "{:.0f}")
rec("Q2-25 network failures", float(q2.fail.sum()), "{:.0f}")
rec("Q2-25 Gen1 share of failures %", g1.fail.sum() / q2.fail.sum() * 100)
HOT = ["Delhi NCR", "Jaipur", "Hyderabad"]
g1h = g1[g1.city.isin(HOT)]
rec("Q2-25 hot-city Gen1 stations", float(g1h.station_id.nunique()), "{:.0f}")
rec("Q2-25 hot-city Gen1 share of failures %", g1h.fail.sum() / q2.fail.sum() * 100)
rec("Q2-25 hot-city Gen1 fail rate %", g1h.fail.sum() / g1h.att.sum() * 100)
cc = q2.groupby(["city", "charger_generation", "grp"]).agg(n=("station_id", "size"), att=("att", "sum"), fail=("fail", "sum"),
                                                          slot_days=("util_slot", "size"))
cc["rate"] = cc.fail / cc.att * 100
print(cc[["n", "att", "rate"]].round(2).to_string())
# Gen1 load confound: utilisation per slot-day and inventory per slot, same city, Gen1 vs Gen2 (Launch only), Q2-25 and Q2-24
print("\n[D] Gen1 vs Gen2 load and capacity, same city (Launch stations)")
for p in ["Q2-24", "Q2-25"]:
    d = sqp(p)
    d = d[d.grp == "Launch"]
    t = d.groupby(["city", "charger_generation"]).apply(lambda x: pd.Series({
        "n": len(x), "att_per_slot_day": x.att.sum() / (x.days * x.slots).sum(), "att_per_station_day": x.att.sum() / x.days.sum(),
        "slots": x.slots.mean(), "inv_per_slot": ((x.inventory_target_2w + x.inventory_target_3w) / x.slots).mean(),
        "fail%": x.fail.sum() / x.att.sum() * 100}), include_groups=False)
    print(p)
    print(t.round(2).to_string())
# what-if: Gen1 at same-city Gen2 rate (Q2-25, all stations of that generation in the city)
g2r = q2[q2.charger_generation == "Gen2"].groupby("city").apply(lambda x: x.fail.sum() / x.att.sum(), include_groups=False)
g1c = g1.groupby("city").agg(att=("att", "sum"), fail=("fail", "sum"))
g1c["avoid"] = g1c.fail - g1c.att * g2r.reindex(g1c.index)
rec("What-if Gen1->same-city Gen2: avoided failures Q2-25", float(g1c.avoid.sum()), "{:,.0f}")
rec("What-if network Q2-25 rate after %", (q2.fail.sum() - g1c.avoid.sum()) / q2.att.sum() * 100)

print("\n[D] demand deciles, Q2-25 (attempts per day with activity)")
q2["dem"] = q2.att / q2.days
q2["dec"] = pd.qcut(q2.dem.rank(method="first"), 10, labels=range(1, 11)).astype(int)
dt_ = q2.groupby(["dec", "grp"]).agg(n=("station_id", "size"), att=("att", "sum"), fail=("fail", "sum"))
dt_["rate"] = dt_.fail / dt_.att * 100
print(dt_.round(2).unstack("grp").to_string())
for dcl in (9, 10):
    for gr in ("Launch", "New"):
        if (dcl, gr) in dt_.index:
            rec(f"Q2-25 D{dcl} {gr} fail rate %", float(dt_.loc[(dcl, gr), "rate"]))
newshare = q2.groupby("dec").apply(lambda x: (x.grp == "New").mean(), include_groups=False)
rec("Q2-25 new-station share, deciles 1-3 min", float(newshare.loc[1:3].min()))
rec("Q2-25 new-station share, deciles 1-3 max", float(newshare.loc[1:3].max()))
dec_rate = q2.groupby("dec").apply(lambda x: x.fail.sum() / x.att.sum(), include_groups=False)
dec_dem = q2.groupby("dec").dem.median()
rec("Q2-25 Spearman(decile demand, decile rate)", float(stats.spearmanr(dec_dem, dec_rate).statistic))
pan = PAN_Q2.copy()
pan["dem24"] = pan.att_24 / pan.days_24
pan["quint"] = pd.qcut(pan.dem24.rank(method="first"), 5, labels=range(1, 6)).astype(int)
qt = pan.groupby("quint").agg(a24=("att_24", "sum"), a25=("att_25", "sum"), f24=("fail_24", "sum"), f25=("fail_25", "sum"))
qt["r24"], qt["r25"] = qt.f24 / qt.a24 * 100, qt.f25 / qt.a25 * 100
qt["chg"], qt["growth"] = qt.r25 - qt.r24, (qt.a25 / qt.a24 - 1) * 100
print(qt.round(2).to_string())
rec("Launch busiest 2024 quintile Q2 rate 2024 %", float(qt.loc[5, "r24"]))
rec("Launch busiest 2024 quintile Q2 rate 2025 %", float(qt.loc[5, "r25"]))
rec("Launch busiest quintile attempt growth %", float(qt.loc[5, "growth"]))
rec("Launch quintiles max |change| pp", float(qt.chg.abs().max()))

# %% [markdown]
# ## E. Queue wait: flat KPI and the survivorship explanation
# If waits of *completed* swaps are flat while abandoned riders wait longer in summer, survivorship is plausible.
# If waits are flat for every event type, every hour and every station type, the field simply carries no load signal.

# %%
print("\n[E] queue wait by month and event type (minutes)")
qwm = (events(["month", "event_type", "queue_wait_sec"]).filter(ET.is_in(["swap_completed", "abandoned_queue", "failed_no_charged_battery"]))
       .group_by("month", "event_type").agg((pl.col("queue_wait_sec").mean() / 60).alias("mean"),
                                            (pl.col("queue_wait_sec").quantile(0.9) / 60).alias("p90"),
                                            pl.col("queue_wait_sec").null_count().alias("nulls"), pl.len().alias("n"))
       .collect(engine="streaming").sort("event_type", "month"))
print(qwm.pivot(on="event_type", index="month", values=["mean", "p90"]).sort("month"))
cm_ = qwm.filter(pl.col("event_type") == "swap_completed")
rec("Completed wait mean, min over months", float(cm_["mean"].min()))
rec("Completed wait mean, max over months", float(cm_["mean"].max()))
rec("Completed wait p90, min over months", float(cm_["p90"].min()))
rec("Completed wait p90, max over months", float(cm_["p90"].max()))
qwq = (events(["month", "event_type", "queue_wait_sec"]).filter(IS_C & pl.col("month").is_between(M(2024, 4), M(2025, 6)))
       .with_columns(pl.col("month").dt.year().alias("yr"), pl.col("month").dt.month().alias("mo"))
       .filter(pl.col("mo").is_between(4, 6)).group_by("yr").agg((pl.col("queue_wait_sec").mean() / 60).alias("mean"),
                                                                  (pl.col("queue_wait_sec").quantile(0.9) / 60).alias("p90"))
       .collect(engine="streaming").sort("yr"))
print(qwq)
qh = (events(["hour", "event_type", "queue_wait_sec", "charger_generation", "city", "month"])
      .filter(ET.is_in(["swap_completed", "abandoned_queue"]) & (pl.col("month") == M(2025, 5)))
      .with_columns(pl.col("city").is_in(HOT).alias("hot"))
      .group_by("event_type", "charger_generation", "hot").agg((pl.col("queue_wait_sec").mean() / 60).alias("mean_min"), pl.len().alias("n"))
      .collect(engine="streaming").sort("event_type", "charger_generation", "hot"))
print("May-25 wait by event type x generation x hot city:")
print(qh)
qhr = (events(["hour", "event_type", "queue_wait_sec"]).filter(IS_C).group_by("hour")
       .agg((pl.col("queue_wait_sec").mean() / 60).alias("mean_min")).collect(engine="streaming").sort("hour"))
rec("Completed wait mean by hour, min", float(qhr["mean_min"].min()))
rec("Completed wait mean by hour, max", float(qhr["mean_min"].max()))

# %% [markdown]
# ## F. cm1 bridge - independent class-level decomposition (2 segments: 2W/3W) + direct rupee checks

# %%
seg = (events(["month", "event_type", "vehicle_class", "partner_id", "city", "supplier_out", "list_price_inr", "discount_inr",
               "amount_charged_inr", "energy_to_recharge_kwh", "energy_cost", "wear_cost", "fixed_cost"])
       .filter(IS_C)
       .with_columns(pl.when(pl.col("partner_id").is_null()).then(pl.lit("Indep"))
                     .when(pl.col("partner_id") == ZIPDROP_ID).then(pl.lit("ZipDrop")).otherwise(pl.lit("OtherP")).alias("bill"),
                     pl.col("city").is_in(PILOT_CITIES).alias("pilot"))
       .group_by("month", "vehicle_class", "bill", "pilot", "supplier_out")
       .agg(pl.len().alias("n"), pl.col("list_price_inr").cast(pl.Float64).sum().alias("L"),
            pl.col("discount_inr").cast(pl.Float64).sum().alias("D"), pl.col("amount_charged_inr").cast(pl.Float64).sum().alias("A"),
            pl.col("energy_to_recharge_kwh").cast(pl.Float64).sum().alias("K"), pl.col("energy_cost").sum().alias("E"),
            pl.col("wear_cost").sum().alias("W"), pl.col("fixed_cost").sum().alias("F"))
       .collect(engine="streaming").to_pandas())
seg["month"] = pd.to_datetime(seg["month"]).dt.date
print(f"\n[F] segment rows {len(seg)}; check revenue vs monthly panel: {abs(seg.A.sum() - mon['rev'].sum()):.2f}")
gref = seg.groupby(["vehicle_class", "supplier_out"]).apply(lambda x: x.W.sum() / x.n.sum(), include_groups=False)
print(gref.round(2))
rec("Ref wear Kyron 2W Rs/swap", float(gref[("2W", "Kyron")]))
rec("Ref wear Cellora 2W Rs/swap", float(gref[("2W", "Cellora")]))
rec("Ref wear Amptek 2W Rs/swap", float(gref[("2W", "Amptek")]))
rec("Ref wear Kyron 3W Rs/swap", float(gref[("3W", "Kyron")]))
rec("Ref wear Cellora 3W Rs/swap", float(gref[("3W", "Cellora")]))
rec("Kyron/Cellora 2W wear ratio", float(gref[("2W", "Kyron")] / gref[("2W", "Cellora")]))
ks = seg.groupby("month").apply(lambda x: x.loc[x.supplier_out == "Kyron", "n"].sum() / x.n.sum() * 100, include_groups=False)
print("Kyron share of issued packs by month %:", ", ".join(f"{m:%b-%y}={v:.1f}" for m, v in ks.items()))
rec("Kyron share Aug-24 %", float(ks.loc[M(2024, 8)]))
rec("Kyron share Sep-24 %", float(ks.loc[M(2024, 9)]))
rec("Kyron share Jun-25 %", float(ks.loc[M(2025, 6)]))
k2 = seg[seg.vehicle_class == "2W"].groupby("month").apply(lambda x: x.loc[x.supplier_out == "Kyron", "n"].sum() / x.n.sum() * 100, include_groups=False)
rec("Kyron share of 2W issues, Jan-24 %", float(k2.loc[M(2024, 1)]))


def pslice(p):
    a, b = PER[p]
    return seg[(seg.month >= a) & (seg.month <= b)]


def class_stats(p):
    d = pslice(p)
    c = d.groupby("vehicle_class")[["n", "L", "D", "A", "K", "E", "W", "F"]].sum()
    out = pd.DataFrame({"n": c.n, "Lm": c.L / c.n, "d": c.D / c.L, "Pm": (c.A - c.L + c.D) / c.n, "Km": c.K / c.n, "tau": c.E / c.K,
                        "Wm": c.W / c.n, "cm1": (c.A - c.E - c.W) / c.n})
    u = d.pivot_table(index="vehicle_class", columns="supplier_out", values="n", aggfunc="sum").fillna(0)
    u = u.div(u.sum(axis=1), axis=0)
    return out, u, d.n.sum(), d.F.sum()


def bridge(p0, p1):
    c0, u0, N0, F0 = class_stats(p0)
    c1, u1, N1, F1 = class_stats(p1)
    w0, w1 = c0.n / N0, c1.n / N1
    sup = sorted(set(u0.columns) | set(u1.columns))
    u0, u1 = u0.reindex(columns=sup, fill_value=0), u1.reindex(columns=sup, fill_value=0)
    g = pd.DataFrame({k: [gref.get((vc, k), 0.0) for vc in c0.index] for k in sup}, index=c0.index)
    share = ((u1 - u0) * g).sum(axis=1)
    comp = {"price": (w1 * (c1.Lm - c0.Lm) * (1 - c0.d)).sum(), "discount(all partners)": -(w1 * c1.Lm * (c1.d - c0.d)).sum(),
            "pilot": (w1 * (c1.Pm - c0.Pm)).sum(), "kWh/swap": -(w1 * (c1.Km - c0.Km) * c0.tau).sum(),
            "tariff": -(w1 * c1.Km * (c1.tau - c0.tau)).sum(), "Kyron share": -(w1 * share).sum(),
            "wear within": -(w1 * ((c1.Wm - c0.Wm) - share)).sum(), "class mix": ((w1 - w0) * c0.cm1).sum()}
    tot0 = (pslice(p0).A.sum() - pslice(p0).E.sum() - pslice(p0).W.sum()) / N0
    tot1 = (pslice(p1).A.sum() - pslice(p1).E.sum() - pslice(p1).W.sum()) / N1
    comp = pd.Series(comp)
    return comp, tot0, tot1, F0 / N0, F1 / N1, N1 / nmonths(p1)


print("\n[F] class-level bridges (Rs per swap)")
for p0, p1 in [("H1-24", "Jul-Aug-24"), ("Jul-Aug-24", "H1-25"), ("H1-24", "H1-25"), ("Q2-24", "Q2-25")]:
    comp, t0, t1, f0, f1, vol = bridge(p0, p1)
    print(f"  {p0} -> {p1}: cm1 {t0:.2f} -> {t1:.2f} (chg {t1 - t0:+.2f}); residual {t1 - t0 - comp.sum():+.2e}; "
          f"fixed/swap {f0:.2f} -> {f1:.2f}; vol {vol:,.0f}/month")
    for k, v in comp.items():
        print(f"      {k:<24} {v:+7.2f}  ({v * vol / 1e6:+.2f} M/month)")
    rec(f"bridge {p0}->{p1} cm1 change", t1 - t0)
    rec(f"bridge {p0}->{p1} Kyron share effect", comp["Kyron share"])
    rec(f"bridge {p0}->{p1} price effect", comp["price"])
    rec(f"bridge {p0}->{p1} pilot effect", comp["pilot"])
    rec(f"bridge {p0}->{p1} kWh effect", comp["kWh/swap"])
    rec(f"bridge {p0}->{p1} discount effect", comp["discount(all partners)"])
    rec(f"bridge {p0}->{p1} Kyron effect M/month", comp["Kyron share"] * vol / 1e6)
    rec(f"bridge {p0}->{p1} cm2 start", t0 - f0)
    rec(f"bridge {p0}->{p1} cm2 end", t1 - f1)

# direct ZipDrop amendment cost and ZipDrop share, H1-25
h = pslice("H1-25")
zd = h[h.bill == "ZipDrop"]
dzd_pre = pslice("Jul-Aug-24")
dzd_pre = dzd_pre[dzd_pre.bill == "ZipDrop"]
rec("ZipDrop discount rate Jul-Aug-24 %", dzd_pre.D.sum() / dzd_pre.L.sum() * 100)
rec("ZipDrop discount rate H1-25 %", zd.D.sum() / zd.L.sum() * 100)
rec("ZipDrop share of H1-25 swaps %", zd.n.sum() / h.n.sum() * 100)
rec("ZipDrop share of partner-billed H1-25 swaps %", zd.n.sum() / h[h.bill != "Indep"].n.sum() * 100)
zd_cost = (zd.D.sum() - zd.L.sum() * dzd_pre.D.sum() / dzd_pre.L.sum())
rec("ZipDrop amendment cost Rs/swap (network, H1-25)", -zd_cost / h.n.sum())
rec("ZipDrop amendment cost M/month (H1-25)", -zd_cost / 6 / 1e6)
pil = h[h.pilot]
rec("Pilot net adjustment Rs per pilot-city swap", (pil.A - pil.L + pil.D).sum() / pil.n.sum())
rec("Pilot net adjustment M/month (H1-25)", (pil.A - pil.L + pil.D).sum() / 6 / 1e6)
# Kyron direct excess wear H1-25 per class
for vc in ["2W", "3W"]:
    k = h[(h.supplier_out == "Kyron") & (h.vehicle_class == vc)]
    rec(f"Kyron {vc} excess wear vs Cellora M/month (H1-25)", -(k.W.sum() - k.n.sum() * gref[(vc, "Cellora")]) / 6 / 1e6)
    rec(f"Kyron {vc} swaps per month (H1-25)", k.n.sum() / 6, "{:,.0f}")

# %% [markdown]
# ## G. Counterfactual: Kyron-issued swaps at Cellora wear per swap of the same class

# %%
seg["W_cf"] = np.where(seg.supplier_out == "Kyron", seg.n * seg.vehicle_class.map({"2W": gref[("2W", "Cellora")], "3W": gref[("3W", "Cellora")]}), seg.W)
print("\n[G] counterfactual")
for p in ["H1-24", "Jul-Aug-24", "H1-25", "Q2-25", "Jun-25"]:
    d = pslice(p)
    n = d.n.sum()
    act_ = (d.A.sum() - d.E.sum() - d.W.sum()) / n
    cf_ = (d.A.sum() - d.E.sum() - d.W_cf.sum()) / n
    rec(f"{p} cm1 actual", act_)
    rec(f"{p} cm1 Kyron-at-Cellora", cf_)
    rec(f"{p} cm2 Kyron-at-Cellora", cf_ - d.F.sum() / n)
    rec(f"{p} uplift M/month", (cf_ - act_) * n / nmonths(p) / 1e6)

# %% [markdown]
# ## H. Wear sensitivity (EOL 70/60, salvage 0/20%)

# %%
bo = (events(["month", "event_type", "battery_out_id"]).filter(IS_C).group_by("month", "battery_out_id").agg(pl.len().alias("n"))
      .collect(engine="streaming"))
bt = batteries().select("battery_id", "supplier", "purchase_cost_inr", "loss_per_cycle_used", "initial_soh_pct", "wear_cost_per_cycle",
                        "retired_date", "retirement_reason", "vehicle_class", "n_cycles", "current_soh_pct")
bo = bo.join(bt, left_on="battery_out_id", right_on="battery_id", how="left")
print(f"\n[H] issued-pack rows {bo.height:,}; unmatched {bo['supplier'].null_count()}")
monp = mon.to_pandas()
monp["month"] = pd.to_datetime(monp["month"]).dt.date
scen = [("EOL70/s0", 70, 0.0), ("EOL70/s20", 70, 0.2), ("EOL60/s0", 60, 0.0), ("EOL60/s20", 60, 0.2)]
for lab, eol, s in scen:
    w = (bo.with_columns((pl.col("purchase_cost_inr") * (1 - s) * pl.col("loss_per_cycle_used") / (pl.col("initial_soh_pct") - eol) * pl.col("n")).alias("w"))
         .group_by("month").agg(pl.col("w").sum()).to_pandas())
    w["month"] = pd.to_datetime(w["month"]).dt.date
    m2 = monp.merge(w, on="month")
    if lab == "EOL70/s0":
        rec("Sensitivity base vs event wear, max |diff| Rs/month", float((m2.w - m2.wr).abs().max()))

    def lvl(p, cm2=False):
        a_, b_ = PER[p]
        d = m2[(m2.month >= a_) & (m2.month <= b_)]
        v = (d.rev.sum() - d.en.sum() - d.w.sum()) / d.sw.sum()
        return v - d.fx.sum() / d.sw.sum() if cm2 else v
    rec(f"{lab} H1-25 cm1", lvl("H1-25"))
    rec(f"{lab} H1-25 cm2", lvl("H1-25", True))
    rec(f"{lab} chg Jul-Aug-24 -> H1-25", lvl("H1-25") - lvl("Jul-Aug-24"))
    rec(f"{lab} chg H1-24 -> H1-25", lvl("H1-25") - lvl("H1-24"))
    rec(f"{lab} chg Q1-24 -> Q1-25 (no post-EOL issues)", lvl("Q1-25") - lvl("Q1-24"))

# %% [markdown]
# ## I. Pack health, energy and km per 2W swap

# %%
h2 = (events(["month", "event_type", "vehicle_class", "soh_in_pct", "soc_in_pct", "energy_to_recharge_kwh", "km_since_last_swap", "supplier_in"])
      .filter(IS_C & (pl.col("vehicle_class") == "2W"))
      .group_by("month").agg(pl.col("soh_in_pct").mean().alias("soh"), pl.col("soc_in_pct").mean().alias("soc"),
                             pl.col("energy_to_recharge_kwh").mean().alias("kwh"), pl.col("km_since_last_swap").mean().alias("km"),
                             pl.col("km_since_last_swap").null_count().alias("km_null"), pl.len().alias("n"),
                             pl.col("soh_in_pct").filter(pl.col("supplier_in") != "Kyron").mean().alias("soh_nonkyron"),
                             pl.col("km_since_last_swap").filter(pl.col("supplier_in") != "Kyron").mean().alias("km_nonkyron"))
      .collect(engine="streaming").sort("month"))
print("\n[I] 2W health by month")
print(h2)
f, l_ = h2.row(0, named=True), h2.row(-1, named=True)
rec("2W kWh/swap Jan-24", f["kwh"])
rec("2W kWh/swap Jun-25", l_["kwh"])
rec("2W SoH in Jan-24", f["soh"])
rec("2W SoH in Jun-25", l_["soh"])
rec("2W SoC in Jan-24", f["soc"])
rec("2W SoC in Jun-25", l_["soc"])
rec("2W km Jan-24", f["km"])
rec("2W km Jun-25", l_["km"])
rec("2W km change %", (l_["km"] / f["km"] - 1) * 100)
rec("2W non-Kyron SoH in Jun-25", l_["soh_nonkyron"])
rec("2W non-Kyron km Jun-25", l_["km_nonkyron"])
rec("2W km null share overall %", h2["km_null"].sum() / h2["n"].sum() * 100)

# %% [markdown]
# ## J. Extra checks

# %%
print("\n[J1] Kyron packs issued after their recorded retirement date")
after = (events(["date", "event_type", "battery_out_id", "soh_out_pct"]).filter(IS_C)
         .join(bt.lazy().select("battery_id", "supplier", "retired_date", "vehicle_class"), left_on="battery_out_id", right_on="battery_id", how="left")
         .with_columns(pl.col("retired_date").cast(pl.Date))
         .filter(pl.col("retired_date").is_not_null())
         .group_by("supplier", "vehicle_class", (pl.col("date") > pl.col("retired_date")).alias("after_retire"))
         .agg(pl.len().alias("swaps"), pl.col("soh_out_pct").mean().alias("soh_out_mean"), pl.col("date").max().alias("last_date"))
         .collect(engine="streaming").sort("supplier", "vehicle_class", "after_retire"))
print(after)
print(bt.group_by("supplier", "vehicle_class", "retirement_reason").agg(pl.len(), pl.col("retired_date").min().alias("rmin"),
                                                                        pl.col("retired_date").max().alias("rmax"),
                                                                        pl.col("current_soh_pct").mean().alias("soh_now")).sort("supplier", "vehicle_class"))
kj = (events(["date", "event_type", "supplier_out", "soh_out_pct", "vehicle_class"]).filter(IS_C & (pl.col("date") >= dt.date(2025, 6, 1)))
      .group_by((pl.col("date") > dt.date(2025, 6, 15)).alias("after_jun15"), "vehicle_class")
      .agg((pl.col("supplier_out") == "Kyron").mean().alias("kyron_share"), pl.col("soh_out_pct").filter(pl.col("supplier_out") == "Kyron").mean().alias("kyron_soh_out"))
      .collect(engine="streaming").sort("vehicle_class", "after_jun15"))
print(kj)
# Non-Kyron fleet health trajectory: share of 2W non-Kyron packs within 5 SoH points of 70% EOL at end of window
nk = bt.filter((pl.col("supplier") != "Kyron") & (pl.col("vehicle_class") == "2W"))
print(f"[J1b] non-Kyron 2W packs: n={nk.height}, current SoH mean {nk['current_soh_pct'].mean():.1f}, "
      f"share <=80% {(nk['current_soh_pct'] <= 80).mean() * 100:.1f}%, share <=75% {(nk['current_soh_pct'] <= 75).mean() * 100:.1f}%")

print("\n[J2] Q2-25 service failures followed by a completed swap by the same rider")
rr = (events(["rider_id", "event_ts", "event_type", "month", "station_id"]).filter(pl.col("month").is_between(*PER["Q2-25"]))
      .select("rider_id", "event_ts", "station_id", IS_C.alias("c"), IS_F.alias("f"))
      .collect(engine="streaming").sort("rider_id", "event_ts"))
rr = rr.with_columns(pl.when(pl.col("c")).then(pl.col("event_ts")).alias("cts")).with_columns(
    pl.col("cts").backward_fill().over("rider_id").alias("next_c"))
ff = rr.filter(pl.col("f")).with_columns(((pl.col("next_c") - pl.col("event_ts")).dt.total_minutes()).alias("gap_min"))
for lim in (60, 180, 1440):
    rec(f"Q2-25 failures followed by own completed swap within {lim} min %", float((ff["gap_min"] <= lim).mean() * 100))
rec("Q2-25 failures never followed by a completed swap in Q2-25 %", float(ff["gap_min"].is_null().mean() * 100))
rec("Q2-25 revenue per completed swap", PP["Q2-25"]["rev"] / PP["Q2-25"]["sw"])
rec("Q2-25 failures x revenue/swap, Rs M (upper bound of lost revenue)", PP["Q2-25"]["fail"] * PP["Q2-25"]["rev"] / PP["Q2-25"]["sw"] / 1e6)
del rr, ff

# %% [markdown]
# ## K. Wear charged after end-of-life (internal consistency of the cost model)
# The model charges purchase cost x loss/cycle / (initial SoH - 70) on EVERY issue. A pack issued below 70% SoH has
# already been fully depreciated under the stated assumption (70% EOL, no salvage), so every later charge is extra.
# Variants: (a) cap - post-EOL issues carry 0 wear (lifetime wear = purchase cost); (b) replacement-equivalent -
# post-EOL issues carry the Cellora wear of the same class (the cost of the pack that should have been issued).

# %%
print("\n[K1] lifetime wear charged / purchase cost, by supplier x class")
lw = (batteries().group_by("supplier", "vehicle_class")
      .agg(pl.len().alias("packs"), pl.col("n_cycles").mean().alias("cycles_mean"), pl.col("expected_cycles_to_eol").mean().alias("cycles_to_eol"),
           ((pl.col("n_cycles") * pl.col("wear_cost_per_cycle")).sum() / pl.col("purchase_cost_inr").sum()).alias("charged_over_cost"),
           pl.col("purchase_cost_inr").sum().alias("cost_total"))
      .sort("supplier", "vehicle_class"))
print(lw)
k2w = lw.filter((pl.col("supplier") == "Kyron") & (pl.col("vehicle_class") == "2W")).row(0, named=True)
rec("Kyron 2W lifetime wear charged / purchase cost", k2w["charged_over_cost"])
rec("Kyron 2W mean cycles issued", k2w["cycles_mean"])
rec("Kyron 2W mean cycles to 70% EOL", k2w["cycles_to_eol"])
pe = (events(["month", "event_type", "supplier_out", "vehicle_class", "soh_out_pct", "wear_cost"]).filter(IS_C)
      .with_columns((pl.col("soh_out_pct") < EOL_SOH).alias("post_eol"))
      .group_by("month", "supplier_out", "vehicle_class", "post_eol").agg(pl.len().alias("n"), pl.col("wear_cost").sum().alias("W"))
      .collect(engine="streaming").to_pandas())
pe["month"] = pd.to_datetime(pe["month"]).dt.date
pe["cref"] = pe.vehicle_class.map({"2W": gref[("2W", "Cellora")], "3W": gref[("3W", "Cellora")]})
pe["W_cap"] = np.where(pe.post_eol, 0.0, pe.W)
pe["W_rep"] = np.where(pe.post_eol, pe.n * pe.cref, pe.W)
pkm = pe[pe.supplier_out == "Kyron"].groupby(["month", "post_eol"]).n.sum().unstack(fill_value=0)
pkm.columns = ["pre" if not c else "post" for c in pkm.columns]
pkm["pct_post_eol"] = pkm["post"] / (pkm["post"] + pkm["pre"]) * 100
print("Kyron issues past 70% EOL by month (Jan-25 on):")
print(pkm[pkm.index >= M(2025, 1)].round(1).to_string())
rec("Kyron issues past EOL Apr-Jun-25 (count)", float(pkm.loc[M(2025, 4):M(2025, 6), "post"].sum()), "{:,.0f}")
rec("Kyron issues past EOL, May-25 %", float(pkm.loc[M(2025, 5), "pct_post_eol"]))
rec("Kyron issues past EOL, Jun-25 %", float(pkm.loc[M(2025, 6), "pct_post_eol"]))
pem = pe.groupby("month")[["n", "W", "W_cap", "W_rep"]].sum()
mm = monp.set_index("month")
for p in ["Jul-Aug-24", "H1-24", "H1-25", "Q2-24", "Q2-25", "Jun-25"]:
    a_, b_ = PER[p]
    d = mm.loc[a_:b_]
    w = pem.loc[a_:b_]
    n = d.sw.sum()
    base_ = (d.rev.sum() - d.en.sum() - w.W.sum()) / n
    cap_ = (d.rev.sum() - d.en.sum() - w.W_cap.sum()) / n
    rep_ = (d.rev.sum() - d.en.sum() - w.W_rep.sum()) / n
    rec(f"{p} cm1 base (uncapped)", base_)
    rec(f"{p} cm1 (a) wear capped at EOL", cap_)
    rec(f"{p} cm1 (b) post-EOL at Cellora wear", rep_)
    rec(f"{p} post-EOL wear Rs per network swap", (w.W.sum() - w.W_cap.sum()) / n)
    rec(f"{p} post-EOL wear M/month", (w.W.sum() - w.W_cap.sum()) / nmonths(p) / 1e6)
    rec(f"{p} cm2 (a) wear capped", cap_ - d.fx.sum() / n)
# Kyron share effect under the variants: per-swap Kyron excess over Cellora wear (same class), period P1 minus P0
for lab, col in [("base", "W"), ("(a) cap", "W_cap"), ("(b) replacement", "W_rep")]:
    def kx(p):
        a_, b_ = PER[p]
        d = pe[(pe.month >= a_) & (pe.month <= b_)]
        k = d[d.supplier_out == "Kyron"]
        return -(k[col].sum() - (k.n * k.cref).sum()) / d.n.sum()
    rec(f"Kyron excess wear Rs/swap H1-25, {lab}", kx("H1-25"))
    rec(f"Kyron excess wear M/month H1-25, {lab}", kx("H1-25") * PP["H1-25"]["sw"] / 6 / 1e6)
    rec(f"Kyron excess-wear change Jul-Aug-24 -> H1-25, {lab}", kx("H1-25") - kx("Jul-Aug-24"))
# lifetime-cost view of the Kyron 2W shortfall (one-off warranty-style number)
cel = batteries().filter((pl.col("supplier") == "Cellora") & (pl.col("vehicle_class") == "2W"))["expected_cycles_to_eol"].mean()
rec("Cellora 2W mean cycles to EOL", float(cel))
rec("Kyron 2W lifetime shortfall vs Cellora life, Rs M (one-off)", k2w["cost_total"] * (1 - k2w["cycles_to_eol"] / cel) / 1e6)
rec("Kyron 2W total purchase cost, Rs M", k2w["cost_total"] / 1e6)

print("\n[K1b] windows with NO post-EOL issues (wear model internally consistent): Q1 same season, Oct-24..Mar-25")
PER["Oct-24-Mar-25"] = (M(2024, 10), M(2025, 3))
PER["Apr-May-Jun-25"] = PER["Q2-25"]
PP["Oct-24-Mar-25"] = P("Oct-24-Mar-25")
for p0, p1 in [("Q1-24", "Q1-25"), ("Jul-Aug-24", "Oct-24-Mar-25")]:
    comp, t0, t1, f0, f1, vol = bridge(p0, p1)
    print(f"  {p0} -> {p1}: cm1 {t0:.2f} -> {t1:.2f} (chg {t1 - t0:+.2f}); fixed/swap {f0:.2f} -> {f1:.2f}; cm2 {t0 - f0:.2f} -> {t1 - f1:.2f}")
    for k, v in comp.items():
        print(f"      {k:<24} {v:+7.2f}  ({v * vol / 1e6:+.2f} M/month)")
    rec(f"bridge {p0}->{p1} cm1 change", t1 - t0)
    rec(f"bridge {p0}->{p1} cm1 end", t1)
    rec(f"bridge {p0}->{p1} Kyron share effect", comp["Kyron share"])
    rec(f"bridge {p0}->{p1} price effect", comp["price"])
    rec(f"bridge {p0}->{p1} Kyron effect M/month", comp["Kyron share"] * vol / 1e6)
    rec(f"bridge {p0}->{p1} cm2 end", t1 - f1)
a_, b_ = PER["Oct-24-Mar-25"]
w_ = pem.loc[a_:b_]
rec("Oct-24..Mar-25 post-EOL wear (should be 0)", float(w_.W.sum() - w_.W_cap.sum()))

print("\n[K2] per-pack energy link: 2W kWh to recharge by returned-pack SoH band, H1-25 (association within one period)")
kb = (events(["month", "event_type", "vehicle_class", "soh_in_pct", "soc_in_pct", "energy_to_recharge_kwh", "km_since_last_swap", "supplier_in"])
      .filter(IS_C & (pl.col("vehicle_class") == "2W") & pl.col("month").is_between(*PER["H1-25"]))
      .with_columns((pl.col("soh_in_pct") // 5 * 5).alias("soh_band"))
      .group_by("soh_band").agg(pl.len().alias("n"), pl.col("energy_to_recharge_kwh").mean().alias("kwh"),
                                pl.col("km_since_last_swap").mean().alias("km"), pl.col("soc_in_pct").mean().alias("soc_in"))
      .collect(engine="streaming").sort("soh_band"))
print(kb)
kbb = kb.filter(pl.col("n") > 5000)
rec("H1-25 2W kWh per swap, lowest populated SoH band", float(kbb["kwh"][0]))
rec("H1-25 2W kWh per swap, highest populated SoH band", float(kbb["kwh"][-1]))
rec("H1-25 2W km per swap, lowest populated SoH band", float(kbb["km"][0]))
rec("H1-25 2W km per swap, highest populated SoH band", float(kbb["km"][-1]))

print("\n[K3] what riders pay per km (2W, completed swaps with valid km)")
pk = (events(["month", "event_type", "vehicle_class", "amount_charged_inr", "km_since_last_swap"])
      .filter(IS_C & (pl.col("vehicle_class") == "2W") & pl.col("km_since_last_swap").is_not_null())
      .group_by("month").agg(pl.col("amount_charged_inr").cast(pl.Float64).sum().alias("rev"), pl.col("km_since_last_swap").cast(pl.Float64).sum().alias("km"))
      .collect(engine="streaming").sort("month").with_columns((pl.col("rev") / pl.col("km")).alias("rs_per_km")))
rec("2W Rs per km Jan-24", float(pk["rs_per_km"][0]))
rec("2W Rs per km Jun-25", float(pk["rs_per_km"][-1]))
rec("2W Rs per km change Jan-24 -> Jun-25 %", float((pk["rs_per_km"][-1] / pk["rs_per_km"][0] - 1) * 100))
h1a_ = pk.filter(pl.col("month").is_between(*PER["H1-24"]))
h1b_ = pk.filter(pl.col("month").is_between(*PER["H1-25"]))
rec("2W Rs per km H1-24 -> H1-25 change %", float(((h1b_["rev"].sum() / h1b_["km"].sum()) / (h1a_["rev"].sum() / h1a_["km"].sum()) - 1) * 100))

print("\n[K4] early detectability: mean SoH of issued 2W packs by supplier, first months of Kyron use")
ed = (events(["month", "event_type", "vehicle_class", "supplier_out", "soh_out_pct"])
      .filter(IS_C & (pl.col("vehicle_class") == "2W") & pl.col("month").is_between(M(2024, 9), M(2025, 1)))
      .group_by("month", "supplier_out").agg(pl.col("soh_out_pct").mean().alias("soh_out")).collect(engine="streaming")
      .pivot(on="supplier_out", index="month", values="soh_out").sort("month"))
print(ed)
rec("Kyron 2W issued SoH Sep-24", float(ed.filter(pl.col("month") == M(2024, 9))["Kyron"].item()))
rec("Kyron 2W issued SoH Nov-24", float(ed.filter(pl.col("month") == M(2024, 11))["Kyron"].item()))
rec("Cellora 2W issued SoH Sep-24", float(ed.filter(pl.col("month") == M(2024, 9))["Cellora"].item()))
rec("Cellora 2W issued SoH Nov-24", float(ed.filter(pl.col("month") == M(2024, 11))["Cellora"].item()))

print("\n[K5] Gen1 what-if with the same-city Gen3 (new-station) rate as the target, Q2-25")
g3r = q2[q2.charger_generation == "Gen3"].groupby("city").apply(lambda x: x.fail.sum() / x.att.sum(), include_groups=False)
g1c["avoid_g3"] = g1c.fail - g1c.att * g3r.reindex(g1c.index)
print(g1c.assign(r_g3=g3r.reindex(g1c.index) * 100).round(2).to_string())
rec("What-if Gen1->same-city Gen3: avoided failures Q2-25", float(g1c.avoid_g3.sum()), "{:,.0f}")
rec("What-if Gen1->same-city Gen3: network Q2-25 rate after %", (q2.fail.sum() - g1c.avoid_g3.sum()) / q2.att.sum() * 100)

# %% [markdown]
# ## Save the recomputed numbers

# %%
pd.DataFrame(OUT, columns=["metric", "value"]).to_csv(TAB / "q1_verify_numbers.csv", index=False)
print(f"\n[done] {len(OUT)} numbers saved to outputs/tables/q1_verify_numbers.csv; runtime {time.time() - T0:.1f}s")
