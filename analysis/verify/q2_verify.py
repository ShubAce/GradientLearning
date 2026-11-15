# %% [markdown]
# # Q2 verification (independent re-derivation)
# Adversarial re-computation of the Q2 service-failure claims from data/clean via src/common loaders only.
# Does NOT import or reuse analysis/q2_service_failures.py. Event outcome flags are re-derived from event_type.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import *  # noqa: E402,F401,F403

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)

HOTC = ["Delhi NCR", "Jaipur", "Hyderabad"]
AJ = [4, 5, 6]


def p(x, d=2):
    return f"{100 * x:.{d}f}%"


def mh(df, a, n1, c, n0):
    """Mantel-Haenszel risk ratio with Greenland-Robins 95% CI. df: one row per stratum."""
    d = df[(df[n1] > 0) & (df[n0] > 0)]
    N = d[n1] + d[n0]
    R = (d[a] * d[n0] / N).sum()
    S = (d[c] * d[n1] / N).sum()
    rr = R / S
    v = ((d[n1] * d[n0] * (d[a] + d[c]) - d[a] * d[c] * N) / N ** 2).sum() / (R * S)
    se = np.sqrt(v)
    return rr, rr * np.exp(-1.96 * se), rr * np.exp(1.96 * se), len(d)


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    i = np.arange(1, n + 1)
    return (2 * (i * x).sum()) / (n * x.sum()) - (n + 1) / n


def topshare(x, frac=0.10):
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    k = int(np.ceil(frac * len(x)))
    return x[:k].sum() / x.sum()


# %% [markdown]
# ## A. Base aggregates (one lazy scan, projected columns)

# %%
et = pl.col("event_type")
EV = events(["station_id", "event_ts", "event_ts_raw", "ts_corrected", "event_type", "queue_wait_sec", "vehicle_class",
             "city", "charger_generation", "expansion_wave", "connectivity_tier", "station_firmware", "sync_mode"])
FLAGS = [
    (et == "failed_no_charged_battery").alias("so"),
    (et == "abandoned_queue").alias("ab"),
    (et == "failed_system_error").alias("se"),
    (et == "cancelled_by_rider").alias("cn"),
    (et == "swap_completed").alias("cp"),
]
EVF = EV.with_columns(FLAGS).with_columns((pl.col("so") | pl.col("ab") | pl.col("se")).alias("fl"))

# Station-hour-class aggregate on CORRECTED time
SHC = (EVF.with_columns(pl.col("event_ts").dt.truncate("1h").alias("hs"))
       .group_by("station_id", "hs", "vehicle_class")
       .agg(pl.len().alias("att"), pl.col("fl").sum().alias("fail"), pl.col("so").sum().alias("so"),
            pl.col("ab").sum().alias("ab"), pl.col("se").sum().alias("se"), pl.col("cp").sum().alias("comp"),
            pl.col("queue_wait_sec").filter(pl.col("cp")).cast(pl.Float64).sum().alias("qws"),
            (pl.col("queue_wait_sec").filter(pl.col("cp")).cast(pl.Float64) ** 2).sum().alias("qws2"))
       .collect(engine="streaming"))
ST = stations().filter(~pl.col("is_test")).select("station_id", "city", "charger_generation", "expansion_wave",
                                                   "slots_2w", "slots_3w", "location_type")
SHC = (SHC.join(ST, on="station_id", how="left")
       .with_columns(pl.col("hs").dt.date().alias("date"), pl.col("hs").dt.truncate("1mo").dt.date().alias("month"),
                     pl.col("hs").dt.hour().alias("hour"), pl.col("hs").dt.weekday().alias("wd")))
SHC = SHC.with_columns(pl.col("month").dt.month().alias("m"), pl.col("month").dt.year().alias("y"))
print("station-hour-class rows", SHC.height, "| unmatched station attrs:", SHC.filter(pl.col("city").is_null()).height)

# %% [markdown]
# ## B. Taxonomy

# %%
N = int(SHC["att"].sum())
F = int(SHC["fail"].sum())
SO, AB, SE, CP = (int(SHC[c].sum()) for c in ["so", "ab", "se", "comp"])
CN = N - F - CP
print(f"[TAX] attempts {N:,}; failures {F:,} = {p(F / N)}; stockouts {SO:,} ({p(SO / N)}, {p(SO / F, 1)} of fails); "
      f"abandoned {AB:,} ({p(AB / N)}, {p(AB / F, 1)}); system errors {SE:,} ({p(SE / N)}, {p(SE / F, 1)}); cancelled {CN:,} ({p(CN / N)})")
for vc in ["2W", "3W"]:
    s = SHC.filter(pl.col("vehicle_class") == vc)
    print(f"[TAX] {vc}: {s['fail'].sum() / s['att'].sum():.4%} of {s['att'].sum():,}")
for dim in ["connectivity_tier", "station_firmware", "sync_mode"]:
    r = EVF.group_by(dim).agg(pl.len().alias("n"), pl.col("se").mean().alias("r")).sort(dim).collect()
    print(f"[TAX] system-error by {dim}: " + ", ".join(f"{a} {p(b, 3)} (n={n:,})" for a, n, b in r.iter_rows()))

# %% [markdown]
# ## C. Baseline and seasonality

# %%
mo = SHC.group_by("month").agg(pl.col("att").sum(), pl.col("fail").sum(), pl.col("so").sum()).sort("month").with_columns(
    (pl.col("fail") / pl.col("att")).alias("fr"))
cool = mo.filter(~pl.col("month").dt.month().is_in(AJ))
hotm = mo.filter(pl.col("month").dt.month().is_in(AJ))
print(f"[SEAS] Jul-Mar (all non-AJ months, n={cool.height}) baseline {p(cool['fail'].sum() / cool['att'].sum())}, "
      f"monthly range {p(cool['fr'].min())}-{p(cool['fr'].max())}")
print(f"[SEAS] Apr-Jun share of attempts {p(hotm['att'].sum() / N, 1)}, of failures {p(hotm['fail'].sum() / F, 1)}")
print("[SEAS] monthly:", ", ".join(f"{m:%b-%y} {p(r, 1)}" for m, r in zip(mo["month"], mo["fr"])))

jm = SHC.filter(~pl.col("m").is_in(AJ))
cg = jm.group_by("city", "charger_generation").agg(pl.col("att").sum(), pl.col("fail").sum()).with_columns(
    (pl.col("fail") / pl.col("att")).alias("fr")).sort("fr")
print(f"[SEAS] Jul-Mar city x gen cells: n={cg.height}, range {p(cg['fr'].min())}-{p(cg['fr'].max())}")
stjm = jm.group_by("station_id").agg(pl.col("att").sum(), pl.col("fail").sum()).with_columns((pl.col("fail") / pl.col("att")).alias("fr"))
print(f"[SEAS] Jul-Mar STATION rates: n={stjm.height}, range {p(stjm['fr'].min())}-{p(stjm['fr'].max())}, "
      f"p5-p95 {p(stjm['fr'].quantile(0.05))}-{p(stjm['fr'].quantile(0.95))}")
# chi-square heterogeneity across stations in Jul-Mar
exp_f = stjm["att"].to_numpy() * jm["fail"].sum() / jm["att"].sum()
chi = (((stjm["fail"].to_numpy() - exp_f) ** 2) / (exp_f * (1 - jm["fail"].sum() / jm["att"].sum()))).sum()
print(f"[SEAS] Jul-Mar station heterogeneity chi2={chi:.1f} on {stjm.height - 1} df, p={stats.chi2.sf(chi, stjm.height - 1):.3g}")
wd_all = SHC.group_by("wd").agg(pl.col("att").sum(), pl.col("fail").sum()).with_columns((pl.col("fail") / pl.col("att")).alias("fr")).sort("wd")
wd_jm = jm.group_by("wd").agg(pl.col("att").sum(), pl.col("fail").sum()).with_columns((pl.col("fail") / pl.col("att")).alias("fr")).sort("wd")
print(f"[SEAS] weekday range FULL period {p(wd_all['fr'].min())}-{p(wd_all['fr'].max())}; Jul-Mar {p(wd_jm['fr'].min())}-{p(wd_jm['fr'].max())}")

hm = SHC.group_by("month", "hour").agg(pl.col("att").sum(), pl.col("fail").sum()).with_columns((pl.col("fail") / pl.col("att")).alias("fr"))
mx = hm.sort("fr", descending=True).row(0, named=True)
mn = hm.sort("fr").row(0, named=True)
print(f"[HM] worst cell {mx['month']:%b-%y} {mx['hour']}:00 {p(mx['fr'], 1)} (att {mx['att']:,}); best {mn['month']:%b-%y} {mn['hour']}:00 "
      f"{p(mn['fr'], 1)} (att {mn['att']:,}); min cell size {hm['att'].min():,}")
EVE = [19, 20, 21, 22]
may = SHC.filter(pl.col("m") == 5)
e1, e0 = may.filter(pl.col("hour").is_in(EVE)), may.filter(~pl.col("hour").is_in(EVE))
print(f"[HM] May 19-22h {p(e1['fail'].sum() / e1['att'].sum(), 1)} vs other hours {p(e0['fail'].sum() / e0['att'].sum(), 1)}")
ev_all = SHC.filter(pl.col("hour").is_in(EVE))
print(f"[HM] 19-22h share of attempts {p(ev_all['att'].sum() / N, 1)}, failures {p(ev_all['fail'].sum() / F, 1)}")
# Jul-Mar evening vs other, by class
for nm, s in [("all", jm)] + [(vc, jm.filter(pl.col("vehicle_class") == vc)) for vc in ["2W", "3W"]]:
    a1, a0 = s.filter(pl.col("hour").is_in(EVE)), s.filter(~pl.col("hour").is_in(EVE))
    print(f"[HM] Jul-Mar {nm}: evening {p(a1['fail'].sum() / a1['att'].sum())} vs other {p(a0['fail'].sum() / a0['att'].sum())}")
a1, a0 = jm.filter(pl.col("hour").is_in(EVE)), jm.filter(~pl.col("hour").is_in(EVE))
print(f"[HM] Jul-Mar 3W share evening {p(a1.filter(pl.col('vehicle_class') == '3W')['att'].sum() / a1['att'].sum(), 1)} vs "
      f"other {p(a0.filter(pl.col('vehicle_class') == '3W')['att'].sum() / a0['att'].sum(), 1)}")

# Firmware clock trap
fw = (EVF.filter(pl.col("ts_corrected"))
      .with_columns(pl.col("event_ts").dt.hour().alias("hc"), pl.col("event_ts_raw").dt.hour().alias("hr"),
                    (pl.col("event_ts") - pl.col("event_ts_raw")).dt.total_minutes().alias("shift_min"))
      .select(pl.len().alias("n"), pl.col("hc").is_in(EVE).mean().alias("eve_c"), pl.col("hr").is_in(EVE).mean().alias("eve_r"),
              pl.col("hc").is_in(EVE).filter(pl.col("fl")).mean().alias("eve_c_f"),
              pl.col("hr").is_in(EVE).filter(pl.col("fl")).mean().alias("eve_r_f"),
              pl.col("shift_min").min().alias("smin"), pl.col("shift_min").max().alias("smax"),
              pl.col("event_ts_raw").min().alias("raw_min"), pl.col("event_ts_raw").max().alias("raw_max"),
              pl.col("station_firmware").n_unique().alias("nfw"))
      .collect())
print("[FW]", fw.to_dicts()[0])
# any v3.2.0 events in window NOT corrected?
chk = (EVF.filter(pl.col("station_firmware") == "v3.2.0", ~pl.col("ts_corrected"),
                  pl.col("event_ts_raw") >= pl.datetime(2025, 3, 10), pl.col("event_ts_raw") < pl.datetime(2025, 4, 15))
       .select(pl.len()).collect().item())
print(f"[FW] v3.2.0 events in bug window left uncorrected: {chk}")

# %% [markdown]
# ## D. City, generation, wave (Apr-Jun)

# %%
aj = SHC.filter(pl.col("m").is_in(AJ))
cy = aj.group_by("city").agg(pl.col("att").sum(), pl.col("fail").sum()).with_columns((pl.col("fail") / pl.col("att")).alias("fr")).sort("fr", descending=True)
print("[CITY] Apr-Jun:", ", ".join(f"{c} {p(r)}" for c, r in zip(cy["city"], cy["fr"])))
cyj = jm.group_by("city").agg(pl.col("att").sum(), pl.col("fail").sum()).with_columns((pl.col("fail") / pl.col("att")).alias("fr")).sort("fr")
print("[CITY] Jul-Mar:", ", ".join(f"{c} {p(r)}" for c, r in zip(cyj["city"], cyj["fr"])))
aj25 = aj.filter(pl.col("y") == 2025)
g = aj25.group_by("city", "month", "charger_generation").agg(pl.col("att").sum(), pl.col("fail").sum()).to_pandas()
w = g.pivot_table(index=["city", "month"], columns="charger_generation", values=["att", "fail"], fill_value=0)
w.columns = [f"{a}_{b}" for a, b in w.columns]
w = w.reset_index()
for exp in ["Gen1", "Gen2"]:
    rr, lo, hi, k = mh(w, f"fail_{exp}", f"att_{exp}", "fail_Gen3", "att_Gen3")
    print(f"[GEN] Apr-Jun 2025 {exp} vs Gen3 MH RR {rr:.2f} ({lo:.2f}-{hi:.2f}), strata={k}")
for c in sorted(w["city"].unique()):
    s = w[w["city"] == c]
    rr, lo, hi, k = mh(s, "fail_Gen1", "att_Gen1", "fail_Gen3", "att_Gen3")
    print(f"[GEN]   {c}: Gen1 vs Gen3 RR {rr:.2f} ({lo:.2f}-{hi:.2f})")
hg = aj.filter(pl.col("city").is_in(HOTC)).group_by("charger_generation").agg(pl.col("att").sum(), pl.col("fail").sum()).with_columns(
    (pl.col("fail") / pl.col("att")).alias("fr")).sort("charger_generation")
print("[GEN] hot cities Apr-Jun (both yrs):", ", ".join(f"{a} {p(b, 2)}" for a, b in zip(hg["charger_generation"], hg["fr"])))
hg25 = aj25.filter(pl.col("city").is_in(HOTC)).group_by("charger_generation").agg(pl.col("att").sum(), pl.col("fail").sum()).with_columns(
    (pl.col("fail") / pl.col("att")).alias("fr")).sort("charger_generation")
print("[GEN] hot cities Apr-Jun 2025 only:", ", ".join(f"{a} {p(b, 2)} (att {n:,})" for a, n, f_, b in hg25.iter_rows()))
# wave within generation, city x month strata (all months where both exist)
for gen, ew, rw in [("Gen2", "Wave1_2024H2", "Launch"), ("Gen3", "Wave2_2025H1", "Wave1_2024H2")]:
    s = SHC.filter(pl.col("charger_generation") == gen).group_by("city", "month", "expansion_wave").agg(pl.col("att").sum(), pl.col("fail").sum()).to_pandas()
    ww = s.pivot_table(index=["city", "month"], columns="expansion_wave", values=["att", "fail"], fill_value=0)
    ww.columns = [f"{a}_{b}" for a, b in ww.columns]
    ww = ww.reset_index()
    rr, lo, hi, k = mh(ww, f"fail_{ew}", f"att_{ew}", f"fail_{rw}", f"att_{rw}")
    print(f"[WAVE] {gen}: {ew} vs {rw} MH RR {rr:.2f} ({lo:.2f}-{hi:.2f}) strata={k}")

# %% [markdown]
# ## E. Weather (city-day)

# %%
CD = city_daily()
cdd = (SHC.group_by("city", "date").agg(pl.col("att").sum(), pl.col("fail").sum(), pl.col("se").sum(), pl.col("so").sum(), pl.col("ab").sum())
       .join(CD, on=["city", "date"], how="left").with_columns(pl.col("date").dt.truncate("1mo").alias("month")))
print(f"[WX] city-days {cdd.height}, unmatched {cdd.filter(pl.col('max_temp_c').is_null()).height}")
for lab, cond in [("<34", pl.col("max_temp_c") < 34), ("34-38", (pl.col("max_temp_c") >= 34) & (pl.col("max_temp_c") < 38)), (">=38", pl.col("max_temp_c") >= 38)]:
    s = cdd.filter(cond)
    print(f"[WX] tmax {lab}: {p(s['fail'].sum() / s['att'].sum())} ({s.height} city-days)")


def mh_day(flag, ev="fail", df=None):
    df = cdd if df is None else df
    gg = df.with_columns(flag.alias("x")).group_by("city", "month", "x").agg(pl.col(ev).sum().alias("e"), pl.col("att").sum().alias("n")).to_pandas()
    ww = gg.pivot_table(index=["city", "month"], columns="x", values=["e", "n"], fill_value=0)
    ww.columns = [f"{a}_{b}" for a, b in ww.columns]
    ww = ww.reset_index()
    if "n_True" not in ww or "n_False" not in ww:
        return (np.nan,) * 4
    return mh(ww, "e_True", "n_True", "e_False", "n_False")


for lab, fl, e in [("tmax>=36", pl.col("max_temp_c") >= 36, "fail"), ("heat_alert", pl.col("heat_alert"), "fail"),
                   ("grid_outage>0", pl.col("grid_outage_hours") > 0, "fail"), ("grid_outage>0 SE", pl.col("grid_outage_hours") > 0, "se"),
                   ("rain>=20", pl.col("rainfall_mm") >= 20, "fail")]:
    rr, lo, hi, k = mh_day(fl, e)
    print(f"[WX] {lab}: MH RR {rr:.2f} ({lo:.2f}-{hi:.2f}) strata={k}")
# heat alert vs temperature
print("[WX] heat_alert by tmax band:", cdd.with_columns(pl.col("max_temp_c").cut([30, 34, 36, 38, 40, 42]).alias("b"))
      .group_by("b").agg(pl.len().alias("days"), pl.col("heat_alert").mean().alias("alert_share")).sort("b").to_dicts())
cdd_hot = cdd.filter(pl.col("max_temp_c") >= 36)
rr, lo, hi, k = mh_day(pl.col("heat_alert"), "fail", cdd_hot)
print(f"[WX] heat_alert within city-month among days tmax>=36: RR {rr:.2f} ({lo:.2f}-{hi:.2f}) strata={k}")
cdd_mid = cdd.filter(pl.col("max_temp_c") < 36)
rr, lo, hi, k = mh_day(pl.col("heat_alert"), "fail", cdd_mid)
print(f"[WX] heat_alert within city-month among days tmax<36: RR {rr:.2f} ({lo:.2f}-{hi:.2f}) strata={k}")
# finer threshold check within hot city-months
tb = cdd.with_columns((pl.col("max_temp_c") // 1).alias("t")).group_by("t").agg(pl.len().alias("d"), pl.col("att").sum(), pl.col("fail").sum()).with_columns(
    (pl.col("fail") / pl.col("att")).alias("fr")).sort("t")
print("[WX] failure by integer tmax:", ", ".join(f"{int(t)}C {p(r, 1)} ({d})" for t, d, a, f_, r in tb.iter_rows() if t >= 30))

# %% [markdown]
# ## F. Queue wait, patience and load

# %%
qd = (EVF.filter(pl.col("cp") | pl.col("ab")).group_by("queue_wait_sec").agg(pl.col("ab").sum().alias("a"), pl.col("cp").sum().alias("c"))
      .sort("queue_wait_sec").collect())
t = qd["queue_wait_sec"].to_numpy().astype(float)
a = qd["a"].to_numpy().astype(float)
c = qd["c"].to_numpy().astype(float)


def wq(w, cnt, q):
    cs = np.cumsum(cnt)
    return w[np.searchsorted(cs, q * cs[-1])]


print(f"[QW] completed median {wq(t, c, 0.5) / 60:.2f} min, mean {(t * c).sum() / c.sum() / 60:.3f}, p90 {wq(t, c, 0.9) / 60:.2f}; "
      f"abandoned median {wq(t, a, 0.5) / 60:.2f} min (ratio {wq(t, a, 0.5) / wq(t, c, 0.5):.2f}); "
      f"abandoners > completed p90: {p(a[t > wq(t, c, 0.9)].sum() / a.sum(), 1)}")
risk = (a + c).sum() - np.r_[0, np.cumsum(a + c)[:-1]]
S = np.cumprod(1 - a / risk)


def S_at(sec):
    i = np.searchsorted(t, sec, side="right") - 1
    return S[i] if i >= 0 else 1.0


med = t[np.argmax(S <= 0.5)] / 60
print(f"[KM] S(5m) {p(S_at(300), 1)}, S(10m) {p(S_at(600), 1)}, S(15m) {p(S_at(900), 1)}; median {med:.2f} min")
mins = (t // 60).astype(int)
dfm = pd.DataFrame({"m": mins, "a": a, "c": c}).groupby("m").sum()
dfm["risk"] = (dfm["a"] + dfm["c"]).sum() - np.r_[0, np.cumsum(dfm["a"] + dfm["c"])[:-1]]
dfm["h"] = dfm["a"] / dfm["risk"]
print("[KM] hazard per minute:", ", ".join(f"m{m} {p(dfm.loc[m, 'h'], 1)}" for m in [5, 8, 10, 15]))
print(f"[QW] max completed wait {t[c > 0].max() / 60:.1f} min, max abandoned {t[a > 0].max() / 60:.1f} min, "
      f"min abandoned {t[a > 0].min() / 60:.2f} min")

# load: completed swaps per slot-hour (station-hour totals)
SH = (SHC.group_by("station_id", "hs").agg(pl.col("att").sum(), pl.col("comp").sum(), pl.col("ab").sum(), pl.col("so").sum(),
                                            pl.col("fail").sum(), pl.col("qws").sum(), pl.col("qws2").sum(),
                                            (pl.col("slots_2w").first() + pl.col("slots_3w").first()).alias("slots"))
      .with_columns((pl.col("comp") / pl.col("slots")).alias("u")))
edges = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
lb = ["<0.1", "0.1-0.2", "0.2-0.3", "0.3-0.5", "0.5-0.75", "0.75-1.0", "1.0-1.5", ">=1.5"]
ub = (SH.filter(pl.col("comp") > 0).with_columns(pl.col("u").cut(edges, labels=lb, left_closed=True).alias("b"))
      .group_by("b").agg(pl.len().alias("h"), pl.col("comp").sum(), pl.col("att").sum(), pl.col("ab").sum(), pl.col("qws").sum(), pl.col("qws2").sum())
      .with_columns((pl.col("qws") / pl.col("comp")).alias("mw"))
      .with_columns((((pl.col("qws2") / pl.col("comp")) - pl.col("mw") ** 2) / pl.col("comp")).sqrt().alias("se"),
                    (pl.col("ab") / pl.col("att")).alias("abr")).sort("b"))
for r in ub.iter_rows(named=True):
    print(f"[LOAD] util {r['b']}: hours {r['h']:,}, mean wait {r['mw']:.1f}s ({r['mw'] - 1.96 * r['se']:.1f}-{r['mw'] + 1.96 * r['se']:.1f}), abandon {p(r['abr'])}")
stw = SHC.group_by("station_id").agg((pl.col("qws").sum() / pl.col("comp").sum()).alias("mw"))
print(f"[LOAD] station mean completed wait range {stw['mw'].min():.1f}-{stw['mw'].max():.1f}s")
# abandonment by attempts-per-slot (demand) bins
dd = (SH.with_columns((pl.col("att") / pl.col("slots")).cut(edges, labels=lb, left_closed=True).alias("b"))
      .group_by("b").agg(pl.col("att").sum(), pl.col("ab").sum(), pl.col("so").sum()).with_columns((pl.col("ab") / pl.col("att")).alias("abr"),
                                                                                                  (pl.col("so") / pl.col("att")).alias("sor")).sort("b"))
print("[LOAD] abandonment by demand/slot bin:", ", ".join(f"{b} {p(r)} (so {p(s)})" for b, a_, ab_, so_, r, s in dd.iter_rows()))

# %% [markdown]
# ## G. Concentration and excess failures

# %%
smc = SHC.group_by("station_id", "month", "vehicle_class").agg(pl.col("att").sum(), pl.col("fail").sum())
base = smc.filter(~pl.col("month").dt.month().is_in(AJ)).group_by("vehicle_class").agg((pl.col("fail").sum() / pl.col("att").sum()).alias("b"))
print("[EXC] class baselines Jul-Mar:", dict(base.iter_rows()))
smc = smc.join(base, on="vehicle_class").with_columns((pl.col("fail") - pl.col("att") * pl.col("b")).alias("exc")).join(ST, on="station_id")
TOT_EXC = smc["exc"].sum()
print(f"[EXC] total net excess {TOT_EXC:,.0f} = {p(TOT_EXC / F, 1)} of failures")
seg = smc.with_columns(
    pl.when(~pl.col("month").dt.month().is_in(AJ)).then(pl.lit("JulMar"))
    .when(pl.col("city").is_in(HOTC)).then(pl.lit("AJ hot ") + pl.col("charger_generation"))
    .otherwise(pl.lit("AJ other")).alias("seg")).group_by("seg").agg(
    pl.col("att").sum(), pl.col("fail").sum(), pl.col("exc").sum(), pl.col("station_id").n_unique().alias("st")).with_columns(
    (pl.col("att") / N).alias("sh_att"), (pl.col("fail") / F).alias("sh_fail"), (pl.col("exc") / TOT_EXC).alias("sh_exc"),
    (pl.col("fail") / pl.col("att")).alias("fr")).sort("sh_exc", descending=True)
print(seg)
# alternative baseline: each city's own Jul-Mar rate (class-specific) -> robustness of the 80% share
base_c = smc.filter(~pl.col("month").dt.month().is_in(AJ)).group_by("city", "vehicle_class").agg((pl.col("fail").sum() / pl.col("att").sum()).alias("bc"))
smc2 = smc.join(base_c, on=["city", "vehicle_class"]).with_columns((pl.col("fail") - pl.col("att") * pl.col("bc")).alias("exc2"))
g1 = smc2.filter(pl.col("month").dt.month().is_in(AJ), pl.col("city").is_in(HOTC), pl.col("charger_generation") == "Gen1")
print(f"[EXC] robustness (city x class baseline): total excess {smc2['exc2'].sum():,.0f}; hot Gen1 AJ share {p(g1['exc2'].sum() / smc2['exc2'].sum(), 1)}")
# per-year
for yy in [2024, 2025]:
    sy = smc.filter(pl.col("month").dt.month().is_in(AJ), pl.col("month").dt.year() == yy)
    g1y = sy.filter(pl.col("city").is_in(HOTC), pl.col("charger_generation") == "Gen1")
    print(f"[EXC] Apr-Jun {yy}: excess {sy['exc'].sum():,.0f}; hot Gen1 share {p(g1y['exc'].sum() / sy['exc'].sum(), 1)}; "
          f"hot Gen1 share of that summer's attempts {p(g1y['att'].sum() / sy['att'].sum(), 1)}")

sc = smc.group_by("station_id").agg(pl.col("att").sum(), pl.col("fail").sum(), pl.col("exc").sum()).join(ST, on="station_id")
print(f"[CONC] stations {sc.height}; Gini att {gini(sc['att']):.3f}, fail {gini(sc['fail']):.3f}, pos excess {gini(sc['exc'].clip(lower_bound=0)):.3f}")
print(f"[CONC] top10% share: att {p(topshare(sc['att']), 1)}, fail {p(topshare(sc['fail']), 1)}, pos excess {p(topshare(sc['exc'].clip(lower_bound=0)), 1)}")
# Gini of station failure RATE (rate-based concentration) and station rate range
sc = sc.with_columns((pl.col("fail") / pl.col("att")).alias("fr"))
print(f"[CONC] Gini of station failure rate {gini(sc['fr']):.3f}; station overall rates {p(sc['fr'].min())}-{p(sc['fr'].max())}")
top15 = sc.sort("exc", descending=True).head(15)
print("[TOP15] cities", dict(top15.group_by("city").len().iter_rows()), "gen", dict(top15.group_by("charger_generation").len().iter_rows()),
      "wave", dict(top15.group_by("expansion_wave").len().iter_rows()), "loc", dict(top15.group_by("location_type").len().iter_rows()))
print(f"[TOP15] share of net excess {p(top15['exc'].sum() / TOT_EXC, 1)}, of attempts {p(top15['att'].sum() / N, 1)}")
t15 = top15["station_id"].to_list()
t15aj = (SHC.filter(pl.col("station_id").is_in(t15)).group_by("station_id", pl.when(pl.col("m").is_in(AJ)).then(pl.col("y").cast(pl.Utf8)).otherwise(pl.lit("JM")).alias("p"))
         .agg(pl.col("att").sum(), pl.col("fail").sum()).with_columns((pl.col("fail") / pl.col("att")).alias("fr")))
for pp in ["2024", "2025", "JM"]:
    print(f"[TOP15] median station rate {pp}: {p(t15aj.filter(pl.col('p') == pp)['fr'].median())}")
# swaps per slot-day: hotspot vs other (station_month)
smo = station_month().select("station_id", "active_days", "completed", "slots_2w", "slots_3w")
spd = smo.group_by("station_id").agg((pl.col("completed").sum() / (pl.col("active_days") * (pl.col("slots_2w") + pl.col("slots_3w"))).sum()).alias("spd"))
print(f"[TOP15] swaps per slot-day median: hotspots {spd.filter(pl.col('station_id').is_in(t15))['spd'].median():.2f} vs others "
      f"{spd.filter(~pl.col('station_id').is_in(t15))['spd'].median():.2f}")
# do other Gen1 hot-city stations look the same? (are the "hotspots" special, or just the hot-city Gen1 population?)
g1h = sc.filter(pl.col("city").is_in(HOTC), pl.col("charger_generation") == "Gen1").with_columns(pl.col("station_id").is_in(t15).alias("top"))
g1a = (SHC.filter(pl.col("m").is_in(AJ), pl.col("city").is_in(HOTC), pl.col("charger_generation") == "Gen1")
       .group_by("station_id").agg(pl.col("att").sum(), pl.col("fail").sum()).with_columns((pl.col("fail") / pl.col("att")).alias("fr"))
       .with_columns(pl.col("station_id").is_in(t15).alias("top")))
print("[TOP15] hot-city Gen1 Apr-Jun station failure rate: top15 median "
      f"{p(g1a.filter(pl.col('top'))['fr'].median())} vs other 21 median {p(g1a.filter(~pl.col('top'))['fr'].median())}; "
      f"att median top {g1a.filter(pl.col('top'))['att'].median():,.0f} vs other {g1a.filter(~pl.col('top'))['att'].median():,.0f}")
print(f"[TOP15] hot-city Gen1 Apr-Jun station rate range {p(g1a['fr'].min())}-{p(g1a['fr'].max())}")

# station-hour Gini vs null
rng = np.random.default_rng(12345)
SHn = SH.with_columns(pl.col("hs").dt.truncate("1mo").dt.date().alias("month")).join(
    smc.group_by("station_id", "month").agg((pl.col("fail").sum() / pl.col("att").sum()).alias("psm")), on=["station_id", "month"], how="left")
sa, sf, ps = SHn["att"].to_numpy(), SHn["fail"].to_numpy(), SHn["psm"].to_numpy()
nf, ns = rng.binomial(sa, F / N), rng.binomial(sa, ps)
print(f"[CONC-SH] station-hours {len(sa):,}; Gini obs {gini(sf):.3f}, null station-month {gini(ns):.3f}, null flat {gini(nf):.3f}; "
      f"top10% obs {p(topshare(sf), 1)}, null sm {p(topshare(ns), 1)}")
del nf, ns, sa, sf, ps, SHn

# %% [markdown]
# ## H. 3W vs 2W

# %%
s3 = SHC.group_by("station_id", "month", "vehicle_class").agg(pl.col("att").sum(), pl.col("fail").sum(), pl.col("so").sum(), pl.col("ab").sum()).to_pandas()
w3 = s3.pivot_table(index=["station_id", "month"], columns="vehicle_class", values=["att", "fail", "so", "ab"], fill_value=0)
w3.columns = [f"{a}_{b}" for a, b in w3.columns]
w3 = w3.reset_index()
for o in ["so", "ab", "fail"]:
    rr, lo, hi, k = mh(w3, f"{o}_3W", "att_3W", f"{o}_2W", "att_2W")
    print(f"[3W] {o}: MH RR 3W vs 2W {rr:.2f} ({lo:.2f}-{hi:.2f}) strata={k}")
for nm, s in [("Jul-Mar", jm), ("Apr-Jun", aj)]:
    r = s.group_by("vehicle_class").agg((pl.col("so").sum() / pl.col("att").sum()).alias("r"), pl.col("att").sum()).sort("vehicle_class")
    d = dict(zip(r["vehicle_class"], r["r"]))
    print(f"[3W] {nm} stockout 3W {p(d['3W'])} vs 2W {p(d['2W'])}, ratio {d['3W'] / d['2W']:.2f}")
# within station-month for Jul-Mar only
w3j = w3[~pd.to_datetime(w3["month"]).dt.month.isin(AJ)]
rr, lo, hi, k = mh(w3j, "so_3W", "att_3W", "so_2W", "att_2W")
print(f"[3W] Jul-Mar stockout MH RR within station-month {rr:.2f} ({lo:.2f}-{hi:.2f})")
z3 = SHC.filter(pl.col("vehicle_class") == "3W").with_columns((pl.col("slots_3w") == 0).alias("z")).group_by("z").agg(
    pl.col("station_id").n_unique().alias("st"), pl.col("att").sum(), pl.col("comp").sum(), (pl.col("so").sum() / pl.col("att").sum()).alias("sor"))
print("[3W] by zero-3W-slot flag:", z3.to_dicts(), "share att", (z3.filter(pl.col("z"))["att"].sum() / z3["att"].sum()))
# impact claim: closing the cool-season 3W stockout gap
j3 = jm.filter(pl.col("vehicle_class") == "3W")
gap = d_j = jm.group_by("vehicle_class").agg((pl.col("so").sum() / pl.col("att").sum()).alias("r"))
dj = dict(gap.iter_rows())
n3j = j3["att"].sum()
print(f"[3W] Jul-Mar 3W attempts {n3j:,} over {j3['month'].n_unique()} months; gap-closing stockouts {n3j * (dj['3W'] - dj['2W']):,.0f} "
      f"(per 9 months {n3j * (dj['3W'] - dj['2W']) / j3['month'].n_unique() * 9:,.0f})")
j3b = j3.filter(pl.col("month") >= pl.date(2024, 7, 1), pl.col("month") <= pl.date(2025, 3, 1))
s_j = jm.filter(pl.col("month") >= pl.date(2024, 7, 1), pl.col("month") <= pl.date(2025, 3, 1)).group_by("vehicle_class").agg(
    (pl.col("so").sum() / pl.col("att").sum()).alias("r"))
dj2 = dict(s_j.iter_rows())
print(f"[3W] Jul-24..Mar-25 only (9 months): 3W att {j3b['att'].sum():,}; gap stockouts {j3b['att'].sum() * (dj2['3W'] - dj2['2W']):,.0f}")

# %% [markdown]
# ## I. Year on year (fixed Launch panel, H1)

# %%
L = SHC.filter(pl.col("expansion_wave") == "Launch", pl.col("m") <= 6)
print(f"[YOY] launch stations {L['station_id'].n_unique()}")
ly = L.group_by("y").agg(pl.col("att").sum(), pl.col("fail").sum()).sort("y")
print(f"[YOY] attempts x{ly['att'][1] / ly['att'][0]:.3f}, failures x{ly['fail'][1] / ly['fail'][0]:.3f}")
yy = L.group_by("station_id", "m", "y").agg(pl.col("att").sum(), pl.col("fail").sum()).to_pandas()
wy = yy.pivot_table(index=["station_id", "m"], columns="y", values=["att", "fail"], fill_value=0)
wy.columns = [f"{a}_{b}" for a, b in wy.columns]
wy = wy.reset_index()
rr, lo, hi, k = mh(wy, "fail_2025", "att_2025", "fail_2024", "att_2024")
print(f"[YOY] same station-month RR 2025 vs 2024 {rr:.3f} ({lo:.3f}-{hi:.3f}) strata={k}")
rr, lo, hi, k = mh(wy[wy["m"].isin(AJ)], "fail_2025", "att_2025", "fail_2024", "att_2024")
print(f"[YOY] Apr-Jun RR {rr:.3f} ({lo:.3f}-{hi:.3f})")
rr, lo, hi, k = mh(wy[~wy["m"].isin(AJ)], "fail_2025", "att_2025", "fail_2024", "att_2024")
print(f"[YOY] Jan-Mar RR {rr:.3f} ({lo:.3f}-{hi:.3f})")
# class-mix adjusted: station x month x class strata
yyc = L.group_by("station_id", "m", "vehicle_class", "y").agg(pl.col("att").sum(), pl.col("fail").sum()).to_pandas()
wyc = yyc.pivot_table(index=["station_id", "m", "vehicle_class"], columns="y", values=["att", "fail"], fill_value=0)
wyc.columns = [f"{a}_{b}" for a, b in wyc.columns]
wyc = wyc.reset_index()
rr, lo, hi, k = mh(wyc, "fail_2025", "att_2025", "fail_2024", "att_2024")
print(f"[YOY] station x month x class RR {rr:.3f} ({lo:.3f}-{hi:.3f})")
mayL = L.filter(pl.col("m") == 5).group_by("y").agg((pl.col("fail").sum() / pl.col("att").sum()).alias("r")).sort("y")
print(f"[YOY] Launch May rate {p(mayL['r'][0], 1)} -> {p(mayL['r'][1], 1)}")
hg1 = SHC.filter(pl.col("city").is_in(HOTC), pl.col("charger_generation") == "Gen1", pl.col("m") == 5).group_by("y").agg(
    pl.col("att").sum(), pl.col("fail").sum(), pl.col("date").n_unique().alias("days"), pl.col("station_id").n_unique().alias("st")).sort("y")
for r in hg1.iter_rows(named=True):
    print(f"[YOY] hot Gen1 May {r['y']}: {r['att'] / (r['days'] * r['st']):.1f} attempts/station-day, rate {p(r['fail'] / r['att'], 1)}")
# busiest 15 by H1-24 attempts
b15 = L.filter(pl.col("y") == 2024).group_by("station_id").agg(pl.col("att").sum()).sort("att", descending=True).head(15)["station_id"].to_list()
rr, lo, hi, k = mh(wy[wy["station_id"].isin(b15)], "fail_2025", "att_2025", "fail_2024", "att_2024")
print(f"[YOY] busiest 15 RR {rr:.3f} ({lo:.3f}-{hi:.3f})")
m25 = SHC.filter(pl.col("month") == pl.date(2025, 5, 1))
tot25 = m25["att"].sum()
print(f"[YOY] May-25 network {p(m25['fail'].sum() / tot25, 1)}; " + ", ".join(
    f"{w_} {p(r, 1)} (share att {p(a_ / tot25, 1)})" for w_, a_, r in m25.group_by("expansion_wave").agg(pl.col("att").sum(), (pl.col("fail").sum() / pl.col("att").sum()).alias("r")).sort("expansion_wave").iter_rows()))

# %% [markdown]
# ## J. Telemetry mechanism

# %%
HR = hourly(["station_id", "hour_start", "telemetry_status", "charger_generation", "city", "charged_2w_min", "charged_3w_min",
             "ambient_temp_c", "cabinet_temp_c", "avg_charge_minutes", "slots_3w"]).collect()
cov = HR.group_by("telemetry_status").len()
print("[TEL] status:", {k: f"{v / HR.height:.4f}" for k, v in cov.iter_rows()})
# do null sensor values appear in 'ok' or 'partial' rows?
print("[TEL] null charged_2w_min by status:", HR.group_by("telemetry_status").agg(pl.col("charged_2w_min").is_null().mean()).to_dicts())
J = HR.join(SH.select("station_id", pl.col("hs").alias("hour_start"), "att", "so", "ab", "fail"), on=["station_id", "hour_start"], how="left").with_columns(
    [pl.col(x).fill_null(0) for x in ["att", "so", "ab", "fail"]])
miss_att = J.filter(pl.col("telemetry_status") == "missing")["att"].sum()
print(f"[TEL] attempts in missing-telemetry hours {miss_att:,} -> covered {p(1 - miss_att / N)}; event station-hours w/o telemetry row: "
      f"{SH.join(HR.select('station_id', pl.col('hour_start').alias('hs')), on=['station_id', 'hs'], how='anti').height}")
JT = J.filter(pl.col("telemetry_status") != "missing")
r = JT.select(pl.corr("ambient_temp_c", "cabinet_temp_c"), (pl.col("cabinet_temp_c") - pl.col("ambient_temp_c")).mean().alias("d"))
print(f"[TEL] r(amb, cab) {r.item(0, 0):.3f}; cabinet - ambient {r.item(0, 1):.2f}C")
for gen in ["Gen1", "Gen2", "Gen3"]:
    s = JT.filter(pl.col("charger_generation") == gen)
    out = []
    for lab, cnd in [("<40", pl.col("cabinet_temp_c") < 40), ("40-45", (pl.col("cabinet_temp_c") >= 40) & (pl.col("cabinet_temp_c") < 45)),
                     (">=45", pl.col("cabinet_temp_c") >= 45), (">=50", pl.col("cabinet_temp_c") >= 50)]:
        b = s.filter(cnd)
        out.append(f"{lab}: chg {b['avg_charge_minutes'].mean():.0f}m, zero2W {p((b['charged_2w_min'] == 0).mean())}, "
                   f"so {p(b['so'].sum() / max(b['att'].sum(), 1))} (h={b.height:,})")
    print(f"[MECH] {gen} | " + " | ".join(out))
# diurnal profile check (is telemetry on the same clock as events?)
dp = (JT.filter(pl.col("city").is_in(HOTC), pl.col("charger_generation") == "Gen1", pl.col("hour_start").dt.month() == 5)
      .group_by(pl.col("hour_start").dt.hour().alias("h")).agg(pl.col("ambient_temp_c").mean().alias("amb"), pl.col("cabinet_temp_c").mean().alias("cab"),
                                                               (pl.col("charged_2w_min") == 0).mean().alias("zero"), pl.col("att").sum(),
                                                               (pl.col("so").sum() / pl.col("att").sum()).alias("sor"),
                                                               (pl.col("fail").sum() / pl.col("att").sum()).alias("fr")).sort("h"))
print("[DIURNAL] hot-city Gen1 May: hour, ambient, cabinet, zero-stock share, attempts, stockout, fail")
for rr_ in dp.iter_rows():
    print("   ", rr_[0], f"{rr_[1]:.1f}", f"{rr_[2]:.1f}", p(rr_[3], 1), f"{rr_[4]:,}", p(rr_[5], 1), p(rr_[6], 1))
# lag analysis on a proper hourly time base (not row shifts)
JL = J.select("station_id", "hour_start", "att", "so", "charged_2w_min", "telemetry_status")
lag_out = []
for Lh in [-3, -2, -1, 0, 1, 2, 3, 4]:
    lagged = JL.select("station_id", (pl.col("hour_start") + pl.duration(hours=Lh)).alias("hour_start"),
                       pl.when(pl.col("telemetry_status") != "missing").then(pl.col("charged_2w_min")).alias("m_lag"))
    z = JL.join(lagged, on=["station_id", "hour_start"], how="left")
    z0, z1 = z.filter(pl.col("m_lag") == 0), z.filter(pl.col("m_lag") > 0)
    lag_out.append((Lh, z0["so"].sum() / z0["att"].sum(), z1["so"].sum() / z1["att"].sum(), z0["att"].sum()))
print("[LAG] stockout rate | stock==0 at t-L vs >0 (attempts in zero group):",
      ", ".join(f"L={l_}: {p(a_)} vs {p(b_)} (n={n_:,})" for l_, a_, b_, n_ in lag_out))
# lag effect controlled for hour of day: hot Gen1 Apr-Jun, compare zero-at-t vs zero-at-t-3 within hour-of-day
z = JL.join(JL.select("station_id", (pl.col("hour_start") + pl.duration(hours=3)).alias("hour_start"), pl.col("charged_2w_min").alias("m3")),
            on=["station_id", "hour_start"], how="left").with_columns(pl.col("hour_start").dt.hour().alias("h"))
zz = z.filter(pl.col("charged_2w_min").is_not_null(), pl.col("m3").is_not_null()).with_columns(
    pl.when((pl.col("charged_2w_min") == 0) & (pl.col("m3") == 0)).then(pl.lit("both0"))
    .when(pl.col("charged_2w_min") == 0).then(pl.lit("now0_only"))
    .when(pl.col("m3") == 0).then(pl.lit("lag0_only")).otherwise(pl.lit("neither")).alias("st")).group_by("st").agg(
    pl.col("att").sum(), pl.col("so").sum(), pl.len().alias("hours")).with_columns((pl.col("so") / pl.col("att")).alias("sor")).sort("st")
print("[LAG] joint states (now, t-3):", zz.to_dicts())
del JL, z, zz

# station-day peak cabinet and GLM (independent build)
import statsmodels.api as sm  # noqa: E402

sdE = SHC.group_by("station_id", "date").agg(pl.col("att").sum(), pl.col("fail").sum(), pl.col("so").sum())
sdT = JT.with_columns(pl.col("hour_start").dt.date().alias("date")).group_by("station_id", "date").agg(pl.col("cabinet_temp_c").max().alias("cmax"))
SD = sdE.join(sdT, on=["station_id", "date"], how="inner").join(ST.select("station_id", "city", "charger_generation"), on="station_id")
print(f"[SD] station-days {SD.height:,} of {sdE.height:,}")
for gen in ["Gen1", "Gen2", "Gen3"]:
    s = SD.filter(pl.col("charger_generation") == gen)
    lo_, hi_ = s.filter(pl.col("cmax") < 40), s.filter(pl.col("cmax") >= 45)
    print(f"[SD] {gen}: <40 stockout {p(lo_['so'].sum() / lo_['att'].sum())} ({lo_.height:,} d) vs >=45 {p(hi_['so'].sum() / hi_['att'].sum())} ({hi_.height:,} d)")
SDp = SD.with_columns(pl.col("cmax").cut([40, 45, 50, 55], labels=["a<40", "b40-45", "c45-50", "d50-55", "e>=55"], left_closed=True).alias("band"),
                      pl.col("date").dt.strftime("%Y-%m").alias("ym"), pl.col("date").dt.weekday().cast(pl.Utf8).alias("wd")).to_pandas()
for gen in ["Gen1", "Gen2", "Gen3"]:
    d = SDp[SDp["charger_generation"] == gen].copy()
    X = pd.get_dummies(d[["band", "city", "ym", "wd"]].astype(str), drop_first=True, dtype=float)
    X = sm.add_constant(X.loc[:, X.sum() > 0])
    yv = np.column_stack([d["fail"], d["att"] - d["fail"]])
    res = sm.GLM(yv, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["station_id"])[0]})
    col = "band_c45-50"
    if col in res.params:
        ci = res.conf_int().loc[col]
        print(f"[GLM] {gen} 45-50 vs <40: OR {np.exp(res.params[col]):.2f} ({np.exp(ci[0]):.2f}-{np.exp(ci[1]):.2f}); n={len(d):,}, band days {(d['band'] == 'c45-50').sum():,}")
    # same model but WITHOUT year-month FE replaced: station FE instead of city (robustness)
del SDp

# 3W zero charged packs
J3 = JT.filter(pl.col("slots_3w") > 0, pl.col("charged_3w_min").is_not_null())
SH3 = SHC.filter(pl.col("vehicle_class") == "3W").select("station_id", pl.col("hs").alias("hour_start"), pl.col("att").alias("a3"), pl.col("so").alias("s3"))
J3 = J3.join(SH3, on=["station_id", "hour_start"], how="left").with_columns(pl.col("a3").fill_null(0), pl.col("s3").fill_null(0))
z0, z1 = J3.filter(pl.col("charged_3w_min") == 0), J3.filter(pl.col("charged_3w_min") > 0)
print(f"[3W-TEL] zero 3W hours {p(z0.height / J3.height)}; 3W stockout {p(z0['s3'].sum() / z0['a3'].sum(), 1)} vs {p(z1['s3'].sum() / z1['a3'].sum(), 1)}")
del J3, SH3

# exposure by city
ex = JT.filter(pl.col("hour_start").dt.month().is_in(AJ)).group_by("city").agg((pl.col("cabinet_temp_c") >= 45).mean().alias("s")).sort("s", descending=True)
print("[MECH] Apr-Jun share of station-hours cabinet>=45:", ", ".join(f"{c} {p(s_, 1)}" for c, s_ in ex.iter_rows()))

# %% [markdown]
# ## K. Tickets

# %%
TK = tickets()
print("[TK] tickets total", TK.height, "in_window", TK.filter(pl.col("in_window")).height, "test", TK.filter(pl.col("is_test")).height,
      "with station", TK.filter(pl.col("station_id").is_not_null()).height)
tkw = TK.filter(pl.col("in_window"), ~pl.col("is_test"))
tst = tkw.filter(pl.col("station_id").is_not_null()).group_by("station_id").agg(
    (pl.col("category") == "no_battery_available").sum().alias("nb"), (pl.col("category") == "long_queue").sum().alias("lq"))
stt = SHC.group_by("station_id").agg(pl.col("att").sum(), pl.col("so").sum(), pl.col("ab").sum(), pl.col("fail").sum()).join(tst, on="station_id", how="left").fill_null(0)
print(f"[TK] station Spearman nb/1k vs stockout {stats.spearmanr(stt['nb'] / stt['att'], stt['so'] / stt['att'])[0]:.3f}; "
      f"lq vs abandon {stats.spearmanr(stt['lq'] / stt['att'], stt['ab'] / stt['att'])[0]:.3f}; "
      f"(nb+lq) vs fail {stats.spearmanr((stt['nb'] + stt['lq']) / stt['att'], stt['fail'] / stt['att'])[0]:.3f}; n={stt.height}")
tm = tkw.group_by("month").agg(pl.len().alias("tk"), (pl.col("category") == "no_battery_available").sum().alias("nb"))
mm = SHC.group_by("month").agg(pl.col("att").sum(), pl.col("so").sum(), pl.col("fail").sum()).join(tm, on="month", how="left").sort("month")
for mth in [pl.date(2024, 1, 1), pl.date(2024, 5, 1), pl.date(2025, 1, 1), pl.date(2025, 5, 1)]:
    r_ = mm.filter(pl.col("month") == mth).row(0, named=True)
    print(f"[TK] {r_['month']:%b-%y}: tickets/1k {1000 * r_['tk'] / r_['att']:.2f}, nb per 100 stockouts {100 * r_['nb'] / r_['so']:.1f}, fail {p(r_['fail'] / r_['att'], 2)}")
print(f"[TK] monthly Spearman nb/1k vs stockout rate {stats.spearmanr(mm['nb'] / mm['att'], mm['so'] / mm['att'])[0]:.3f} (n={mm.height})")
cat = tkw.filter(pl.col("month").dt.month() <= 3).group_by(pl.col("month").dt.year().alias("y"), "category").len()
att_q1 = SHC.filter(pl.col("m") <= 3).group_by("y").agg(pl.col("att").sum())
cat = cat.join(att_q1, on="y").with_columns((1000 * pl.col("len") / pl.col("att")).alias("per1k")).sort("category", "y")
print("[TK] Jan-Mar per 1k by category:", ", ".join(f"{c}-{y_}: {v:.2f}" for y_, c, n_, a_, v in cat.iter_rows()))

# %% [markdown]
# ## L. Recommendation impact arithmetic

# %%
g1aj = aj.filter(pl.col("city").is_in(HOTC), pl.col("charger_generation") == "Gen1")
g3aj = aj.filter(pl.col("city").is_in(HOTC), pl.col("charger_generation") == "Gen3")
r1 = g1aj["fail"].sum() / g1aj["att"].sum()
r3 = g3aj["fail"].sum() / g3aj["att"].sum()
print(f"[IMP] hot Gen1 AJ rate {p(r1)} (att {g1aj['att'].sum():,}); hot Gen3 AJ rate {p(r3)} (att {g3aj['att'].sum():,}, "
      f"years {sorted(g3aj['y'].unique().to_list())}, by year {g3aj.group_by('y').agg(pl.col('att').sum()).sort('y').to_dicts()})")
tot_avoid = g1aj["att"].sum() * (r1 - r3)
print(f"[IMP] avoidable over both summers {tot_avoid:,.0f} ({p(tot_avoid / F, 1)} of all failures); per summer avg {tot_avoid / 2:,.0f}")
for yy_ in [2024, 2025]:
    s = g1aj.filter(pl.col("y") == yy_)
    print(f"[IMP]   {yy_}: att {s['att'].sum():,}, rate {p(s['fail'].sum() / s['att'].sum())}, avoidable at Gen3 rate {s['att'].sum() * (s['fail'].sum() / s['att'].sum() - r3):,.0f}")
# MH-based counterfactual (Gen1 -> Gen3 within city-month, 2025): failures / RR
w25h = w[w["city"].isin(HOTC)]
rrh, loh, hih, _ = mh(w25h, "fail_Gen1", "att_Gen1", "fail_Gen3", "att_Gen3")
g125 = g1aj.filter(pl.col("y") == 2025)
print(f"[IMP] hot-city 2025 MH RR Gen1 vs Gen3 {rrh:.2f} ({loh:.2f}-{hih:.2f}); MH counterfactual saving 2025 summer = "
      f"{g125['fail'].sum() * (1 - 1 / rrh):,.0f}")
# hot Gen3 vs Gen1 load difference (attempts per station-day)
for gen in ["Gen1", "Gen3"]:
    s = aj25.filter(pl.col("city").is_in(HOTC), pl.col("charger_generation") == gen)
    sd_ = s.group_by("station_id", "date").agg(pl.col("att").sum())
    print(f"[IMP] hot {gen} Apr-Jun 2025 attempts per active station-day {sd_['att'].mean():.1f}")
t15a = aj.filter(pl.col("station_id").is_in(t15))
print(f"[IMP] top15 AJ att {t15a['att'].sum():,}, rate {p(t15a['fail'].sum() / t15a['att'].sum())}, avoidable per summer "
      f"{t15a['att'].sum() * (t15a['fail'].sum() / t15a['att'].sum() - r3) / 2:,.0f}")
aj_att = aj["att"].sum()
st36 = SHC.filter(pl.col("city").is_in(HOTC), pl.col("charger_generation") == "Gen1")
print(f"[FRAME] hot Gen1 AJ = {p(g1aj['att'].sum() / aj_att, 1)} of Apr-Jun attempts; the 36 stations hold {p(st36['att'].sum() / N, 1)} of ALL attempts "
      f"and {p(st36['fail'].sum() / F, 1)} of all failures (all months)")

# %% [markdown]
# ## M. Does the hourly telemetry stock level explain event-level stockouts? (tests the last link of the mechanism)

# %%
SH2 = SHC.filter(pl.col("vehicle_class") == "2W").select("station_id", pl.col("hs").alias("hour_start"), "att", "so", "ab", "city",
                                                          "charger_generation")
J2 = (HR.filter(pl.col("telemetry_status") != "missing").select("station_id", "hour_start", "charged_2w_min", "cabinet_temp_c")
      .join(SH2, on=["station_id", "hour_start"], how="inner")
      .with_columns(pl.col("hour_start").dt.date().alias("date"), pl.col("hour_start").dt.hour().alias("h"),
                    pl.col("hour_start").dt.truncate("1mo").alias("mo"), (pl.col("charged_2w_min") == 0).alias("z")))
z0, z5 = J2.filter(pl.col("z")), J2.filter(pl.col("charged_2w_min") >= 5)
print(f"[HOURLY] pooled 2W: zero-stock hours stockout {p(z0['so'].sum() / z0['att'].sum())}, abandon {p(z0['ab'].sum() / z0['att'].sum())} | "
      f">=5 packs {p(z5['so'].sum() / z5['att'].sum())}, {p(z5['ab'].sum() / z5['att'].sum())}")


def mh_flag(df, keys, flag, ev="so"):
    gg = df.group_by(keys + [flag]).agg(pl.col("att").sum(), pl.col(ev).sum().alias("e")).to_pandas()
    ww = gg.pivot_table(index=keys, columns=flag, values=["att", "e"], fill_value=0)
    ww.columns = [f"{a}_{b}" for a, b in ww.columns]
    return mh(ww.reset_index(), "e_True", "att_True", "e_False", "att_False")


for lab, keys in [("station-day", ["station_id", "date"]), ("station-day x evening", ["station_id", "date", "eve"]),
                  ("city x gen x month x hour", ["city", "charger_generation", "mo", "h"])]:
    rr, lo, hi, k = mh_flag(J2.with_columns(pl.col("h").is_in(EVE).alias("eve")), keys, "z")
    print(f"[HOURLY] 2W stockout, zero-stock hour vs stocked hour, within {lab}: MH RR {rr:.2f} ({lo:.2f}-{hi:.2f}) strata={k}")
    rr, lo, hi, k = mh_flag(J2.with_columns(pl.col("h").is_in(EVE).alias("eve")), keys, "z", "ab")
    print(f"[HOURLY] 2W abandonment, same comparison within {lab}: MH RR {rr:.2f} ({lo:.2f}-{hi:.2f})")
hd = J2.filter(pl.col("city").is_in(HOTC), pl.col("charger_generation") == "Gen1", pl.col("hour_start").dt.month().is_in(AJ))
hd = hd.join(hd.group_by("station_id", "date").agg(pl.col("cabinet_temp_c").max().alias("pk")), on=["station_id", "date"])
hot = hd.filter(pl.col("pk") >= 45)
prof = hot.group_by("h").agg((pl.col("so").sum() / pl.col("att").sum()).alias("so"), pl.col("z").mean().alias("zero")).sort("h")
print("[HOURLY] hot-city Gen1 Apr-Jun, peak cabinet >=45C days, 2W stockout | zero-stock share by hour: " +
      "; ".join(f"{h}h {100 * s_:.1f}|{100 * z_:.0f}" for h, s_, z_ in prof.iter_rows()))
for lab, cnd in [("0-18h", pl.col("h") <= 18), ("12-17h", pl.col("h").is_between(12, 17)), ("19-22h", pl.col("h").is_in(EVE))]:
    s = hot.filter(cnd)
    print(f"[HOURLY]   {lab}: stockout {p(s['so'].sum() / s['att'].sum())}, zero-stock share {p(s['z'].mean(), 1)}")
for zz in [True, False]:
    s = hot.filter(pl.col("z") == zz)
    print(f"[HOURLY]   zero-stock={zz}: stockout {p(s['so'].sum() / s['att'].sum())} (2W att {s['att'].sum():,})")
eg = (hot.filter(pl.col("h").is_in(EVE)).with_columns(pl.col("charged_2w_min").cut([0.5, 2.5, 5.5], labels=["0", "1-2", "3-5", "6+"]).alias("b"))
      .group_by("b").agg(pl.col("att").sum(), (pl.col("so").sum() / pl.col("att").sum()).alias("so")).sort("b"))
print("[HOURLY] same days, 19-22h, stockout by min charged 2W packs: " + ", ".join(f"{b}: {p(s_)} (att {a_:,})" for b, a_, s_ in eg.iter_rows()))
nt = (hd.filter(pl.col("h") <= 8).with_columns((pl.col("pk") >= 45).alias("hotday")).group_by("hotday")
      .agg((pl.col("so").sum() / pl.col("att").sum()).alias("so"), pl.col("charged_2w_min").mean().alias("ms"), pl.col("att").sum()).sort("hotday"))
print("[HOURLY] same stations 0-8h (cabinets full): " + ", ".join(f"hot day={a_}: stockout {p(b_)}, mean min stock {c_:.1f} (att {d_:,})" for a_, b_, c_, d_ in nt.iter_rows()))
TL = HR.filter(pl.col("telemetry_status") != "missing").select("station_id", "hour_start", "charged_2w_min")
for Lh in [0, 1, 2, 3, 4, 6]:
    lg = TL.select("station_id", (pl.col("hour_start") + pl.duration(hours=Lh)).alias("hour_start"), (pl.col("charged_2w_min") == 0).alias("zl"))
    zj = J2.join(lg, on=["station_id", "hour_start"], how="inner")
    rr, lo, hi, k = mh_flag(zj, ["city", "charger_generation", "mo", "h"], "zl")
    raw0, raw1 = zj.filter(pl.col("zl")), zj.filter(~pl.col("zl"))
    print(f"[HOURLY] zero stock {Lh}h earlier: pooled stockout {p(raw0['so'].sum() / raw0['att'].sum())} vs {p(raw1['so'].sum() / raw1['att'].sum())}; "
          f"within city x gen x month x hour MH RR {rr:.2f} ({lo:.2f}-{hi:.2f})")
del J2, hd, hot, TL, SH2

# %% [markdown]
# ## N. Extra checks / missed angles

# %%
# class-adjusted station heterogeneity in Jul-Mar
bcl = dict(base.iter_rows())
sj = jm.group_by("station_id", "vehicle_class").agg(pl.col("att").sum(), pl.col("fail").sum()).with_columns(
    (pl.col("att") * pl.col("vehicle_class").replace_strict(bcl, return_dtype=pl.Float64)).alias("exp"),
    (pl.col("att") * pl.col("vehicle_class").replace_strict(bcl, return_dtype=pl.Float64) * (1 - pl.col("vehicle_class").replace_strict(bcl, return_dtype=pl.Float64))).alias("var"))
ss = sj.group_by("station_id").agg(pl.col("fail").sum(), pl.col("exp").sum(), pl.col("var").sum(), pl.col("att").sum())
chi_c = (((ss["fail"] - ss["exp"]) ** 2) / ss["var"]).sum()
print(f"[N] Jul-Mar station heterogeneity AFTER class mix: chi2={chi_c:.1f} on {ss.height - 1} df, p={stats.chi2.sf(chi_c, ss.height - 1):.3g}; "
      f"class-standardised O/E range {(ss['fail'] / ss['exp']).min():.2f}-{(ss['fail'] / ss['exp']).max():.2f}")
# hot-city Gen1 Apr-Jun: are the top-15 special within city x month x class, or just big?
ref = (aj.filter(pl.col("city").is_in(HOTC), pl.col("charger_generation") == "Gen1").group_by("city", "month", "vehicle_class")
       .agg((pl.col("fail").sum() / pl.col("att").sum()).alias("r")))
hs_ = (aj.filter(pl.col("city").is_in(HOTC), pl.col("charger_generation") == "Gen1").group_by("station_id", "city", "month", "vehicle_class")
       .agg(pl.col("att").sum(), pl.col("fail").sum()).join(ref, on=["city", "month", "vehicle_class"])
       .with_columns((pl.col("att") * pl.col("r")).alias("e"), (pl.col("att") * pl.col("r") * (1 - pl.col("r"))).alias("v"))
       .group_by("station_id", "city").agg(pl.col("fail").sum(), pl.col("e").sum(), pl.col("v").sum(), pl.col("att").sum())
       .with_columns((pl.col("fail") / pl.col("e")).alias("oe"), pl.col("station_id").is_in(t15).alias("top")))
chi_h = (((hs_["fail"] - hs_["e"]) ** 2) / hs_["v"]).sum()
print(f"[N] 36 hot-city Gen1 stations, Apr-Jun, vs own city x month x class rate: chi2={chi_h:.1f} on {hs_.height - 3} df "
      f"(p={stats.chi2.sf(chi_h, hs_.height - 3):.3g}); O/E range {hs_['oe'].min():.2f}-{hs_['oe'].max():.2f}; "
      f"top-15 median O/E {hs_.filter(pl.col('top'))['oe'].median():.2f} vs other 21 {hs_.filter(~pl.col('top'))['oe'].median():.2f}; "
      f"top-15 median AJ attempts {hs_.filter(pl.col('top'))['att'].median():,.0f} vs {hs_.filter(~pl.col('top'))['att'].median():,.0f}")
print("[N] top-15 by city vs hot Gen1 stations by city:", dict(hs_.filter(pl.col("top")).group_by("city").len().iter_rows()),
      dict(hs_.group_by("city").len().iter_rows()))

# rider-level: what happens after a failure? (recovery) and first-attempt failure vs return
R = (events(["rider_id", "event_ts", "event_type", "signup_date"]).sort("rider_id", "event_ts")
     .with_columns(et.is_in(["failed_no_charged_battery", "abandoned_queue", "failed_system_error"]).alias("fl"),
                   (et == "swap_completed").alias("cp"))
     .with_columns(pl.when(pl.col("cp")).then(pl.col("event_ts")).alias("cts"))
     .with_columns(pl.col("cts").shift(-1).over("rider_id").alias("cts_n"), pl.int_range(pl.len()).over("rider_id").alias("k"))
     .with_columns(pl.col("cts_n").backward_fill().over("rider_id").alias("next_cts"))
     .with_columns(((pl.col("next_cts") - pl.col("event_ts")).dt.total_minutes()).alias("gap"))
     .select("rider_id", "event_ts", "fl", "cp", "gap", "k", "signup_date")
     .collect(engine="streaming"))
for lab, cnd in [("after a service failure", pl.col("fl")), ("after a completed swap", pl.col("cp"))]:
    s = R.filter(cnd, pl.col("event_ts") < pl.datetime(2025, 6, 1))
    print(f"[N] {lab}: next completed swap by same rider within 60 min {p((s['gap'] <= 60).mean(), 1)}, within 24 h {p((s['gap'] <= 1440).mean(), 1)}, "
          f"median gap {s['gap'].median() / 60:.1f} h, never again {p(s['gap'].is_null().mean(), 1)} (n={s.height:,})")
first = R.filter(pl.col("k") == 0, pl.col("event_ts") < pl.datetime(2025, 5, 1), pl.col("fl") | pl.col("cp"),
                 pl.col("signup_date") >= pl.date(2024, 1, 1))
for lab, cnd in [("first attempt failed", pl.col("fl")), ("first attempt completed", pl.col("cp"))]:
    s = first.filter(cnd)
    # for a completed first attempt, gap is to the NEXT completed swap (k>=1); comparable 'came back and swapped' within 30 days
    print(f"[N] riders whose {lab}: n={s.height:,}; completed a (further) swap within 30 days {p((s['gap'] <= 30 * 1440).mean(), 1)}; never {p(s['gap'].is_null().mean(), 1)}")
del R, first
print("DONE")
