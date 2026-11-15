# %% [markdown]
# # Q3 verification (independent re-derivation)
# Re-derives the q3 headline and key findings from data/clean via src/common.py loaders. Written from scratch; it
# does not import or copy analysis/q3_stations_geography.py. Each block prints "CHECK <claim>: analyst X | ours Y".

# %%
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import *  # noqa: E402,F403

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
T0 = time.time()
HOT = ["Delhi NCR", "Jaipur", "Hyderabad"]


def out(*a):
    print(f"[{time.time()-T0:6.1f}s]", *a, flush=True)


def chk(name, analyst, ours):
    print(f"CHECK {name}: analyst {analyst} | ours {ours}", flush=True)


ST = stations().filter(~pl.col("is_test")).to_pandas()
ST["slots"] = ST.slots_2w + ST.slots_3w
out("stations:", len(ST), ST.groupby(["expansion_wave", "charger_generation"]).size().to_dict())

# %% [markdown]
# ## A. Station-month aggregates rebuilt from raw events (not station_month.parquet)

# %%
EVM = (events(["station_id", "month", "is_completed", "is_stockout", "is_abandoned", "is_system_error", "is_service_failure"])
       .group_by("station_id", "month")
       .agg(pl.len().alias("att"), pl.col("is_completed").sum().alias("comp"), pl.col("is_stockout").sum().alias("so"),
            pl.col("is_service_failure").sum().alias("fail"),
            (pl.col("is_stockout") | pl.col("is_abandoned") | pl.col("is_system_error")).sum().alias("fail2"))
       .collect(engine="streaming").to_pandas())
assert (EVM.fail == EVM.fail2).all(), "is_service_failure != stockout|abandoned|system_error"
EVM["month"] = pd.to_datetime(EVM.month)
EVM = EVM.merge(ST[["station_id", "city", "zone", "expansion_wave", "charger_generation", "location_type", "slots"]], on="station_id")
SMF = station_month().to_pandas(); SMF["month"] = pd.to_datetime(SMF.month)
cross = EVM.merge(SMF[["station_id", "month", "attempts", "stockouts", "abandoned", "system_errors"]], on=["station_id", "month"])
out("events vs station_month cross-check: attempts identical", bool((cross.att == cross.attempts).all()),
    "| failures identical", bool((cross.fail == cross.stockouts + cross.abandoned + cross.system_errors).all()))


def per(d, a, b):
    return d[(d.month >= pd.Timestamp(a)) & (d.month <= pd.Timestamp(b))]


Q224, Q225, H124 = per(EVM, "2024-04-01", "2024-06-01"), per(EVM, "2025-04-01", "2025-06-01"), per(EVM, "2024-01-01", "2024-06-01")


def fr(d):
    return 100 * d.fail.sum() / d.att.sum()


g1hot = set(ST[(ST.charger_generation == "Gen1") & ST.city.isin(HOT)].station_id)
leg = set(ST[ST.expansion_wave == "Launch"].station_id)
g = Q225[Q225.station_id.isin(g1hot)]
chk("hot-city Gen1 stations", 36, len(g1hot))
chk("Q2-25 failed attempts (36 Gen1 / network)", "40,723 of 77,445", f"{int(g.fail.sum()):,} of {int(Q225.fail.sum()):,}")
chk("share of failures / share of attempts", "52.6% / 27.4%",
    f"{g.fail.sum()/Q225.fail.sum():.1%} / {g.att.sum()/Q225.att.sum():.1%}")
chk("network failure rate Q2-24 -> Q2-25", "9.52 -> 7.92", f"{fr(Q224):.2f} -> {fr(Q225):.2f}")
chk("legacy failure rate Q2-24 -> Q2-25", "9.52 -> 9.55", f"{fr(Q224[Q224.station_id.isin(leg)]):.2f} -> {fr(Q225[Q225.station_id.isin(leg)]):.2f}")
chk("new stations Q2-25", "5.08", f"{fr(Q225[~Q225.station_id.isin(leg)]):.2f}")
chk("36 hot-city Gen1 Q2-24 -> Q2-25", "15.48 -> 15.20",
    f"{fr(Q224[Q224.station_id.isin(g1hot)]):.2f} -> {fr(Q225[Q225.station_id.isin(g1hot)]):.2f}")

# hotspot = top quartile of H1-24 failure rate among Launch stations; busiest = top decile of H1-24 attempts per slot-day
SD_H1 = SMF[(SMF.month >= "2024-01-01") & (SMF.month <= "2024-06-01")].assign(sd=lambda d: d.active_days * (d.slots_2w + d.slots_3w)) \
    .groupby("station_id").sd.sum()
h1 = H124[H124.station_id.isin(leg)].groupby("station_id").agg(att=("att", "sum"), fail=("fail", "sum"))
h1["rate"] = h1.fail / h1.att
h1["util"] = h1.att / h1.index.map(SD_H1)
hotspot = set(h1[h1.rate >= h1.rate.quantile(0.75)].index)
busiest = set(h1[h1.util >= h1.util.quantile(0.90)].index)
chk("hotspot count / all Gen1 hot-city", "23 / all 23", f"{len(hotspot)} / {len(hotspot & g1hot)}")
chk("hotspots Q2-24 -> Q2-25", "16.70 -> 16.61",
    f"{fr(Q224[Q224.station_id.isin(hotspot)]):.2f} -> {fr(Q225[Q225.station_id.isin(hotspot)]):.2f}")
chk("busiest 10% legacy Q2-24 -> Q2-25", "9 stations, 10.09 -> 9.67",
    f"{len(busiest)} stations, {fr(Q224[Q224.station_id.isin(busiest)]):.2f} -> {fr(Q225[Q225.station_id.isin(busiest)]):.2f}")
off = EVM[~EVM.month.dt.month.isin([4, 5, 6])]
grp = np.select([off.station_id.isin(hotspot), off.station_id.isin(leg)], ["hotspot", "other_legacy"], "new")
chk("off-season (not Apr-Jun) failure rate by group", "3.85-3.93%",
    {k: round(fr(off[grp == k]), 2) for k in ["hotspot", "other_legacy", "new"]})
# composition / load: did legacy load change? (dilution vs relief)
q24l, q25l = Q224[Q224.station_id.isin(leg)], Q225[Q225.station_id.isin(leg)]
sd_q = SMF.assign(sd=lambda d: d.active_days * (d.slots_2w + d.slots_3w))
sd24 = sd_q[(sd_q.month >= "2024-04-01") & (sd_q.month <= "2024-06-01") & sd_q.station_id.isin(leg)].sd.sum()
sd25 = sd_q[(sd_q.month >= "2025-04-01") & (sd_q.month <= "2025-06-01") & sd_q.station_id.isin(leg)].sd.sum()
out(f"LEGACY LOAD: attempts Q2-24 {q24l.att.sum():,} -> Q2-25 {q25l.att.sum():,} ({q25l.att.sum()/q24l.att.sum()-1:+.1%}); "
    f"attempts/slot-day {q24l.att.sum()/sd24:.2f} -> {q25l.att.sum()/sd25:.2f}; network attempts {Q224.att.sum():,} -> {Q225.att.sum():,} "
    f"({Q225.att.sum()/Q224.att.sum()-1:+.1%}); new-station share of Q2-25 attempts {1-q25l.att.sum()/Q225.att.sum():.1%}")
out(f"LEGACY stockout rate Q2-24 {100*q24l.so.sum()/q24l.att.sum():.2f}% -> Q2-25 {100*q25l.so.sum()/q25l.att.sum():.2f}%")
# by month within season (May-24 vs May-25 etc.) for legacy and hotspots
for m24, m25 in [("2024-04-01", "2025-04-01"), ("2024-05-01", "2025-05-01"), ("2024-06-01", "2025-06-01")]:
    a = EVM[EVM.month == m24]; b = EVM[EVM.month == m25]
    out(f"  {m24[:7]} vs {m25[:7]}: network {fr(a):.2f}->{fr(b):.2f} | legacy {fr(a[a.station_id.isin(leg)]):.2f}->"
        f"{fr(b[b.station_id.isin(leg)]):.2f} | hotspots {fr(a[a.station_id.isin(hotspot)]):.2f}->{fr(b[b.station_id.isin(hotspot)]):.2f}")

# %% [markdown]
# ## B. Thermal derating (telemetry_status == 'ok' only; blanks never filled)

# %%
HR = (hourly(["station_id", "city", "charger_generation", "expansion_wave", "season", "month", "date", "hour", "telemetry_status",
              "cabinet_temp_c", "ambient_temp_c", "avg_charge_minutes", "packs_quarantined"])
      .filter(pl.col("telemetry_status") == "ok").collect())
n_all = hourly(["station_id"]).select(pl.len()).collect().item()
nulls_ok = HR.select(pl.col("cabinet_temp_c").is_null().sum(), pl.col("avg_charge_minutes").is_null().sum()).row(0)
chk("telemetry-ok station-hours", "1,415,325 (96.8%)", f"{HR.height:,} ({HR.height/n_all:.1%}); nulls in ok rows (cab, charge) {nulls_ok}")
hinge = {}
for gen in ["Gen1", "Gen2", "Gen3"]:
    d = HR.filter(pl.col("charger_generation") == gen)
    T = d["cabinet_temp_c"].to_numpy().astype(np.float64); y = d["avg_charge_minutes"].to_numpy().astype(np.float64)
    syy = ((y - y.mean()) ** 2).sum(); best = None
    for k in np.arange(25, 52.01, 0.25):
        x = np.maximum(T - k, 0); xm = x.mean(); sxx = ((x - xm) ** 2).sum()
        if sxx == 0:
            continue
        b = ((x - xm) * (y - y.mean())).sum() / sxx; a = y.mean() - b * xm
        sse = syy - b * b * sxx
        if best is None or sse < best[0]:
            best = (sse, k, a, b)
    sse, k, a, b = best
    hinge[gen] = dict(n=len(y), knot=k, base=a, slope=b, at50=a + b * max(0, 50 - k), r2=1 - sse / syy)
H_ = pd.DataFrame(hinge).T
out("hinge fit (0.25 C grid):\n" + H_.round(3).to_string())
chk("Gen1 hinge", "88.1 base, +9.46/C above 41.5, 168.5 at 50, R2 0.884",
    f"{H_.loc['Gen1','base']:.1f} base, +{H_.loc['Gen1','slope']:.2f}/C above {H_.loc['Gen1','knot']}, {H_.loc['Gen1','at50']:.1f} at 50, R2 {H_.loc['Gen1','r2']:.3f}")
chk("Gen2 / Gen3 at 50 C", "73.6 (0.96/C) / 43.7 (0.18/C)",
    f"{H_.loc['Gen2','at50']:.1f} ({H_.loc['Gen2','slope']:.2f}/C, knot {H_.loc['Gen2','knot']}) / {H_.loc['Gen3','at50']:.1f} ({H_.loc['Gen3','slope']:.2f}/C, knot {H_.loc['Gen3','knot']})")
chk("Gen1 capacity at 50 C", "52%", f"{H_.loc['Gen1','base']/H_.loc['Gen1','at50']:.1%}")
# empirical binned means at 49-51 C (does the hinge describe the tail?)
tail = (HR.filter((pl.col("cabinet_temp_c") >= 49) & (pl.col("cabinet_temp_c") < 51)).group_by("charger_generation")
        .agg(pl.len(), pl.col("avg_charge_minutes").mean()).sort("charger_generation"))
out("empirical mean charge minutes, cabinet 49-51 C:\n" + str(tail))
g1s = HR.filter((pl.col("charger_generation") == "Gen1") & (pl.col("season") == "Summer"))
hot_g1s = g1s.filter(pl.col("cabinet_temp_c") > 40)
chk("Gen1 summer hot share; charge min hot vs not", "19.8%; 131.1 vs 88.0",
    f"{hot_g1s.height/g1s.height:.1%}; {hot_g1s['avg_charge_minutes'].mean():.1f} vs {g1s.filter(pl.col('cabinet_temp_c') <= 40)['avg_charge_minutes'].mean():.1f}")
hh = HR.filter(pl.col("season") == "Summer").group_by("hour").agg((pl.col("cabinet_temp_c") > 40).mean().alias("s")).sort("hour")
wmax = HR.filter(pl.col("season") == "Winter").select((pl.col("cabinet_temp_c") > 40).mean()).item()
chk("summer hot share at 15:00 / 06:00 / winter", "50.0% / 0% / 0%",
    f"{hh.filter(pl.col('hour')==15)['s'][0]:.1%} / {hh.filter(pl.col('hour')==6)['s'][0]:.1%} / {wmax:.2%}; peak hour {hh.sort('s', descending=True)['hour'][0]}")
cs = HR.filter(pl.col("season") == "Summer").group_by("city").agg((pl.col("cabinet_temp_c") > 40).mean().alias("hot")).sort("hot", descending=True)
chk("summer hot share by city", "DEL 26.7 JAI 26.5 HYD 21.8 PUN 12.5 BLR 8.2 MUM 5.3",
    " ".join(f"{c[:3]} {100*v:.1f}" for c, v in cs.iter_rows()))
# is cabinet heat itself a generation attribute? same city, same hour, by generation
cg = (HR.filter(pl.col("season") == "Summer").group_by("city", "charger_generation")
      .agg((pl.col("cabinet_temp_c") > 40).mean().alias("hot"), (pl.col("cabinet_temp_c") - pl.col("ambient_temp_c")).mean().alias("cab_minus_amb"))
      .sort("city", "charger_generation"))
out("summer hot share and cabinet-minus-ambient by city x generation:\n" + str(cg))
qq = (HR.with_columns((pl.col("cabinet_temp_c") > 40).alias("hot")).group_by("hot")
      .agg(pl.len(), pl.col("packs_quarantined").sum().alias("pack_hours"), (pl.col("packs_quarantined") > 0).mean().alias("share_hours")))
chk("quarantined packs hot vs other hours", "1,315 vs 16", str(qq.sort("hot").to_dicts()))

# %% [markdown]
# ## C. Heat -> stockouts: Gen1 station-days vs same-city same-day Gen2/Gen3

# %%
SDY = (events(["station_id", "date", "is_stockout", "is_service_failure"]).group_by("station_id", "date")
       .agg(pl.len().alias("att"), pl.col("is_stockout").sum().alias("so"), pl.col("is_service_failure").sum().alias("fail"))
       .collect(engine="streaming"))
HD = HR.group_by("station_id", "date").agg(pl.len().alias("ok_h"), (pl.col("cabinet_temp_c") > 40).sum().alias("hot_h"))
gen_map = pl.from_pandas(ST[["station_id", "city", "charger_generation"]])
SDY = SDY.join(gen_map, on="station_id").join(HD, on=["station_id", "date"], how="left")
ctl = (SDY.filter(pl.col("charger_generation") != "Gen1").group_by("city", "date")
       .agg((pl.col("so").sum() / pl.col("att").sum()).alias("c_so"), (pl.col("fail").sum() / pl.col("att").sum()).alias("c_fail")))
G1 = (SDY.filter((pl.col("charger_generation") == "Gen1") & (pl.col("ok_h") >= 20)).join(ctl, on=["city", "date"])
      .with_columns(pl.when(pl.col("hot_h") == 0).then(pl.lit("0")).when(pl.col("hot_h") <= 3).then(pl.lit("1-3"))
                    .when(pl.col("hot_h") <= 7).then(pl.lit("4-7")).otherwise(pl.lit("8+")).alias("b"))).to_pandas()
bk = G1.groupby("b").apply(lambda d: pd.Series({"n": len(d), "g1_so": 100 * d.so.sum() / d.att.sum(),
                                                "ctl_so": 100 * (d.c_so * d.att).sum() / d.att.sum(),
                                                "g1_fail": 100 * d.fail.sum() / d.att.sum(),
                                                "ctl_fail": 100 * (d.c_fail * d.att).sum() / d.att.sum()}))
out("Gen1 station-days by hot hours:\n" + bk.round(2).to_string())
chk("8+ hot hours stockout Gen1 vs ctl; failures", "12.2 vs 4.8; 18.1 vs 7.5",
    f"{bk.loc['8+','g1_so']:.1f} vs {bk.loc['8+','ctl_so']:.1f}; {bk.loc['8+','g1_fail']:.1f} vs {bk.loc['8+','ctl_fail']:.1f}")
chk("0 hot hours stockout Gen1 vs ctl", "2.35 vs 2.32", f"{bk.loc['0','g1_so']:.2f} vs {bk.loc['0','ctl_so']:.2f}")
G1["ex"] = G1.so / G1.att - G1.c_so
G1["mon"] = pd.to_datetime(G1.date).dt.strftime("%Y-%m")
mh = smf.wls("ex ~ hot_h + C(station_id) + C(mon)", data=G1, weights=G1.att).fit(cov_type="cluster", cov_kwds={"groups": G1.station_id})
ci = mh.conf_int().loc["hot_h"] * 100
chk("within-station pp per hot hour", "+0.81 (0.78-0.85), n 25,170", f"{100*mh.params['hot_h']:.3f} ({ci[0]:.3f}-{ci[1]:.3f}), n {int(mh.nobs):,}")
# does the control group itself also have hot hours? (Gen2/3 heat in the same city-days with 8+ Gen1 hot hours)
c_hot = (SDY.filter(pl.col("charger_generation") != "Gen1").join(
    pl.from_pandas(G1[G1.b == "8+"][["city", "date"]].drop_duplicates()).with_columns(pl.col("date").cast(pl.Date)),
    on=["city", "date"]).select(pl.col("hot_h").mean()).item())
out(f"Gen2/3 stations on those same city-days average {c_hot:.1f} hot hours (ctl not heat-free)")

# %% [markdown]
# ## D. Station-month WLS: attributes vs failure rate, base vs heat x generation

# %%
hm = HR.group_by("station_id", "month").agg((pl.col("cabinet_temp_c") > 40).mean().alias("hot_share")).to_pandas()
hm["month"] = pd.to_datetime(hm.month)
P = SMF.merge(hm, on=["station_id", "month"], how="left")
P["slots"] = P.slots_2w + P.slots_3w
P["failures"] = P.stockouts + P.abandoned + P.system_errors
P = P[(P.active_days >= 15) & (P.attempts >= 100)].copy()
P["fail_pp"] = 100 * P.failures / P.attempts
P["log_util"] = np.log(P.attempts / (P.active_days * P.slots))
P["hot_pp"] = 100 * P.hot_share
P["mon"] = P.month.dt.strftime("%Y-%m")
P["gen"] = P.charger_generation; P["loc"] = P.location_type; P["host"] = P.host_type; P["conn"] = P.connectivity_tier; P["wave"] = P.expansion_wave
TK = tickets().filter(pl.col("in_window") & ~pl.col("is_test") & pl.col("station_id").is_not_null()).to_pandas()
TK["month"] = pd.to_datetime(TK.month)
tkm = TK.groupby(["station_id", "month"]).size().rename("tickets").reset_index()
P = P.merge(tkm, on=["station_id", "month"], how="left").fillna({"tickets": 0})
P["tk1k"] = 1000 * P.tickets / P.completed
out(f"panel n={len(P)}, missing hot_share rows {P.hot_pp.isna().sum()}; tickets in window at real stations {len(TK):,}")
rhs = ("C(gen, Treatment('Gen2')) + C(loc, Treatment('commercial_hub')) + C(host, Treatment('kirana')) + "
       "C(conn, Treatment('good')) + C(wave, Treatment('Launch')) + log_util + C(city) + C(mon)")
Pm = P.dropna(subset=["hot_pp"])
mb = smf.wls(f"fail_pp ~ {rhs}", data=Pm, weights=Pm.attempts).fit(cov_type="cluster", cov_kwds={"groups": Pm.station_id})
mt = smf.wls(f"fail_pp ~ {rhs} + hot_pp + C(gen, Treatment('Gen2')):hot_pp", data=Pm, weights=Pm.attempts).fit(
    cov_type="cluster", cov_kwds={"groups": Pm.station_id})


def co(m, t):
    k = [x for x in m.params.index if x == t]
    return m.params[k[0]] if k else np.nan


G1T = "C(gen, Treatment('Gen2'))[T.Gen1]"; W2T = "C(wave, Treatment('Launch'))[T.Wave2_2025H1]"
chk("base model n / Gen1 / Wave2 / Delhi / Jaipur", "2,003 / +1.35 / -2.52 / +2.64 / +3.05",
    f"{int(mb.nobs):,} / {co(mb, G1T):+.2f} (p={mb.pvalues[G1T]:.4f}) / {co(mb, W2T):+.2f} / {co(mb, 'C(city)[T.Delhi NCR]'):+.2f} / {co(mb, 'C(city)[T.Jaipur]'):+.2f}")
s2 = mt.params["hot_pp"]
s1 = s2 + mt.params[f"{G1T}:hot_pp"]; s3 = s2 + mt.params["C(gen, Treatment('Gen2'))[T.Gen3]:hot_pp"]
chk("R2 base -> thermal", "0.552 -> 0.945", f"{mb.rsquared:.3f} -> {mt.rsquared:.3f}")
chk("slopes Gen1/Gen2/Gen3 per pp hot", "0.359 / 0.154 / 0.069", f"{s1:.3f} / {s2:.3f} / {s3:.3f}")
chk("thermal Gen1 main / Wave2 / max |city|", "-0.25 / +0.20 n.s. / <=0.29",
    f"{co(mt, G1T):+.2f} / {co(mt, W2T):+.2f} (p={mt.pvalues[W2T]:.2f}) / "
    f"{max(abs(v) for k, v in mt.params.items() if k.startswith('C(city)')):.2f}")
others = {k: round(v, 2) for k, v in mb.params.items() if k.startswith(("C(loc", "C(host", "C(conn"))}
chk("base loc/host/conn |coef| <= 0.23 all n.s.", "yes", f"max |coef| {max(abs(v) for v in others.values()):.2f}; "
    f"min p {min(mb.pvalues[k] for k in others):.3f}")
# robustness: station fixed effects (within-station heat slope by generation)
mfe = smf.wls("fail_pp ~ hot_pp + C(gen, Treatment('Gen2')):hot_pp + log_util + C(station_id) + C(mon)", data=Pm, weights=Pm.attempts).fit(
    cov_type="cluster", cov_kwds={"groups": Pm.station_id})
out(f"ROBUSTNESS station-FE heat slopes: Gen2 {mfe.params['hot_pp']:.3f}, Gen1 {mfe.params['hot_pp'] + mfe.params[f'{G1T}:hot_pp']:.3f}, "
    f"Gen3 {mfe.params['hot_pp'] + mfe.params['C(gen, Treatment(\'Gen2\'))[T.Gen3]:hot_pp']:.3f}")
# raw month-only R2 for scale: how much of R2 is month FE alone?
m0 = smf.wls("fail_pp ~ C(mon)", data=Pm, weights=Pm.attempts).fit()
mh_only = smf.wls("fail_pp ~ C(mon) + hot_pp + C(gen, Treatment('Gen2')):hot_pp", data=Pm, weights=Pm.attempts).fit()
out(f"R2 month FE only {m0.rsquared:.3f}; month FE + heat x gen only {mh_only.rsquared:.3f}")
# queue wait and system errors by attribute
qwa = {}
for c in ["gen", "loc", "host", "conn", "wave", "city"]:
    gq = P.groupby(c).apply(lambda d: pd.Series({"qw": np.average(d.q_wait_completed_mean.fillna(0), weights=d.completed),
                                                 "se": 100 * d.system_errors.sum() / d.attempts.sum()}))
    qwa[c] = (gq.qw.min(), gq.qw.max(), gq.se.min(), gq.se.max())
chk("queue wait / system error ranges", "243.3-244.2 s / 0.28-0.31%",
    f"{min(v[0] for v in qwa.values()):.1f}-{max(v[1] for v in qwa.values()):.1f} s / {min(v[2] for v in qwa.values()):.2f}-{max(v[3] for v in qwa.values()):.2f}%")

# %% [markdown]
# ## E. Complaints vs failures

# %%
Pt = P.dropna(subset=["tk1k"])
mk = smf.wls("tk1k ~ fail_pp + C(city) + C(mon)", data=Pt, weights=Pt.completed).fit(cov_type="cluster", cov_kwds={"groups": Pt.station_id})
c_ = mk.conf_int().loc["fail_pp"]
chk("tickets per 1K swaps per +1 pp failure", "1.005 (0.961-1.049); mean 11.48",
    f"{mk.params['fail_pp']:.3f} ({c_[0]:.3f}-{c_[1]:.3f}); mean {1000*P.tickets.sum()/P.completed.sum():.2f}")
out(f"  => one extra ticket per {10/mk.params['fail_pp']:.1f} extra failed attempts (per 1K attempts), i.e. NOT one ticket per failure")
q25p = P[P.month >= "2025-04-01"]
tg = q25p.groupby("gen").apply(lambda d: 1000 * d.tickets.sum() / d.completed.sum())
chk("Q2-25 tickets per 1K by gen", "Gen1 21.38 / Gen2 13.21 / Gen3 13.49", " / ".join(f"{k} {v:.2f}" for k, v in tg.items()))
# per-attempt denominator (mechanical: failures shrink completed)
tga = q25p.groupby("gen").apply(lambda d: 1000 * d.tickets.sum() / d.attempts.sum())
out("  Q2-25 tickets per 1K ATTEMPTS by gen: " + " / ".join(f"{k} {v:.2f}" for k, v in tga.items()))
cat = TK.assign(gen=TK.station_id.map(ST.set_index("station_id").charger_generation)).query("month >= '2025-04-01'") \
    .groupby(["gen", "category"]).size().unstack(fill_value=0)
out("Q2-25 ticket categories by generation:\n" + cat.to_string())

# %% [markdown]
# ## F. Placement vs need (zones)

# %%
z = ST.assign(L=np.where(ST.expansion_wave == "Launch", ST.slots, 0), W1=np.where(ST.expansion_wave == "Wave1_2024H2", ST.slots, 0),
              W2=np.where(ST.expansion_wave == "Wave2_2025H1", ST.slots, 0),
              Lst=(ST.expansion_wave == "Launch").astype(int), W1st=(ST.expansion_wave == "Wave1_2024H2").astype(int),
              W2st=(ST.expansion_wave == "Wave2_2025H1").astype(int)).groupby(["city", "zone"])[["L", "W1", "W2", "Lst", "W1st", "W2st"]].sum().reset_index()
RD = riders().to_pandas()
rz = RD[RD.first_attempt_ts < pd.Timestamp("2024-07-01")].groupby("home_zone").size().rename("riders_h1")
z = z.merge(rz, left_on="zone", right_index=True, how="left").fillna({"riders_h1": 0})
zf = lambda d: d.groupby("zone").agg(att=("att", "sum"), fail=("fail", "sum"))  # noqa: E731
zh1 = zf(H124); zq2 = zf(Q224); zh2 = zf(per(EVM, "2024-07-01", "2024-12-01"))
zh2_leg = zf(per(EVM[EVM.station_id.isin(leg)], "2024-07-01", "2024-12-01"))
z["fr_h1"] = z.zone.map(zh1.fail / zh1.att); z["f_h1"] = z.zone.map(zh1.fail)
z["fr_q2"] = z.zone.map(zq2.fail / zq2.att); z["f_q2"] = z.zone.map(zq2.fail)
z["fr_h2"] = z.zone.map(zh2.fail / zh2.att); z["fr_h2_leg"] = z.zone.map(zh2_leg.fail / zh2_leg.att); z["f_h2"] = z.zone.map(zh2.fail)
zc = z.groupby("city")[["L", "W1", "W2", "riders_h1", "f_h1"]].sum()
sh = lambda c: zc.loc[HOT, c].sum() / zc[c].sum()  # noqa: E731
zc["new"] = zc.W1 + zc.W2
chk("hot cities: share H1-24 failures / riders / legacy slots / new slots", "67.8 / 49.0 / 48.8 / 50.4",
    f"{sh('f_h1'):.1%} / {sh('riders_h1'):.1%} / {sh('L'):.1%} / {zc.loc[HOT,'new'].sum()/zc.new.sum():.1%}")
zl = z[z.Lst > 0].copy()
out(f"zones with legacy stations: {len(zl)}; zones without: {z[z.Lst==0].zone.tolist()} riders {int(z[z.Lst==0].riders_h1.sum())}, "
    f"W1 st {int(z[z.Lst==0].W1st.sum())}, W2 st {int(z[z.Lst==0].W2st.sum())}")
zl["t1"] = pd.qcut(zl.fr_h1.rank(method="first"), 3, labels=["low", "mid", "high"])
hi = zl[zl.t1 == "high"]
chk("W1 high-need tercile: share failures / share W1 slots", "49.1 / 35.6",
    f"{hi.f_h1.sum()/zl.f_h1.sum():.1%} / {hi.W1.sum()/z.W1.sum():.1%} ({int(hi.W1.sum())} slots)")
r1 = stats.spearmanr(zl.fr_h1, zl.W1 / zl.L)
r2_h2 = stats.spearmanr(zl.fr_h2, zl.W2 / zl.L)
r2_h2leg = stats.spearmanr(zl.fr_h2_leg, zl.W2 / zl.L)
r2_q2 = stats.spearmanr(zl.fr_q2, zl.W2 / zl.L)
r2_h1 = stats.spearmanr(zl.fr_h1, zl.W2 / zl.L)
chk("Spearman W1 (H1-24 rate vs new/legacy slots)", "0.15 (p 0.46)", f"{r1.statistic:.2f} (p {r1.pvalue:.2f})")
chk("Spearman W2 (H2-24 rate, all stations in zone)", "-0.45 (p 0.03)", f"{r2_h2.statistic:.2f} (p {r2_h2.pvalue:.2f})")
out(f"  W2 robustness: legacy-only H2-24 rate rho {r2_h2leg.statistic:.2f} (p {r2_h2leg.pvalue:.2f}); same-season Apr-Jun-24 rate "
    f"rho {r2_q2.statistic:.2f} (p {r2_q2.pvalue:.2f}); H1-24 rate rho {r2_h1.statistic:.2f} (p {r2_h1.pvalue:.2f})")
out(f"  H2-24 zone failure-rate spread: mean {100*zl.fr_h2.mean():.2f}%, SD {100*zl.fr_h2.std():.2f} pp vs H1-24 SD {100*zl.fr_h1.std():.2f} pp; "
    f"Spearman(H1 rate, H2 rate) = {stats.spearmanr(zl.fr_h1, zl.fr_h2).statistic:.2f}")
rc1 = stats.spearmanr(zl.f_h1, zl.W1); rc2 = stats.spearmanr(zl.f_h2, zl.W2)
chk("Spearman failure counts vs new slots W1 / W2", "0.40 / 0.43", f"{rc1.statistic:.2f} / {rc2.statistic:.2f}")
# W2 share to hot-city / high-need zones (same-season need)
zl["tq2"] = pd.qcut(zl.fr_q2.rank(method="first"), 3, labels=["low", "mid", "high"])
hq = zl[zl.tq2 == "high"]
out(f"  W2 by same-season (Apr-Jun-24) need tercile: high tercile has {hq.f_q2.sum()/zl.f_q2.sum():.1%} of Q2-24 failures, "
    f"{hq.W2.sum()/z.W2.sum():.1%} of W2 slots; W2 hot-city slot share {zc.loc[HOT,'W2'].sum()/zc.W2.sum():.1%}; "
    f"W1 hot-city slot share {zc.loc[HOT,'W1'].sum()/zc.W1.sum():.1%}")
# Gen1 placement of new stations: did any new slots go to zones dominated by Gen1 hotspots?
zl["gen1_hot_slots"] = zl.zone.map(ST[ST.station_id.isin(hotspot)].groupby("zone").slots.sum()).fillna(0)
out(f"  zones containing hotspot stations: {int((zl.gen1_hot_slots>0).sum())}, they got {zl[zl.gen1_hot_slots>0].W1.sum()+zl[zl.gen1_hot_slots>0].W2.sum():.0f} "
    f"of {z.W1.sum()+z.W2.sum():.0f} new slots ({(zl[zl.gen1_hot_slots>0].W1.sum()+zl[zl.gen1_hot_slots>0].W2.sum())/(z.W1.sum()+z.W2.sum()):.1%}); "
    f"they hold {zl[zl.gen1_hot_slots>0].L.sum()/zl.L.sum():.1%} of legacy slots")

# %% [markdown]
# ## G. Wave economics Apr-Jun 2025 (station_month: fixed cost is the brief's pro-rated rent + maintenance)

# %%
E = SMF[(SMF.month >= "2025-04-01")].copy()
E["sd"] = E.active_days * (E.slots_2w + E.slots_3w)
E["failures"] = E.stockouts + E.abandoned + E.system_errors
wv = E.groupby("expansion_wave").agg(n=("station_id", "nunique"), comp=("completed", "sum"), att=("attempts", "sum"), fc=("fixed_cost", "sum"),
                                     sd=("sd", "sum"), fail=("failures", "sum"))
wv["util"] = wv.comp / wv.sd; wv["att_util"] = wv.att / wv.sd; wv["fcps"] = wv.fc / wv.comp; wv["fail_pct"] = 100 * wv.fail / wv.att
out("Q2-25 by wave:\n" + wv.round(2).to_string())
chk("swaps/slot-day & fixed cost/swap L / W1 / W2", "4.54/11.33, 3.74/14.73, 5.51/10.71",
    ", ".join(f"{wv.loc[w,'util']:.2f}/{wv.loc[w,'fcps']:.2f}" for w in ["Launch", "Wave1_2024H2", "Wave2_2025H1"]))
new_ = wv.drop("Launch")
chk("new stations share of fixed cost / swaps; network fcps", "40.0% / 37.7%; 11.76",
    f"{new_.fc.sum()/wv.fc.sum():.1%} / {new_.comp.sum()/wv.comp.sum():.1%}; {wv.fc.sum()/wv.comp.sum():.2f}")
exc = (wv.loc["Wave1_2024H2", "fcps"] - wv.loc["Launch", "fcps"]) * wv.loc["Wave1_2024H2", "comp"]
chk("W1 excess fixed cost", "5.05 lakh/qtr (1.68/month)", f"{exc/1e5:.2f} lakh/qtr ({exc/3e5:.2f}/month)")
lw = ST.groupby(["expansion_wave", "location_type"]).size().unstack(fill_value=0)
out("site types by wave:\n" + lw.to_string())
rent = ST.groupby("expansion_wave").monthly_rent_inr.mean()
chk("mean rent W1 vs Launch", "18,649 vs 17,510", f"{rent['Wave1_2024H2']:,.0f} vs {rent['Launch']:,.0f}")
lf = H124[H124.station_id.isin(leg)].groupby("location_type").fail.sum()
chk("legacy H1-24 failure share fuel pump / residential / commercial / market", "1.5 / 5.7 / 47.3 / 37.5",
    " / ".join(f"{100*lf.get(k,0)/lf.sum():.1f}" for k in ["highway_fuel_pump", "residential", "commercial_hub", "market"]))
lfs = ST[ST.expansion_wave == "Launch"].groupby("location_type").slots.sum()
out("  legacy slot share by site type: " + ", ".join(f"{k} {100*v/lfs.sum():.1f}%" for k, v in lfs.items()))
lr = H124[H124.station_id.isin(leg)].groupby("location_type").apply(lambda d: 100 * d.fail.sum() / d.att.sum())
out("  legacy H1-24 failure RATE by site type: " + ", ".join(f"{k} {v:.2f}%" for k, v in lr.items()))
lu = SMF[(SMF.month <= "2024-06-01") & SMF.station_id.isin(leg)].assign(sd=lambda d: d.active_days * (d.slots_2w + d.slots_3w)) \
    .groupby("location_type").apply(lambda d: d.attempts.sum() / d.sd.sum())
out("  legacy H1-24 attempts/slot-day by site type: " + ", ".join(f"{k} {v:.2f}" for k, v in lu.items()))
# base-model ticket coefficients (Gen1, hot cities) quoted by the analyst
mkb = smf.wls(f"tk1k ~ {rhs}", data=Pm, weights=Pm.completed).fit(cov_type="cluster", cov_kwds={"groups": Pm.station_id})
chk("tickets base model Gen1 / DEL / JAI / HYD vs BLR", "+1.26 (p<0.001) / +3.5..+5.2",
    f"{mkb.params[G1T]:+.2f} (p={mkb.pvalues[G1T]:.4f}) / {mkb.params['C(city)[T.Delhi NCR]']:+.2f} / {mkb.params['C(city)[T.Jaipur]']:+.2f} / "
    f"{mkb.params['C(city)[T.Hyderabad]']:+.2f}")
unserved = z[(z.Lst == 0) & (z.riders_h1 > 0)].zone.tolist()
Eu = E[E.station_id.isin(ST[ST.zone.isin(unserved)].station_id)]
En = E[E.expansion_wave != "Launch"]
chk("unserved-zone new stations util vs all new", "11.56 vs 4.56 (2.54x)",
    f"{Eu.completed.sum()/Eu.sd.sum():.2f} vs {En.completed.sum()/En.sd.sum():.2f} ({(Eu.completed.sum()/Eu.sd.sum())/(En.completed.sum()/En.sd.sum()):.2f}x), "
    f"n stations {Eu.station_id.nunique()}")
# ramp vs same-city legacy, by months since commissioning (completed AND attempts)
R = SMF[SMF.active_days >= 15].copy(); R["sd"] = R.active_days * (R.slots_2w + R.slots_3w)
legcm = R[R.expansion_wave == "Launch"].groupby(["city", "month"]).agg(lc=("completed", "sum"), la=("attempts", "sum"), lsd=("sd", "sum")).reset_index()
R = R.merge(legcm, on=["city", "month"])
R["age"] = (R.month.dt.year - pd.to_datetime(R.commissioned_date).dt.year) * 12 + R.month.dt.month - pd.to_datetime(R.commissioned_date).dt.month
rr = (R[R.expansion_wave != "Launch"].groupby(["expansion_wave", "age"])
      .apply(lambda d: pd.Series({"n": d.station_id.nunique(), "rel_comp": (d.completed.sum() / d.sd.sum()) / ((d.lc / d.lsd * d.sd).sum() / d.sd.sum()),
                                  "rel_att": (d.attempts.sum() / d.sd.sum()) / ((d.la / d.lsd * d.sd).sum() / d.sd.sum())})).reset_index())
out("ramp (relative to same-city same-month legacy; completed vs attempts):\n" + rr[rr.n >= 20].round(3).to_string(index=False))
w1r = rr[(rr.expansion_wave == "Wave1_2024H2") & (rr.n >= 20) & (rr.age.between(1, 8))]
w2r = rr[(rr.expansion_wave == "Wave2_2025H1") & (rr.n >= 20)]
chk("W1 rel util months 1-8 / W2 rel util", "82-85% / 116-131%",
    f"{100*w1r.rel_comp.min():.0f}-{100*w1r.rel_comp.max():.0f}% / {100*w2r.rel_comp.min():.0f}-{100*w2r.rel_comp.max():.0f}% "
    f"(attempts basis: W1 {100*w1r.rel_att.min():.0f}-{100*w1r.rel_att.max():.0f}%, W2 {100*w2r.rel_att.min():.0f}-{100*w2r.rel_att.max():.0f}%)")
# same-season, same-city comparison W2 vs legacy Gen2/Gen3-only (strip legacy Gen1 stockouts)
E2 = E.merge(ST[["station_id", "zone"]], on="station_id")
for nm, d in [("legacy all", E2[E2.expansion_wave == "Launch"]), ("legacy non-Gen1", E2[(E2.expansion_wave == "Launch") & (E2.charger_generation != "Gen1")]),
              ("W1", E2[E2.expansion_wave == "Wave1_2024H2"]), ("W2", E2[E2.expansion_wave == "Wave2_2025H1"])]:
    out(f"  Q2-25 {nm}: completed/slot-day {d.completed.sum()/d.sd.sum():.2f}, attempts/slot-day {d.attempts.sum()/d.sd.sum():.2f}, "
        f"failure {100*d.failures.sum()/d.attempts.sum():.2f}%, slots/station {(d.slots_2w+d.slots_3w).groupby(d.station_id).first().mean():.1f}")

# %% [markdown]
# ## H. DiD design A (same-season long difference for Launch stations)

# %%
import datetime as _dt  # noqa: E402
hq_ = HR.filter(pl.col("month").is_in([_dt.date(y, m, 1) for y in (2024, 2025) for m in (4, 5, 6)])) \
    .with_columns(pl.col("month").dt.year().alias("yr")).group_by("station_id", "yr").agg((pl.col("cabinet_temp_c") > 40).mean().alias("hot")).to_pandas()
hq_ = hq_.pivot(index="station_id", columns="yr", values="hot")
A = Q224[Q224.station_id.isin(leg)].groupby("station_id").agg(a0=("att", "sum"), f0=("fail", "sum")).join(
    Q225[Q225.station_id.isin(leg)].groupby("station_id").agg(a1=("att", "sum"), f1=("fail", "sum")))
A["d_fail"] = 100 * (A.f1 / A.a1 - A.f0 / A.a0)
A["d_hot"] = 100 * (hq_[2025] - hq_[2024]).reindex(A.index)
A = A.join(ST.set_index("station_id")[["zone", "city", "charger_generation"]])
A["intensity"] = A.zone.map(z.set_index("zone").eval("(W1 + W2) / L"))
mA = smf.ols("d_fail ~ intensity + d_hot * C(charger_generation) + C(city)", data=A.reset_index()).fit(
    cov_type="cluster", cov_kwds={"groups": A.reset_index().zone})
cA = mA.conf_int().loc["intensity"]
chk("DiD A per +1.0 intensity", "-0.26 (-0.46 to -0.05)", f"{mA.params['intensity']:.2f} ({cA[0]:.2f} to {cA[1]:.2f}), p={mA.pvalues['intensity']:.3f}, "
    f"median intensity {A.intensity.median():.2f}")
out(f"  heat change Q2-25 vs Q2-24 at legacy stations: mean d_hot {A.d_hot.mean():+.2f} pp; at hot-city Gen1 {A[A.index.isin(g1hot)].d_hot.mean():+.2f} pp")
mA0 = smf.ols("d_fail ~ intensity + C(city)", data=A.reset_index()).fit(cov_type="cluster", cov_kwds={"groups": A.reset_index().zone})
out(f"  DiD A without heat controls: {mA0.params['intensity']:.2f} (p={mA0.pvalues['intensity']:.3f})")

# %% [markdown]
# ## I. Competitor entry DiD (weekly, same city x wave controls)

# %%
WKD = (events(["station_id", "week", "is_completed", "rider_id"]).filter((pl.col("week") >= pl.date(2024, 10, 28)) & (pl.col("week") <= pl.date(2025, 6, 23)))
       .group_by("station_id", "week").agg(pl.col("is_completed").sum().alias("comp"), pl.col("rider_id").n_unique().alias("rid"))
       .collect(engine="streaming").to_pandas())
WKD["week"] = pd.to_datetime(WKD.week)
WKD = WKD[WKD.week != pd.Timestamp("2025-01-13")].merge(ST[["station_id", "city", "expansion_wave", "commissioned_date", "competitor_within_1_5km_since"]], on="station_id")
WKD["treat"] = WKD.competitor_within_1_5km_since.notna().astype(int)
WKD["stratum"] = WKD.city + "|" + WKD.expansion_wave
pre_open = pd.to_datetime(WKD.commissioned_date) < pd.Timestamp("2024-10-21")
tstrata = WKD[(WKD.treat == 1) & pre_open].stratum.unique()
CP = WKD[WKD.stratum.isin(tstrata) & pre_open].copy()
CP["post"] = (CP.week > pd.Timestamp("2025-01-13")).astype(int); CP["tp"] = CP.treat * CP.post
CP["y"] = np.log(CP.comp.clip(lower=1)); CP["yr"] = np.log(CP.rid.clip(lower=1))
CP["sw"] = CP.stratum + CP.week.dt.strftime("%Y%m%d")
mc = smf.ols("y ~ tp + C(station_id) + C(sw)", data=CP).fit(cov_type="cluster", cov_kwds={"groups": CP.station_id})
mr = smf.ols("yr ~ tp + C(station_id) + C(sw)", data=CP).fit(cov_type="cluster", cov_kwds={"groups": CP.station_id})
cc = mc.conf_int().loc["tp"]
chk("competitor sample", "8 treated / 21 controls / 34 weeks", f"{CP[CP.treat==1].station_id.nunique()} / {CP[CP.treat==0].station_id.nunique()} / {CP.week.nunique()}")
chk("competitor DiD swaps", "-3.9% (-10.7 to +3.3), p 0.28",
    f"{100*(np.exp(mc.params['tp'])-1):.1f}% ({100*(np.exp(cc[0])-1):.1f} to {100*(np.exp(cc[1])-1):+.1f}), p {mc.pvalues['tp']:.2f}")
chk("competitor DiD riders", "-1.3% (p 0.72)", f"{100*(np.exp(mr.params['tp'])-1):.1f}% (p {mr.pvalues['tp']:.2f})")
sl = CP.groupby(["station_id", "stratum", "treat", "post"]).y.mean().unstack("post"); sl["d"] = sl[1] - sl[0]; sl = sl.reset_index()


def did(lbl):
    num = den = 0.0
    for s, gg in sl.assign(l=lbl).groupby("stratum"):
        nt = gg.l.sum()
        if 0 < nt < len(gg):
            num += nt * (gg.d[gg.l == 1].mean() - gg.d[gg.l == 0].mean()); den += nt
    return num / den


rng = np.random.default_rng(7)
act = did(sl.treat.values)
perm = []
for _ in range(2000):
    l = sl.treat.values.copy()
    for s in sl.stratum.unique():
        ix = np.where(sl.stratum.values == s)[0]; l[ix] = rng.permutation(l[ix])
    perm.append(did(l))
chk("competitor permutation p", "0.22", f"{(np.abs(perm) >= abs(act) - 1e-12).mean():.2f} (did {100*(np.exp(act)-1):.1f}%)")
# trend-adjusted: allow station-specific linear trends (pre-trend sensitivity)
CP["t"] = (CP.week - CP.week.min()).dt.days / 7
mtr = smf.ols("y ~ tp + C(station_id) + C(station_id):t + C(sw)", data=CP).fit(cov_type="cluster", cov_kwds={"groups": CP.station_id})
out(f"  with station-specific linear trends: {100*(np.exp(mtr.params['tp'])-1):.1f}% (p {mtr.pvalues['tp']:.2f})")

# %% [markdown]
# ## J. Counterfactual sanity check: actual Q2-25 failure rates of Gen2/Gen3 in the same hot cities

# %%
Qh = Q225[Q225.city.isin(HOT)]
for gname, d in [("Gen1 (36)", Qh[Qh.charger_generation == "Gen1"]), ("Gen2 all", Qh[Qh.charger_generation == "Gen2"]),
                 ("Gen2 Launch", Qh[(Qh.charger_generation == "Gen2") & (Qh.expansion_wave == "Launch")]),
                 ("Gen3", Qh[Qh.charger_generation == "Gen3"])]:
    out(f"  hot cities Q2-25 {gname}: {fr(d):.2f}% failure, {d.station_id.nunique()} stations, attempts {d.att.sum():,}")
g1a = Qh[Qh.charger_generation == "Gen1"]
for alt, rate_alt in [("Gen2 Launch", fr(Qh[(Qh.charger_generation == "Gen2") & (Qh.expansion_wave == "Launch")])), ("Gen3", fr(Qh[Qh.charger_generation == "Gen3"]))]:
    av = (fr(g1a) - rate_alt) / 100 * g1a.att.sum()
    out(f"  naive same-city counterfactual Gen1 -> {alt}: -{fr(g1a)-rate_alt:.2f} pp, {av:,.0f} failed attempts avoided ({av/Q225.fail.sum():.1%} of network)")
chk("model counterfactual Gen2-like / Gen3-like", "-7.36 pp, 19,718 (25.5%) / -10.36 pp, 27,751 (35.8%)", "see naive same-city rates above")
out(f"TOTAL {time.time()-T0:.1f}s")

# %% [markdown]
# ## K. Extra checks: DiD B, rider-level competitor, small-n flags, load sensitivity, quoted descriptives

# %%
# DiD design B: Wave-1 staggered TWFE on Launch stations, Jan-2024..Jan-2025
w1open = ST[ST.expansion_wave == "Wave1_2024H2"].groupby("zone").commissioned_date.min()
B = SMF[SMF.station_id.isin(leg) & (SMF.month <= "2025-01-01") & (SMF.active_days >= 15)].merge(ST[["station_id", "zone"]], on="station_id")
B["w1"] = pd.to_datetime(B.zone.map(w1open))
B["tz"] = B.w1.notna().astype(int)
B = B[~((B.tz == 1) & (B.month.dt.to_period("M") == B.w1.dt.to_period("M")))]
B["post"] = ((B.tz == 1) & (B.month > B.w1)).astype(int)
B["fpp"] = 100 * (B.stockouts + B.abandoned + B.system_errors) / B.attempts
B["cm"] = B.city + B.month.dt.strftime("%Y%m"); B["gm"] = B.charger_generation + B.month.dt.strftime("%Y%m")
mB = smf.wls("fpp ~ post + C(station_id) + C(cm) + C(gm)", data=B, weights=B.attempts).fit(cov_type="cluster", cov_kwds={"groups": B.station_id})
cB = mB.conf_int().loc["post"]
chk("DiD B (W1 TWFE) failure pp", "+0.145 (-0.13 to +0.42); 74 vs 16", f"{mB.params['post']:+.3f} ({cB[0]:.2f} to {cB[1]:+.2f}); "
    f"{B[B.tz==1].station_id.nunique()} vs {B[B.tz==0].station_id.nunique()} stations, control zones {B[B.tz==0].zone.nunique()}")

# load sensitivity: within-station, does utilisation move failure rates once heat x gen is held fixed?
out(f"ROBUSTNESS station-FE log_util coef (heat x gen + month FE): {mfe.params['log_util']:+.3f} pp per log-unit (p={mfe.pvalues['log_util']:.2f})")
# excess (above off-season baseline) failures concentrated at the 36 stations
base = fr(off) / 100
ex_net = Q225.fail.sum() - base * Q225.att.sum(); ex_g1 = g.fail.sum() - base * g.att.sum()
out(f"EXCESS failures Q2-25 above the {100*base:.2f}% off-season baseline: network {ex_net:,.0f}; 36 hot-city Gen1 {ex_g1:,.0f} ({ex_g1/ex_net:.1%})")
# unserved-zone stations individually
uu = Eu.groupby("station_id").apply(lambda d: pd.Series({"util": d.completed.sum() / d.sd.sum(), "slots": (d.slots_2w + d.slots_3w).iloc[0],
                                                         "fcps": d.fixed_cost.sum() / d.completed.sum()}))
out("unserved-zone new stations (Q2-25):\n" + uu.round(2).join(ST.set_index("station_id")[["zone", "expansion_wave", "location_type"]]).to_string())
# Bengaluru Wave-2 next to competitor
bw = E[(E.city == "Bengaluru") & (E.expansion_wave == "Wave2_2025H1")].copy()
bw["comp_near"] = bw.station_id.isin(ST[ST.competitor_within_1_5km_since.notna()].station_id)
bwr = bw.groupby("comp_near").apply(lambda d: pd.Series({"n": d.station_id.nunique(), "util": d.completed.sum() / d.sd.sum()}))
chk("BLR W2 competitor-adjacent vs other", "3.84 (n=2) vs 5.89 (n=4)", bwr.round(2).to_dict())
# raw city failure ranking Q2-25 and >Rs20 fixed cost stations
cr = Q225.groupby("city").apply(fr).sort_values(ascending=False)
chk("Q2-25 raw city failure rates", "JAI 12.10 DEL 11.47 HYD 9.59 ... MUM 4.02", " ".join(f"{k[:3]} {v:.2f}" for k, v in cr.items()))
st_fc = E.groupby(["station_id", "expansion_wave"]).apply(lambda d: d.fixed_cost.sum() / d.completed.sum()).rename("fcps").reset_index()
chk("stations > Rs20 fixed cost/swap new vs legacy", "16 vs 10",
    f"{int(((st_fc.fcps > 20) & (st_fc.expansion_wave != 'Launch')).sum())} vs {int(((st_fc.fcps > 20) & (st_fc.expansion_wave == 'Launch')).sum())}")
cmw = E.groupby("expansion_wave").apply(lambda d: pd.Series({"cm1": d.cm1.sum() / d.completed.sum(), "cm2": d.cm2.sum() / d.completed.sum()}))
chk("CM1/CM2 per swap by wave (L/W1/W2)", "-5.61/-16.94, -3.52/-18.25, -7.01/-17.72",
    ", ".join(f"{cmw.loc[w,'cm1']:.2f}/{cmw.loc[w,'cm2']:.2f}" for w in ["Launch", "Wave1_2024H2", "Wave2_2025H1"]))

# rider-level competitor check (riders whose most-used pre-entry station is in the DiD sample)
RWk = (events(["rider_id", "station_id", "week", "is_completed", "city"])
       .filter(pl.col("is_completed") & pl.col("city").is_in(["Bengaluru", "Pune"]) & (pl.col("week") >= pl.date(2024, 10, 28))
               & (pl.col("week") <= pl.date(2025, 6, 23)) & (pl.col("week") != pl.date(2025, 1, 13)))
       .with_columns((pl.col("week") > pl.date(2025, 1, 13)).alias("post")).group_by("rider_id", "station_id", "post").agg(pl.len().alias("n"))
       .collect(engine="streaming").to_pandas())
pre_ = RWk[~RWk.post].sort_values("n", ascending=False).drop_duplicates("rider_id").set_index("rider_id").station_id
hm_ = pd.DataFrame({"home": pre_}).join(CP.drop_duplicates("station_id").set_index("station_id")[["treat", "stratum"]], on="home").dropna()
tot_ = RWk.groupby(["rider_id", "post"]).n.sum().unstack(fill_value=0)
hm_ = hm_.join(tot_)
npre, npost = CP[CP.post == 0].week.nunique(), CP[CP.post == 1].week.nunique()
hm_["d"] = hm_[True] / npost - hm_[False] / npre; hm_["act"] = (hm_[True] > 0).astype(int)
mrd = smf.ols("d ~ treat + C(stratum)", data=hm_).fit(cov_type="cluster", cov_kwds={"groups": hm_.home})
mra = smf.ols("act ~ treat + C(stratum)", data=hm_).fit(cov_type="cluster", cov_kwds={"groups": hm_.home})
chk("rider-level: riders treated/control; d swaps/wk; still active", "665/1,921; -0.023 (-0.152..0.106); +0.5 pp (p 0.67)",
    f"{int(hm_.treat.sum())}/{int((1-hm_.treat).sum())}; {mrd.params['treat']:+.3f} ({mrd.conf_int().loc['treat',0]:.3f}..{mrd.conf_int().loc['treat',1]:.3f}); "
    f"{100*mra.params['treat']:+.1f} pp (p {mra.pvalues['treat']:.2f})")
out(f"TOTAL {time.time()-T0:.1f}s")

# %% [markdown]
# ## L. Site type vs location: what drives new-station utilisation?

# %%
NS = E[E.expansion_wave != "Launch"].groupby(["station_id", "expansion_wave", "location_type", "city"]).agg(
    comp=("completed", "sum"), att=("attempts", "sum"), sd=("sd", "sum")).reset_index().merge(ST[["station_id", "zone"]], on="station_id")
NS["util"] = NS.comp / NS.sd
zz = z.set_index("zone")
NS["riders_per_slot_all"] = NS.zone.map(zz.riders_h1 / (zz.L + zz.W1 + zz.W2))
NS["unserved"] = NS.zone.isin(unserved).astype(int)
tab = NS.groupby(["expansion_wave", "location_type", "unserved"]).apply(lambda d: pd.Series({"n": len(d), "util": d.comp.sum() / d.sd.sum()}))
out("new stations Q2-25 utilisation by wave x site type x unserved zone:\n" + tab.round(2).to_string())
fp = NS[(NS.location_type == "highway_fuel_pump")]
out(f"fuel-pump new stations: unserved zones {fp[fp.unserved==1].comp.sum()/fp[fp.unserved==1].sd.sum():.2f} (n={int(fp.unserved.sum())}) vs "
    f"served zones {fp[fp.unserved==0].comp.sum()/fp[fp.unserved==0].sd.sum():.2f} (n={int((1-fp.unserved).sum())})")
ms = smf.wls("util ~ np.log(riders_per_slot_all) + C(location_type, Treatment('market')) + C(expansion_wave) + C(city)", data=NS, weights=NS.sd).fit(cov_type="HC1")
out("WLS new-station util ~ log(resident riders per zone slot) + site type + wave + city:\n" +
    pd.DataFrame({"coef": ms.params, "p": ms.pvalues}).round(3).loc[lambda d: ~d.index.str.startswith("C(city")].to_string() + f"\n R2 {ms.rsquared:.3f}, n {int(ms.nobs)}")
ms2 = smf.wls("util ~ np.log(riders_per_slot_all) + C(location_type, Treatment('market')) + C(expansion_wave) + C(city)", data=NS[NS.unserved == 0],
              weights=NS[NS.unserved == 0].sd).fit(cov_type="HC1")
out("  same, excluding the 3 unserved-zone stations:\n" +
    pd.DataFrame({"coef": ms2.params, "p": ms2.pvalues}).round(3).loc[lambda d: ~d.index.str.startswith("C(city")].to_string())
w2x = E[(E.expansion_wave == "Wave2_2025H1") & ~E.station_id.isin(Eu.station_id)]
out(f"W2 excluding its 2 unserved-zone fuel pumps: {w2x.completed.sum()/w2x.sd.sum():.2f} swaps/slot-day, Rs {w2x.fixed_cost.sum()/w2x.completed.sum():.2f}/swap "
    f"(n={w2x.station_id.nunique()}) vs legacy 4.54 / 11.33")
w1x = E[(E.expansion_wave == "Wave1_2024H2") & ~E.station_id.isin(Eu.station_id)]
out(f"W1 excluding PUN-106: {w1x.completed.sum()/w1x.sd.sum():.2f} swaps/slot-day, Rs {w1x.fixed_cost.sum()/w1x.completed.sum():.2f}/swap")

# %% [markdown]
# ## M. Competitor DiD confound: own Wave-2 openings in the treated zones during the post period

# %%
w2s = ST[ST.expansion_wave == "Wave2_2025H1"][["station_id", "zone", "slots", "commissioned_date"]]
out("W2 stations in zones of the competitor DiD sample:\n" + w2s[w2s.zone.isin(ST[ST.station_id.isin(CP.station_id)].zone)].sort_values(["zone", "commissioned_date"]).to_string(index=False))
CPz = CP.merge(ST[["station_id", "zone"]], on="station_id")
def w2_open(row_zone, wk):
    d = w2s[(w2s.zone == row_zone) & (pd.to_datetime(w2s.commissioned_date) <= wk)]
    return d.slots.sum()
key = CPz[["zone", "week"]].drop_duplicates()
key["w2_slots_zone"] = [w2_open(a, b) for a, b in zip(key.zone, key.week)]
CPz = CPz.merge(key, on=["zone", "week"])
legsl = ST[ST.expansion_wave != "Wave2_2025H1"].groupby("zone").slots.sum()
CPz["w2_int"] = CPz.w2_slots_zone / CPz.zone.map(legsl)
pz = CPz[CPz.post == 1].groupby("treat").w2_int.mean()
out(f"mean own-W2 slot intensity in station's zone, post period: treated {pz.get(1):.3f} vs control {pz.get(0):.3f}")
mc2 = smf.ols("y ~ tp + w2_int + C(station_id) + C(sw)", data=CPz).fit(cov_type="cluster", cov_kwds={"groups": CPz.station_id})
c2 = mc2.conf_int().loc["tp"]
out(f"competitor DiD controlling for own W2 intensity in zone: {100*(np.exp(mc2.params['tp'])-1):.1f}% ({100*(np.exp(c2[0])-1):.1f} to {100*(np.exp(c2[1])-1):+.1f}), "
    f"p {mc2.pvalues['tp']:.2f}; w2_int coef {mc2.params['w2_int']:+.3f} (p {mc2.pvalues['w2_int']:.2f})")
# Wave-2 market-only economics (title of fig 4 says 'Wave-2 market sites Rs 10.7')
w2m = E[(E.expansion_wave == "Wave2_2025H1") & E.station_id.isin(ST[ST.location_type == "market"].station_id)]
out(f"W2 market sites only: {w2m.completed.sum()/w2m.sd.sum():.2f} swaps/slot-day, Rs {w2m.fixed_cost.sum()/w2m.completed.sum():.2f}/swap, n={w2m.station_id.nunique()}")
w2fp = E[(E.expansion_wave == "Wave2_2025H1") & E.station_id.isin(ST[ST.location_type == "highway_fuel_pump"].station_id)]
out(f"W2 fuel-pump sites: {w2fp.completed.sum()/w2fp.sd.sum():.2f} swaps/slot-day, Rs {w2fp.fixed_cost.sum()/w2fp.completed.sum():.2f}/swap, n={w2fp.station_id.nunique()}")
w1fr = E[(E.expansion_wave == "Wave1_2024H2") & E.station_id.isin(ST[ST.location_type.isin(["highway_fuel_pump", "residential"])].station_id)]
out(f"W1 fuel-pump+residential: {w1fr.completed.sum()/w1fr.sd.sum():.2f} swaps/slot-day, Rs {w1fr.fixed_cost.sum()/w1fr.completed.sum():.2f}/swap, n={w1fr.station_id.nunique()}")
out(f"TOTAL {time.time()-T0:.1f}s")

# %% [markdown]
# ## N. Is the hotspot failure RATE load-sensitive? (if not, siting new stations could not fix it)

# %%
for nm, ids in [("hotspots (23)", hotspot), ("36 hot-city Gen1", g1hot), ("legacy (90)", leg)]:
    a0, a1 = Q224[Q224.station_id.isin(ids)], Q225[Q225.station_id.isin(ids)]
    s0 = sd_q[(sd_q.month >= "2024-04-01") & (sd_q.month <= "2024-06-01") & sd_q.station_id.isin(ids)].sd.sum()
    s1 = sd_q[(sd_q.month >= "2025-04-01") & (sd_q.month <= "2025-06-01") & sd_q.station_id.isin(ids)].sd.sum()
    out(f"{nm}: attempts {a0.att.sum():,} -> {a1.att.sum():,} ({a1.att.sum()/a0.att.sum()-1:+.1%}); attempts/slot-day {a0.att.sum()/s0:.2f} -> "
        f"{a1.att.sum()/s1:.2f}; failure {fr(a0):.2f}% -> {fr(a1):.2f}%")
g1p = Pm[Pm.gen == "Gen1"]
mg1 = smf.wls("fail_pp ~ hot_pp + log_util + C(station_id) + C(mon)", data=g1p, weights=g1p.attempts).fit(cov_type="cluster", cov_kwds={"groups": g1p.station_id})
out(f"Gen1 station-months, station + month FE: log_util {mg1.params['log_util']:+.3f} pp (p={mg1.pvalues['log_util']:.2f}); hot_pp {mg1.params['hot_pp']:+.3f}")
out(f"TOTAL {time.time()-T0:.1f}s")
for nm, ids in [("network", set(EVM.station_id)), ("legacy", leg), ("hotspots (23)", hotspot), ("36 hot-city Gen1", g1hot)]:
    f0, f1 = int(Q224[Q224.station_id.isin(ids)].fail.sum()), int(Q225[Q225.station_id.isin(ids)].fail.sum())
    out(f"ABSOLUTE failed attempts Apr-Jun 2024 -> 2025, {nm}: {f0:,} -> {f1:,} ({f1/f0-1:+.1%})")
