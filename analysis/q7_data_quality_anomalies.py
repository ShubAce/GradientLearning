# %% [markdown]
# # Q7 · Data-quality audit, cleaning impact and anomalies
#
# **Why this matters.** Cleaning & analytical rigour is 20% of the score, and every other question's numbers rest on
# the decisions taken in `src/clean.py`. This notebook
# 1. independently re-tests each documented cleaning decision and sizes how far each issue would move headline KPIs
#    if it were left untreated (before vs after),
# 2. hunts for **undocumented** issues (physically impossible battery movements, retired packs still issued, tariff
#    arithmetic, slot metadata, ...),
# 3. flags anomalous stations (funnel plot, controlled for city x month x vehicle class), battery lots and time
#    windows, plus firmware/connectivity irregularities.
#
# Deliverable: `outputs/tables/q7_data_quality_scorecard.csv` (issue | detection | rows | % | decision | KPI impact).
# Conventions: *raw* = `data/parquet` as delivered, *clean* = `data/clean` after `src/clean.py`.
# Every number quoted in `outputs/findings/q7.md` is printed by this script.

# %%
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:                       # inside a notebook
    ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
from common import *  # noqa: E402,F401,F403  (paths, loaders, constants, palette, set_style, save_fig)

set_style()
T0 = time.time()
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(70)
RAW_EV = PARQ / "swap_events.parquet"
LOG = {s["step"]: s for s in json.loads((TAB / "data_quality_log.json").read_text(encoding="utf-8"))["steps"]}
N_RAW = int(LOG["exclude_test_stations"]["raw_events"])
N_EV = events(["event_id"]).select(pl.len()).collect().item()
SC: list[dict] = []                     # scorecard rows


def add_issue(issue, kind, method, rows, denom, unit, decision, impact, short, severity):
    SC.append(dict(issue=issue, kind=kind, detection_method=method, rows_affected=int(rows), unit=unit,
                   denominator=int(denom), pct_of_data=round(100 * rows / denom, 4) if denom else np.nan,
                   decision=decision, impact_on_kpis=impact, impact_short=short, severity=severity))


def hdr(t):
    print(f"\n{'=' * 110}\n{t}   [{time.time() - T0:5.1f}s]\n{'=' * 110}", flush=True)


def as_i32(col, start):                 # 'RD-1000123' -> 1000123 etc. (memory-light keys)
    return pl.col(col).str.slice(start).cast(pl.Int32)


print(f"raw attempts {N_RAW:,} | clean attempts {N_EV:,}")

# %% [markdown]
# ## 1 · Firmware v3.2.0 timestamp bug — does the +5h30m fix restore the true hour-of-day profile?
# Test 1: inside the bug window the corrected profile of v3.2.0 stations should match v3.2.1 stations (same weeks,
# unaffected). Test 2 (independent field): the pilot tariff engine bills on *true* local time. We read its schedule off
# the data (modal tariff code by corrected hour; Q5 documents the same bands): PEAK 12:00–13:59 and 19:00–22:59,
# OFFPEAK 23:00–05:59 and 14:00–16:59, STD otherwise. After the fix, the code of every v3.2.0 pilot-city row must
# match the schedule at its corrected hour. Test 3: window edges — compare v3.2.0 vs v3.2.1 volumes in 30-min bins
# around the start and end of the window, then identify mis-timed rows one by one with the tariff schedule.

# %%
hdr("1. Firmware v3.2.0 timestamp fix")
BUG_A, BUG_B = dt.date(2025, 3, 10), dt.date(2025, 4, 14)
fw = (events(["station_firmware", "event_ts_raw", "hour", "city", "tariff_code", "ts_corrected", "is_service_failure"])
      .filter(pl.col("event_ts_raw").dt.date().is_between(BUG_A, BUG_B))
      .with_columns(pl.col("event_ts_raw").dt.hour().alias("hour_raw"))
      .collect())
n_shift = int(fw["ts_corrected"].sum())
v0 = fw.filter(pl.col("station_firmware") == "v3.2.0")
v1 = fw.filter(pl.col("station_firmware") == "v3.2.1")
H = np.arange(24)


def hcounts(df, col):
    c = df.group_by(col).len()
    out = np.zeros(24)
    out[c[col].to_numpy().astype(int)] = c["len"].to_numpy()
    return out


def night(p): return 100 * p[:6].sum()
def peakwin(p): return 100 * p[12:23].sum()
def meanh(p): return float((H * p).sum())
def tvd(p, q): return 100 * 0.5 * np.abs(p - q).sum()


n_raw, n_fix, n_ref = hcounts(v0, "hour_raw"), hcounts(v0, "hour"), hcounts(v1, "hour")
p_raw, p_fix, p_ref = n_raw / n_raw.sum(), n_fix / n_fix.sum(), n_ref / n_ref.sum()
chi_fix = stats.chi2_contingency(np.vstack([n_fix, n_ref]))
chi_raw = stats.chi2_contingency(np.vstack([n_raw, n_ref]))
na_raw, na_fix = hcounts(fw, "hour_raw"), hcounts(fw, "hour")
pa_raw, pa_fix = na_raw / na_raw.sum(), na_fix / na_fix.sum()
fr_raw = v0.group_by("hour_raw").agg(pl.len().alias("n"), pl.col("is_service_failure").mean().alias("r")).filter(pl.col("n") >= 1000).sort("r", descending=True)
fr_fix = v0.group_by("hour").agg(pl.len().alias("n"), pl.col("is_service_failure").mean().alias("r")).filter(pl.col("n") >= 1000).sort("r", descending=True)
pk = v0.filter((pl.col("tariff_code") == "PEAK") & pl.col("city").is_in(PILOT_CITIES))
pk_in_raw = 100 * pk["hour_raw"].is_between(12, 22).mean()
pk_in_fix = 100 * pk["hour"].is_between(12, 22).mean()
print(f"events shifted by clean.py: {n_shift:,} = {100 * n_shift / N_EV:.2f}% of clean attempts "
      f"(v3.2.0 attempts in window {v0.height:,}; v3.2.1 {v1.height:,})")
print(f"v3.2.0 share of attempts at 00-05h: raw {night(p_raw):.1f}% -> fixed {night(p_fix):.1f}% (v3.2.1 reference {night(p_ref):.1f}%)")
print(f"v3.2.0 mean event hour: raw {meanh(p_raw):.2f} -> fixed {meanh(p_fix):.2f} (reference {meanh(p_ref):.2f})")
print(f"v3.2.0 share in 12:00-22:59 (pricing peak window): raw {peakwin(p_raw):.1f}% -> fixed {peakwin(p_fix):.1f}% (reference {peakwin(p_ref):.1f}%)")
print(f"v3.2.0 busiest hour: raw {int(np.argmax(p_raw))}h ({100 * p_raw.max():.1f}%) -> fixed {int(np.argmax(p_fix))}h "
      f"({100 * p_fix.max():.1f}%); reference {int(np.argmax(p_ref))}h ({100 * p_ref.max():.1f}%)")
print(f"total-variation distance to reference: raw {tvd(p_raw, p_ref):.1f} pts vs fixed {tvd(p_fix, p_ref):.2f} pts "
      f"(chi2 p: raw {chi_raw.pvalue:.2g}, fixed {chi_fix.pvalue:.2g})")
print(f"NETWORK (all stations) in window: 00-05h share raw {night(pa_raw):.1f}% -> {night(pa_fix):.1f}%; "
      f"12-22h share raw {peakwin(pa_raw):.1f}% -> {peakwin(pa_fix):.1f}%; TVD {tvd(pa_raw, pa_fix):.1f} pts")
print(f"v3.2.0 worst failure-rate hour (>=1,000 attempts): raw {fr_raw['hour_raw'][0]}h ({100 * fr_raw['r'][0]:.1f}%) "
      f"vs fixed {fr_fix['hour'][0]}h ({100 * fr_fix['r'][0]:.1f}%)")
print(f"(coarse check) PEAK-coded attempts at v3.2.0 pilot stations in window n={pk.height:,}: inside 12-22h on raw clock "
      f"{pk_in_raw:.1f}% vs corrected clock {pk_in_fix:.1f}% -- 12:00-22:59 is NOT the tariff window, see the true schedule below")

# The TRUE pilot tariff schedule, read off the data: modal tariff_code by corrected hour for independent riders in the
# pilot cities after 1-Oct-2024 (partners are billed under PARTNER, see section 8).
tzl = (events(["city", "date", "hour", "event_ts_raw", "tariff_code", "is_independent", "station_firmware"])
       .filter(pl.col("city").is_in(PILOT_CITIES) & (pl.col("date") >= PILOT_START) & pl.col("is_independent"))
       .with_columns(pl.col("hour").cast(pl.Int32)))
hb = (tzl.group_by("hour", "tariff_code").len().collect()
      .pivot(on="tariff_code", index="hour", values="len").fill_null(0).sort("hour"))
TCODES = [c for c in ["PEAK", "OFFPEAK", "STD"] if c in hb.columns]
modal = {int(r["hour"]): max(TCODES, key=lambda c: r[c]) for r in hb.iter_rows(named=True)}
purity = {int(r["hour"]): max(r[c] for c in TCODES) / sum(r[c] for c in TCODES) for r in hb.iter_rows(named=True)}
PKH = sorted(h for h, c in modal.items() if c == "PEAK")
OPH = sorted(h for h, c in modal.items() if c == "OFFPEAK")
STH = sorted(h for h, c in modal.items() if c == "STD")
print(f"TRUE tariff schedule (modal code by corrected hour; pilot-city independents, post-pilot): PEAK {PKH}; OFFPEAK {OPH}; STD {STH}; "
      f"lowest single-hour purity {100 * min(purity.values()):.3f}%")


def sched(h):
    return pl.when(h.is_in(PKH)).then(pl.lit("PEAK")).when(h.is_in(OPH)).then(pl.lit("OFFPEAK")).otherwise(pl.lit("STD"))


contra = (tzl.filter(sched(pl.col("hour")) != pl.col("tariff_code"))
          .select("station_firmware", "date", "event_ts_raw", "tariff_code", "hour").collect())
print(f"ROW-LEVEL EDGE PROOF: tariff/clock contradictions in the whole post-pilot period (corrected clock): {contra.height}; "
      f"firmware {sorted(contra['station_firmware'].unique().to_list())}; dates {sorted(str(d) for d in contra['date'].unique().to_list())}; "
      f"logged {contra['event_ts_raw'].min()} .. {contra['event_ts_raw'].max()} (true time +5h30m: "
      f"{contra['event_ts_raw'].min() + dt.timedelta(hours=5, minutes=30)} .. {contra['event_ts_raw'].max() + dt.timedelta(hours=5, minutes=30)}); "
      f"billed {sorted(contra['tariff_code'].unique().to_list())}")
win_t = (tzl.filter(pl.col("event_ts_raw").dt.date().is_between(BUG_A, BUG_B) & (pl.col("station_firmware") == "v3.2.0"))
         .with_columns(sched(pl.col("event_ts_raw").dt.hour().cast(pl.Int32)).alias("exp_raw"), sched(pl.col("hour")).alias("exp_fix")).collect())
agree_raw = 100 * float((win_t["exp_raw"] == win_t["tariff_code"]).mean())
agree_fix = 100 * float((win_t["exp_fix"] == win_t["tariff_code"]).mean())
pk_t = win_t.filter(pl.col("tariff_code") == "PEAK")
pkt_raw = 100 * float(pk_t["event_ts_raw"].dt.hour().cast(pl.Int32).is_in(PKH).mean())
pkt_fix = 100 * float(pk_t["hour"].is_in(PKH).mean())
print(f"TARIFF CLOCK on the true bands (v3.2.0 pilot-city independents in the bug window): PEAK rows inside true PEAK hours raw {pkt_raw:.1f}% "
      f"vs fixed {pkt_fix:.1f}% (n={pk_t.height:,}); tariff code agrees with the hour schedule raw {agree_raw:.1f}% vs fixed {agree_fix:.1f}% "
      f"(n={win_t.height:,})")

# window edges: is the bug window defined on logged or on true time?
bd = (events(["station_firmware", "event_ts_raw", "event_ts"])
      .filter(pl.col("event_ts_raw").is_between(dt.datetime(2025, 3, 1), dt.datetime(2025, 3, 10, 23, 59, 59))
              | pl.col("event_ts_raw").is_between(dt.datetime(2025, 4, 14), dt.datetime(2025, 4, 15, 23, 59, 59)))
      .with_columns((pl.col("event_ts_raw").dt.hour().cast(pl.Int32) * 60 + pl.col("event_ts_raw").dt.minute().cast(pl.Int32)).alias("mod_raw"),
                    pl.col("event_ts_raw").dt.date().alias("d_raw"))
      .collect())


def nfw(df, f): return df.filter(pl.col("station_firmware") == f).height


EVE = 18 * 60 + 30                      # 18:30 = 24:00 - 5h30m
pre_eve = bd.filter(pl.col("d_raw").is_between(dt.date(2025, 3, 1), dt.date(2025, 3, 8)) & (pl.col("mod_raw") >= EVE))
r_eve = nfw(pre_eve, "v3.2.0") / nfw(pre_eve, "v3.2.1")
eve9 = bd.filter((pl.col("d_raw") == dt.date(2025, 3, 9)) & (pl.col("mod_raw") >= EVE))
obs_eve9, exp_eve9 = nfw(eve9, "v3.2.0"), r_eve * nfw(eve9, "v3.2.1")
pre_ngt = bd.filter(pl.col("d_raw").is_between(dt.date(2025, 3, 2), dt.date(2025, 3, 9)) & (pl.col("mod_raw") < 330))
r_ngt = nfw(pre_ngt, "v3.2.0") / nfw(pre_ngt, "v3.2.1")
gap10 = bd.filter(pl.col("event_ts").is_between(dt.datetime(2025, 3, 10, 0, 0), dt.datetime(2025, 3, 10, 5, 29, 59)))
obs_gap, exp_gap = nfw(gap10, "v3.2.0"), r_ngt * nfw(gap10, "v3.2.1")
end14 = bd.filter(pl.col("event_ts_raw").is_between(dt.datetime(2025, 4, 14, 18, 30), dt.datetime(2025, 4, 14, 23, 59, 59)))
obs_end, exp_end = nfw(end14, "v3.2.0"), r_eve * nfw(end14, "v3.2.1")
edge_excess = obs_eve9 - exp_eve9
print(f"EDGE start: v3.2.0 attempts logged 2025-03-09 18:30-23:59 = {obs_eve9:,} vs expected {exp_eve9:,.0f} from v3.2.1 volume "
      f"(excess {edge_excess:+,.0f}, z = {edge_excess / np.sqrt(exp_eve9):.1f})")
print(f"EDGE start: v3.2.0 attempts at corrected 2025-03-10 00:00-05:29 = {obs_gap:,} vs expected {exp_gap:,.0f}  -> "
      f"the first ~5.5 h of 10-Mar were logged on 9-Mar evening and NOT shifted by clean.py")
print(f"EDGE end: v3.2.0 attempts logged 2025-04-14 18:30-23:59 = {obs_end:,} vs expected {exp_end:,.0f} if the window were on "
      f"logged time -> window is defined on TRUE local date (10-Mar..14-Apr); clean.py end edge is handled correctly")
print(f"EDGE impact: ~{edge_excess:.0f} mis-timed attempts = {100 * edge_excess / N_EV:.4f}% of clean attempts; {contra.height} of them are "
      f"identifiable row by row (pilot-city independents whose tariff code contradicts the logged hour)")
pd.DataFrame({"hour": H, "v320_raw_share": p_raw, "v320_fixed_share": p_fix, "v321_reference_share": p_ref,
              "network_raw_share": pa_raw, "network_fixed_share": pa_fix}).to_csv(TAB / "q7_firmware_hour_profile.csv", index=False)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [2.6, 1], "wspace": 0.25})
ymax = 100 * max(p_raw.max(), p_fix.max(), p_ref.max()) * 1.18
ax.axvspan(-0.5, 5.5, color=PALETTE["light"], alpha=0.55, lw=0, zorder=0)
ax.text(2.5, ymax * 0.985, "00:00–05:59", ha="center", va="top", fontsize=10, color="#555555")
ax.plot(H, 100 * p_ref, color=PALETTE["muted"], lw=1.8, label="v3.2.1 stations, same weeks (unaffected reference)", zorder=2)
ax.plot(H, 100 * p_raw, color=PALETTE["accent"], lw=2, marker="o", ms=5, label="v3.2.0 stations, as logged (raw)", zorder=3)
ax.plot(H, 100 * p_fix, color=PALETTE["primary"], lw=2.4, marker="o", ms=5, label="v3.2.0 stations, after +5h30m fix", zorder=4)
ax.set_ylim(0, ymax); ax.set_xlim(-0.5, 23.5); ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("Hour of day (local time)"); ax.set_ylabel("Share of the group's swap attempts (%)")
ax.set_title(f"Hour-of-day profile, {BUG_A:%d %b}–{BUG_B:%d %b %Y}  ({v0.height:,} v3.2.0 attempts)", fontsize=11, fontweight="normal")
ax.annotate(f"Raw: {night(p_raw):.1f}% of attempts\nlogged at 00–05 h", xy=(3, 100 * p_raw[3]), xytext=(0.0, ymax * 0.50),
            fontsize=10, color="#333333", arrowprops=dict(arrowstyle="-", color="#777777", lw=0.8))
ax.annotate(f"Fixed: {night(p_fix):.1f}% (reference {night(p_ref):.1f}%)\nmean hour {meanh(p_raw):.1f} → {meanh(p_fix):.1f} (ref {meanh(p_ref):.1f})",
            xy=(20, 100 * p_fix[20]), xytext=(12.2, ymax * 0.90), fontsize=10, color="#333333",
            arrowprops=dict(arrowstyle="-", color="#777777", lw=0.8))
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), fontsize=9.5)
xg, wb = np.arange(2), 0.36
raw_v, fix_v = [pkt_raw, agree_raw], [pkt_fix, agree_fix]
b_raw = ax2.bar(xg - wb / 2, raw_v, width=wb, color=PALETTE["accent"], label="as logged (raw clock)")
b_fix = ax2.bar(xg + wb / 2, fix_v, width=wb, color=PALETTE["primary"], label="after +5h30m fix")
for b, v in list(zip(b_raw, raw_v)) + list(zip(b_fix, fix_v)):
    ax2.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%" if v < 99.95 else "100%", ha="center", va="bottom", fontsize=11.5,
             fontweight="bold", color="#333333")
ax2.set_xticks(xg, [f"PEAK rows inside\ntrue PEAK hours\n(n = {pk_t.height:,})", f"Tariff code matches\nthe hour schedule\n(n = {win_t.height:,})"], fontsize=9.5)
ax2.set_ylim(0, 138); ax2.set_yticks([0, 25, 50, 75, 100]); ax2.set_ylabel("v3.2.0 pilot-city rows in the bug window (%)")
ax2.legend(loc="upper left", fontsize=9.5)
ax2.set_title("Independent clock: the pilot tariff schedule\n(PEAK 12–13 h & 19–22 h; OFFPEAK 23–05 h & 14–16 h)", fontsize=11, fontweight="normal")
ax2.grid(axis="x", visible=False)
ax2.text(0.0, -0.19, f"Whole pilot period: only {contra.height} rows contradict the schedule —\nv3.2.0 rows logged 9-Mar-25 "
         f"{contra['event_ts_raw'].min():%H:%M}–{contra['event_ts_raw'].max():%H:%M}, billed OFFPEAK\n(the unshifted first hours of 10-Mar)",
         transform=ax2.transAxes, ha="left", va="top", fontsize=9.5, color="#333333")
fig.suptitle(f"Firmware bug logged {night(p_raw):.0f}% of v3.2.0 swaps in the small hours; the +5h30m fix restores the true profile "
             f"and the tariff clock ({pkt_raw:.1f}% → {pkt_fix:.0f}%)", x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.03)
save_fig(fig, "q7_01_firmware_hour_profile")
add_issue("Firmware v3.2.0 timestamp bug (10-Mar..14-Apr-2025)", "documented",
          "Hour profile vs v3.2.1 stations in same weeks; true pilot tariff schedule as independent clock; 30-min edge bins; row-level tariff contradictions",
          n_shift, N_EV, "attempts", "Shift +5h30m (clean.py)",
          f"Untreated: {night(p_raw):.1f}% of v3.2.0 attempts at 00-05h (true {night(p_fix):.1f}%); network 12-22h share "
          f"{peakwin(pa_raw):.1f}% vs {peakwin(pa_fix):.1f}% in window; PEAK rows inside true PEAK hours (12-13h, 19-22h) {pkt_raw:.1f}% raw vs "
          f"{pkt_fix:.1f}% fixed; code-schedule agreement {agree_raw:.1f}% vs {agree_fix:.1f}%. Edge: ~{max(edge_excess, 0):.0f} attempts of "
          f"10-Mar 00:00-05:29 left unshifted, {contra.height} of them identified row by row (logged 9-Mar evening, billed OFFPEAK).",
          f"00-05h share {night(p_raw):.1f}%→{night(p_fix):.1f}%; tariff clock {pkt_raw:.1f}%→{pkt_fix:.0f}%", "high")

# %% [markdown]
# ## 2 · Test stations STN-TST-01/02 — what would keeping them do to Bengaluru and network KPIs?

# %%
hdr("2. Test stations")
tst = pl.read_parquet(CLEAN / "test_station_events.parquet", columns=["station_id", "rider_id", "event_ts", "event_type", "amount_charged_inr"])
t_att, t_comp = tst.height, int((tst["event_type"] == "swap_completed").sum())
t_fail = int(tst["event_type"].is_in(FAIL_TYPES).sum()); t_rev = float(tst["amount_charged_inr"].sum())
t_zero = int((tst["amount_charged_inr"] == 0).sum()); t_zero_comp = int(((tst["amount_charged_inr"] == 0) & (tst["event_type"] == "swap_completed")).sum())
t_night = 100 * float((tst["event_ts"].dt.hour() < 6).mean())
agg = [pl.len().alias("att"), pl.col("is_completed").sum().alias("comp"), pl.col("is_service_failure").sum().alias("fail"),
       pl.col("revenue").sum().alias("rev"), (pl.col("event_ts").dt.hour() < 6).mean().alias("night")]
ev2 = events(["city", "month", "is_completed", "is_service_failure", "revenue", "event_ts"])
blr = ev2.filter(pl.col("city") == "Bengaluru").select(agg).collect().row(0, named=True)
net = ev2.select(agg).collect().row(0, named=True)
blr_m = ev2.filter(pl.col("city") == "Bengaluru").group_by("month").agg(pl.col("is_completed").sum().alias("comp"), pl.col("revenue").sum().alias("rev")).collect()
tst_m = (tst.filter(pl.col("event_type") == "swap_completed").group_by(pl.col("event_ts").dt.truncate("1mo").dt.date().alias("month"))
         .agg(pl.len().alias("t_comp"), pl.col("amount_charged_inr").sum().alias("t_rev")))
bm = blr_m.join(tst_m, on="month", how="left").fill_null(0).sort("month")
g0 = bm.filter(pl.col("month") == dt.date(2024, 1, 1)).row(0, named=True); g1 = bm.filter(pl.col("month") == dt.date(2025, 6, 1)).row(0, named=True)
growth_wo, growth_w = g1["comp"] / g0["comp"], (g1["comp"] + g1["t_comp"]) / (g0["comp"] + g0["t_comp"])
n_blr_st = stations().filter((pl.col("city") == "Bengaluru") & ~pl.col("is_test")).height
sm_blr = station_month().filter(pl.col("city") == "Bengaluru")
blr_st_month = float(sm_blr["completed"].sum() / sm_blr.height)
tst_st_month = t_comp / (2 * 18)
rows = []
for k, lab in [("att", "attempts"), ("comp", "completed swaps"), ("rev", "revenue (INR)")]:
    tv = {"att": t_att, "comp": t_comp, "rev": t_rev}[k]
    rows.append(dict(kpi=f"Bengaluru {lab}", without_test=blr[k], with_test=blr[k] + tv, pct_change=100 * tv / blr[k]))
    rows.append(dict(kpi=f"Network {lab}", without_test=net[k], with_test=net[k] + tv, pct_change=100 * tv / net[k]))
fr_b0, fr_b1 = 100 * blr["fail"] / blr["att"], 100 * (blr["fail"] + t_fail) / (blr["att"] + t_att)
rows += [dict(kpi="Bengaluru service-failure rate (%)", without_test=fr_b0, with_test=fr_b1, pct_change=fr_b1 - fr_b0),
         dict(kpi="Bengaluru revenue per swap (INR)", without_test=blr["rev"] / blr["comp"], with_test=(blr["rev"] + t_rev) / (blr["comp"] + t_comp),
              pct_change=100 * ((blr["rev"] + t_rev) / (blr["comp"] + t_comp) / (blr["rev"] / blr["comp"]) - 1)),
         dict(kpi="Bengaluru swaps growth Jun-25 / Jan-24 (x)", without_test=growth_wo, with_test=growth_w, pct_change=100 * (growth_w / growth_wo - 1))]
tsi = pd.DataFrame(rows); tsi.to_csv(TAB / "q7_test_station_impact.csv", index=False)
rdx = riders().select("rider_id", "n_attempts", "last_attempt_ts")
tr = tst.group_by("rider_id").agg(pl.len().alias("n_tst"), pl.col("event_ts").max().alias("last_tst")).join(rdx, on="rider_id", how="left")
tr_only = int(tr["n_attempts"].is_null().sum())
tr_later = tr.filter(pl.col("last_tst") > pl.col("last_attempt_ts") + pl.duration(days=30)).height
tr_share = float(tr["n_tst"].sum() / (tr["n_tst"].sum() + tr["n_attempts"].fill_null(0).sum()))
print(f"test stations: {t_att:,} attempts, {t_comp:,} completed, revenue INR {t_rev:,.0f}, zero-value rows {t_zero:,} "
      f"(of which completed {t_zero_comp:,}); failures {t_fail:,} ({100 * t_fail / t_att:.1f}%)")
print(f"share of attempts at 00-05h: test {t_night:.1f}% vs network {100 * net['night']:.1f}% -> NOT 'unusual hours'")
print(f"test-station completed swaps per station-month {tst_st_month:,.0f} vs Bengaluru real-station average {blr_st_month:,.0f} "
      f"({n_blr_st} real Bengaluru stations)")
print(tsi.round(3).to_string(index=False))
print(f"riders seen at test stations: {tr.height:,}; test-only riders {tr_only}; their attempts at test stations = {100 * tr_share:.1f}% of their total; "
      f"riders whose last test-station visit is >30 d after their last real attempt: {tr_later}")
rev_b = tsi.loc[tsi.kpi == "Bengaluru revenue (INR)", "pct_change"].iloc[0]
rev_n = tsi.loc[tsi.kpi == "Network revenue (INR)", "pct_change"].iloc[0]
add_issue("Test stations STN-TST-01/02 hold full paid activity", "documented",
          "station_id prefix; compare value/hours vs documentation", t_att, N_RAW, "attempts",
          "Exclude from all KPIs (kept in test_station_events.parquet)",
          f"Kept in: Bengaluru revenue +{rev_b:.1f}%, network revenue +{rev_n:.1f}% (INR {t_rev / 1e6:.2f}M), Bengaluru failure rate "
          f"{fr_b0:.2f}%→{fr_b1:.2f}%, Bengaluru growth {growth_wo:.2f}x→{growth_w:.2f}x; only {t_zero:,} zero-value rows (all non-completed) and "
          f"a normal night share ({t_night:.1f}%) — they behave like a real Bengaluru station ({tst_st_month:,.0f} vs {blr_st_month:,.0f} swaps/station-month) "
          f"used by {tr.height:,} real riders, so the exclusion is a labelled-asset decision, not an obvious-junk one.",
          f"Bengaluru revenue +{rev_b:.1f}%, network +{rev_n:.1f}% (₹{t_rev / 1e6:.1f}M)", "high")

# %% [markdown]
# ## 3 · km_since_last_swap outliers — delivered-range statistics with vs without the null rule (<0 or >200 km)

# %%
hdr("3. km outliers")
bt = batteries()
assert bt["battery_id"].str.slice(7).n_unique() == bt.height, "battery numeric ids not unique"
bmap = bt.select(as_i32("battery_id", 7).alias("bid"), "supplier")
km = (pl.scan_parquet(RAW_EV)
      .filter(~pl.col("station_id").str.starts_with("STN-TST") & pl.col("km_since_last_swap").is_not_null())
      .select(pl.col("km_since_last_swap").cast(pl.Float64).alias("km"), pl.col("battery_in_id").str.slice(4, 2).alias("vc"),
              as_i32("battery_in_id", 7).alias("bid"), as_i32("rider_id", 3).alias("r"))
      .collect()
      .join(bmap, on="bid", how="left")
      .with_columns(((pl.col("km") < 0) | (pl.col("km") > 200)).alias("bad")))


def kstats(df):
    return (df.group_by("vc").agg(pl.len().alias("n"), pl.col("km").mean().alias("mean"), pl.col("km").median().alias("median"),
                                  pl.col("km").std().alias("sd"), pl.col("km").quantile(0.99).alias("p99"),
                                  pl.col("km").min().alias("min"), pl.col("km").max().alias("max")).sort("vc"))


k_all = kstats(km).with_columns(pl.lit("raw (with outliers)").alias("version"))
k_val = kstats(km.filter(~pl.col("bad"))).with_columns(pl.lit("clean (0-200 km)").alias("version"))
kt = pl.concat([k_all, k_val]).to_pandas(); kt.to_csv(TAB / "q7_km_outlier_impact.csv", index=False)
n_neg, n_hi = int((km["km"] < 0).sum()), int((km["km"] > 200).sum())
neg_rng = (float(km.filter(pl.col("km") < 0)["km"].min()), float(km.filter(pl.col("km") < 0)["km"].max()))
hi_rng = (float(km.filter(pl.col("km") > 200)["km"].min()), float(km.filter(pl.col("km") > 200)["km"].max()))
val_rng = (float(km.filter(~pl.col("bad"))["km"].min()), float(km.filter(~pl.col("bad"))["km"].max()))
n_zero_km = int((km["km"] == 0).sum())
pr = km.group_by("r").agg(pl.len().alias("n"), pl.col("bad").sum().alias("b"))
p_bad = float(km["bad"].mean()); e_b = pr["n"] * p_bad
disp_km = float((((pr["b"] - e_b) ** 2) / (e_b * (1 - p_bad))).sum() / (pr.height - 1))
sup_km = (km.filter(pl.col("vc") == "2W").group_by("supplier")
          .agg(pl.col("bad").mean().alias("bad_share"), pl.col("km").mean().alias("mean_raw"),
               pl.col("km").filter(~pl.col("bad")).mean().alias("mean_clean"), pl.col("km").filter(~pl.col("bad")).median().alias("median_clean"))
          .sort("supplier"))
print(f"invalid km: {n_neg:,} negative (range {neg_rng[0]:.0f}..{neg_rng[1]:.0f}) + {n_hi:,} >200 km (range {hi_rng[0]:.0f}..{hi_rng[1]:.0f}) "
      f"= {n_neg + n_hi:,} of {km.height:,} non-null readings ({100 * (n_neg + n_hi) / km.height:.2f}%)")
print(f"valid readings span {val_rng[0]:.1f}-{val_rng[1]:.1f} km; outliers are {neg_rng[0]:.0f}..{neg_rng[1]:.0f} km and {hi_rng[0]:.0f}..{hi_rng[1]:.0f} km; "
      f"no readings between {val_rng[1]:.1f} and {hi_rng[0]:.0f} km and none in [0, {val_rng[0]:.1f}) (exact zeros: {n_zero_km}) -> "
      f"the <0 / >200 cut sits in empty gaps, so the rule is not arbitrary")
print(kt.round(2).to_string(index=False))
print(f"rider dispersion index of invalid readings = {disp_km:.2f} (1.0 = random across riders; >>1 = vehicle-specific odometer faults)")
print(sup_km.to_pandas().round(3).to_string(index=False))
m2 = {r["vc"]: r for r in k_all.iter_rows(named=True)}; c2 = {r["vc"]: r for r in k_val.iter_rows(named=True)}
add_issue("km_since_last_swap outliers (odometer resets)", "documented",
          "Value range; empty gap 120-330 km; rider dispersion test", n_neg + n_hi, km.height, "readings",
          "Set to null (<0 or >200 km), flag km_invalid",
          f"2W mean {m2['2W']['mean']:.1f}→{c2['2W']['mean']:.1f} km, sd {m2['2W']['sd']:.1f}→{c2['2W']['sd']:.1f}; 3W mean "
          f"{m2['3W']['mean']:.1f}→{c2['3W']['mean']:.1f}; medians unchanged; spread evenly across riders (dispersion {disp_km:.2f}) and suppliers.",
          f"2W mean range {m2['2W']['mean']:.1f}→{c2['2W']['mean']:.1f} km; sd {m2['2W']['sd']:.0f}→{c2['2W']['sd']:.0f}", "medium")
del km

# %% [markdown]
# ## 4 · SoC/SoH readings above 100% (sensor drift) — clip at 100?

# %%
hdr("4. SoC/SoH > 100%")
RC = ["soc_in_pct", "soh_in_pct", "soc_out_pct", "soh_out_pct"]
raw_nt = pl.scan_parquet(RAW_EV).filter(~pl.col("station_id").str.starts_with("STN-TST"))
cl = raw_nt.select(*[(pl.col(c) > 100).sum().alias(f"{c}|n") for c in RC], *[pl.col(c).max().alias(f"{c}|max") for c in RC],
                   *[pl.col(c).mean().alias(f"{c}|mean_raw") for c in RC], *[pl.col(c).clip(upper_bound=100).mean().alias(f"{c}|mean_clip") for c in RC],
                   *[pl.col(c).count().alias(f"{c}|nn") for c in RC],
                   pl.any_horizontal([pl.col(c) > 100 for c in RC]).sum().alias("rows_any")).collect().row(0, named=True)
n_read_over = sum(cl[f"{c}|n"] for c in RC); n_read = sum(cl[f"{c}|nn"] for c in RC)
for c in RC:
    print(f"{c:12s}: >100 = {cl[f'{c}|n']:>5,} of {cl[f'{c}|nn']:,} (max {cl[f'{c}|max']:.1f}); mean raw {cl[f'{c}|mean_raw']:.4f} -> clipped {cl[f'{c}|mean_clip']:.4f}")
sc4 = raw_nt.group_by("station_id").agg(pl.len().alias("n"), pl.any_horizontal([pl.col(c) > 100 for c in RC]).sum().alias("over")).collect()
p4 = sc4["over"].sum() / sc4["n"].sum(); e4 = sc4["n"] * p4
disp4 = float((((sc4["over"] - e4) ** 2) / (e4 * (1 - p4))).sum() / (sc4.height - 1))
over_m = (raw_nt.group_by(pl.col("event_ts").dt.truncate("1mo").alias("m"))
          .agg(pl.len().alias("n"), pl.any_horizontal([pl.col(c) > 100 for c in RC]).sum().alias("over")).collect().sort("m")
          .with_columns((100 * pl.col("over") / pl.col("n")).alias("pct")))
sc4q1 = (raw_nt.filter(pl.col("event_ts") < dt.datetime(2024, 4, 1)).group_by("station_id")
         .agg(pl.len().alias("n"), pl.any_horizontal([pl.col(c) > 100 for c in RC]).sum().alias("over")).collect())
p4q = sc4q1["over"].sum() / sc4q1["n"].sum(); e4q = sc4q1["n"] * p4q
disp4q = float((((sc4q1["over"] - e4q) ** 2) / (e4q * (1 - p4q))).sum() / (sc4q1.height - 1))
print(f"readings >100: {n_read_over:,} of {n_read:,} readings ({100 * n_read_over / n_read:.3f}%) on {cl['rows_any']:,} attempts; "
      f"station dispersion index {disp4:.2f} over the full period")
print(f"share of attempts with a >100 reading by month: {over_m['pct'][0]:.2f}% ({over_m['m'][0]:%b-%y}) -> {over_m['pct'][-1]:.2f}% ({over_m['m'][-1]:%b-%y}); "
      f"within Jan-Mar 2024 only, station dispersion index = {disp4q:.2f} -> concentration reflects pack age (new packs read >100), not faulty cabinets")
d_soh = cl["soh_out_pct|mean_raw"] - cl["soh_out_pct|mean_clip"]
add_issue("SoC/SoH readings >100% (sensor drift)", "documented", "Range check (max 103.0%); by-month share; station dispersion within a fixed period",
          n_read_over, n_read, "readings", "Clip at 100, flag reading_clipped",
          f"Mean issued-pack SoH moves by {d_soh:.4f} pts. Readings >100 fade from {over_m['pct'][0]:.2f}% of attempts ({over_m['m'][0]:%b-%y}) to "
          f"{over_m['pct'][-1]:.2f}% ({over_m['m'][-1]:%b-%y}) as packs age; within Q1-2024 they are spread across stations like chance "
          f"(dispersion {disp4q:.2f}) -> drift on new packs, no faulty cabinets, no KPI impact.",
          f"mean SoH shift {d_soh:.4f} pts (negligible)", "low")

# %% [markdown]
# ## 5 · riders.home_city spellings — city counts before vs after standardisation

# %%
hdr("5. home_city standardisation")
rr = pl.read_parquet(PARQ / "riders.parquet").select("rider_id", "home_city").join(riders().select("rider_id", "home_city_clean", "zone_city"), on="rider_id")
raw_exact = rr.group_by("home_city").len().rename({"home_city": "city", "len": "raw_exact_match"})
clean_ct = rr.group_by("home_city_clean").len().rename({"home_city_clean": "city", "len": "clean"})
variants = (rr.filter(pl.col("home_city") != pl.col("home_city_clean")).group_by("home_city_clean")
            .agg(pl.col("home_city").unique().sort().alias("v")).with_columns(pl.col("v").list.eval(pl.element().map_elements(repr, return_dtype=pl.String)).list.join(", "))
            .rename({"home_city_clean": "city", "v": "variants_absorbed"}))
ct = (pl.DataFrame({"city": CITIES}).join(raw_exact, on="city", how="left").join(clean_ct, on="city", how="left").join(variants, on="city", how="left")
      .with_columns((100 * (pl.col("clean") / pl.col("raw_exact_match") - 1)).round(2).alias("pct_undercount_if_raw")))
ct.write_csv(TAB / "q7_city_standardisation.csv")
n_recode = int((rr["home_city"] != rr["home_city_clean"]).sum()); n_var = rr["home_city"].n_unique()
mism = int((rr["home_city_clean"] != rr["zone_city"]).sum())
print(ct.to_pandas().to_string(index=False))
print(f"raw spellings {n_var} -> 6 cities; riders recoded {n_recode:,} ({100 * n_recode / rr.height:.2f}%); mismatch vs home_zone prefix: {mism}")
add_issue("riders.home_city inconsistent spellings", "documented", "Value list; cross-check vs home_zone prefix (0 mismatches)",
          n_recode, rr.height, "riders", "Map 21 spellings → 6 cities",
          f"Raw exact-match counts undercount cities by {ct['pct_undercount_if_raw'].min():.1f}–{ct['pct_undercount_if_raw'].max():.1f}%; "
          f"{n_recode} riders would fall into 15 phantom 'cities'.",
          f"city rider counts +{ct['pct_undercount_if_raw'].min():.1f}% to +{ct['pct_undercount_if_raw'].max():.1f}%", "medium")

# %% [markdown]
# ## 6 · Telemetry blanks — where are they, and what would zero-filling do to stock levels and stockout-hours?
# A "stockout-hour" = a station-hour whose minimum charged 2W packs is 0. Zero-filling a blank would silently turn every
# missing hour into a stockout-hour. We also test whether blank hours are *informative* (MNAR): do riders see more
# failures in the hours when the cabinet failed to report?

# %%
hdr("6. Telemetry missingness")
st_all = stations()
hr_raw = (pl.scan_parquet(PARQ / "station_hourly_status.parquet")
          .select("station_id", "telemetry_status", "charged_2w_min", "charged_2w_avg", "hour_start")
          .join(st_all.lazy().select("station_id", "connectivity_tier", "firmware_version", "is_test"), on="station_id")
          .filter(~pl.col("is_test")))
tele = (hr_raw.group_by("connectivity_tier").agg(
    pl.len().alias("station_hours"), (pl.col("telemetry_status") == "missing").sum().alias("missing"),
    (pl.col("telemetry_status") == "partial").sum().alias("partial"),
    (pl.col("charged_2w_min") == 0).sum().alias("stockout_hours_nullaware"),
    pl.col("charged_2w_avg").mean().alias("avg_charged_2w_nullaware"), pl.col("charged_2w_avg").fill_null(0).mean().alias("avg_charged_2w_zerofill"))
    .collect().sort("connectivity_tier")
    .with_columns((pl.col("stockout_hours_nullaware") + pl.col("missing")).alias("stockout_hours_zerofill"))
    .with_columns((100 * pl.col("missing") / pl.col("station_hours")).alias("missing_pct"),
                  (100 * pl.col("partial") / pl.col("station_hours")).alias("partial_pct"),
                  (100 * pl.col("stockout_hours_nullaware") / (pl.col("station_hours") - pl.col("missing"))).alias("stockout_hour_rate_nullaware"),
                  (100 * pl.col("stockout_hours_zerofill") / pl.col("station_hours")).alias("stockout_hour_rate_zerofill"),
                  (100 * (pl.col("avg_charged_2w_zerofill") / pl.col("avg_charged_2w_nullaware") - 1)).alias("stock_level_bias_pct")))
tele.write_csv(TAB / "q7_telemetry_zero_fill_bias.csv")
print(tele.to_pandas().round(3).to_string(index=False))
tot_h, tot_miss = int(tele["station_hours"].sum()), int(tele["missing"].sum())
so_na, so_zf = int(tele["stockout_hours_nullaware"].sum()), int(tele["stockout_hours_zerofill"].sum())
poor = tele.filter(pl.col("connectivity_tier") == "poor").row(0, named=True)
print(f"missing station-hours {tot_miss:,} of {tot_h:,} ({100 * tot_miss / tot_h:.2f}%); share at poor-tier stations "
      f"{100 * poor['missing'] / tot_miss:.1f}%; 'partial' rows carry complete values (no blanks)")
print(f"stockout-hours network: null-aware {so_na:,} -> zero-filled {so_zf:,} (x{so_zf / so_na:.2f}); poor tier "
      f"{poor['stockout_hours_nullaware']:,} -> {poor['stockout_hours_zerofill']:,} (x{poor['stockout_hours_zerofill'] / poor['stockout_hours_nullaware']:.2f})")
# is missingness informative? (poor-tier stations only, event outcomes in the same station-hour)
eh = (events(["station_id", "date", "hour", "is_stockout", "is_service_failure", "connectivity_tier"])
      .filter(pl.col("connectivity_tier") == "poor")
      .group_by("station_id", "date", pl.col("hour").cast(pl.Int32))
      .agg(pl.len().alias("att"), pl.col("is_stockout").sum().alias("so"), pl.col("is_service_failure").sum().alias("fail")).collect())
hp = (hourly(["station_id", "date", "hour", "telemetry_status", "connectivity_tier", "month"]).filter(pl.col("connectivity_tier") == "poor")
      .with_columns(pl.col("hour").cast(pl.Int32)).collect())
mh = hp.join(eh, on=["station_id", "date", "hour"], how="left").with_columns(pl.col("att").fill_null(0), pl.col("so").fill_null(0), pl.col("fail").fill_null(0))
mnar = (mh.group_by("telemetry_status").agg(pl.len().alias("hours"), pl.col("att").mean().alias("attempts_per_hour"),
                                             (pl.col("so").sum() / pl.col("att").sum() * 100).alias("stockout_rate_pct"),
                                             (pl.col("fail").sum() / pl.col("att").sum() * 100).alias("failure_rate_pct")).sort("telemetry_status"))
print("poor-tier stations — event outcomes by telemetry status of the same station-hour:")
print(mnar.to_pandas().round(3).to_string(index=False))
mm = mh.with_columns((pl.col("telemetry_status") == "missing").alias("miss"))
ct_m = mm.group_by("miss").agg(pl.col("so").sum(), pl.col("att").sum()).sort("miss")
tab_m = np.array([[r["so"], r["att"] - r["so"]] for r in ct_m.iter_rows(named=True)])
chi_m = stats.chi2_contingency(tab_m)
print(f"chi2 test stockout rate missing vs reported hours: p = {chi_m.pvalue:.3g}")
miss_fw = (hr_raw.filter(pl.col("connectivity_tier") == "poor").group_by("firmware_version")
           .agg((pl.col("telemetry_status") == "missing").mean().alias("missing_share"), pl.len()).collect())
miss_month = (hr_raw.group_by(pl.col("hour_start").dt.truncate("1mo").alias("m")).agg((pl.col("telemetry_status") == "missing").mean().alias("missing_share"))
              .collect().sort("m"))
print("poor tier missing share by firmware:", {r["firmware_version"]: round(100 * r["missing_share"], 2) for r in miss_fw.iter_rows(named=True)})
print(f"monthly missing share range: {100 * miss_month['missing_share'].min():.2f}%..{100 * miss_month['missing_share'].max():.2f}%")
mrow = {r["telemetry_status"]: r for r in mnar.iter_rows(named=True)}
add_issue("Telemetry blanks (telemetry_status = missing)", "documented",
          "Null profile by status x connectivity tier; zero-fill simulation; outcome test in blank hours",
          tot_miss, tot_h, "station-hours", "Keep null; null-aware means; report coverage",
          f"All blanks at poor-tier stations ({100 * poor['missing'] / tot_miss:.0f}%). Zero-fill would add {tot_miss:,} fake stockout-hours "
          f"(network x{so_zf / so_na:.2f}, poor tier x{poor['stockout_hours_zerofill'] / poor['stockout_hours_nullaware']:.2f}) and cut poor-tier "
          f"stock level by {abs(poor['stock_level_bias_pct']):.1f}%. Blank hours are not worse for riders (stockout "
          f"{mrow['missing']['stockout_rate_pct']:.2f}% vs {mrow['ok']['stockout_rate_pct']:.2f}% when reported).",
          f"zero-fill → stockout-hours x{so_zf / so_na:.1f} (poor tier x{poor['stockout_hours_zerofill'] / poor['stockout_hours_nullaware']:.1f})", "high")

fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"wspace": 0.3})
tiers = ["good", "fair", "poor"]
td = {r["connectivity_tier"]: r for r in tele.iter_rows(named=True)}
xs = np.arange(3)
axa.bar(xs - 0.2, [td[t]["partial_pct"] for t in tiers], width=0.38, color=PALETTE["warn"], label="partial (values complete)")
axa.bar(xs + 0.2, [td[t]["missing_pct"] for t in tiers], width=0.38, color=PALETTE["accent"], label="missing (all sensors blank)")
for i, t in enumerate(tiers):
    axa.text(i + 0.2, td[t]["missing_pct"] + 0.1, f"{td[t]['missing_pct']:.1f}%", ha="center", va="bottom", fontsize=10, color="#333333")
    axa.text(i - 0.2, td[t]["partial_pct"] + 0.1, f"{td[t]['partial_pct']:.1f}%", ha="center", va="bottom", fontsize=10, color="#333333")
axa.set_xticks(xs, [f"{t}\n{td[t]['station_hours'] / 1e3:.0f}K hrs" for t in tiers])
axa.set_ylabel("Share of station-hours (%)"); axa.set_xlabel("Station connectivity tier")
axa.set_title("Telemetry gaps exist only at poor-connectivity stations", fontsize=11.5)
axa.legend(loc="upper left")
axb.bar(xs - 0.2, [td[t]["stockout_hour_rate_nullaware"] for t in tiers], width=0.38, color=PALETTE["primary"], label="blanks kept as null (correct)")
axb.bar(xs + 0.2, [td[t]["stockout_hour_rate_zerofill"] for t in tiers], width=0.38, color=PALETTE["accent"], label="blanks zero-filled (wrong)")
for i, t in enumerate(tiers):
    axb.text(i - 0.2, td[t]["stockout_hour_rate_nullaware"] + 0.05, f"{td[t]['stockout_hour_rate_nullaware']:.2f}%", ha="center", va="bottom", fontsize=10, color="#333333")
    axb.text(i + 0.2, td[t]["stockout_hour_rate_zerofill"] + 0.05, f"{td[t]['stockout_hour_rate_zerofill']:.2f}%", ha="center", va="bottom", fontsize=10, color="#333333")
axb.set_xticks(xs, tiers); axb.set_xlabel("Station connectivity tier")
axb.set_ylabel("Station-hours with zero charged 2W packs (%)")
axb.set_title("Zero-filling would invent stockouts where the network is least visible", fontsize=11.5)
axb.legend(loc="upper left")
fig.suptitle(f"Never zero-fill telemetry: it would add {tot_miss:,} fake stockout-hours (x{so_zf / so_na:.1f} network-wide)",
             x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.02)
save_fig(fig, "q7_04_telemetry_zero_fill_bias")
del hp, mh, eh

# %% [markdown]
# ## 7 · Near-duplicate offline-sync retries — confirm the documented rule and stress-test looser definitions
# Consecutive attempts of the same rider (sorted by time). If sync retries were duplicated, gaps between repeat
# records at the same station would pile up at a few seconds–2 minutes and be over-represented in offline_batch rows.

# %%
hdr("7. Near-duplicates")
DUPC = ["r", "s", "event_ts", "et", "bi", "bo", "off"]
pairs = (events(["event_id", "rider_id", "station_id", "event_ts", "event_type", "battery_in_id", "battery_out_id", "sync_mode", "vehicle_class"])
         .select(as_i32("rider_id", 3).alias("r"), pl.col("station_id").cast(pl.Categorical).alias("s"), "event_ts",
                 pl.col("event_type").cast(pl.Categorical).alias("et"), as_i32("battery_in_id", 7).alias("bi"),
                 as_i32("battery_out_id", 7).alias("bo"), (pl.col("sync_mode") == "offline_batch").alias("off"),
                 (pl.col("vehicle_class") == "3W").alias("w3"), pl.col("event_id").str.slice(3).cast(pl.Int64).alias("eid"))
         .sort("r", "event_ts", "eid")                      # event_id breaks timestamp ties -> deterministic pairs
         .with_columns([pl.col(c).shift(1).alias(c + "_p") for c in DUPC])
         .filter(pl.col("r") == pl.col("r_p"))
         .with_columns((pl.col("event_ts") - pl.col("event_ts_p")).dt.total_seconds().alias("gap"))
         .filter(pl.col("gap") <= 600)
         .collect())
n_consec = N_EV - riders().filter(pl.col("n_attempts").is_not_null()).height
same_s, same_t = pl.col("s") == pl.col("s_p"), pl.col("et") == pl.col("et_p")
same_b = pl.col("bi").eq_missing(pl.col("bi_p")) & pl.col("bo").eq_missing(pl.col("bo_p"))
any_off = pl.col("off") | pl.col("off_p")
g = pl.col("gap")
DEFS = {
    "D0 documented: same station+type+both batteries, <=180 s, >=1 offline_batch": same_s & same_t & same_b & (g <= 180) & any_off,
    "D1 ignore battery IDs: same station+type, <=180 s, >=1 offline_batch": same_s & same_t & (g <= 180) & any_off,
    "D2 any sync mode: same station+type, <=180 s": same_s & same_t & (g <= 180),
    "D3 any event type: same station, <=180 s": same_s & (g <= 180),
    "D4 same returned pack (battery_in), <=300 s, any station": (pl.col("bi") == pl.col("bi_p")) & (g <= 300),
    "D5 identical timestamp (0 s gap)": g == 0,
    "D6 wide window: same station+type, <=600 s": same_s & same_t,
}
dup_rows = []
for name, expr in DEFS.items():
    sub = pairs.filter(expr.fill_null(False))
    dup_rows.append(dict(definition=name, pairs=sub.height, pct_of_attempts=100 * sub.height / N_EV,
                         share_offline_involved=float(sub.select((pl.col("off") | pl.col("off_p")).mean()).item()) if sub.height else np.nan))
dupt = pd.DataFrame(dup_rows); dupt.to_csv(TAB / "q7_near_duplicate_definitions.csv", index=False)
print(f"consecutive same-rider pairs in data: {n_consec:,}; pairs <=600 s apart: {pairs.height:,}")
print(dupt.round(4).to_string(index=False))
st_off = events(["station_id", "sync_mode"]).group_by("station_id").agg((pl.col("sync_mode") == "offline_batch").mean().alias("p")).collect()
d2 = (pairs.filter((same_s & same_t & (g <= 180)).fill_null(False)).with_columns(pl.col("s").cast(pl.String).alias("station_id"))
      .join(st_off, on="station_id", how="left"))
exp_off = float((1 - (1 - d2["p"]) ** 2).mean()); obs_off = float(d2.select((pl.col("off") | pl.col("off_p")).mean()).item())
edges = np.arange(0, 601, 30)
ss = pairs.filter((same_s & same_t).fill_null(False))
h_off, _ = np.histogram(ss.filter(any_off)["gap"].to_numpy(), bins=edges)
h_rt, _ = np.histogram(ss.filter(~any_off)["gap"].to_numpy(), bins=edges)
h_all = h_off + h_rt
chi_u = stats.chisquare(h_all)
dens_short, dens_long = h_all[:4].mean(), h_all[4:].mean()
seq = (pairs.filter((same_s & (g <= 180)).fill_null(False)).group_by(pl.col("et_p").cast(pl.String).alias("previous"), pl.col("et").cast(pl.String).alias("next"))
       .len().sort("len", descending=True))
d2_types = d2.group_by(pl.col("et").cast(pl.String).alias("later_record_type")).len()
d2_comp = int(d2_types.filter(pl.col("later_record_type") == "swap_completed")["len"].sum())
d2_fail = int(d2_types.filter(pl.col("later_record_type").is_in(FAIL_TYPES))["len"].sum())
base_fail = events(["is_service_failure"]).select(pl.col("is_service_failure").sum()).collect().item()
fr_before, fr_after = 100 * base_fail / N_EV, 100 * (base_fail - d2_fail) / (N_EV - d2.height)
print(f"D2 pairs: offline_batch involved {100 * obs_off:.1f}% vs {100 * exp_off:.1f}% expected from station offline shares (no excess)")
print(f"if the later record of every D2 pair were dropped: completed swaps -{d2_comp:,} ({100 * d2_comp / 3586675:.2f}%), "
      f"service-failure rate {fr_before:.3f}% -> {fr_after:.3f}%")
print(f"gap histogram (same station+type, 30-s bins): density 0-120 s = {dens_short:.0f}/bin vs 120-600 s = {dens_long:.0f}/bin "
      f"(ratio {dens_short / dens_long:.2f}); chi2 uniformity p = {chi_u.pvalue:.3g}")
print("same-station pairs <=180 s by sequence:"); print(seq.head(8))
# A sync retry re-sends the SAME record, so it would carry the same battery IDs: the documented rule can see retries.
# Check the battery-ID signature directly: same (non-null) battery_in within 300 s, any station, vs chance coincidence.
n_pack = {w3_: int(bt.filter(pl.col("pack_type").str.starts_with("3W" if w3_ else "2W")).height) for w3_ in (False, True)}
p300 = pairs.filter((g <= 300) & pl.col("bi").is_not_null() & pl.col("bi_p").is_not_null())
d4_obs = int(p300.filter(pl.col("bi") == pl.col("bi_p")).height)
d4_exp = sum(int((p300["w3"] == w3_).sum()) / n_pack[w3_] for w3_ in (False, True))
print(f"same non-null battery_in within 300 s (any station): {d4_obs} pairs network-wide vs {d4_exp:.1f} expected by chance "
      f"({p300.height:,} same-rider pairs <=300 s with both packs recorded; {n_pack[False]:,} 2W / {n_pack[True]:,} 3W packs) "
      f"-> no record was re-sent: there are no retries under any rule, so the documented rule is adequate")
d0n = int(dupt.iloc[0]["pairs"])
add_issue("Near-duplicate offline-sync retries", "documented",
          "Documented rule + 6 looser definitions; gap-distribution, offline-share and same-battery tests", d0n, N_EV, "attempts",
          "None removed (rule applied, 0 hits)",
          f"0 duplicates under the documented rule and none under any looser rule: the same station + type rule (D2) finds "
          f"{int(dupt.iloc[2]['pairs']):,} pairs ({dupt.iloc[2]['pct_of_attempts']:.3f}%) with a flat gap density (0-120 s vs 120-600 s ratio "
          f"{dens_short / dens_long:.2f}) and no offline_batch excess ({100 * obs_off:.1f}% vs {100 * exp_off:.1f}% expected); only {d4_obs} "
          f"same-battery_in pairs ≤300 s exist network-wide ({d4_exp:.1f} expected by chance) -> chance coincidences, not retries. Dropping D2 "
          f"pairs anyway would cut swaps by {100 * d2_comp / 3586675:.2f}% and move the failure rate {fr_before:.3f}%→{fr_after:.3f}%.",
          f"0 duplicates; looser rules ≤{dupt.iloc[3]['pct_of_attempts']:.2f}% & look like chance", "low")

fig, ax = plt.subplots(figsize=(12, 5.2))
centers = edges[:-1] + 15
ax.bar(centers, h_rt, width=27, color=PALETTE["primary"], label="both records realtime")
ax.bar(centers, h_off, width=27, bottom=h_rt, color=PALETTE["warn"], label="≥1 record offline_batch")
ax.axvspan(0, 180, color=PALETTE["light"], alpha=0.6, lw=0, zorder=0)
ax.text(90, max(h_all) * 1.13, "documented retry window\n(≤180 s)", ha="center", va="top", fontsize=10, color="#555555")
ax.axhline(h_all.mean(), color="#333333", lw=1.2, label=f"mean {h_all.mean():.0f} pairs per bin (flat = chance)")
ax.set_ylim(0, max(h_all) * 1.3); ax.set_xlim(0, 600)
ax.set_xlabel("Seconds between consecutive attempts of the same rider at the same station, same event type")
ax.set_ylabel("Pairs per 30-second bin")
ax.legend(loc="upper right", ncol=3, fontsize=9.5)
ax.set_title(f"Documented rule: 0 duplicates. Same station + type rule (D2): {int(dupt.iloc[2]['pairs']):,} pairs ≤180 s, but the density is flat "
             f"(0–120 s / 120–600 s = {dens_short / dens_long:.2f}).\nA re-sent record would repeat the battery IDs: only {d4_obs} same-battery pairs "
             f"≤300 s exist network-wide ({d4_exp:.1f} expected by chance)", fontsize=11, fontweight="normal")
fig.suptitle("No hidden sync-retry duplicates: repeat records show no spike at short gaps", x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.0)
save_fig(fig, "q7_05_near_duplicate_gap_test")
del pairs, ss, d2, p300

# %% [markdown]
# ## 8 · Revenue logic — non-completed attempts and tariff arithmetic
# Non-completed attempts carry list_price/discount/tariff_code but amount_charged = 0. We size the error from using
# list − discount, then test the identity amount = list − discount (+15 PEAK, −8 OFFPEAK) row by row.

# %%
hdr("8. Revenue logic & tariff arithmetic")
rv = (events(["event_id", "rider_id", "event_ts", "is_completed", "tariff_code", "list_price_inr", "discount_inr", "amount_charged_inr",
              "partner_id", "plan_type", "city", "date", "hour"])
      .with_columns((pl.col("amount_charged_inr") - (pl.col("list_price_inr") - pl.col("discount_inr"))).round(1).alias("res")))
rsum = rv.group_by("is_completed").agg(pl.len().alias("n"), pl.col("list_price_inr").is_not_null().sum().alias("list_populated"),
                                       pl.col("amount_charged_inr").sum().alias("amount"), (pl.col("list_price_inr") - pl.col("discount_inr")).sum().alias("list_minus_disc")).collect()
rc_ = {r["is_completed"]: r for r in rsum.iter_rows(named=True)}
rev_true, naive_extra = rc_[True]["amount"], rc_[False]["list_minus_disc"]
print(f"non-completed attempts {rc_[False]['n']:,}: list price populated {rc_[False]['list_populated']:,}, amount charged sum {rc_[False]['amount']:.0f}")
print(f"revenue (amount on completed) INR {rev_true:,.0f}; using list-discount on ALL attempts would add INR {naive_extra:,.0f} (+{100 * naive_extra / rev_true:.1f}%)")
fp = partners().select("partner_id", "peak_surcharge_billable")
cmp_ = rv.filter(pl.col("is_completed")).join(fp.lazy(), on="partner_id", how="left")
expl = (pl.when(pl.col("tariff_code") == "PEAK").then(pl.col("res") == 15).when(pl.col("tariff_code") == "OFFPEAK").then(pl.col("res") == -8)
        .when(pl.col("tariff_code") == "PARTNER").then(pl.col("res").is_in([0.0, 15.0, -8.0])).otherwise(pl.col("res") == 0))
tar = (cmp_.with_columns(expl.alias("explained"))
       .group_by("tariff_code").agg(pl.len().alias("n"), (pl.col("res").abs() > 0.01).sum().alias("amount_ne_list_minus_disc"),
                                    (~pl.col("explained")).sum().alias("unexplained"), pl.col("res").filter(~pl.col("explained")).sum().alias("unexpl_net_inr"),
                                    pl.col("res").filter(~pl.col("explained")).abs().sum().alias("unexpl_gross_inr"),
                                    (pl.col("res") == 15).sum().alias("plus15"), (pl.col("res") == -8).sum().alias("minus8"))
       .collect().sort("tariff_code"))
print(tar.to_pandas().to_string(index=False))
hid = (cmp_.filter(pl.col("tariff_code") == "PARTNER")
       .group_by("peak_surcharge_billable", pl.col("city").is_in(PILOT_CITIES).alias("pilot_city"), (pl.col("date") >= PILOT_START).alias("post_pilot"))
       .agg(pl.len().alias("n"), (pl.col("res") == 15).sum().alias("plus15"), (pl.col("res") == -8).sum().alias("minus8"))
       .collect().sort("peak_surcharge_billable", "pilot_city", "post_pilot"))
print("PARTNER rows with hidden peak (+15) / off-peak (-8) adjustments:"); print(hid)
n_p15 = int(tar.filter(pl.col("tariff_code") == "PARTNER")["plus15"].item()); n_m8 = int(tar.filter(pl.col("tariff_code") == "PARTNER")["minus8"].item())
partner_pilot = int(hid.filter(pl.col("pilot_city") & pl.col("post_pilot"))["n"].sum())
n_unexpl, net_unexpl, gross_unexpl = int(tar["unexplained"].sum()), float(tar["unexpl_net_inr"].sum()), float(tar["unexpl_gross_inr"].sum())
n_comp = int(tar["n"].sum())
print(f"hidden partner adjustments: +15 on {n_p15:,} and -8 on {n_m8:,} PARTNER swaps = {100 * (n_p15 + n_m8) / partner_pilot:.1f}% of "
      f"post-pilot partner swaps in pilot cities ({partner_pilot:,}); net INR {15 * n_p15 - 8 * n_m8:,.0f}")
# denominators: how much pilot exposure does tariff_code miss?
partner_pilot_y = int(hid.filter(pl.col("pilot_city") & pl.col("post_pilot") & (pl.col("peak_surcharge_billable") == "Y"))["n"].sum())
n_coded_pilot = int(tar.filter(pl.col("tariff_code").is_in(["PEAK", "OFFPEAK"]))["n"].sum())
n_coded_partner = int(cmp_.filter(pl.col("tariff_code").is_in(["PEAK", "OFFPEAK"]) & pl.col("partner_id").is_not_null()).select(pl.len()).collect().item())
n_pilot_all = n_coded_pilot + n_p15 + n_m8
on_sched = (cmp_.filter((pl.col("tariff_code") == "PARTNER") & pl.col("res").is_in([15.0, -8.0]))
            .group_by("res").agg(pl.len().alias("n"), pl.when(pl.col("res") == 15).then(pl.col("hour").cast(pl.Int32).is_in(PKH))
                                 .otherwise(pl.col("hour").cast(pl.Int32).is_in(OPH)).mean().alias("on_schedule"))
            .collect())
osd = {float(r["res"]): 100 * r["on_schedule"] for r in on_sched.iter_rows(named=True)}
print(f"tariff_code labels {n_coded_partner} partner swaps PEAK/OFFPEAK -> it misses 100% of the {n_p15 + n_m8:,} pilot-priced partner swaps. "
      f"They are {100 * (n_p15 + n_m8) / partner_pilot:.1f}% of {partner_pilot:,} post-pilot partner swaps in Bengaluru/Pune, "
      f"{100 * (n_p15 + n_m8) / partner_pilot_y:.1f}% of the {partner_pilot_y:,} by Y-flag partners, and {100 * (n_p15 + n_m8) / n_pilot_all:.1f}% "
      f"of all {n_pilot_all:,} pilot-priced completed swaps ({n_coded_pilot:,} coded PEAK/OFFPEAK + {n_p15 + n_m8:,} hidden)")
print(f"hidden +15 rows on true PEAK hours: {osd[15.0]:.2f}%; hidden -8 rows on true OFFPEAK hours: {osd[-8.0]:.2f}% -> they ARE the pilot schedule")
print(f"unexplained +-5/+-10 deviations: {n_unexpl:,} swaps ({100 * n_unexpl / n_comp:.3f}%), net INR {net_unexpl:,.0f}, gross INR {gross_unexpl:,.0f}")
# do unexplained deviations trigger billing disputes? (same rider, ticket within 7 days after the swap)
bdt = tickets().filter(pl.col("category") == "billing_dispute").select("rider_id", "created_ts")
devs = cmp_.filter(~expl).select("event_id", "rider_id", "event_ts").collect()
base = cmp_.filter(expl & (pl.col("event_id").hash(7) % 40 == 0)).select("event_id", "rider_id", "event_ts").collect()


def dispute_rate(df):
    j = (df.join(bdt, on="rider_id", how="inner")
         .filter((pl.col("created_ts") >= pl.col("event_ts")) & (pl.col("created_ts") <= pl.col("event_ts") + pl.duration(days=7))))
    return j["event_id"].n_unique() / df.height, j["event_id"].n_unique()


dr_dev, k_dev = dispute_rate(devs); dr_base, k_base = dispute_rate(base)
z_bd = stats.chi2_contingency(np.array([[k_dev, devs.height - k_dev], [k_base, base.height - k_base]]))
print(f"billing-dispute ticket within 7 d: after unexplained deviation {100 * dr_dev:.2f}% (n={devs.height:,}) vs normal swaps "
      f"{100 * dr_base:.2f}% (sample n={base.height:,}); chi2 p = {z_bd.pvalue:.3g}")
pp = (cmp_.group_by("plan_type", "tariff_code").agg(pl.len().alias("n"), pl.col("amount_charged_inr").mean().alias("mean_amount"),
                                                    pl.col("discount_inr").mean().alias("mean_discount")).collect().sort("plan_type", "tariff_code"))
print(pp)
raw_codes = pl.scan_parquet(RAW_EV).group_by("tariff_code").len().collect()
raw_pm = pl.scan_parquet(RAW_EV).group_by("payment_mode").len().collect()
n_prepaid = int(pp.filter(pl.col("plan_type") == "prepaid_pack")["n"].sum())
print(f"tariff codes present in raw: {sorted(raw_codes['tariff_code'].to_list())}; PREPAID rows 0, PROMO_FREE rows 0")
print(f"payment_mode values: {raw_pm.to_dicts()}")
print(f"prepaid_pack riders' completed swaps: {n_prepaid:,} ({100 * n_prepaid / n_comp:.1f}%), all billed at STD/PEAK/OFFPEAK with discount 0")
add_issue("Price fields populated on non-completed attempts", "documented", "amount vs list/discount by event_type",
          rc_[False]["n"], N_EV, "attempts", "Revenue = amount_charged on completed swaps only",
          f"Using list − discount on every attempt would overstate revenue by INR {naive_extra / 1e6:.1f}M (+{100 * naive_extra / rev_true:.1f}%).",
          f"list-based revenue would be +{100 * naive_extra / rev_true:.1f}% (₹{naive_extra / 1e6:.1f}M)", "medium")
add_issue("Peak/off-peak adjustments hidden inside PARTNER tariff code", "undocumented",
          "Row identity amount = list − discount; residual +15/−8 by partner flag, city, period",
          n_p15 + n_m8, n_comp, "swaps", "Derive pilot exposure from the residual, not tariff_code",
          f"{n_p15:,} partner swaps carry the +15 surcharge and {n_m8:,} the −8 discount, on the true PEAK/OFFPEAK hours "
          f"{osd[15.0]:.0f}%/{osd[-8.0]:.2f}% of the time, while coded PARTNER: tariff_code labels none of them (misses 100% of partner "
          f"exposure). They are {100 * (n_p15 + n_m8) / partner_pilot:.1f}% of post-pilot partner swaps in Bengaluru/Pune "
          f"({100 * (n_p15 + n_m8) / partner_pilot_y:.1f}% of Y-flag partners' swaps) and {100 * (n_p15 + n_m8) / n_pilot_all:.1f}% of all "
          f"{n_pilot_all:,} pilot-priced swaps; N-flag partners never charged. Net INR {(15 * n_p15 - 8 * n_m8) / 1e6:.2f}M.",
          f"tariff_code hides {100 * (n_p15 + n_m8) / n_pilot_all:.0f}% of pilot-priced swaps ({n_p15 + n_m8:,}; ₹{(15 * n_p15 - 8 * n_m8) / 1e6:.2f}M)", "high")
add_issue("Unexplained ±₹5/±₹10 billing deviations", "undocumented", "Residual after documented tariff rules",
          n_unexpl, n_comp, "swaps", "Keep amount_charged as revenue; flag for billing audit",
          f"Net INR {net_unexpl:,.0f} (gross INR {gross_unexpl:,.0f}); billing-dispute ticket within 7 d {100 * dr_dev:.2f}% vs {100 * dr_base:.2f}% for normal swaps.",
          f"net ₹{net_unexpl / 1e3:.0f}K; disputes {100 * dr_dev:.1f}% vs {100 * dr_base:.1f}%", "low")
add_issue("PREPAID / PROMO_FREE tariff codes absent; prepaid riders pay standard tariffs", "undocumented",
          "Code inventory vs data dictionary; tariff by plan_type", n_prepaid, n_comp, "swaps",
          "Treat plan_type as a label; no prepaid discount modelled",
          f"{n_prepaid:,} prepaid_pack swaps billed exactly like pay-as-you-go (same tariffs, discount 0) — prepaid revenue cannot be separated.",
          "prepaid pricing not observable", "medium")
add_issue("payment_mode constant ('partner_invoice' on 100%)", "undocumented", "Distinct values (also for pay-as-you-go riders)",
          int(raw_pm["len"].sum()), N_RAW, "attempts", "Drop field; use plan_type/tariff", "No payment-channel analysis possible.",
          "field unusable", "low")

# %% [markdown]
# ## 9 · CSAT — naive mean vs inverse-probability-weighted (IPW) mean
# CSAT is missing for 65.6% of tickets and response depends on resolution speed and status (MNAR w.r.t. the naive
# mean). We model P(response | speed decile, category, channel, priority) on resolved tickets, re-weight responders by
# 1/p, and bound the unobservable part (unresolved tickets have 0% response).

# %%
hdr("9. CSAT naive vs IPW")
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

tk = tickets()
tkw = tk.filter(pl.col("in_window"))
st_resp = tkw.group_by("resolution_status").agg(pl.len().alias("n"), pl.col("has_csat").mean().alias("response_rate")).sort("resolution_status")
print(st_resp)
res = tkw.filter(pl.col("resolution_status") == "resolved").select("resolution_hours", "category", "channel", "priority", "has_csat", "csat_score").to_pandas()
res["speed_decile"] = pd.qcut(res["resolution_hours"], 10, labels=False) + 1
y = res["has_csat"].to_numpy()
X = pd.get_dummies(res[["speed_decile", "category", "channel", "priority"]].astype(str), drop_first=True).astype(float)
lr = LogisticRegression(max_iter=3000).fit(X, y)
ph = lr.predict_proba(X)[:, 1]
auc = roc_auc_score(y, ph)
sc_ = res["csat_score"].to_numpy()
naive = float(np.nanmean(sc_[y]))
w = 1 / ph[y]
ipw = float(np.sum(w * sc_[y]) / w.sum())
cells = res["speed_decile"].astype(str) + "|" + res["category"] + "|" + res["channel"]
codes, _u = pd.factorize(cells)
K = codes.max() + 1
s0 = np.where(y, sc_, 0.0)


def poststrat(idx):
    Nc = np.bincount(codes[idx], minlength=K); Rc = np.bincount(codes[idx], weights=y[idx], minlength=K)
    Sc = np.bincount(codes[idx], weights=s0[idx], minlength=K)
    ok = Rc > 0
    return float(np.sum(Nc[ok] * Sc[ok] / Rc[ok]) / Nc[ok].sum())


rng = np.random.default_rng(59500)
n_res = len(res)
ps = poststrat(np.arange(n_res))
boot = np.array([poststrat(rng.integers(0, n_res, n_res)) for _ in range(400)])
ci_ps = np.percentile(boot, [2.5, 97.5])
boot_naive = np.array([np.mean(rng.choice(sc_[y], y.sum())) for _ in range(400)])
ci_nv = np.percentile(boot_naive, [2.5, 97.5])
by_dec = res.groupby("speed_decile").agg(n=("has_csat", "size"), response_rate=("has_csat", "mean"), csat_mean=("csat_score", "mean"),
                                         hours_median=("resolution_hours", "median")).reset_index()
by_cat = res.groupby("category").agg(n=("has_csat", "size"), response_rate=("has_csat", "mean"), csat_mean=("csat_score", "mean")).reset_index()
n_unres = int(tkw.filter(pl.col("resolution_status") == "unresolved").height)
pop = n_res + n_unres
bounds = {s: (n_res * ipw + n_unres * s) / pop for s in (1.0, 2.0, 3.0, 4.0, 5.0)}
print(by_dec.round(3).to_string(index=False)); print(by_cat.round(3).to_string(index=False))
print(f"response model AUC = {auc:.3f} (resolved tickets, n={n_res:,})")
print(f"CSAT naive mean {naive:.3f} (95% CI {ci_nv[0]:.3f}-{ci_nv[1]:.3f}); IPW {ipw:.3f}; post-stratified {ps:.3f} (95% CI {ci_ps[0]:.3f}-{ci_ps[1]:.3f})")
print(f"unresolved tickets {n_unres:,} ({100 * n_unres / pop:.1f}% of resolved+unresolved) have 0% response. SCENARIO: if they would score "
      f"1/2/3 the ticket-population CSAT would be {bounds[1.0]:.2f}/{bounds[2.0]:.2f}/{bounds[3.0]:.2f}; the unrestricted range "
      f"(any score 1-5) is {bounds[1.0]:.2f}-{bounds[5.0]:.2f}")
csat_tab = pd.DataFrame([dict(estimator="naive mean of responders", value=naive, ci_low=ci_nv[0], ci_high=ci_nv[1]),
                         dict(estimator="IPW (logistic response model, resolved tickets)", value=ipw, ci_low=np.nan, ci_high=np.nan),
                         dict(estimator="post-stratified (speed decile x category x channel)", value=ps, ci_low=ci_ps[0], ci_high=ci_ps[1])]
                        + [dict(estimator=f"scenario: all resolved+unresolved if unresolved scored {s:.0f}", value=v, ci_low=np.nan, ci_high=np.nan) for s, v in bounds.items()])
csat_tab.to_csv(TAB / "q7_csat_estimates.csv", index=False)
n_csat_miss = int(tk["csat_score"].null_count())
add_issue("CSAT mostly missing, not at random", "documented", "Response rate by speed decile/status; logistic response model; IPW + post-stratification",
          n_csat_miss, tk.height, "tickets", "Report IPW/post-stratified CSAT for resolved tickets + bound for unresolved",
          f"Naive {naive:.2f} vs IPW {ipw:.2f} (post-strat {ps:.2f}, CI {ci_ps[0]:.2f}-{ci_ps[1]:.2f}): re-weighting barely moves it because scores "
          f"vary little with speed; the real gap is the {n_unres:,} unresolved tickets with 0% response (scenario: population CSAT "
          f"{bounds[1.0]:.2f}-{bounds[3.0]:.2f} if they scored 1-3; unrestricted range {bounds[1.0]:.2f}-{bounds[5.0]:.2f}).",
          f"naive {naive:.2f} vs IPW {ipw:.2f}; scenario with unresolved {bounds[1.0]:.2f}–{bounds[3.0]:.2f}", "medium")

fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.3})
axa.bar(by_dec["speed_decile"], 100 * by_dec["response_rate"], color=PALETTE["primary"], width=0.7)
for _, r in by_dec.iterrows():
    axa.text(r["speed_decile"], 100 * r["response_rate"] + 0.8, f"{100 * r['response_rate']:.0f}%", ha="center", va="bottom", fontsize=9.5, color="#333333")
axa.set_xticks(by_dec["speed_decile"], [f"D{int(d)}\n{h:.0f}h" for d, h in zip(by_dec["speed_decile"], by_dec["hours_median"])], fontsize=9)
axa.set_xlabel("Resolution-speed decile (median hours to resolve)"); axa.set_ylabel("Tickets with a CSAT score (%)")
axa.set_ylim(0, 100 * by_dec["response_rate"].max() * 1.25)
axa.set_title("Slow-resolved tickets rarely get scored; unresolved never do", fontsize=11.5)
labels = ["Naive mean", "IPW", "Post-stratified"]
vals = [naive, ipw, ps]
axb.errorbar([0], [naive], yerr=[[naive - ci_nv[0]], [ci_nv[1] - naive]], fmt="o", color=PALETTE["muted"], ms=9, capsize=4)
axb.plot([1], [ipw], "o", color=PALETTE["primary"], ms=9)
axb.errorbar([2], [ps], yerr=[[ps - ci_ps[0]], [ci_ps[1] - ps]], fmt="o", color=PALETTE["primary"], ms=9, capsize=4)
axb.vlines(3, bounds[1.0], bounds[5.0], color=PALETTE["light"], lw=6)
axb.vlines(3, bounds[1.0], bounds[3.0], color=PALETTE["accent"], lw=6)
axb.text(3.12, (bounds[1.0] + bounds[3.0]) / 2, f"{bounds[1.0]:.2f}–{bounds[3.0]:.2f}\n(scenario)", va="center", fontsize=10, color="#333333")
axb.text(3.12, bounds[5.0], f"{bounds[5.0]:.2f} if\nscored 5", va="center", fontsize=9, color="#555555")
for i, v in enumerate(vals):
    axb.text(i + 0.12, v, f"{v:.2f}", va="center", fontsize=10, color="#333333")
axb.set_xticks([0, 1, 2, 3], ["Naive\nmean", "IPW", "Post-\nstratified", "Scenario:\nunresolved\nscored 1–3"], fontsize=9.5)
axb.set_xlim(-0.5, 3.9); axb.set_ylabel("Mean CSAT (1–5)")
axb.set_title("Re-weighting moves CSAT little;\nunresolved tickets are the blind spot", fontsize=11.5)
fig.suptitle(f"CSAT is missing not at random: naive {naive:.2f} describes resolved tickets only", x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.02)
save_fig(fig, "q7_06_csat_ipw")

# %% [markdown]
# ## 10 · Undocumented issues
# Physical plausibility (battery IDs, retirement, commissioning), metadata consistency (3W slots, firmware field),
# integrity checks (sign-up / commissioning / decommissioning dates), tickets, queue caps and energy accounting.

# %%
hdr("10. Undocumented issues")
# 10a battery IDs are not physically tracked
bp = (events(["event_id", "rider_id", "event_ts", "battery_in_id", "battery_out_id", "city", "is_completed"]).filter(pl.col("is_completed"))
      .select(as_i32("rider_id", 3).alias("r"), "event_ts", as_i32("battery_in_id", 7).alias("bi"), as_i32("battery_out_id", 7).alias("bo"),
              pl.col("city").cast(pl.Categorical).alias("c"), pl.col("event_id").str.slice(3).cast(pl.Int64).alias("eid")))
chain = (bp.sort("r", "event_ts", "eid").with_columns(pl.col("r").shift(1).alias("r_p"), pl.col("bo").shift(1).alias("bo_p"))
         .filter(pl.col("r") == pl.col("r_p")).select((pl.col("bi") == pl.col("bo_p")).mean().alias("match"), pl.len().alias("n")).collect().row(0, named=True))
packc = (bp.sort("bo", "event_ts", "eid")                    # event_id breaks timestamp ties -> deterministic pairs
         .with_columns(pl.col("bo").shift(1).alias("bo_p"), pl.col("c").shift(1).alias("c_p"), pl.col("event_ts").shift(1).alias("ts_p"))
         .filter(pl.col("bo") == pl.col("bo_p"))
         .with_columns((pl.col("event_ts") - pl.col("ts_p")).dt.total_seconds().alias("gs"))   # strict seconds (total_minutes truncates)
         .select(pl.len().alias("pairs"), (pl.col("c") != pl.col("c_p")).sum().alias("diff_city"),
                 ((pl.col("c") != pl.col("c_p")) & (pl.col("gs") <= 3600)).sum().alias("diff_city_60"),
                 ((pl.col("c") != pl.col("c_p")) & (pl.col("gs") <= 600)).sum().alias("diff_city_10"),
                 (pl.col("gs") <= 3600).sum().alias("any_60"), (pl.col("gs").median() / 60).alias("median_gap_min")).collect().row(0, named=True))
print(f"returned pack == pack issued to the same rider last time: {100 * chain['match']:.3f}% of {chain['n']:,} consecutive swaps (physical tracking would give ~100%)")
print(f"consecutive issues of the same pack: {packc['pairs']:,}; in a different city {packc['diff_city']:,} ({100 * packc['diff_city'] / packc['pairs']:.1f}%); "
      f"different city within 60 min {packc['diff_city_60']:,} ({100 * packc['diff_city_60'] / packc['pairs']:.1f}%), within 10 min {packc['diff_city_10']:,}; "
      f"re-issued within 60 min anywhere {packc['any_60']:,}; median gap {packc['median_gap_min']:.0f} min")
add_issue("Battery IDs not physically tracked (packs 'teleport' between cities)", "undocumented",
          "Same pack issued in 2 cities ≤60 min apart; returned pack vs last issued pack per rider",
          packc["diff_city_60"], packc["pairs"], "pack re-issues", "Per-pack SoH valid; per-pack location / rotation / turnaround NOT valid",
          f"{packc['diff_city_60']:,} re-issues of a pack in another city within 60 min ({packc['diff_city_10']:,} within 10 min); riders return "
          f"the pack they were last given only {100 * chain['match']:.2f}% of the time.",
          f"{packc['diff_city_60']:,} impossible cross-city re-issues ≤60 min", "high")

# 10b commissioning / manufacture dates vs usage
btx = bt.with_columns((pl.col("commission_date") - pl.col("first_seen").dt.date()).dt.total_days().alias("days_before_commission"),
                      (pl.col("first_seen").dt.date() < pl.col("manufacture_date")).alias("used_before_manufacture"))
ubc = btx.filter(pl.col("used_before_commission").fill_null(False))
ubc_sup = btx.group_by("supplier").agg(pl.len().alias("packs"), pl.col("used_before_commission").fill_null(False).mean().alias("share_used_before_commission"),
                                       pl.col("used_before_manufacture").fill_null(False).mean().alias("share_used_before_manufacture")).sort("supplier")
n_ubm = int(btx["used_before_manufacture"].fill_null(False).sum())
print(f"packs used before recorded commission_date: {ubc.height:,} of {bt.height:,} ({100 * ubc.height / bt.height:.1f}%); median {ubc['days_before_commission'].median():.0f} days early; "
      f"used before manufacture_date: {n_ubm:,}")
print(ubc_sup)
add_issue("Packs used before recorded commission date", "undocumented", "first_seen in swaps vs batteries.commission_date / manufacture_date",
          ubc.height, bt.height, "packs", "Age from first_seen; ignore commission_date",
          f"{100 * ubc.height / bt.height:.0f}% of packs used a median {ubc['days_before_commission'].median():.0f} days before commissioning "
          f"({n_ubm:,} even before manufacture) — commission-based age would be negative.",
          f"pack age from commission_date wrong for {100 * ubc.height / bt.height:.0f}% of packs", "medium")

# 10c retired packs still issued
ret = bt.filter(pl.col("retired_date").is_not_null()).select(as_i32("battery_id", 7).alias("bo"), "retired_date")
sub70 = (events(["battery_out_id", "date", "soh_out_pct", "supplier_out", "is_completed"]).filter(pl.col("is_completed"))
         .select(as_i32("battery_out_id", 7).alias("bo"), "date", "soh_out_pct", "supplier_out")
         .join(ret.lazy(), on="bo", how="left")
         .select((pl.col("date") > pl.col("retired_date")).sum().alias("after_retire"),
                 pl.col("bo").filter(pl.col("date") > pl.col("retired_date")).n_unique().alias("packs_after"),
                 pl.col("soh_out_pct").filter(pl.col("date") > pl.col("retired_date")).median().alias("soh_after"),
                 (pl.col("soh_out_pct") < EOL_SOH).sum().alias("issued_below_eol"),
                 pl.col("date").filter(pl.col("soh_out_pct") < EOL_SOH).min().alias("first_below_eol"),
                 (pl.col("supplier_out") == "Kyron").filter(pl.col("soh_out_pct") < EOL_SOH).mean().alias("kyron_share_below_eol"))
         .collect().row(0, named=True))
print(f"swaps issuing a pack AFTER its recorded retirement (2025-06-15): {sub70['after_retire']:,} ({sub70['packs_after']:,} packs, median SoH {sub70['soh_after']:.1f}%)")
print(f"swaps issuing a pack below 70% SoH (EOL): {sub70['issued_below_eol']:,} ({100 * sub70['issued_below_eol'] / n_comp:.2f}% of {n_comp:,} completed), "
      f"first on {sub70['first_below_eol']}, Kyron share {100 * sub70['kyron_share_below_eol']:.1f}%")
# Cost-model consequence: wear charged on packs that are already past the 70% end of life. Under the brief's usage-based
# book depreciation (70% EOL, no salvage) such a pack is fully written off, so this charge is a sensitivity, not a fact:
# a replacement-cost view would keep it. (Cross-question: Q1 cm1 trend, Q4 Kyron economics.)
wm = (events(["month", "is_completed", "soh_out_pct", "wear_cost", "cm1"]).filter(pl.col("is_completed"))
      .group_by("month").agg(pl.len().alias("swaps"), (pl.col("soh_out_pct") < EOL_SOH).sum().alias("sub70_swaps"),
                             pl.col("wear_cost").filter(pl.col("soh_out_pct") < EOL_SOH).sum().alias("wear_past_eol_inr"),
                             pl.col("cm1").sum().alias("cm1_inr"))
      .with_columns((pl.col("cm1_inr") / pl.col("swaps")).alias("cm1_per_swap"),
                    ((pl.col("cm1_inr") + pl.col("wear_past_eol_inr")) / pl.col("swaps")).alias("cm1_per_swap_no_wear_past_eol"),
                    (pl.col("wear_past_eol_inr") / pl.col("swaps")).alias("wear_past_eol_per_swap"))
      .collect().sort("month"))
wm.write_csv(TAB / "q7_wear_past_eol_sensitivity.csv")
w_tot = float(wm["wear_past_eol_inr"].sum()); w_n = int(wm["sub70_swaps"].sum())
w_mean = float(events(["is_completed", "soh_out_pct", "wear_cost"]).filter(pl.col("is_completed") & (pl.col("soh_out_pct") < EOL_SOH))
               .select(pl.col("wear_cost").mean()).collect().item())
wmd = {(r["month"].year, r["month"].month): r for r in wm.iter_rows(named=True)}
print(f"WEAR PAST EOL: the cost model charges INR {w_tot:,.0f} of wear (INR {w_mean:.1f}/swap) on the {w_n:,} swaps that issued a pack already below 70% SoH")
print(wm.filter(pl.col("month") >= dt.date(2025, 3, 1)).select("month", "swaps", "sub70_swaps", "wear_past_eol_inr", "wear_past_eol_per_swap", "cm1_per_swap",
                                                             "cm1_per_swap_no_wear_past_eol").to_pandas().round(2).to_string(index=False))
print(f"sensitivity (70%-EOL, no-salvage book depreciation -> no wear past EOL): cm1/swap Apr-25 {wmd[(2025, 4)]['cm1_per_swap']:+.1f} -> "
      f"{wmd[(2025, 4)]['cm1_per_swap_no_wear_past_eol']:+.1f}, May-25 {wmd[(2025, 5)]['cm1_per_swap']:+.1f} -> {wmd[(2025, 5)]['cm1_per_swap_no_wear_past_eol']:+.1f}, "
      f"Jun-25 {wmd[(2025, 6)]['cm1_per_swap']:+.1f} -> {wmd[(2025, 6)]['cm1_per_swap_no_wear_past_eol']:+.1f} INR; wear past EOL = INR "
      f"{wmd[(2025, 5)]['wear_past_eol_per_swap']:.1f} per May-25 swap and INR {wmd[(2025, 6)]['wear_past_eol_per_swap']:.1f} per Jun-25 swap")
add_issue("Packs below the 70% EOL SoH still issued (incl. 'retired' packs after 15-Jun-2025)", "undocumented",
          "SoH of issued packs vs 70% EOL; join issued pack to retired_date; wear cost charged on those swaps", sub70["issued_below_eol"], n_comp, "swaps",
          "Keep (real swaps); flag; treat retirement record as administrative; size wear-past-EOL as a cm1 sensitivity",
          f"{sub70['issued_below_eol']:,} swaps issued sub-70% packs (first {sub70['first_below_eol']}, all Kyron); {sub70['after_retire']:,} came after "
          f"retirement, using {sub70['packs_after']:,} 'retired' packs at median SoH {sub70['soh_after']:.1f}%. The cost model charges INR "
          f"{w_tot / 1e6:.1f}M of wear on them (INR {w_mean:.1f}/swap); without it cm1/swap would be {wmd[(2025, 5)]['cm1_per_swap_no_wear_past_eol']:+.1f} "
          f"(May-25) and {wmd[(2025, 6)]['cm1_per_swap_no_wear_past_eol']:+.1f} (Jun-25) instead of {wmd[(2025, 5)]['cm1_per_swap']:+.1f} / "
          f"{wmd[(2025, 6)]['cm1_per_swap']:+.1f}.",
          f"{sub70['issued_below_eol']:,} sub-EOL swaps; ₹{w_tot / 1e6:.1f}M wear charged past EOL", "high")

# 10d 3W swaps at stations with no 3W slots
s3 = st_all.filter(~pl.col("is_test")).select("station_id", "slots_3w")
e3 = (events(["station_id", "vehicle_class", "is_stockout"]).join(s3.lazy(), on="station_id")
      .group_by((pl.col("slots_3w") == 0).alias("no_3w_slots"), "vehicle_class")
      .agg(pl.len().alias("attempts"), pl.col("is_stockout").mean().alias("stockout_rate"), pl.col("station_id").n_unique().alias("stations"))
      .collect().sort("no_3w_slots", "vehicle_class"))
print(e3)
n3_bad = int(e3.filter(pl.col("no_3w_slots") & (pl.col("vehicle_class") == "3W"))["attempts"].item())
n_st0 = s3.filter(pl.col("slots_3w") == 0).height
sh3 = {r["no_3w_slots"]: r["attempts"] for r in e3.filter(pl.col("vehicle_class") == "3W").iter_rows(named=True)}
tot3 = {k: int(e3.filter(pl.col("no_3w_slots") == k)["attempts"].sum()) for k in (True, False)}
print(f"3W attempts at the {n_st0} stations with slots_3w = 0: {n3_bad:,} ({100 * n3_bad / N_EV:.1f}% of attempts); 3W share there "
      f"{100 * sh3[True] / tot3[True]:.1f}% vs {100 * sh3[False] / tot3[False]:.1f}% at stations with 3W slots")
add_issue("3W swaps at stations with zero 3W slots", "undocumented", "Event vehicle class vs stations.slots_3w",
          n3_bad, N_EV, "attempts", "Flag; do not use slots_3w / inventory_target_3w for 3W capacity",
          f"{n_st0} stations list 0 3W slots yet serve {n3_bad:,} 3W attempts (3W share {100 * sh3[True] / tot3[True]:.1f}% vs "
          f"{100 * sh3[False] / tot3[False]:.1f}% elsewhere, same stockout rate) — slot metadata does not constrain 3W activity; "
          f"swaps_per_slot_day and 3W capacity metrics are unreliable.",
          f"{n3_bad / 1e3:.0f}K 3W attempts at 0-slot stations; capacity metrics unreliable", "medium")

# 10e firmware field is a static station attribute, not the firmware at event time
fws = events(["station_id", "station_firmware", "date"]).group_by("station_id").agg(pl.col("station_firmware").n_unique().alias("n_fw")).collect()
pre_fw = events(["station_firmware", "date"]).filter((pl.col("station_firmware") == "v3.2.0") & (pl.col("date") < BUG_A)).select(pl.len()).collect().item()
print(f"stations whose events show >1 firmware value: {int((fws['n_fw'] > 1).sum())} of {fws.height}; v3.2.0-labelled attempts dated before the "
      f"v3.2.0 install (10-Mar-2025): {pre_fw:,}")
add_issue("station_firmware is static per station (not 'firmware at event time')", "undocumented",
          "Distinct firmware per station over time vs stations.firmware_updated_date", pre_fw, N_EV, "attempts",
          "Use only as station attribute; no pre/post firmware comparisons",
          f"All {fws.height} stations carry one value for 18 months; {pre_fw:,} attempts are labelled v3.2.0 before it was installed.",
          "firmware field cannot date firmware changes", "low")

# 10f integrity checks that PASS
integ = (events(["date", "signup_date", "commissioned_date", "station_id", "battery_in_id", "rider_vehicle_class"])
         .join(st_all.lazy().select("station_id", "decommissioned_date"), on="station_id", how="left")
         .select((pl.col("date") < pl.col("signup_date")).sum().alias("before_signup"),
                 (pl.col("date") < pl.col("commissioned_date")).sum().alias("before_commissioning"),
                 (pl.col("date") > pl.col("decommissioned_date")).sum().alias("after_decommissioning"),
                 (pl.col("battery_in_id").str.slice(4, 2) != pl.col("rider_vehicle_class")).sum().alias("pack_vs_rider_class_mismatch"))
         .collect().row(0, named=True))
print("integrity checks (violations):", integ)
add_issue("Integrity checks: events before sign-up / commissioning, after decommissioning, pack-vs-vehicle class", "check passed",
          "Date and class cross-checks on every attempt", sum(integ.values()), N_EV, "attempts", "No action",
          "0 violations in all four checks — rider/station timelines are consistent.", "0 violations (4 checks)", "pass")

# 10g tickets
rdt = riders().select("rider_id", "signup_date", "first_attempt_ts")
tki = tk.join(rdt, on="rider_id", how="left")
t_out, t_test = int((~tk["in_window"]).sum()), int(tk["is_test"].sum())
t_bs = int((tki["created_date"] < tki["signup_date"]).sum()); t_bf = int((tki["created_ts"] < tki["first_attempt_ts"]).sum())
t_other = int((tk["category"] == "other").sum())
other_cm = tk.filter(pl.col("category") == "other")["rider_comment"].unique().sort().to_list()
print(f"tickets: outside window {t_out}, referencing test stations {t_test}, created before rider sign-up {t_bs}, before rider's first attempt {t_bf}")
print(f"category 'other' = {t_other:,} tickets ({100 * t_other / tk.height:.1f}%), consisting only of {len(other_cm)} vehicle-blaming templates: {other_cm}")
add_issue("Tickets outside window / at test stations / before sign-up", "undocumented", "created_ts vs window, station is_test, rider signup",
          t_out + t_test + t_bs, tk.height, "tickets", "Exclude from in-window ticket KPIs (flags in_window, is_test)",
          f"{t_out} outside Jan-24..Jun-25, {t_test} at test stations, {t_bs} before the rider signed up ({t_bf} before first attempt).",
          f"{t_out + t_test + t_bs} tickets to exclude/flag", "low")
add_issue("Ticket category 'other' hides vehicle/battery complaints", "documented", "Distinct comment templates per category",
          t_other, tk.height, "tickets", "Re-classify via comment text (ticket-text analysis)",
          f"All {t_other:,} 'other' tickets are 3 vehicle-blaming templates ('{other_cm[0]}', ...) — likely range/battery issues mis-filed.",
          f"{100 * t_other / tk.height:.0f}% of tickets in an uninformative 'other' bucket", "medium")

# 10h queue-wait caps
qc = (events(["event_type", "queue_wait_sec"]).group_by("event_type")
      .agg(pl.col("queue_wait_sec").max().alias("max_s"), (pl.col("queue_wait_sec") == pl.col("queue_wait_sec").max()).sum().alias("at_max"),
           pl.len().alias("n"), pl.col("queue_wait_sec").quantile(0.99).alias("p99"), pl.col("queue_wait_sec").mean().alias("mean"))
      .collect().sort("event_type"))
print(qc)
n_cap = int(qc.filter(pl.col("event_type").is_in(["swap_completed", "abandoned_queue"]))["at_max"].sum())
cap_c = qc.filter(pl.col("event_type") == "swap_completed").row(0, named=True); cap_a = qc.filter(pl.col("event_type") == "abandoned_queue").row(0, named=True)
add_issue("queue_wait_sec hard caps (1,800 s completed / 2,400 s abandoned)", "undocumented", "Point mass at the maximum by event type",
          n_cap, N_EV, "attempts", "Keep; treat as right-censored; report medians/p90",
          f"{cap_c['at_max']} completed swaps at exactly {cap_c['max_s']} s and {cap_a['at_max']} abandons at {cap_a['max_s']} s; p99 waits "
          f"({cap_c['p99']:.0f}/{cap_a['p99']:.0f} s) sit below the caps, so means/p90 are unaffected.", "negligible (censoring only)", "low")

# 10i energy accounting
en = (events(["energy_to_recharge_kwh", "soc_in_pct", "soh_in_pct", "battery_in_id", "is_completed", "charger_generation"])
      .filter(pl.col("is_completed"))
      .with_columns(pl.when(pl.col("battery_in_id").str.slice(4, 2) == "3W").then(4.8).otherwise(2.1).alias("cap"))
      .with_columns(((100 - pl.col("soc_in_pct")) / 100 * pl.col("cap") * pl.col("soh_in_pct") / 100).alias("need"))
      .with_columns((pl.col("energy_to_recharge_kwh") / pl.col("need")).alias("k")))
ek = en.select(pl.col("k").quantile(0.01).alias("p01"), pl.col("k").median().alias("median"), pl.col("k").quantile(0.99).alias("p99"),
               ((pl.col("k") < 0.7) | (pl.col("k") > 1.6)).sum().alias("implausible"), pl.corr("energy_to_recharge_kwh", "need").alias("r"),
               pl.len().alias("n")).collect().row(0, named=True)
ekg = en.group_by("charger_generation").agg(pl.col("k").median().alias("grid_kwh_per_kwh_needed")).collect().sort("charger_generation")
egd = {r["charger_generation"]: r["grid_kwh_per_kwh_needed"] for r in ekg.iter_rows(named=True)}
print(f"grid kWh per kWh needed to refill to 100% (= (100-SoC_in) x capacity x SoH): p01 {ek['p01']:.3f}, median {ek['median']:.3f}, p99 {ek['p99']:.3f}; "
      f"rows outside 0.7-1.6 (implausible) {ek['implausible']:,}; corr(energy, need) = {ek['r']:.3f}")
print(ekg)
print(f"Gen1 chargers draw {100 * (egd['Gen1'] / egd['Gen3'] - 1):.1f}% more grid energy per kWh delivered than Gen3")
add_issue("Energy-to-recharge vs SoC/SoH/capacity consistency", "check passed", "Ratio of grid kWh to kWh needed per swap; by charger generation",
          ek["implausible"], ek["n"], "swaps", "No action (energy cost per swap trustworthy)",
          f"r = {ek['r']:.3f}; median {ek['median']:.2f} grid kWh per kWh needed (p01–p99 {ek['p01']:.2f}–{ek['p99']:.2f}); {ek['implausible']:,} rows outside 0.7–1.6. "
          f"Efficiency rises with charger generation (Gen1 {egd['Gen1']:.3f}, Gen2 {egd['Gen2']:.3f}, Gen3 {egd['Gen3']:.3f}).",
          f"consistent (r = {ek['r']:.2f}); Gen1 uses {100 * (egd['Gen1'] / egd['Gen3'] - 1):.0f}% more energy than Gen3", "pass")

# 10j KYC
rk = riders().select("rider_id", "kyc_verified", "n_attempts", "n_swaps")
kyc_r = int((~rk["kyc_verified"]).sum()); kyc_ev = int(rk.filter(~pl.col("kyc_verified"))["n_attempts"].fill_null(0).sum())
kyc_sw = int(rk.filter(~pl.col("kyc_verified"))["n_swaps"].fill_null(0).sum())
print(f"KYC-unverified riders: {kyc_r} riders with {kyc_ev:,} attempts / {kyc_sw:,} completed swaps ({100 * kyc_ev / N_EV:.1f}% of attempts)")
add_issue("Riders without KYC verification swapping", "undocumented", "riders.kyc_verified vs activity", kyc_ev, N_EV, "attempts",
          "Keep; flag for compliance", f"{kyc_r} unverified riders made {kyc_sw:,} completed swaps.", "compliance flag, no KPI effect", "low")

# %% [markdown]
# ## 11 · Anomalies
# ### 11a Stations with extreme service-failure rates — funnel plot controlled for city x month x vehicle class
# Expected failures for each station = Σ attempts x peer failure rate of the same city, month and vehicle class
# (peers exclude the station itself), i.e. indirect standardisation; this removes seasonality, city mix and the timing of
# commissioning (Wave-2 sites only lived through the hot months). Limits are widened for over-dispersion
# (Spiegelhalter 2005: φ from 10%-winsorised z-scores). Stage 2 asks whether the excess spread is explained by station
# design. Q2/Q3 show Gen1 chargers derate in heat, so the design effect may be heat-conditional: we fit the additive
# design model (spec A) and one with a charger-generation x hot-city interaction (spec B; hot = Delhi NCR, Jaipur,
# Hyderabad), and only call a station anomalous if it sits outside the limits of the better-specified model.

# %%
hdr("11a. Station failure-rate funnel")
smv = (events(["station_id", "city", "month", "vehicle_class", "is_service_failure"])
       .group_by("station_id", "city", "month", "vehicle_class").agg(pl.len().alias("att"), pl.col("is_service_failure").sum().alias("fail")).collect())
cellv = smv.group_by("city", "month", "vehicle_class").agg(pl.col("att").sum().alias("c_att"), pl.col("fail").sum().alias("c_fail"))
smv = (smv.join(cellv, on=["city", "month", "vehicle_class"])
       .with_columns(((pl.col("c_fail") - pl.col("fail")) / (pl.col("c_att") - pl.col("att"))).fill_nan(None).alias("peer"))
       .with_columns((pl.col("att") * pl.col("peer")).alias("exp")))
fun = (smv.group_by("station_id").agg(pl.col("att").sum().alias("attempts"), pl.col("fail").sum().alias("observed"), pl.col("exp").sum().alias("expected"))
       .with_columns((pl.col("observed") / pl.col("expected")).alias("smr"), ((pl.col("observed") - pl.col("expected")) / pl.col("expected").sqrt()).alias("z")))
zz = fun["z"].to_numpy(); lo, hi = np.percentile(zz, [10, 90])
phi = float(np.mean(np.clip(zz, lo, hi) ** 2))
fun = fun.with_columns((pl.col("z") / np.sqrt(max(phi, 1.0))).alias("z_adj"))
sm_util = station_month().group_by("station_id").agg(pl.col("swaps_per_slot_day").mean().alias("swaps_per_slot_day"))
fun = (fun.join(st_all.select("station_id", "city", "expansion_wave", "charger_generation", "location_type", "host_type", "connectivity_tier",
                              "slots_2w", "slots_3w", "inventory_target_2w", "competitor_within_1_5km_since", "firmware_version"), on="station_id", how="left")
       .join(sm_util, on="station_id", how="left")
       .with_columns(pl.when(pl.col("z_adj") > 3.09).then(pl.lit("high outlier")).when(pl.col("z_adj") < -3.09).then(pl.lit("low outlier"))
                     .otherwise(pl.lit("within limits")).alias("flag"))
       .sort("z_adj", descending=True))
n_hi_raw, n_lo_raw = int((fun["z"] > 3.09).sum()), int((fun["z"] < -3.09).sum())
n_hi, n_lo = int((fun["flag"] == "high outlier").sum()), int((fun["flag"] == "low outlier").sum())
rho = stats.spearmanr(fun["z_adj"].to_numpy(), fun["swaps_per_slot_day"].to_numpy(), nan_policy="omit")
print(f"stations {fun.height}; overdispersion phi = {phi:.2f}; beyond 99.8% limits: Poisson {n_hi_raw} high / {n_lo_raw} low; "
      f"after overdispersion {n_hi} high / {n_lo} low")
print(f"Spearman rho(z_adj, swaps per slot-day) = {rho.statistic:.2f} (p = {rho.pvalue:.2g})")
show = ["station_id", "city", "expansion_wave", "charger_generation", "location_type", "connectivity_tier", "inventory_target_2w",
        "swaps_per_slot_day", "attempts", "observed", "expected", "smr", "z_adj"]
print(fun.filter(pl.col("flag") != "within limits").select(show).to_pandas().round(2).to_string(index=False))
for col in ["charger_generation", "expansion_wave", "location_type", "connectivity_tier"]:
    t_ = (fun.group_by(col).agg(pl.len().alias("stations"), (pl.col("flag") == "high outlier").sum().alias("high"),
                                (pl.col("observed").sum() / pl.col("expected").sum()).alias("pooled_smr")).sort(col))
    print(t_.to_pandas().round(3).to_string(index=False))

# Is the generation effect constant, or does it exist only under heat? (Q2/Q3: Gen1 derates above ~41 C cabinet temp.)
# Stage-1 pooled O/E by hot city x Apr-Jun x charger generation.
HOT = ["Delhi NCR", "Jaipur", "Hyderabad"]               # the three cities with the Apr-Jun heat regime (Q2/Q3)
GENS = ["Gen1", "Gen2", "Gen3"]
smv_g = (smv.join(st_all.select("station_id", "charger_generation"), on="station_id", how="left")
         .with_columns(pl.col("city").is_in(HOT).alias("hot_city"), pl.col("month").dt.month().is_in([4, 5, 6]).alias("apr_jun")))
oe_ctx = (smv_g.group_by("hot_city", "apr_jun", "charger_generation")
          .agg((pl.col("fail").sum() / pl.col("exp").sum()).alias("pooled_oe"), pl.col("att").sum().alias("attempts"),
               pl.col("station_id").n_unique().alias("stations"))
          .sort("hot_city", "apr_jun", "charger_generation"))
oe_ctx.write_csv(TAB / "q7_station_oe_heat_generation.csv")
oed = {(r["hot_city"], r["apr_jun"], r["charger_generation"]): r["pooled_oe"] for r in oe_ctx.iter_rows(named=True)}
oe_out = [v for (h_, s_, g_), v in oed.items() if not s_]
print("stage-1 pooled O/E by hot city x Apr-Jun x charger generation:")
print(oe_ctx.pivot(on="charger_generation", index=["hot_city", "apr_jun"], values="pooled_oe").sort("hot_city", "apr_jun")
      .to_pandas().round(3).to_string(index=False))
print(f"outside Apr-Jun every generation sits at O/E {min(oe_out):.2f}-{max(oe_out):.2f}; Apr-Jun hot cities Gen1/Gen2/Gen3 = "
      f"{oed[(True, True, 'Gen1')]:.2f}/{oed[(True, True, 'Gen2')]:.2f}/{oed[(True, True, 'Gen3')]:.2f}; Apr-Jun cool cities = "
      f"{oed[(False, True, 'Gen1')]:.2f}/{oed[(False, True, 'Gen2')]:.2f}/{oed[(False, True, 'Gen3')]:.2f}")

# Stage 2: quasi-Poisson GLM with the stage-1 expectation as offset + station design & load.
# Spec A (original): additive design terms. Spec B: + charger generation x hot city (the heat-conditional design effect).
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

fd = fun.to_pandas()
fd["hot"] = fd["city"].isin(HOT)
fd["log_util"] = np.log(fd["swaps_per_slot_day"]); fd["log_E"] = np.log(fd["expected"])
apd = station_month().group_by("station_id").agg((pl.col("attempts").sum() / pl.col("active_days").sum()).alias("apd"))
fd["log_apd"] = np.log(fd["station_id"].map(dict(zip(apd["station_id"].to_list(), apd["apd"].to_list()))))
SPEC_A = "observed ~ C(charger_generation) + C(expansion_wave) + C(location_type) + C(connectivity_tier) + log_util"
SPEC_B = "observed ~ C(charger_generation) * hot + C(expansion_wave) + C(location_type) + C(connectivity_tier) + log_util"


def fit_stage2(spec):
    m = smf.glm(spec, data=fd, family=sm.families.Poisson(), offset=fd["log_E"]).fit(scale="X2")
    mu_ = m.fittedvalues
    return m, mu_, (fd["observed"] - mu_) / np.sqrt(mu_ * m.scale)


def describe_fit(lab, m, mu_, z_):
    out = dict(model=lab, scale=float(m.scale), dev_expl=100 * (1 - m.deviance / m.null_deviance),
               band_pct=100 * 3.09 * np.sqrt(m.scale / np.median(mu_)), outside=fd.loc[np.abs(z_) > 3.09, "station_id"].tolist())
    print(f"{lab}: residual scale {out['scale']:.2f}; deviance explained {out['dev_expl']:.1f}%; 99.8% band at the median station "
          f"+/-{out['band_pct']:.0f}%; stations outside: {len(out['outside'])} {out['outside']}")
    return out


glmA, muA, zA = fit_stage2(SPEC_A)
glmB, muB, zB = fit_stage2(SPEC_B)
glmC, muC, zC = fit_stage2(SPEC_B.replace("log_util", "log_apd"))
fitA = describe_fit("stage 2 spec A (design + load, no heat term)", glmA, muA, zA)
fitB = describe_fit("stage 2 spec B (+ generation x hot city)     ", glmB, muB, zB)
fitC = describe_fit("robustness: spec B with attempts per active day as load (no slot metadata)", glmC, muC, zC)
smallA = (muA < 300).to_numpy()
n_small, n_small_w2 = int(smallA.sum()), int((fd.loc[smallA, "expansion_wave"] == "Wave2_2025H1").sum())
n_small_cool, oe_smallA = int((~fd.loc[smallA, "hot"]).sum()), float(fd.loc[smallA, "observed"].sum() / muA[smallA].sum())
oe_smallB = float(fd.loc[smallA, "observed"].sum() / muB[smallA].sum())
print(f"spec A leaves {n_small} small stations (fitted failures < 300; {n_small_w2} Wave2, {n_small_cool} in cool cities) at pooled "
      f"observed/fit {oe_smallA:.2f}; spec B fits the same stations at {oe_smallB:.2f}")
xt = pd.crosstab(fd["expansion_wave"], fd["charger_generation"])
print("stations by wave x generation (Wave2 is all Gen3; only the Wave1 Gen3 sites separate the Gen3 and Wave2 terms):")
print(xt.to_string())


def rr_table(m, lab):
    ci = m.conf_int()
    return pd.DataFrame({"model": lab, "term": m.params.index, "rate_ratio": np.exp(m.params.values), "p": m.pvalues.values,
                         "ci_low": np.exp(ci[0].values), "ci_high": np.exp(ci[1].values)})


rr_tab = pd.concat([rr_table(glmA, "A design + load"), rr_table(glmB, "B + generation x hot city"),
                    rr_table(glmC, "C = B with attempts/day load")], ignore_index=True)
rr_tab.to_csv(TAB / "q7_station_failure_glm.csv", index=False)
print(rr_tab.round(3).to_string(index=False))
pB = glmB.params
gen_rr = {(g_, h_): float(np.exp(pB.get(f"C(charger_generation)[T.{g_}]", 0.0) + (pB.get(f"C(charger_generation)[T.{g_}]:hot[T.True]", 0.0) if h_ else 0.0)))
          for g_ in ["Gen2", "Gen3"] for h_ in (False, True)}
print(f"spec B rate ratios vs Gen1 (same city-month peers): cool cities Gen2 {gen_rr[('Gen2', False)]:.2f}, Gen3 {gen_rr[('Gen3', False)]:.2f}; "
      f"hot cities Gen2 {gen_rr[('Gen2', True)]:.2f}, Gen3 {gen_rr[('Gen3', True)]:.2f}")
rrA = rr_tab[rr_tab.model == "A design + load"].set_index("term")
print(f"spec A (heat-averaged) rate ratios: Gen3 {rrA.loc['C(charger_generation)[T.Gen3]', 'rate_ratio']:.2f} "
      f"({rrA.loc['C(charger_generation)[T.Gen3]', 'ci_low']:.2f}-{rrA.loc['C(charger_generation)[T.Gen3]', 'ci_high']:.2f}), "
      f"Wave2 {rrA.loc['C(expansion_wave)[T.Wave2_2025H1]', 'rate_ratio']:.2f} "
      f"({rrA.loc['C(expansion_wave)[T.Wave2_2025H1]', 'ci_low']:.2f}-{rrA.loc['C(expansion_wave)[T.Wave2_2025H1]', 'ci_high']:.2f})")

# primary stage-2 model = spec B
glm, scale2 = glmB, float(glmB.scale)
fd["mu_A"], fd["z_A"] = muA, zA
fd["mu"] = muB
fd["ratio_struct"] = fd["observed"] / fd["mu"]
fd["z_struct"] = zB
fd["flag_struct"] = np.where(fd["z_struct"] > 3.09, "high anomaly", np.where(fd["z_struct"] < -3.09, "low anomaly", "within limits"))
fd.sort_values("z_struct", ascending=False).to_csv(TAB / "q7_station_failure_funnel.csv", index=False)
n_hi2, n_lo2 = int((fd["flag_struct"] == "high anomaly").sum()), int((fd["flag_struct"] == "low anomaly").sum())
dev_share = 1 - glmB.deviance / glmB.null_deviance
print(f"stage 2 (spec B): residual scale {scale2:.2f} (vs {phi:.1f} at stage 1 and {fitA['scale']:.2f} under spec A); deviance explained "
      f"{100 * dev_share:.1f}%; anomalies beyond 99.8%: {n_hi2} high / {n_lo2} low")
fd["mu_C"], fd["z_C"] = muC, zC
anom = fd.loc[fd.flag_struct != "within limits"]
print(anom[["station_id", "city", "expansion_wave", "charger_generation", "location_type", "host_type", "connectivity_tier",
            "swaps_per_slot_day", "observed", "expected", "mu", "ratio_struct", "z_struct", "z_A", "z_C"]].round(2).to_string(index=False))
cellfit = (fd.groupby(["expansion_wave", "charger_generation", "hot"])
           .apply(lambda d: pd.Series({"stations": len(d), "obs_over_fit_B": d["observed"].sum() / d["mu"].sum(),
                                       "obs_over_fit_A": d["observed"].sum() / d["mu_A"].sum()}), include_groups=False).reset_index())
print("pooled observed / fit by wave x generation x hot city (a well-specified model gives ~1.00 in every cell):")
print(cellfit.round(3).to_string(index=False))
for sid in anom["station_id"]:
    s_ = (smv_g.filter(pl.col("station_id") == sid).group_by("apr_jun")
          .agg((pl.col("fail").sum() / pl.col("exp").sum()).alias("oe"), pl.col("att").sum().alias("attempts")).sort("apr_jun"))
    peers_ = oe_ctx.filter(pl.col("hot_city") == bool(anom.loc[anom.station_id == sid, "hot"].iloc[0]))
    g_sid = anom.loc[anom.station_id == sid, "charger_generation"].iloc[0]
    print(f"{sid}: stage-1 O/E Jul-Mar {s_.filter(~pl.col('apr_jun'))['oe'].item():.2f}, Apr-Jun {s_.filter(pl.col('apr_jun'))['oe'].item():.2f} "
          f"(its generation x city-heat peers: Jul-Mar {peers_.filter(~pl.col('apr_jun') & (pl.col('charger_generation') == g_sid))['pooled_oe'].item():.2f}, "
          f"Apr-Jun {peers_.filter(pl.col('apr_jun') & (pl.col('charger_generation') == g_sid))['pooled_oe'].item():.2f})")

GEN_COLORS = {"Gen1": SERIES[0], "Gen2": SERIES[3], "Gen3": SERIES[2]}
fig, (axL, axM, axR) = plt.subplots(1, 3, figsize=(22, 6.8), gridspec_kw={"width_ratios": [1.2, 1.05, 1.2], "wspace": 0.24})
ex = np.linspace(fd["expected"].min() * 0.8, fd["expected"].max() * 1.15, 300)
for zc, lw_ in [(1.96, 1.0), (3.09, 1.6)]:
    axL.plot(ex, 1 + zc / np.sqrt(ex), color=PALETTE["muted"], lw=lw_); axL.plot(ex, 1 - zc / np.sqrt(ex), color=PALETTE["muted"], lw=lw_)
axL.text(ex[0], 1 + 3.09 / np.sqrt(ex[0]) + 0.03, "99.8% chance limit", ha="left", va="bottom", fontsize=9.5, color="#555555")
axL.axhline(1, color="#333333", lw=1)
for gname, gc in GEN_COLORS.items():
    d_ = fd[fd["charger_generation"] == gname]
    axL.scatter(d_["expected"], d_["smr"], s=34, color=gc, alpha=0.85, edgecolor="white", linewidth=0.5,
                label=f"{gname} (pooled O/E {d_['observed'].sum() / d_['expected'].sum():.2f})", zorder=3)
axL.set_xscale("log"); axL.set_ylim(0.2, max(2.2, fd["smr"].max() * 1.08))
axL.set_xlabel("Expected failures from city × month × vehicle-class peers (log)")
axL.set_ylabel("Observed / expected service failures")
axL.legend(loc="upper right", title="charger generation", fontsize=9.5, title_fontsize=9.5)
axL.set_title(f"1 · Controlled for city × month × vehicle class: {n_hi_raw} high / {n_lo_raw} low\nbeyond chance (over-dispersion φ = {phi:.0f})",
              fontsize=11)
ctx_lab = [("Jul–Mar\ncool cities", False, False), ("Jul–Mar\nhot cities", True, False), ("Apr–Jun\ncool cities", False, True),
           ("Apr–Jun\nhot cities", True, True)]
xc, bw = np.arange(len(ctx_lab)), 0.26
for j, g_ in enumerate(GENS):
    vals_ = [oed.get((h_, s_, g_), np.nan) for _, h_, s_ in ctx_lab]
    bars_ = axM.bar(xc + (j - 1) * bw, vals_, width=bw, color=GEN_COLORS[g_], label=g_)
    for b_, v_ in zip(bars_, vals_):
        axM.text(b_.get_x() + b_.get_width() / 2, v_ + 0.02, f"{v_:.2f}", ha="center", va="bottom", fontsize=8.3, color="#333333")
axM.axhline(1, color="#333333", lw=1)
axM.set_xticks(xc, [c_[0] for c_ in ctx_lab], fontsize=9.5)
axM.set_ylabel("Pooled observed / expected failures")
axM.set_ylim(0, max(float(oe_ctx["pooled_oe"].max()) * 1.25, 1.45))
axM.legend(loc="upper left", title="charger generation", fontsize=9.5, title_fontsize=9.5, ncol=3)
axM.set_title("2 · The generation gap appears only under heat\n(hot = Delhi NCR, Jaipur, Hyderabad)", fontsize=11)
axM.grid(axis="x", visible=False)
ex2 = np.linspace(min(fd["mu"].min(), fd["mu_A"].min()) * 0.8, fd["mu"].max() * 1.15, 300)
axR.plot(ex2, 1 + 3.09 * np.sqrt(glmA.scale / ex2), color=PALETTE["muted"], lw=1.2, ls="--",
         label=f"old 99.8% limit, no heat term (±{fitA['band_pct']:.0f}% at median)")
axR.plot(ex2, 1 - 3.09 * np.sqrt(glmA.scale / ex2), color=PALETTE["muted"], lw=1.2, ls="--")
for zc, lw_ in [(1.96, 1.0), (3.09, 1.6)]:
    axR.plot(ex2, 1 + zc * np.sqrt(scale2 / ex2), color=PALETTE["muted"], lw=lw_,
             label=f"99.8% limit with heat × generation (±{fitB['band_pct']:.0f}% at median)" if zc > 3 else None)
    axR.plot(ex2, 1 - zc * np.sqrt(scale2 / ex2), color=PALETTE["muted"], lw=lw_)
axR.axhline(1, color="#333333", lw=1)
for k_, (gname, gc) in enumerate(GEN_COLORS.items()):
    d_ = fd[(fd["charger_generation"] == gname) & (fd.flag_struct == "within limits")]
    axR.scatter(d_["mu"], d_["ratio_struct"], s=30, color=gc, alpha=0.6, edgecolor="white", linewidth=0.5, zorder=3,
                label=f"{int((fd.flag_struct == 'within limits').sum())} stations within limits (colour = generation)" if k_ == 0 else None)
for fl_name, col_ in [("high anomaly", PALETTE["accent"]), ("low anomaly", PALETTE["good"])]:
    d_ = fd[fd.flag_struct == fl_name]
    if len(d_):
        axR.scatter(d_["mu"], d_["ratio_struct"], s=80, color=col_, edgecolor="#333333", linewidth=0.9, label=f"{fl_name} ({len(d_)})", zorder=4)
    for _, r in d_.iterrows():
        extra = "; borderline: inside\nthe limits with a slot-free load measure" if abs(r["z_C"]) <= 3.09 else ""
        axR.annotate(f"{r['station_id']} ({r['charger_generation']}, {r['expansion_wave'].split('_')[0]}, {r['location_type'].replace('_', ' ')}):\n"
                     f"{r['ratio_struct']:.2f}x its fit, z = {r['z_struct']:.2f}{extra}", (r["mu"], r["ratio_struct"]),
                     xytext=(300, 1.86 if fl_name.startswith("high") else 0.3), textcoords="data", va="bottom", fontsize=9.5, color="#222222",
                     arrowprops=dict(arrowstyle="-", color="#777777", lw=0.8))
top_ex = fd.reindex(fd["z_struct"].abs().sort_values(ascending=False).index).head(4)
print("most extreme stations under spec B:")
print(top_ex[["station_id", "city", "expansion_wave", "charger_generation", "location_type", "observed", "mu", "ratio_struct", "z_struct"]].round(2).to_string(index=False))
axR.set_xscale("log"); axR.set_ylim(axL.get_ylim())
axR.set_xlabel("Expected failures after design, load and generation × hot-city terms (log)")
axR.set_ylabel("Observed / model-adjusted expected")
axR.legend(loc="lower right", fontsize=9.5, frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
axR.set_title(f"3 · Adding generation × hot city: {n_hi2 + n_lo2} station{'s' if n_hi2 + n_lo2 != 1 else ''} outside the 99.8% limits\n(residual dispersion {fitA['scale']:.1f} → "
              f"{scale2:.1f}; deviance explained {fitA['dev_expl']:.0f}% → {100 * dev_share:.0f}%)", fontsize=11)
fig.suptitle(f"Station failure gaps are heat × charger generation, not rogue sites: in hot-city Apr–Jun Gen1 runs {oed[(True, True, 'Gen1')]:.2f}x "
             f"its peers and Gen3 {oed[(True, True, 'Gen3')]:.2f}x; " + (f"only {', '.join(anom['station_id'])} stays anomalous" if len(anom) else "no station stays anomalous"),
             x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.02)
save_fig(fig, "q7_03_station_failure_funnel")

# %% [markdown]
# ### 11b Battery cohorts — manufacturing lots with unusual degradation; does BMS firmware matter?
# Per-pack SoH loss per cycle (from clean.py's per-pack OLS slope, packs with ≥50 cycles) compared with the pack's own
# supplier x pack-type peers (Mann–Whitney, Benjamini–Hochberg). Lot codes are also checked against usage dates.

# %%
hdr("11b. Battery lots & BMS firmware")
bl = bt.filter((pl.col("n_cycles") >= 50) & pl.col("soh_loss_per_cycle").is_not_null()).with_columns(
    pl.col("manufacturing_lot").str.slice(3, 4).alias("lot_yymm"))
lot_rows = []
for (sup, pt), grp in bl.group_by(["supplier", "pack_type"]):
    for lot, lg in grp.group_by("manufacturing_lot"):
        lot = lot[0] if isinstance(lot, tuple) else lot
        rest = grp.filter(pl.col("manufacturing_lot") != lot)["soh_loss_per_cycle"].to_numpy()
        x_ = lg["soh_loss_per_cycle"].to_numpy()
        p_ = stats.mannwhitneyu(x_, rest).pvalue if len(x_) >= 5 and len(rest) >= 5 else np.nan
        lot_rows.append(dict(supplier=sup, pack_type=pt, lot=lot, packs=len(x_), lot_median=float(np.median(x_)), peer_median=float(np.median(rest)),
                             rel_diff_pct=100 * (np.median(x_) / np.median(rest) - 1), p=p_,
                             first_seen=lg["first_seen"].min(), lot_code_month=lg["lot_yymm"][0]))
lots = pd.DataFrame(lot_rows).sort_values(["supplier", "pack_type", "lot"])
pv = lots["p"].to_numpy(); ok = ~np.isnan(pv)
q = np.full(len(pv), np.nan)
if ok.sum():
    order = np.argsort(pv[ok]); m_ = ok.sum(); ranked = pv[ok][order] * m_ / (np.arange(m_) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]; tmp = np.empty(m_); tmp[order] = np.minimum(ranked, 1); q[ok] = tmp
lots["q_bh"] = q
lots["flag"] = np.where((lots["q_bh"] < 0.01) & (lots["rel_diff_pct"].abs() > 10), "unusual", "")
lots.to_csv(TAB / "q7_battery_lot_anomalies.csv", index=False)
print(lots.round(4).to_string(index=False))
n_lot_flag = int((lots["flag"] == "unusual").sum())
lot_dt = (bt.with_columns(pl.col("manufacturing_lot").str.slice(3, 2).cast(pl.Int32).alias("yy"), pl.col("manufacturing_lot").str.slice(5, 2).cast(pl.Int32).alias("mm"))
          .with_columns(pl.when(pl.col("mm").is_between(1, 12)).then(pl.date(2000 + pl.col("yy"), pl.col("mm").clip(1, 12), 1)).alias("lot_month")))
n_lot_bad = int(lot_dt.filter(pl.col("lot_month").is_null())["manufacturing_lot"].n_unique())
bad_lots = sorted(lot_dt.filter(pl.col("lot_month").is_null())["manufacturing_lot"].unique().to_list())
lv = lot_dt.filter(pl.col("lot_month").is_not_null() & pl.col("first_seen").is_not_null())
lot_before = float(lv.select((pl.col("first_seen").dt.date() < pl.col("lot_month")).mean()).item())
mfg_match = float(lv.select((pl.col("manufacture_date").dt.truncate("1mo") == pl.col("lot_month")).mean()).item())
ret_lot = (bt.filter(pl.col("supplier") == "Kyron").group_by("pack_type", "manufacturing_lot")
           .agg(pl.len().alias("packs"), pl.col("retired_date").is_not_null().sum().alias("retired"), pl.col("current_soh_pct").median().alias("soh_now"))
           .sort("pack_type", "manufacturing_lot"))
print(f"lots flagged unusual vs own supplier x pack peers (BH q<0.01 and |diff|>10%): {n_lot_flag} of {len(lots)}: "
      f"{lots.loc[lots.flag == 'unusual', 'lot'].tolist()} ({lots.loc[lots.flag == 'unusual', 'pack_type'].tolist()})")
print(f"lot codes that are not a valid yymm: {n_lot_bad} ({bad_lots}); for valid codes, code month == manufacture month for "
      f"{100 * mfg_match:.1f}% of packs and {100 * lot_before:.1f}% of packs were first used before their lot-code month")
print("Kyron lots — packs, retired on 15-Jun-2025, median current SoH:"); print(ret_lot)
fl_lots = lots.loc[lots.flag == "unusual", "lot"].tolist()
still = bt.filter((pl.col("supplier") == "Kyron") & (pl.col("pack_type") == "3W_4.8kWh") & pl.col("manufacturing_lot").is_in(fl_lots) & pl.col("retired_date").is_null())
print(f"packs from the fast-degrading Kyron 3W lots still in service on 30-Jun-2025: {still.height} (current SoH {still['current_soh_pct'].min():.1f}-"
      f"{still['current_soh_pct'].max():.1f}%, median loss/cycle {still['soh_loss_per_cycle'].median():.4f})")
bms_rows = []
for (sup, pt), grp in bl.group_by(["supplier", "pack_type"]):
    arrs = [g_["soh_loss_per_cycle"].to_numpy() for _, g_ in grp.group_by("bms_firmware")]
    meds = {k_[0] if isinstance(k_, tuple) else k_: float(np.median(g_["soh_loss_per_cycle"].to_numpy())) for k_, g_ in grp.group_by("bms_firmware")}
    kw = stats.kruskal(*arrs)
    bms_rows.append(dict(supplier=sup, pack_type=pt, packs=grp.height, kruskal_p=kw.pvalue, **{f"median_{k}": v for k, v in sorted(meds.items())}))
bms = pd.DataFrame(bms_rows).sort_values(["pack_type", "supplier"])
bms.to_csv(TAB / "q7_bms_firmware_test.csv", index=False)
print(bms.round(4).to_string(index=False))
kyr = bl.filter((pl.col("supplier") == "Kyron") & (pl.col("pack_type") == "2W_2.1kWh"))["soh_loss_per_cycle"]
oth = bl.filter((pl.col("supplier") != "Kyron") & (pl.col("pack_type") == "2W_2.1kWh"))["soh_loss_per_cycle"]
print(f"Kyron 2W median loss/cycle {kyr.median():.4f} vs Cellora+Amptek 2W {oth.median():.4f} (x{kyr.median() / oth.median():.2f}); "
      f"Kyron 2W lot medians range {lots[(lots.supplier == 'Kyron') & (lots.pack_type == '2W_2.1kWh')]['lot_median'].min():.4f}-"
      f"{lots[(lots.supplier == 'Kyron') & (lots.pack_type == '2W_2.1kWh')]['lot_median'].max():.4f}")
# Supplier-wide or lot cohort? Kyron 2W packs come only from the lot codes that are also the fast-fading Kyron 3W lots.
k2_lots = sorted(bt.filter((pl.col("supplier") == "Kyron") & (pl.col("pack_type") == "2W_2.1kWh"))["manufacturing_lot"].unique().to_list())
n_k2 = bt.filter((pl.col("supplier") == "Kyron") & (pl.col("pack_type") == "2W_2.1kWh")).height
k3_all = lots[(lots.supplier == "Kyron") & (lots.pack_type == "3W_4.8kWh")]
k3_late = k3_all[~k3_all["lot"].isin(k2_lots)]
k3_fast = k3_all[k3_all["lot"].isin(k2_lots)]
late_med = float(bl.filter((pl.col("supplier") == "Kyron") & (pl.col("pack_type") == "3W_4.8kWh") & ~pl.col("manufacturing_lot").is_in(k2_lots))["soh_loss_per_cycle"].median())
print(f"all {n_k2:,} Kyron 2W packs come from lots {k2_lots}; the Kyron 3W lots flagged as fast are {sorted(fl_lots)} (same codes: "
      f"{set(k2_lots) == set(fl_lots)}). Kyron 3W lots with those codes fade {k3_fast['lot_median'].min():.4f}-{k3_fast['lot_median'].max():.4f} "
      f"pts/cycle; the later Kyron 3W lots ({k3_late['lot'].min()}..{k3_late['lot'].max()}, {int(k3_late['packs'].sum())} packs) fade normally "
      f"({k3_late['lot_median'].min():.4f}-{k3_late['lot_median'].max():.4f}; pooled median {late_med:.4f}) -> a 2407-09 cohort defect; "
      f"'supplier-wide' cannot be tested for 2W because no later Kyron 2W lot exists")
# Exposure-matched check for the Kyron 3W lots: is the fast fade an artefact of different exposure (cycles, calendar)?
k3ids = bt.filter((pl.col("supplier") == "Kyron") & (pl.col("pack_type") == "3W_4.8kWh")).select(pl.col("battery_id").alias("battery_out_id"), "manufacturing_lot")
trj = (events(["battery_out_id", "event_ts", "soh_out_pct", "is_completed"]).filter(pl.col("is_completed"))
       .join(k3ids.lazy(), on="battery_out_id").collect()
       .sort("battery_out_id", "event_ts").with_columns(pl.int_range(1, pl.len() + 1).over("battery_out_id").alias("cyc")))
e150 = (trj.filter(pl.col("cyc") <= 150).group_by("battery_out_id", "manufacturing_lot")
        .agg(pl.len().alias("n"), (-pl.cov("cyc", "soh_out_pct") / pl.col("cyc").var()).alias("slope"), pl.col("event_ts").min().alias("t0"))
        .filter(pl.col("n") == 150).with_columns(pl.col("manufacturing_lot").is_in(k2_lots).alias("cohort_2407_09"))
        .group_by("cohort_2407_09").agg(pl.len().alias("packs"), pl.col("slope").median().alias("slope_first150"), pl.col("t0").median().alias("median_first_use"))
        .sort("cohort_2407_09"))
h125 = (trj.filter(pl.col("event_ts") >= dt.datetime(2025, 1, 1)).with_columns(pl.int_range(1, pl.len() + 1).over("battery_out_id").alias("c2"))
        .group_by("battery_out_id", "manufacturing_lot").agg(pl.len().alias("n"), (-pl.cov("c2", "soh_out_pct") / pl.col("c2").var()).alias("slope"))
        .filter(pl.col("n") >= 50).with_columns(pl.col("manufacturing_lot").is_in(k2_lots).alias("cohort_2407_09"))
        .group_by("cohort_2407_09").agg(pl.len().alias("packs"), pl.col("slope").median().alias("slope_jan_jun_2025")).sort("cohort_2407_09"))
e1 = {r["cohort_2407_09"]: r for r in e150.iter_rows(named=True)}; h1d = {r["cohort_2407_09"]: r for r in h125.iter_rows(named=True)}
print(f"exposure-matched Kyron 3W check: SoH-loss slope over each pack's first 150 cycles {e1[True]['slope_first150']:.4f} (KY-2407/08/09, "
      f"{e1[True]['packs']} packs) vs {e1[False]['slope_first150']:.4f} (later lots, {e1[False]['packs']} packs; median first use "
      f"{e1[True]['median_first_use']:%d-%b-%Y} vs {e1[False]['median_first_use']:%d-%b-%Y}); same calendar window Jan-Jun 2025: "
      f"{h1d[True]['slope_jan_jun_2025']:.4f} vs {h1d[False]['slope_jan_jun_2025']:.4f}")
pd.concat([e150.to_pandas().assign(check="first 150 cycles"), h125.to_pandas().assign(check="Jan-Jun 2025 swaps only")]).to_csv(
    TAB / "q7_kyron3w_exposure_check.csv", index=False)
del trj

# %% [markdown]
# ### 11c Time windows — month-boundary failure steps vs temperature, the 1-Jul-2024 price-vs-heat contrast, heat/flood days,
# Kyron retirement, competitor entry, firmware window. Temperature comes from two independent sources (city weather and
# station telemetry); the 1-Jul-2024 base-price rise hit every city on the same day, so a cross-city contrast separates price from heat.

# %%
hdr("11c. Time-window anomalies")
dn = (events(["date", "is_service_failure"]).group_by("date")
      .agg(pl.len().alias("att"), pl.col("is_service_failure").mean().alias("rate")).collect().sort("date"))
cdd = city_daily()
# two independent temperature sources: city weather (mean over the 6 cities) and station telemetry (cabinet ambient)
t_city = cdd.group_by("date").agg(pl.col("max_temp_c").mean().alias("tmax_city"))
t_amb = hourly(["date", "ambient_temp_c"]).group_by("date").agg(pl.col("ambient_temp_c").mean().alias("amb_station")).collect()
dn = dn.join(t_city, on="date", how="left").join(t_amb, on="date", how="left").to_pandas()
dn["med"] = dn["rate"].rolling(29, center=True, min_periods=15).median()
dn["mad"] = (dn["rate"] - dn["med"]).abs().rolling(29, center=True, min_periods=15).median() * 1.4826
dn["z"] = (dn["rate"] - dn["med"]) / dn["mad"]
ctx = (cdd.group_by("date").agg(pl.col("heat_alert").sum().alias("heat_cities"), pl.col("flood_disruption").sum().alias("flood_cities"),
                                pl.col("festival_or_event").drop_nulls().unique().str.join(", ").alias("events"),
                                pl.col("grid_outage_hours").mean().alias("grid_outage_h"), pl.col("max_temp_c").max().alias("max_temp"))
       .to_pandas())
dn = dn.merge(ctx, on="date", how="left")
flag_days = dn[dn["z"] > 3.5].copy()
print(f"days flagged (rate > rolling-29d median + 3.5 robust SD): {len(flag_days)}")
print(flag_days[["date", "att", "rate", "med", "z", "heat_cities", "flood_cities", "events", "max_temp"]].round(4).to_string(index=False))
# regime shifts: 7-day mean after vs before each 1st of month, against the same contrast around the 16th (control),
# with the temperature step over the same 7+7 days
dts = pd.to_datetime(dn["date"])
step_rows = []
for m0 in pd.date_range("2024-02-01", "2025-06-01", freq="MS"):
    for lab, pivot in [("1st of month", m0), ("16th (control)", m0 + pd.Timedelta(days=15))]:
        bm_, am_ = (dts >= pivot - pd.Timedelta(days=7)) & (dts < pivot), (dts >= pivot) & (dts < pivot + pd.Timedelta(days=7))
        step_rows.append(dict(pivot=pivot.date(), kind=lab, before=100 * dn.loc[bm_, "rate"].mean(), after=100 * dn.loc[am_, "rate"].mean(),
                              step_pts=100 * (dn.loc[am_, "rate"].mean() - dn.loc[bm_, "rate"].mean()),
                              tmax_before=dn.loc[bm_, "tmax_city"].mean(), tmax_after=dn.loc[am_, "tmax_city"].mean(),
                              tmax_step_c=dn.loc[am_, "tmax_city"].mean() - dn.loc[bm_, "tmax_city"].mean(),
                              amb_before=dn.loc[bm_, "amb_station"].mean(), amb_after=dn.loc[am_, "amb_station"].mean(),
                              amb_step_c=dn.loc[am_, "amb_station"].mean() - dn.loc[bm_, "amb_station"].mean(),
                              att_step_pct=100 * (dn.loc[am_, "att"].mean() / dn.loc[bm_, "att"].mean() - 1)))
steps = pd.DataFrame(step_rows)
steps.to_csv(TAB / "q7_failure_regime_steps.csv", index=False)
st1 = steps[steps["kind"] == "1st of month"]
st16 = steps[steps["kind"] == "16th (control)"]
big = st1[st1["step_pts"].abs() > 1]
ctrl_max, ctrl_tmax = st16["step_pts"].abs().max(), st16["tmax_step_c"].abs().max()
print("1st-of-month steps (7-day means), failure rate and temperature:")
print(st1[["pivot", "before", "after", "step_pts", "tmax_before", "tmax_after", "tmax_step_c", "amb_step_c", "att_step_pct"]].round(2).to_string(index=False))
print(f"around the 16th (control): largest |failure step| {ctrl_max:.2f} pts, largest |city max-temp step| {ctrl_tmax:.2f} C; "
      f"on the 1st: largest |temp step| {st1['tmax_step_c'].abs().max():.2f} C")
sx = {(r["pivot"], r["kind"]): r for _, r in steps.iterrows()}
jul_step = sx[(dt.date(2024, 7, 1), "1st of month")]
mar24, mar25 = sx[(dt.date(2024, 3, 1), "1st of month")], sx[(dt.date(2025, 3, 1), "1st of month")]
print(f"1-Jul-2024: failures {jul_step['before']:.2f}% -> {jul_step['after']:.2f}%, mean city max temp {jul_step['tmax_before']:.1f} -> "
      f"{jul_step['tmax_after']:.1f} C ({jul_step['tmax_step_c']:+.1f}), station ambient {jul_step['amb_before']:.1f} -> {jul_step['amb_after']:.1f} C, "
      f"network attempts/day {jul_step['att_step_pct']:+.1f}%")
print(f"March control: 1-Mar-24 temp {mar24['tmax_step_c']:+.1f} C -> failure step {mar24['step_pts']:+.2f} pts; 1-Mar-25 temp {mar25['tmax_step_c']:+.1f} C -> "
      f"{mar25['step_pts']:+.2f} pts (both months stay below the heat threshold)")
mrate = (events(["month", "is_service_failure"]).group_by("month").agg(pl.len().alias("att"), (100 * pl.col("is_service_failure").mean()).alias("fail_pct"))
         .collect().sort("month").with_columns(pl.col("month").dt.month().replace_strict(SEASON_OF_MONTH).alias("season_label")))
print(mrate.to_pandas().round(2).to_string(index=False))
mis_att = int(mrate.filter(pl.col("month").dt.month().is_in([3, 6]))["att"].sum())
mr = {(r["month"].year, r["month"].month): r["fail_pct"] for r in mrate.iter_rows(named=True)}
print(f"season-label mismatch: March is 'Summer' but runs {mr[(2024, 3)]:.2f}%/{mr[(2025, 3)]:.2f}% (2024/2025), June is 'Monsoon' but runs "
      f"{mr[(2024, 6)]:.2f}%/{mr[(2025, 6)]:.2f}%; attempts in Mar+Jun = {mis_att:,}")

# city level: threshold, within-month variation, and the 1-Jul-2024 cross-city contrast (same price rise everywhere)
cdy = (events(["city", "date", "is_service_failure"]).group_by("city", "date").agg(pl.len().alias("att"), pl.col("is_service_failure").mean().alias("rate")).collect()
       .join(cdd.select("city", "date", "heat_alert", "flood_disruption", "max_temp_c"), on=["city", "date"], how="left")
       .with_columns(pl.col("date").dt.truncate("1mo").alias("m")))
cmt = (cdy.group_by("city", "m").agg(pl.col("max_temp_c").mean().alias("tmax"), pl.col("rate").mean().alias("rate"), pl.col("att").sum().alias("att"))
       .to_pandas())
cmt["band"] = pd.cut(cmt["tmax"], [-np.inf, 34, 36, 38, np.inf], labels=["<=34 C", "34-36 C", "36-38 C", ">38 C"])
thr = (cmt.groupby("band", observed=False).agg(city_months=("rate", "size"), fail_pct=("rate", lambda s: 100 * s.mean()),
                                               tmax_mean=("tmax", "mean")).reset_index())
thr.to_csv(TAB / "q7_failure_temperature_threshold.csv", index=False)
thd = {r["band"]: r for _, r in thr.iterrows()}
sd_within = float(cdy.with_columns((pl.col("max_temp_c") - pl.col("max_temp_c").mean().over("city", "m")).alias("d"))["d"].std())
r_cm = float(np.corrcoef(cmt["tmax"], cmt["rate"])[0, 1])
print("city-month failure rate (mean of daily rates) by city-month mean max temp:"); print(thr.round(2).to_string(index=False))
print(f"corr(city-month mean max temp, failure rate) = {r_cm:.3f} over {len(cmt)} city-months; within a city-month the SD of daily max temp is "
      f"only {sd_within:.2f} C -> the weather varies at month level, so day-level heat tests cannot separate heat from month")
P0 = BASE_PRICE_CHANGE
jul = (cdy.filter(pl.col("date").is_between(P0 - dt.timedelta(days=7), P0 + dt.timedelta(days=6)))
       .group_by("city", (pl.col("date") >= P0).alias("post"))
       .agg(pl.col("max_temp_c").mean().alias("tmax"), (100 * pl.col("rate").mean()).alias("fail_pct"), pl.col("att").mean().alias("att_day"))
       .to_pandas())
jw = jul.pivot(index="city", columns="post", values=["tmax", "fail_pct", "att_day"])
jw.columns = [f"{a}_{'after' if b else 'before'}" for a, b in jw.columns]
jw = jw.reset_index()
jw["hot_city"] = jw["city"].isin(HOT)
jw["tmax_change_c"] = jw["tmax_after"] - jw["tmax_before"]
jw["fail_change_pts"] = jw["fail_pct_after"] - jw["fail_pct_before"]
jw = jw.sort_values(["hot_city", "fail_pct_before"], ascending=[False, False]).reset_index(drop=True)
jw.to_csv(TAB / "q7_jul24_cross_city.csv", index=False)
print("1-Jul-2024 (base-price rise everywhere), 7 days before vs after, by city:")
print(jw[["city", "hot_city", "tmax_before", "tmax_after", "tmax_change_c", "fail_pct_before", "fail_pct_after", "fail_change_pts",
          "att_day_before", "att_day_after"]].round(2).to_string(index=False))
hot_j, cool_j = jw[jw.hot_city], jw[~jw.hot_city]
print(f"hot cities: failures fell {hot_j['fail_change_pts'].min():+.2f} to {hot_j['fail_change_pts'].max():+.2f} pts with max temp "
      f"{hot_j['tmax_change_c'].min():+.1f} to {hot_j['tmax_change_c'].max():+.1f} C; cool cities under the same price rise: "
      f"{cool_j['fail_change_pts'].min():+.2f} to {cool_j['fail_change_pts'].max():+.2f} pts (temp {cool_j['tmax_change_c'].min():+.1f} to "
      f"{cool_j['tmax_change_c'].max():+.1f} C) -> the July drop is heat-driven, not price-driven")
add_issue("Season labels (Summer = Mar–May) misaligned with the Apr–Jun heat regime", "analysis trap",
          "Daily failure rate; 7-day steps at month boundaries vs mid-month control; city weather and station ambient temperature; cross-city contrast",
          mis_att, N_EV, "attempts",
          "Control seasonality with city-month temperature (or city x month effects), not SEASON_OF_MONTH",
          f"Failure rate steps on the 1st of Apr/May/Jun/Jul because the weather is month-level: mean city max temp steps "
          f"{sx[(dt.date(2024, 4, 1), '1st of month')]['tmax_step_c']:+.1f}/{sx[(dt.date(2024, 5, 1), '1st of month')]['tmax_step_c']:+.1f}/"
          f"{sx[(dt.date(2024, 6, 1), '1st of month')]['tmax_step_c']:+.1f}/{jul_step['tmax_step_c']:+.1f} C on the same days (≤{ctrl_tmax:.2f} C around the 16th). "
          f"The 1-Jul-24 drop ({jul_step['before']:.1f}%→{jul_step['after']:.1f}%) happened only in hot cities; cool cities under the same price rise "
          f"moved {cool_j['fail_change_pts'].min():+.1f} to {cool_j['fail_change_pts'].max():+.1f} pts -> heat, not price. March ('Summer') runs "
          f"{mr[(2024, 3)]:.1f}%, June ('Monsoon') {mr[(2024, 6)]:.1f}%: season dummies blur the regime.",
          "failures track month-level heat (>~34–36 °C), not season labels", "medium")
# heat-alert days within a city-month: structurally weak test (weather barely varies within a month)
h1 = cdy.filter(pl.col("heat_alert")).group_by("city", "m").agg(pl.col("rate").mean().alias("r_heat"), pl.len().alias("d_heat"))
h0 = cdy.filter(~pl.col("heat_alert")).group_by("city", "m").agg(pl.col("rate").mean().alias("r_no"), pl.len().alias("d_no"))
hh = h1.join(h0, on=["city", "m"], how="inner").with_columns((pl.col("r_heat") - pl.col("r_no")).alias("diff")).to_pandas()
tt = stats.ttest_1samp(hh["diff"], 0)
ci_h = stats.t.interval(0.95, len(hh) - 1, loc=hh["diff"].mean(), scale=stats.sem(hh["diff"]))
heat_eff = float(hh["diff"].mean())
print(f"heat-alert days vs other days in the same city-month: {100 * heat_eff:+.2f} pts (unweighted mean of {len(hh)} city-months, 95% CI "
      f"{100 * ci_h[0]:+.2f}..{100 * ci_h[1]:+.2f}, p = {tt.pvalue:.2g}) -> uninformative: within-month temperature SD is {sd_within:.2f} C")
fl = cdy.filter(pl.col("flood_disruption")).select("city", "date", "rate").to_pandas()
fl_diffs = []
for _, r in fl.iterrows():
    ref = cdy.filter((pl.col("city") == r["city"]) & (pl.col("date") - r["date"]).dt.total_days().abs().is_between(1, 7) & ~pl.col("flood_disruption"))
    fl_diffs.append(r["rate"] - ref["rate"].mean())
print(f"flood days ({len(fl)}): failure rate vs same city +-7 days: mean diff {100 * np.mean(fl_diffs):+.2f} pts (range {100 * min(fl_diffs):+.2f}..{100 * max(fl_diffs):+.2f})")
RET = dt.date(2025, 6, 15)
kr = (events(["date", "is_stockout", "is_completed", "supplier_out"])
      .filter(pl.col("date").is_between(RET - dt.timedelta(days=7), RET + dt.timedelta(days=7)) & (pl.col("date") != RET))
      .group_by((pl.col("date") > RET).alias("post"))
      .agg(pl.len().alias("att"), pl.col("is_stockout").mean().alias("stockout_rate"),
           (pl.col("supplier_out") == "Kyron").filter(pl.col("is_completed")).mean().alias("kyron_share_issued")).collect().sort("post"))
print("Kyron 'retirement' 15-Jun-2025, 7 days before vs after:"); print(kr)
CE = COMPETITOR_ENTRY
cp = (events(["date", "station_id", "competitor_since", "is_service_failure"])
      .filter(pl.col("date").is_between(CE - dt.timedelta(days=56), CE + dt.timedelta(days=55)))
      .with_columns(pl.col("competitor_since").is_not_null().alias("treated"), (pl.col("date") >= CE).alias("post"))
      .group_by("treated", "post").agg(pl.len().alias("att"), pl.struct("station_id", "date").n_unique().alias("station_days"),
                                       pl.col("is_service_failure").mean().alias("fail_rate"))
      .with_columns((pl.col("att") / pl.col("station_days")).alias("att_per_station_day")).collect().sort("treated", "post"))
cpd = {(r["treated"], r["post"]): r for r in cp.iter_rows(named=True)}
did_vol = (cpd[(True, True)]["att_per_station_day"] / cpd[(True, False)]["att_per_station_day"]) / (cpd[(False, True)]["att_per_station_day"] / cpd[(False, False)]["att_per_station_day"])
print(cp); print(f"competitor entry DiD (8 weeks pre/post): attempts per station-day ratio-of-ratios = {did_vol:.3f}")
fe = (events(["date", "station_firmware", "is_system_error"])
      .filter(pl.col("date").is_between(dt.date(2025, 2, 1), dt.date(2025, 5, 19)))
      .with_columns(pl.when(pl.col("date") < BUG_A).then(pl.lit("1 pre (1 Feb-9 Mar)")).when(pl.col("date") <= BUG_B).then(pl.lit("2 bug window"))
                    .otherwise(pl.lit("3 post (15 Apr-19 May)")).alias("period"))
      .group_by("station_firmware", "period").agg(pl.len().alias("att"), pl.col("is_system_error").sum().alias("sys_err"))
      .with_columns((100 * pl.col("sys_err") / pl.col("att")).alias("sys_err_pct")).collect().sort("station_firmware", "period"))
print(fe)
fed = {(r["station_firmware"], r["period"][0]): r for r in fe.iter_rows(named=True)}
did_fw = (fed[("v3.2.0", "2")]["sys_err_pct"] - fed[("v3.2.0", "1")]["sys_err_pct"]) - (fed[("v3.2.1", "2")]["sys_err_pct"] - fed[("v3.2.1", "1")]["sys_err_pct"])
print(f"system-error DiD (v3.2.0 vs v3.2.1, bug window minus pre): {did_fw:+.3f} pts")
twc = pd.DataFrame([
    dict(check="1-Jul-24 hot cities: failure change 7d after - before (pts, mean of 3)", effect=hot_j["fail_change_pts"].mean(),
         low=hot_j["fail_change_pts"].min(), high=hot_j["fail_change_pts"].max(), p=np.nan, n=len(hot_j), note="range over cities; same price rise"),
    dict(check="1-Jul-24 cool cities: failure change 7d after - before (pts, mean of 3)", effect=cool_j["fail_change_pts"].mean(),
         low=cool_j["fail_change_pts"].min(), high=cool_j["fail_change_pts"].max(), p=np.nan, n=len(cool_j), note="range over cities; same price rise"),
    dict(check="heat-alert day vs same city-month (pts, unweighted)", effect=100 * heat_eff, low=100 * ci_h[0], high=100 * ci_h[1], p=tt.pvalue,
         n=len(hh), note=f"95% CI; uninformative: within city-month temp SD {sd_within:.2f} C"),
    dict(check="flood day vs same city +-7 d (pts)", effect=100 * np.mean(fl_diffs), low=100 * min(fl_diffs), high=100 * max(fl_diffs), p=np.nan,
         n=len(fl), note="range over flood days"),
    dict(check="Kyron retirement: stockout rate 7d after - before (pts)", effect=100 * (kr["stockout_rate"][1] - kr["stockout_rate"][0]), low=np.nan,
         high=np.nan, p=np.nan, n=int(kr["att"].sum()), note=""),
    dict(check="competitor entry DiD attempts/station-day (ratio)", effect=did_vol, low=np.nan, high=np.nan, p=np.nan, n=int(cp["att"].sum()), note=""),
    dict(check="firmware window DiD system-error rate (pts)", effect=did_fw, low=np.nan, high=np.nan, p=np.nan, n=int(fe["att"].sum()), note="")])
twc.to_csv(TAB / "q7_time_window_checks.csv", index=False)

TEMP_C = SERIES[7]
fig, (ax, axr) = plt.subplots(1, 2, figsize=(17.5, 6.0), gridspec_kw={"width_ratios": [2.75, 1], "wspace": 0.42})
ax.axvspan(pd.Timestamp(BUG_A), pd.Timestamp(BUG_B), color=PALETTE["light"], alpha=0.7, lw=0, zorder=0)
ax.plot(dts, 100 * dn["rate"], color=PALETTE["primary"], lw=0.9, alpha=0.5, label="daily service-failure rate (left)")
ax.plot(dts, 100 * dn["med"], color=PALETTE["primary"], lw=2.2, label="rolling 29-day median (left)")
ax.scatter(pd.to_datetime(flag_days["date"]), 100 * flag_days["rate"], s=46, color=PALETTE["accent"], zorder=4,
           label=f"anomalous day (robust z > 3.5): {len(flag_days)}")
ymax_t = 100 * dn["rate"].max() * 1.18
axt = ax.twinx()
axt.plot(dts, dn["tmax_city"], color=TEMP_C, lw=1.5, alpha=0.9, label="mean city max temp (°C, right)")
axt.spines["right"].set_visible(True); axt.grid(False)
axt.set_ylabel("Mean daily max temperature across 6 cities (°C)", color=TEMP_C)
axt.tick_params(axis="y", colors=TEMP_C)
axt.set_ylim(dn["tmax_city"].min() - 2, dn["tmax_city"].max() + 9)
for pv in [dt.date(2024, 4, 1), dt.date(2024, 5, 1), dt.date(2024, 7, 1)]:
    r_ = sx[(pv, "1st of month")]
    axt.annotate(f"{r_['tmax_step_c']:+.1f} °C\n{r_['step_pts']:+.1f} pts", xy=(pd.Timestamp(pv), r_["tmax_after"] if r_["tmax_step_c"] > 0 else r_["tmax_before"]),
                 xytext=(-46, 10) if r_["tmax_step_c"] > 0 else (10, 34), textcoords="offset points", fontsize=9.5, color="#333333")
for d_, lab in [(BASE_PRICE_CHANGE, "base-price rise"), (PILOT_START, "peak pilot"), (CE, "competitor entry"), (RET, "Kyron 'retired'")]:
    ax.axvline(pd.Timestamp(d_), color="#555555", lw=1)
    ax.text(pd.Timestamp(d_), ymax_t * 0.99, f" {lab}", rotation=90, va="top", ha="left", fontsize=9.5, color="#333333")
ax.text(pd.Timestamp(BUG_A) + pd.Timedelta(days=17), ymax_t * 0.62, "firmware\nbug window", ha="center", fontsize=9.5, color="#555555")
ax.set_ylim(0, ymax_t); ax.set_ylabel("Service failures (% of attempts)"); ax.set_xlabel("")
h_a, l_a = ax.get_legend_handles_labels(); h_b, l_b = axt.get_legend_handles_labels()
ax.legend(h_a + h_b, l_a + l_b, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4, fontsize=9.5)
ax.set_title(f"Mean city max temp (right axis) steps on the same days as failures; around the 16th it moves ≤{ctrl_tmax:.2f} °C.\n"
             f"Threshold: failure rate {thd['<=34 C']['fail_pct']:.2f}% in city-months ≤34 °C, {thd['36-38 C']['fail_pct']:.2f}% at 36–38 °C, "
             f"{thd['>38 C']['fail_pct']:.1f}% above 38 °C.\nFirmware window, competitor entry and Kyron 'retirement' leave no step.",
             fontsize=10.5, fontweight="normal")
yy = np.arange(len(jw))
for i_, r_ in jw.iterrows():
    col_ = PALETTE["accent"] if r_["hot_city"] else PALETTE["primary"]
    axr.plot([r_["fail_pct_before"], r_["fail_pct_after"]], [i_, i_], color=col_, lw=2, alpha=0.6, zorder=2)
    axr.scatter([r_["fail_pct_before"]], [i_], s=60, facecolor="white", edgecolor=col_, linewidth=1.8, zorder=3)
    axr.scatter([r_["fail_pct_after"]], [i_], s=60, color=col_, zorder=4)
    axr.text(max(r_["fail_pct_before"], r_["fail_pct_after"]) + 0.6, i_, f"{r_['tmax_change_c']:+.1f} °C", va="center", fontsize=9.5, color="#333333")
axr.set_yticks(yy, [f"{c_} ({'hot' if h_ else 'cool'})" for c_, h_ in zip(jw["city"], jw["hot_city"])])
axr.invert_yaxis(); axr.set_xlim(0, jw[["fail_pct_before", "fail_pct_after"]].max().max() + 7)
axr.set_xlabel("Failure rate (%): ○ 24–30 Jun → ● 1–7 Jul 2024\n(label = change in mean max temp)")
axr.set_title("Same price rise on 1-Jul-24, different outcome:\nfailures fell only in the hot cities", fontsize=11)
axr.grid(axis="y", visible=False)
fig.suptitle("Failure steps track month-level temperature steps; the 1-Jul-24 drop happened only in hot cities", x=0.01, ha="left",
             fontsize=14, fontweight="bold", y=1.03)
save_fig(fig, "q7_07_time_window_anomalies")

# %% [markdown]
# ### 11d Firmware- and connectivity-linked irregularities

# %%
hdr("11d. Connectivity / firmware irregularities")
ctier = (events(["connectivity_tier", "sync_mode", "is_system_error", "is_service_failure"]).group_by("connectivity_tier")
         .agg(pl.len().alias("attempts"), (100 * (pl.col("sync_mode") == "offline_batch").mean()).alias("offline_batch_pct"),
              (100 * pl.col("is_system_error").mean()).alias("system_error_pct"), (100 * pl.col("is_service_failure").mean()).alias("service_failure_pct"))
         .collect().join(tele.select("connectivity_tier", "missing_pct"), on="connectivity_tier").sort("connectivity_tier"))
cfw = (events(["station_firmware", "sync_mode", "is_system_error"]).group_by("station_firmware")
       .agg(pl.len().alias("attempts"), (100 * (pl.col("sync_mode") == "offline_batch").mean()).alias("offline_batch_pct"),
            (100 * pl.col("is_system_error").mean()).alias("system_error_pct")).collect().sort("station_firmware"))
csync = (events(["sync_mode", "is_system_error", "connectivity_tier"]).group_by("connectivity_tier", "sync_mode")
         .agg(pl.len().alias("attempts"), (100 * pl.col("is_system_error").mean()).alias("system_error_pct")).collect().sort("connectivity_tier", "sync_mode"))
print(ctier); print(cfw); print(csync)
pd.concat([ctier.to_pandas().assign(group="connectivity_tier"), cfw.to_pandas().assign(group="firmware")]).to_csv(TAB / "q7_connectivity_firmware.csv", index=False)
tab_se = ctier.select("attempts", "system_error_pct").to_numpy()
cnt_se = np.column_stack([tab_se[:, 0] * tab_se[:, 1] / 100, tab_se[:, 0] * (1 - tab_se[:, 1] / 100)])
chi_se = stats.chi2_contingency(cnt_se)
print(f"system-error rate differs by tier: chi2 p = {chi_se.pvalue:.3g}")

fig = plt.figure(figsize=(15, 5.6))
gs = fig.add_gridspec(1, 4, width_ratios=[2.3, 1, 1, 1], wspace=0.42)
axl = fig.add_subplot(gs[0, 0])
grp_order = [("Cellora", "2W_2.1kWh"), ("Amptek", "2W_2.1kWh"), ("Kyron", "2W_2.1kWh"), ("Cellora", "3W_4.8kWh"), ("Amptek", "3W_4.8kWh"), ("Kyron", "3W_4.8kWh")]
for i, (sup, pt) in enumerate(grp_order):
    d_ = lots[(lots.supplier == sup) & (lots.pack_type == pt)]
    jit = np.random.default_rng(i).uniform(-0.18, 0.18, len(d_))
    axl.scatter(d_["lot_median"], i + jit, s=np.clip(d_["packs"], 20, 160), color=SUPPLIER_COLORS[sup], alpha=0.8, edgecolor="white", linewidth=0.6, zorder=3)
    fl_ = d_[d_["flag"] == "unusual"]
    axl.scatter(fl_["lot_median"], i + jit[(d_["flag"] == "unusual").to_numpy()], s=200, facecolor="none", edgecolor="#333333", linewidth=1.2, zorder=4)
axl.set_yticks(range(len(grp_order)), [f"{s} {p[:2]}" for s, p in grp_order]); axl.invert_yaxis()
axl.set_xlabel("Median SoH loss per cycle in the lot (pts)")
ul = lots[lots["flag"] == "unusual"]
axl.annotate(f"{', '.join(ul['lot'])} (3W)\n{ul['packs'].sum()} packs, x{(ul['lot_median'] / ul['peer_median']).mean():.1f} peers",
             xy=(ul["lot_median"].min(), 5), xytext=(0.052, 4.35), fontsize=9.5, color="#333333",
             arrowprops=dict(arrowstyle="-", color="#777777", lw=0.8))
axl.annotate(f"all {n_k2:,} Kyron 2W packs = KY-2407/08/09\n(x{kyr.median() / oth.median():.1f} Cellora/Amptek 2W)",
             xy=(float(kyr.median()), 2), xytext=(0.040, 1.45), va="center", fontsize=9.5, color="#333333",
             arrowprops=dict(arrowstyle="-", color="#777777", lw=0.8))
axl.set_title(f"Lots KY-2407/08/09 fade fast in both pack types;\nlater Kyron 3W lots fade normally "
              f"({k3_late['lot_median'].min():.3f}–{k3_late['lot_median'].max():.3f})", fontsize=11)
for j, (col, lab) in enumerate([("offline_batch_pct", "offline_batch rows (%)"), ("system_error_pct", "system errors (% attempts)"), ("missing_pct", "telemetry missing (% hours)")]):
    a_ = fig.add_subplot(gs[0, j + 1])
    vals_ = [ctier.filter(pl.col("connectivity_tier") == t)[col].item() for t in tiers]
    a_.bar(tiers, vals_, color=[PALETTE["primary"], PALETTE["warn"], PALETTE["accent"]], width=0.6)
    for k_, v_ in enumerate(vals_):
        a_.text(k_, v_ * 1.02 + 1e-6, f"{v_:.2f}" if v_ < 10 else f"{v_:.1f}", ha="center", va="bottom", fontsize=10, color="#333333")
    a_.set_title(lab, fontsize=10.5); a_.set_ylim(0, max(vals_) * 1.25 + 1e-6); a_.set_xlabel("connectivity tier")
fig.suptitle("Fast fading is a Kyron lot cohort (KY-2407/08/09), not proven supplier-wide; weak links cause offline sync and blank telemetry, not system errors",
             x=0.01, ha="left", fontsize=13.5, fontweight="bold", y=1.03)
save_fig(fig, "q7_08_battery_lots_connectivity")

# %% [markdown]
# ## 12 · Data-quality scorecard (deliverable) and its visual

# %%
hdr("12. Scorecard")
add_issue("Telemetry 'partial' status rows", "undocumented", "Null profile by telemetry_status", int(tele["partial"].sum()), tot_h, "station-hours",
          "Treat as usable", f"{int(tele['partial'].sum()):,} 'partial' rows have every field populated; only 'missing' rows are blank.",
          "no blanks — usable", "low")
sev_rank = {"high": 0, "medium": 1, "low": 2, "pass": 3}
scd = pd.DataFrame(SC)
scd["sev_rank"] = scd["severity"].map(sev_rank)
scd = scd.sort_values(["sev_rank", "pct_of_data"], ascending=[True, False]).drop(columns="sev_rank").reset_index(drop=True)
scd.insert(0, "id", [f"DQ{i + 1:02d}" for i in range(len(scd))])
scd.to_csv(TAB / "q7_data_quality_scorecard.csv", index=False)
print(scd[["id", "issue", "kind", "rows_affected", "unit", "pct_of_data", "severity", "impact_short"]].to_string(index=False))
print(f"issues: {len(scd)} ({(scd.kind == 'documented').sum()} documented, {(scd.kind == 'undocumented').sum()} undocumented, "
      f"{(scd.kind == 'analysis trap').sum()} analysis trap, {(scd.kind == 'check passed').sum()} checks passed); high severity {(scd.severity == 'high').sum()}")

n = len(scd)
fig = plt.figure(figsize=(20, 0.46 * n + 1.8))
AX0, AXW = 0.30, 0.12                              # bar axis position/width in figure fraction
ax = fig.add_axes([AX0, 0.03, AXW, 0.89])


def fx(f):                                         # figure-fraction x -> bar-axis fraction (for blended transforms)
    return (f - AX0) / AXW


sev_col = {"high": PALETTE["accent"], "medium": PALETTE["warn"], "low": PALETTE["muted"], "pass": PALETTE["good"]}
pcts = scd["pct_of_data"].to_numpy(dtype=float)
for i, r in scd.iterrows():
    if r["rows_affected"] > 0:
        ax.barh(i, max(pcts[i], 1e-4), color=sev_col[r["severity"]], height=0.62, left=1e-4)
ax.set_xscale("log"); ax.set_xlim(1e-4, 150); ax.set_ylim(n - 0.5, -1.0)
ax.set_yticks([]); ax.set_xticks([1e-3, 1e-1, 10], ["0.001", "0.1", "10"])
ax.set_xlabel("% of table affected (log)")
ax.grid(axis="y", visible=False)
trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
tag = {"documented": "D", "undocumented": "U", "check passed": "✓", "analysis trap": "T"}
COLS = {"rows": fx(0.428), "decision": fx(0.525), "impact": fx(0.735)}
for i, r in scd.iterrows():
    lab = r["issue"] if len(r["issue"]) <= 64 else r["issue"][:62] + "…"
    ax.text(-0.05, i, f"{r['id']} [{tag[r['kind']]}]  {lab}", transform=trans, ha="right", va="center", fontsize=10, color="#222222")
    ax.text(COLS["rows"], i, f"{r['rows_affected']:,} {r['unit']}", transform=trans, ha="left", va="center", fontsize=9.6, color="#333333")
    dec = r["decision"] if len(r["decision"]) <= 50 else r["decision"][:48] + "…"
    ax.text(COLS["decision"], i, dec, transform=trans, ha="left", va="center", fontsize=9.6, color="#444444")
    imp = r["impact_short"] if len(r["impact_short"]) <= 64 else r["impact_short"][:62] + "…"
    ax.text(COLS["impact"], i, imp, transform=trans, ha="left", va="center", fontsize=9.6, color="#222222")
hy = -0.95
ax.text(-0.05, hy, "Issue   [D documented · U undocumented · T analysis trap · ✓ check]", transform=trans, ha="right", va="center", fontsize=10.5, fontweight="bold")
ax.text(COLS["rows"], hy, "Rows affected", transform=trans, ha="left", va="center", fontsize=10.5, fontweight="bold")
ax.text(COLS["decision"], hy, "Decision", transform=trans, ha="left", va="center", fontsize=10.5, fontweight="bold")
ax.text(COLS["impact"], hy, "Impact on KPIs if untreated / note", transform=trans, ha="left", va="center", fontsize=10.5, fontweight="bold")
handles = [plt.Rectangle((0, 0), 1, 1, color=sev_col[s]) for s in ["high", "medium", "low", "pass"]]
fig.legend(handles, ["high severity", "medium", "low", "check passed"], loc="upper right", ncol=4, bbox_to_anchor=(0.99, 1.0), fontsize=10)
n_issue = int((scd.kind != "check passed").sum()); n_high = int((scd.severity == "high").sum())
print(f"scorecard figure: {n_issue} issues audited (excluding passed checks), {n_high} high severity; Poisson funnel outliers total {n_hi_raw + n_lo_raw}")
fig.suptitle(f"Data-quality scorecard: {n_issue} issues audited; {n_high} high-severity ones would distort a headline KPI or invalidate an analysis if untreated",
             x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.0)
save_fig(fig, "q7_02_dq_scorecard")

# %% [markdown]
# ## Summary of figures and tables

# %%
hdr("Done")
for f_ in sorted(FIG.glob("q7_*.png")):
    print("figure:", f_.relative_to(ROOT))
for f_ in sorted(TAB.glob("q7_*.csv")):
    print("table :", f_.relative_to(ROOT))
print(f"runtime {time.time() - T0:.1f}s")
