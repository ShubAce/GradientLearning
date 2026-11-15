# %% [markdown]
# # Q9 - Synthesis and budget decision (lead analyst)
# **Question.** What is actually driving VoltRelay's service failures, new-rider churn and margin erosion, where does
# performance vary most, and which of the four budget proposals - (a) more stations, (b) more batteries,
# (c) network-wide pricing rollout, (d) long-term exclusive with the largest fleet partner - does the evidence support?
#
# **Approach.**
# 1. Re-derive, with fresh code on the clean data, the headline numbers the integrated story rests on (scale, same-season
#    failures, cm1 per swap under three wear conventions, failure concentration), so every analysis is quoted from one
#    consistent base.
# 2. Re-test the points where the eight analyses disagreed (inventory/pre-charging as a heat lever, the heat trigger,
#    the date the non-Kyron fleet reaches end of life, the margin level in 2025).
# 3. Size every budget option in the same units with the shared cost model: Apr-Jun failure-rate points, new-rider
#    non-retention points, cm1 per swap, Rs lakh per month (H1-2025 run-rate volume) and one-off cost.
# 4. Save the key-number table and two synthesis figures (causal chain, budget scorecard).
#
# Margin levels are *under our usage-based battery-depreciation assumption* (no salvage, 70% SoH end of life); trends and
# differences are the robust part. Numbers marked "cited" come from the verified q1-q8 analyses and are printed here
# so that every number in the synthesis is printed by this script.

# %%
import sys
import datetime as dt
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:  # running inside a notebook
    ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from common import *  # noqa: E402,F401

set_style()
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(24); pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(60)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)

QID = "q9"
HOT = ["Delhi NCR", "Jaipur", "Hyderabad"]
M = lambda y, m: dt.date(y, m, 1)  # noqa: E731
LAKH = 1e5
CRORE = 1e7
WIN = {
    "H1-24": [M(2024, m) for m in range(1, 7)],
    "Q1-24": [M(2024, m) for m in (1, 2, 3)],
    "Q2-24": [M(2024, m) for m in (4, 5, 6)],
    "Jul-Aug-24": [M(2024, 7), M(2024, 8)],
    "Oct-24..Mar-25": [M(2024, m) for m in (10, 11, 12)] + [M(2025, m) for m in (1, 2, 3)],
    "Q1-25": [M(2025, m) for m in (1, 2, 3)],
    "Q2-25": [M(2025, m) for m in (4, 5, 6)],
    "H1-25": [M(2025, m) for m in range(1, 7)],
}
CUR_TERMS = [M(2024, 11), M(2024, 12)] + [M(2025, m) for m in range(1, 7)]  # after the ZipDrop amendment (q5 window)
PEAK_H = [12, 13, 19, 20, 21, 22]                 # data-verified pilot bands (q5, q7)
OFF_H = [23, 0, 1, 2, 3, 4, 5, 14, 15, 16]


def show(title, df):
    print(f"\n--- {title}")
    print(df)


# %% [markdown]
# ## 1. How the network performed: one consistent set of headline numbers
# Monthly panel by vehicle class from the clean events (test stations excluded, firmware timestamps fixed). For every
# window we compute cm1 per swap three ways: the shared per-cycle wear model (base); wear stopped at 70% EOL (cap);
# and post-EOL issues charged at the Cellora class wear (the better economic comparator, q1). The Kyron gap replaces each
# Kyron-issued swap's wear with the Cellora wear per swap of the same class (q1's counterfactual).

# %%
ev_cols = ["month", "vehicle_class", "supplier_out", "is_completed", "is_service_failure", "revenue", "energy_cost",
           "wear_cost", "fixed_cost", "soh_out_pct", "expansion_wave"]
mon = (events(ev_cols)
       .with_columns(pe=pl.col("is_completed") & (pl.col("soh_out_pct") < EOL_SOH),
                     ky=pl.col("is_completed") & (pl.col("supplier_out") == "Kyron"),
                     cel=pl.col("is_completed") & (pl.col("supplier_out") == "Cellora"),
                     launch=pl.col("expansion_wave") == "Launch")
       .group_by(["month", "vehicle_class"])
       .agg(attempts=pl.len(), swaps=pl.col("is_completed").sum(), failures=pl.col("is_service_failure").sum(),
            revenue=pl.col("revenue").sum(), energy=pl.col("energy_cost").sum(), wear=pl.col("wear_cost").sum(),
            fixed=pl.col("fixed_cost").sum(),
            pe_swaps=pl.col("pe").sum(), pe_wear=pl.col("wear_cost").filter(pl.col("pe")).sum(),
            ky_swaps=pl.col("ky").sum(), ky_wear=pl.col("wear_cost").filter(pl.col("ky")).sum(),
            cel_swaps=pl.col("cel").sum(), cel_wear=pl.col("wear_cost").filter(pl.col("cel")).sum(),
            l_att=pl.col("launch").sum(), l_fail=(pl.col("launch") & pl.col("is_service_failure")).sum())
       .sort(["month", "vehicle_class"])
       .collect(engine="streaming"))
REF = {r["vehicle_class"]: r["cel_wear"] / r["cel_swaps"]
       for r in mon.group_by("vehicle_class").agg(pl.col("cel_wear").sum(), pl.col("cel_swaps").sum()).iter_rows(named=True)}
print("Cellora reference wear per swap (full window, Rs):", {k: round(v, 2) for k, v in REF.items()})


def window(months):
    d = mon.filter(pl.col("month").is_in(months))
    s = d.select(pl.exclude("month", "vehicle_class").sum()).row(0, named=True)
    by = {r["vehicle_class"]: r for r in d.group_by("vehicle_class").agg(pl.col("pe_swaps").sum(), pl.col("ky_swaps").sum()).iter_rows(named=True)}
    n, sw = len(months), s["swaps"]
    cm1 = (s["revenue"] - s["energy"] - s["wear"]) / sw
    pe_ref = sum(by.get(c, {}).get("pe_swaps", 0) * REF[c] for c in REF)
    ky_ref = sum(by.get(c, {}).get("ky_swaps", 0) * REF[c] for c in REF)
    gap = (s["ky_wear"] - ky_ref) / sw
    return dict(months=n, attempts=s["attempts"], swaps=sw, swaps_pm=sw / n, failures=s["failures"],
                fail_rate=s["failures"] / s["attempts"], launch_fail_rate=s["l_fail"] / s["l_att"],
                revenue_pm=s["revenue"] / n, rev_ps=s["revenue"] / sw, energy_ps=s["energy"] / sw, wear_ps=s["wear"] / sw,
                fixed_ps=s["fixed"] / sw, cm1_ps=cm1, cm2_ps=cm1 - s["fixed"] / sw,
                cm1_cap_ps=cm1 + s["pe_wear"] / sw, cm1_celwear_ps=cm1 + (s["pe_wear"] - pe_ref) / sw,
                post_eol_swaps=s["pe_swaps"], post_eol_wear=s["pe_wear"],
                kyron_gap_ps=gap, cm1_allcellora_ps=cm1 + gap, cm2_allcellora_ps=cm1 - s["fixed"] / sw + gap)


W = {k: window(v) for k, v in WIN.items()}
wtab = pd.DataFrame(W).T
show("Window KPIs (per swap in Rs; rates as shares)", wtab[["swaps_pm", "fail_rate", "launch_fail_rate", "rev_ps", "cm1_ps",
                                                           "cm1_cap_ps", "cm1_celwear_ps", "cm2_ps", "kyron_gap_ps",
                                                           "cm1_allcellora_ps", "cm2_allcellora_ps", "post_eol_swaps"]].round(3))
wtab.to_csv(TAB / f"{QID}_window_kpis.csv")

h24, h25 = W["H1-24"], W["H1-25"]
mult = {k: h25[k] / h24[k] for k in ("attempts", "swaps", "failures", "revenue_pm")}
print("\nH1-25 vs H1-24 multiples:", {k: round(v, 2) for k, v in mult.items()})
m_all = mon.group_by("month").agg(pl.col("attempts", "swaps", "failures", "revenue").sum()).sort("month")
jan24 = m_all.filter(pl.col("month") == M(2024, 1)).row(0, named=True)
jun25 = m_all.filter(pl.col("month") == M(2025, 6)).row(0, named=True)
naive_fail_x, naive_swap_x = jun25["failures"] / jan24["failures"], jun25["swaps"] / jan24["swaps"]
print(f"Jun-25: {jun25['swaps']:,} completed swaps, revenue Rs{jun25['revenue']/1e6:.2f}M (Jan-24: {jan24['swaps']:,}, Rs{jan24['revenue']/1e6:.2f}M)")
print(f"Naive Jan-24 -> Jun-25: failures x{naive_fail_x:.2f}, swaps x{naive_swap_x:.2f} "
      f"({jan24['failures']/jan24['attempts']:.2%} -> {jun25['failures']/jun25['attempts']:.2%})")
print(f"Same season Apr-Jun: network {W['Q2-24']['fail_rate']:.2%} -> {W['Q2-25']['fail_rate']:.2%}; "
      f"Launch panel {W['Q2-24']['launch_fail_rate']:.2%} -> {W['Q2-25']['launch_fail_rate']:.2%}; "
      f"failed attempts {W['Q2-24']['failures']:,} -> {W['Q2-25']['failures']:,} "
      f"(+{W['Q2-25']['failures']/W['Q2-24']['failures']-1:.1%}); swaps x{W['Q2-25']['swaps']/W['Q2-24']['swaps']:.2f}")
print(f"cm1/swap same season Q1-24 {W['Q1-24']['cm1_ps']:+.2f} -> Q1-25 {W['Q1-25']['cm1_ps']:+.2f}; "
      f"Kyron gap Q1-25 {W['Q1-25']['kyron_gap_ps']:+.2f}/swap; Q1-25 with Cellora-grade packs: cm1 "
      f"{W['Q1-25']['cm1_allcellora_ps']:+.2f}, cm2 {W['Q1-25']['cm2_allcellora_ps']:+.2f}")
print(f"H1-25 cm1/swap: per-cycle model {h25['cm1_ps']:+.2f} | post-EOL at Cellora wear {h25['cm1_celwear_ps']:+.2f} | "
      f"wear stops at EOL {h25['cm1_cap_ps']:+.2f}; post-EOL swaps {h25['post_eol_swaps']:,}, wear charged on them "
      f"Rs{h25['post_eol_wear']/1e6:.2f}M")

# %% [markdown]
# ## 2. Where failures concentrate (Apr-Jun) and what a Gen1 fix is worth
# Excess = failures above the Jul-Mar network rate. The Gen1 counterfactual applies the same-city, same-month Gen3
# (or Gen2) failure rate to the 36 hot-city Gen1 stations' Apr-Jun 2025 attempts (a naive like-for-like check that
# agrees with q2's Mantel-Haenszel and q3's model estimates). Generation is confounded with site age, so this is an
# upper-end association, to be validated by a matched pilot.

# %%
st = stations().filter(~pl.col("is_test"))
hot_g1 = st.filter(pl.col("city").is_in(HOT) & (pl.col("charger_generation") == "Gen1"))["station_id"].to_list()
print("hot-city Gen1 stations:", len(hot_g1), st.filter(pl.col("station_id").is_in(hot_g1)).group_by("city").len().sort("city").rows())
offseason = mon.filter(~pl.col("month").dt.month().is_in([4, 5, 6])).select(pl.col("failures").sum() / pl.col("attempts").sum()).item()
print(f"Jul-Mar baseline failure rate: {offseason:.2%}")

aj = (events(["month", "city", "charger_generation", "is_service_failure", "is_completed", "revenue"])
      .filter(pl.col("month").dt.month().is_in([4, 5, 6]))
      .with_columns(year=pl.col("month").dt.year(),
                    seg=pl.when(pl.col("city").is_in(HOT)).then(pl.lit("Hot city ") + pl.col("charger_generation"))
                    .otherwise(pl.lit("Other cities")))
      .group_by(["year", "month", "city", "charger_generation", "seg"])
      .agg(att=pl.len(), fail=pl.col("is_service_failure").sum(), swaps=pl.col("is_completed").sum(), rev=pl.col("revenue").sum())
      .collect(engine="streaming"))
seg = (aj.group_by(["year", "seg"]).agg(pl.col("att", "fail", "swaps", "rev").sum())
       .with_columns(excess=pl.col("fail") - pl.col("att") * offseason)
       .with_columns(rate=pl.col("fail") / pl.col("att"),
                     sh_att=pl.col("att") / pl.col("att").sum().over("year"),
                     sh_fail=pl.col("fail") / pl.col("fail").sum().over("year"),
                     sh_excess=pl.col("excess") / pl.col("excess").clip(lower_bound=0).sum().over("year"))
       .sort(["year", "seg"]))
show("Apr-Jun failure concentration by segment", seg)
seg.write_csv(TAB / f"{QID}_concentration_aprjun.csv")
hg25 = seg.filter((pl.col("year") == 2025) & (pl.col("seg") == "Hot city Gen1")).row(0, named=True)
both = seg.group_by("seg").agg(pl.col("att", "fail", "excess").sum()).with_columns(
    sh_att=pl.col("att") / pl.col("att").sum(), sh_excess=pl.col("excess") / pl.col("excess").clip(lower_bound=0).sum())
hg_both = both.filter(pl.col("seg") == "Hot city Gen1").row(0, named=True)
print(f"Hot-city Gen1, Apr-Jun 2025: {hg25['sh_att']:.1%} of attempts, {hg25['sh_fail']:.1%} of failures, "
      f"{hg25['sh_excess']:.1%} of excess, rate {hg25['rate']:.2%}; both summers: {hg_both['sh_att']:.1%} of Apr-Jun attempts, "
      f"{hg_both['sh_excess']:.1%} of excess")

a25 = aj.filter((pl.col("year") == 2025) & pl.col("city").is_in(HOT))
rate_cm = a25.select("month", "city", "charger_generation", (pl.col("fail") / pl.col("att")).alias("r"))
g1 = a25.filter(pl.col("charger_generation") == "Gen1")
for g in ("Gen2", "Gen3"):
    g1 = g1.join(rate_cm.filter(pl.col("charger_generation") == g).select("month", "city", pl.col("r").alias(f"r_{g}")),
                 on=["month", "city"], how="left")
cf = g1.with_columns(avoid_g2=pl.col("fail") - pl.col("att") * pl.col("r_Gen2"),
                     avoid_g3=pl.col("fail") - pl.col("att") * pl.col("r_Gen3"))
show("Gen1 counterfactual by city-month (Apr-Jun 2025)", cf.select("month", "city", "att", "fail", "r_Gen2", "r_Gen3", "avoid_g2", "avoid_g3"))
cf.write_csv(TAB / f"{QID}_gen1_counterfactual.csv")
AV2, AV3 = cf["avoid_g2"].sum(), cf["avoid_g3"].sum()
G1F = cf["fail"].sum()
A_G2, A_G3 = AV2 / G1F, AV3 / G1F                      # share of hot-city Gen1 failures avoided
aj25 = aj.filter(pl.col("year") == 2025).select(pl.col("att", "fail", "swaps", "rev").sum()).row(0, named=True)
NET_AJ25 = aj25["fail"] / aj25["att"]
rate_after2, rate_after3 = (aj25["fail"] - AV2) / aj25["att"], (aj25["fail"] - AV3) / aj25["att"]
REV_PS_HG1 = hg25["rev"] / hg25["swaps"]
print(f"Avoided failed attempts, Apr-Jun 2025: Gen2-like {AV2:,.0f} ({AV2/aj25['fail']:.1%} of network failures), "
      f"Gen3-like {AV3:,.0f} ({AV3/aj25['fail']:.1%}); network Apr-Jun rate {NET_AJ25:.2%} -> {rate_after2:.2%} / {rate_after3:.2%}")
print(f"Share of hot-city Gen1 failures avoided: Gen2-like {A_G2:.1%}, Gen3-like {A_G3:.1%}; revenue per swap at these "
      f"stations Rs{REV_PS_HG1:.2f}; revenue recovered per summer if each failure is a lost swap: Rs{AV2*REV_PS_HG1/1e6:.2f}M-"
      f"Rs{AV3*REV_PS_HG1/1e6:.2f}M")
TICKETS_PER_FAIL = 1 / 9.9                      # cited, q3: about one ticket per 9.9 extra failed attempts
print(f"Tickets avoided per summer (q3 slope, cited): {AV2*TICKETS_PER_FAIL:,.0f}-{AV3*TICKETS_PER_FAIL:,.0f}")

# %% [markdown]
# ## 3. Contradictions between analyses, re-tested
# **C1. Is hourly inventory (pre-charging, more packs) a heat lever?** q3 and q6 recommend pre-charging and
# pre-positioning packs; q2 (verified) says hourly stock does not predict failing hours. We re-test at the 36 hot-city
# Gen1 stations, Apr-Jun, on station-hours with telemetry ok and at least one 2W attempt: 2W stockout rate by the
# hour's minimum charged-2W stock, within day type (hot day = the day's peak cabinet temperature >= 45 C) and window.

# %%
evh = (events(["station_id", "date", "hour", "month", "vehicle_class", "is_stockout"])
       .filter(pl.col("station_id").is_in(hot_g1) & pl.col("month").dt.month().is_in([4, 5, 6]) & (pl.col("vehicle_class") == "2W"))
       .group_by(["station_id", "date", "hour"]).agg(att=pl.len(), so=pl.col("is_stockout").sum())
       .collect(engine="streaming"))
hh = (hourly(["station_id", "date", "hour", "charged_2w_min", "cabinet_temp_c", "telemetry_status"])
      .filter(pl.col("station_id").is_in(hot_g1) & pl.col("date").dt.month().is_in([4, 5, 6]))
      .collect())
peak = hh.group_by(["station_id", "date"]).agg(peak_cab=pl.col("cabinet_temp_c").max())
inv = (evh.join(hh.filter((pl.col("telemetry_status") == "ok") & pl.col("charged_2w_min").is_not_null()),
                on=["station_id", "date", "hour"], how="inner")
       .join(peak, on=["station_id", "date"], how="left")
       .with_columns(day=pl.when(pl.col("peak_cab") >= 45).then(pl.lit("hot day (peak cabinet >=45C)")).otherwise(pl.lit("other Apr-Jun day")),
                     win=pl.when(pl.col("hour").is_between(19, 22)).then(pl.lit("19-22h")).otherwise(pl.lit("other hours")),
                     stock=pl.when(pl.col("charged_2w_min") <= 0).then(pl.lit("0"))
                     .when(pl.col("charged_2w_min") <= 2).then(pl.lit("1-2"))
                     .when(pl.col("charged_2w_min") <= 5).then(pl.lit("3-5")).otherwise(pl.lit("6+"))))
crude = inv.group_by(pl.col("stock") == "0").agg(pl.col("att", "so").sum()).with_columns(rate=pl.col("so") / pl.col("att"))
show("C1 pooled (crude): stockout rate, zero-stock hour vs stocked", crude)
inv_tab = (inv.group_by(["day", "win", "stock"]).agg(pl.col("att", "so").sum(), hours=pl.len())
           .with_columns(rate=pl.col("so") / pl.col("att")).sort(["day", "win", "stock"]))
show("C1 within day type x window: 2W stockout rate by charged-2W stock", inv_tab)
inv_tab.write_csv(TAB / f"{QID}_inventory_check.csv")
hot_eve = inv_tab.filter((pl.col("day").str.starts_with("hot")) & (pl.col("win") == "19-22h"))
hot_oth = inv_tab.filter((pl.col("day").str.starts_with("hot")) & (pl.col("win") == "other hours"))
print(f"Hot days, 19-22h: stockout rate across stock buckets {hot_eve['rate'].min():.1%}-{hot_eve['rate'].max():.1%}; "
      f"other hours {hot_oth['rate'].min():.1%}-{hot_oth['rate'].max():.1%}")

# %% [markdown]
# **C2. Which heat trigger?** q3 suggests acting on forecast heat-alert days; q2 says the alert fires only around 42 C.
# City-day failure rate by daily maximum temperature, with the share of days carrying an official heat alert.

# %%
cdy = (events(["city", "date", "is_service_failure"]).group_by(["city", "date"])
       .agg(att=pl.len(), fail=pl.col("is_service_failure").sum()).collect(engine="streaming")
       .join(city_daily().select("city", pl.col("date").cast(pl.Date), "max_temp_c", pl.col("heat_alert").cast(pl.Float64)),
             on=["city", "date"], how="left"))
heat = (cdy.with_columns(tbin=pl.col("max_temp_c").cut([34, 36, 38, 42], labels=["<34C", "34-36C", "36-38C", "38-42C", ">=42C"], left_closed=True))
        .group_by("tbin").agg(days=pl.len(), att=pl.col("att").sum(), fail=pl.col("fail").sum(), alert_share=pl.col("heat_alert").mean())
        .with_columns(rate=pl.col("fail") / pl.col("att")).sort("tbin"))
show("C2 failure rate by city-day max temperature", heat)
heat.write_csv(TAB / f"{QID}_heat_threshold.csv")
HEAT = {r["tbin"]: r for r in heat.iter_rows(named=True)}

# %% [markdown]
# **C3. When does the non-Kyron fleet reach end of life?** q1 extrapolated returned-pack SoH (about Feb-2026); q4
# projected Cellora/Amptek life at about 832 days (about Apr-2026). We project each pack from its own calendar fade
# (initial SoH minus SoH on 30-Jun-2025, over days since first seen; q4 shows fade is calendar-driven) and, as a second
# method, extrapolate the last six months of issued-pack SoH for non-Kyron 2W packs.

# %%
ASOF = dt.datetime(2025, 6, 30)
BAD_LOTS = ["KY-2407", "KY-2408", "KY-2409"]
bt = (batteries()
      .with_columns(days=(pl.lit(ASOF) - pl.col("first_seen")).dt.total_seconds() / 86400,
                    grp=pl.when(pl.col("manufacturing_lot").is_in(BAD_LOTS)).then(pl.lit("Kyron bad lots"))
                    .when(pl.col("supplier") == "Kyron").then(pl.lit("Kyron later lots")).otherwise(pl.lit("Cellora/Amptek")))
      .with_columns(fade_pd=(pl.col("initial_soh_pct") - pl.col("current_soh_pct")) / pl.col("days"))
      .with_columns(days_to_eol=(pl.col("current_soh_pct") - EOL_SOH) / pl.col("fade_pd"),
                    life_days=(pl.col("initial_soh_pct") - EOL_SOH) / pl.col("fade_pd")))
btp = bt.to_pandas()
btp["eol_date"] = pd.Timestamp(ASOF) + pd.to_timedelta(btp["days_to_eol"], unit="D")
fleet = (btp.groupby(["grp", "vehicle_class"])
         .agg(packs=("battery_id", "size"), cost_mean=("purchase_cost_inr", "mean"), cost_total=("purchase_cost_inr", "sum"),
              soh_now=("current_soh_pct", "mean"), fade_pd_median=("fade_pd", "median"), life_days_median=("life_days", "median"),
              eol_p10=("eol_date", lambda s: s.quantile(0.10)), eol_p50=("eol_date", lambda s: s.quantile(0.50)),
              eol_p90=("eol_date", lambda s: s.quantile(0.90)),
              share_eol_by_apr26=("eol_date", lambda s: (s < pd.Timestamp("2026-04-01")).mean()))
         .reset_index())
show("C3 fleet end-of-life projection (per-pack calendar fade)", fleet)
fleet.to_csv(TAB / f"{QID}_fleet_eol_projection.csv", index=False)
nk2 = fleet[(fleet.grp == "Cellora/Amptek") & (fleet.vehicle_class == "2W")].iloc[0]
nk3 = fleet[(fleet.grp == "Cellora/Amptek") & (fleet.vehicle_class == "3W")].iloc[0]
bad = fleet[fleet.grp == "Kyron bad lots"]
bad_capex = float(bad.cost_total.sum())
soh_trend = (events(["month", "vehicle_class", "supplier_out", "is_completed", "soh_out_pct"])
             .filter(pl.col("is_completed") & (pl.col("vehicle_class") == "2W") & (pl.col("supplier_out") != "Kyron")
                     & pl.col("month").is_in(WIN["H1-25"]))
             .group_by("month").agg(soh=pl.col("soh_out_pct").mean()).sort("month").collect(engine="streaming"))
x = np.arange(len(soh_trend)); slope, icpt = np.polyfit(x, soh_trend["soh"].to_numpy(), 1)
months_to_70 = (soh_trend["soh"][-1] - EOL_SOH) / -slope
trend_date = pd.Timestamp("2025-06-15") + pd.DateOffset(months=int(np.floor(months_to_70)))
print(f"Issued non-Kyron 2W SoH Jan-Jun 2025: {soh_trend['soh'][0]:.1f} -> {soh_trend['soh'][-1]:.1f} "
      f"({slope:+.2f} pts/month) -> 70% in {months_to_70:.1f} months (about {trend_date:%b-%Y})")
print(f"Per-pack calendar fade: non-Kyron 2W median EOL {nk2.eol_p50:%b-%Y} (p10 {nk2.eol_p10:%b-%Y}, p90 {nk2.eol_p90:%b-%Y}); "
      f"{nk2.share_eol_by_apr26:.0%} before Apr-2026; renewal capex Rs{nk2.cost_total/CRORE:.2f} Cr for {nk2.packs:,} packs; "
      f"3W non-Kyron median EOL {nk3.eol_p50:%b-%Y}, Rs{nk3.cost_total/CRORE:.2f} Cr for {nk3.packs:,} packs")
print(f"Kyron bad-lot replacement capex (1:1): Rs{bad_capex/CRORE:.2f} Cr for {int(bad.packs.sum()):,} packs")

# %% [markdown]
# **C4. Kyron margin: q1's +Rs16.8/swap vs q4's "no cm1 gain from replacement".** Both hold; they answer different
# questions. q1 compares buying Kyron-grade vs Cellora-grade packs (a procurement choice, clean window Q1-25). q4
# compares replacing packs that are already past end of life vs running them on, which under book depreciation look free
# (wear stops at EOL) but deliver about 26% less range. The forward budget question is q1's: packs must be replaced at EOL
# anyway, and the grade bought decides about Rs16.8 of cm1 per swap at a 25% share of issues.

# %%
print(f"Kyron-grade vs Cellora-grade wear gap, Q1-25 (clean window): {W['Q1-25']['kyron_gap_ps']:+.2f} Rs/swap; "
      f"Oct-24..Mar-25 {W['Oct-24..Mar-25']['kyron_gap_ps']:+.2f}; at H1-25 volume Rs"
      f"{W['Q1-25']['kyron_gap_ps']*h25['swaps_pm']/LAKH:.1f} L/month")

# %% [markdown]
# ## 4. New-rider retention: how much would the Gen1 fix move it?
# New riders = signed up from 2024-01-01; day 0 = first clean attempt; early failures = service failures in days 0-13;
# retained = a completed swap in days 31-60 (q6's outcome; eligible = first attempt before 2025-05-01). For each rider we
# compute the probability of still having 2+ early failures if failures at the 36 hot-city Gen1 stations in Apr-Jun were
# avoided at the Gen3-like (or Gen2-like) share from section 2, then convert the drop in riders with 2+ early failures
# into retention points with q6's adjusted effect (2+ vs 0-1 OR 1.67 [1.43-1.94]).

# %%
first = events(["rider_id", "event_ts"]).group_by("rider_id").agg(first_ts=pl.col("event_ts").min())
newr = riders().filter(pl.col("signup_date") >= dt.date(2024, 1, 1)).select("rider_id")
rid = (events(["rider_id", "event_ts", "station_id", "is_service_failure", "is_completed"])
       .join(newr.lazy(), on="rider_id", how="semi")
       .join(first, on="rider_id", how="inner")
       .with_columns(day=((pl.col("event_ts") - pl.col("first_ts")).dt.total_seconds() / 86400).floor().cast(pl.Int32),
                     hotg1=pl.col("station_id").is_in(hot_g1) & pl.col("event_ts").dt.month().is_in([4, 5, 6]))
       .filter(pl.col("day") <= 60)
       .group_by("rider_id")
       .agg(first_ts=pl.col("first_ts").first(), n_early=(pl.col("day") <= 13).sum(),
            k=((pl.col("day") <= 13) & pl.col("is_service_failure")).sum(),
            h=((pl.col("day") <= 13) & pl.col("is_service_failure") & pl.col("hotg1")).sum(),
            ret=((pl.col("day") >= 31) & pl.col("is_completed")).any(),
            sw3160=((pl.col("day") >= 31) & pl.col("is_completed")).sum())
       .collect(engine="streaming"))
elig = rid.filter(pl.col("first_ts") < dt.datetime(2025, 5, 1))
e = elig.with_columns(two=pl.col("k") >= 2, notret=~pl.col("ret"))
grp_rates = e.group_by("two").agg(n=pl.len(), nonret=pl.col("notret").mean()).sort("two")
show("q6 cross-check: D31-60 non-retention by early failures (0-1 vs 2+)", grp_rates)
P0 = grp_rates.filter(~pl.col("two"))["nonret"].item()
P2 = grp_rates.filter(pl.col("two"))["nonret"].item()
print(f"eligible new riders {e.height:,}; overall non-retention {e['notret'].mean():.1%}; 2+ early failures: "
      f"{e['two'].sum():,} riders ({e['two'].mean():.1%}); non-retention {P2:.1%} vs {P0:.1%}")


def or_to_pp(orv, p0):
    o = p0 / (1 - p0) * orv
    return o / (1 + o) - p0


EFF = {lab: or_to_pp(v, P0) for lab, v in (("central", 1.67), ("lo", 1.43), ("hi", 1.94))}   # cited ORs from q6
print("Effect of moving a rider from 2+ to 0-1 early failures (pp of non-retention):", {k: round(v * 100, 2) for k, v in EFF.items()})


def p2plus_after(k, h, a):
    """Probability of 2+ early failures when each hot-city Gen1 Apr-Jun failure is avoided with probability a."""
    o = k - h
    p0 = np.where(o == 0, a ** h, 0.0)
    p1 = np.where(o == 1, a ** h, 0.0) + np.where(o == 0, h * (1 - a) * a ** np.maximum(h - 1, 0), 0.0)
    return 1 - p0 - p1


def ret_effect(df, label):
    k, h = df["k"].to_numpy().astype(float), df["h"].to_numpy().astype(float)
    before = (k >= 2).mean()
    out = dict(cohort=label, riders=df.height, share_2plus=before, hot_share_of_early_failures=h.sum() / max(k.sum(), 1))
    for tag, a in (("gen2", A_G2), ("gen3", A_G3)):
        d = before - p2plus_after(k, h, a).mean()
        out[f"drop_2plus_{tag}"] = d
        for lab, eff in EFF.items():
            out[f"nonret_pp_{tag}_{lab}"] = -d * eff * 100
    return out


rid = rid.with_columns(m=pl.col("first_ts").dt.month(), y=pl.col("first_ts").dt.year())
cohorts = [ret_effect(elig, "All eligible new riders (Jan-24..Apr-25 entry)"),
           ret_effect(elig.filter(pl.col("first_ts").dt.month().is_in([4, 5, 6])), "Summer-entry, eligible (Apr-Jun 2024, Apr 2025)"),
           ret_effect(rid.filter((pl.col("y") == 2025) & pl.col("m").is_in([4, 5, 6]) & (pl.col("first_ts") < dt.datetime(2025, 6, 17))),
                      "Summer-2025 entry (Apr 1 - Jun 16, full 14-day window)")]
rtab = pd.DataFrame(cohorts)
show("Retention effect of a Gen1 fix (negative = fewer riders lost)", rtab.round(4))
rtab.to_csv(TAB / f"{QID}_retention_counterfactual.csv", index=False)
S25 = rtab.iloc[2]
n_summer25 = rid.filter((pl.col("y") == 2025) & pl.col("m").is_in([4, 5, 6])).height
ret_g3, ret_g2 = -S25["nonret_pp_gen3_central"] / 100 * n_summer25, -S25["nonret_pp_gen2_central"] / 100 * n_summer25
sw_per_ret = e.filter(pl.col("ret"))["sw3160"].mean()
print(f"Summer-2025 new riders: {n_summer25:,}; riders kept per summer by the fix: {ret_g2:.0f} (Gen2-like) to "
      f"{ret_g3:.0f} (Gen3-like); swaps per retained rider in days 31-60: {sw_per_ret:.1f}")

# %% [markdown]
# ## 5. Budget levers in common units (shared cost model, H1-2025 run-rate volume)
# **5a. Pricing rollout (proposal c).** q5's difference-in-differences effects per swap (independents +Rs7.90, billable
# partners +Rs5.43) applied to our own H1-2025 volumes in the four non-pilot cities, plus the surcharge that
# non-billable partners (ZipDrop, Swiggle Go) would pay if billable (+Rs15 peak, -Rs8 off-peak, data-verified bands).

# %%
q5r = pd.read_csv(TAB / "q5_budget_pricing_rollout.csv")
DID_I = float(q5r.loc[q5r["item"].str.startswith("Revenue/cm1: Independent"), "did_rs_per_swap"].iloc[0])
DID_Y = float(q5r.loc[q5r["item"].str.startswith("Revenue/cm1: Partner Y"), "did_rs_per_swap"].iloc[0])
pf = partners().select("partner_id", "partner_name", "peak_surcharge_billable", "vehicle_class")
pv = (events(["month", "city", "partner_id", "hour", "is_completed", "list_price_inr"])
      .filter(pl.col("is_completed") & pl.col("month").is_in(WIN["H1-25"]))
      .join(pf.lazy().select("partner_id", "peak_surcharge_billable"), on="partner_id", how="left")
      .with_columns(segm=pl.when(pl.col("partner_id").is_null()).then(pl.lit("Independent"))
                    .when(pl.col("peak_surcharge_billable") == "Y").then(pl.lit("Partner Y")).otherwise(pl.lit("Partner N")),
                    band=pl.when(pl.col("hour").is_in(PEAK_H)).then(pl.lit("PEAK"))
                    .when(pl.col("hour").is_in(OFF_H)).then(pl.lit("OFFPEAK")).otherwise(pl.lit("STD")),
                    pilot=pl.col("city").is_in(PILOT_CITIES))
      .group_by(["pilot", "segm", "band", "partner_id"]).agg(n=pl.len(), list_sum=pl.col("list_price_inr").sum())
      .collect(engine="streaming"))
ctl = pv.filter(~pl.col("pilot"))
indep_pm = ctl.filter(pl.col("segm") == "Independent")["n"].sum() / 6
y_pm = ctl.filter(pl.col("segm") == "Partner Y")["n"].sum() / 6
n_peak_ctl = ctl.filter((pl.col("segm") == "Partner N") & (pl.col("band") == "PEAK"))["n"].sum() / 6
n_off_ctl = ctl.filter((pl.col("segm") == "Partner N") & (pl.col("band") == "OFFPEAK"))["n"].sum() / 6
ROLL_PM = indep_pm * DID_I + y_pm * DID_Y
BILL_PM = n_peak_ctl * 15 - n_off_ctl * 8
n_peak_all = pv.filter((pl.col("segm") == "Partner N") & (pl.col("band") == "PEAK"))["n"].sum() / 6
n_off_all = pv.filter((pl.col("segm") == "Partner N") & (pl.col("band") == "OFFPEAK"))["n"].sum() / 6
BILL_ALL_PM = n_peak_all * 15 - n_off_all * 8
print(f"Rollout (four non-pilot cities, H1-25 volume): independents {indep_pm:,.0f}/month x Rs{DID_I:.2f} + billable partners "
      f"{y_pm:,.0f}/month x Rs{DID_Y:.2f} = Rs{ROLL_PM/LAKH:.1f} L/month (= cm1; costs do not change with price), "
      f"Rs{ROLL_PM/h25['swaps_pm']:.2f} per network swap")
print(f"Surcharge not collected from non-billable partners: four cities Rs{BILL_PM/LAKH:.2f} L/month "
      f"({n_peak_ctl:,.0f} peak, {n_off_ctl:,.0f} off-peak swaps/month); all six cities Rs{BILL_ALL_PM/LAKH:.2f} L/month")
all_peak = pv.filter(pl.col("band") == "PEAK")["n"].sum()
zd = pv.filter(pl.col("partner_id") == ZIPDROP_ID)
ZD_PM = zd["n"].sum() / 6
ZD_PEAK_SHARE_NET = zd.filter(pl.col("band") == "PEAK")["n"].sum() / all_peak
ZD_PEAK_SHARE_OWN = zd.filter(pl.col("band") == "PEAK")["n"].sum() / zd["n"].sum()
print(f"ZipDrop: {ZD_PM:,.0f} swaps/month ({ZD_PM/h25['swaps_pm']:.1%} of network); {ZD_PEAK_SHARE_OWN:.1%} of its swaps in "
      f"PEAK bands = {ZD_PEAK_SHARE_NET:.1%} of all network PEAK-band swaps (H1-25)")

# %% [markdown]
# **5b. Partner terms (proposal d) and 3W pricing.** Steady-state cm1 replaces the energy and wear of Kyron-issued swaps
# with the non-Kyron averages of the same vehicle class (Nov-24..Jun-25, after the ZipDrop amendment; q5's definition).

# %%
pe = (events(["month", "partner_id", "vehicle_class", "supplier_out", "is_completed", "revenue", "energy_cost", "wear_cost",
              "fixed_cost", "list_price_inr"])
      .filter(pl.col("is_completed") & pl.col("month").is_in(CUR_TERMS))
      .with_columns(kyr=pl.col("supplier_out") == "Kyron"))
nkavg = pe.filter(~pl.col("kyr")).group_by("vehicle_class").agg(e_nk=pl.col("energy_cost").mean(), w_nk=pl.col("wear_cost").mean()).collect()
pss = (pe.join(nkavg.lazy(), on="vehicle_class")
       .with_columns(e_ss=pl.when(pl.col("kyr")).then(pl.col("e_nk")).otherwise(pl.col("energy_cost")),
                     w_ss=pl.when(pl.col("kyr")).then(pl.col("w_nk")).otherwise(pl.col("wear_cost")))
       .group_by("partner_id")
       .agg(swaps=pl.len(), rev=pl.col("revenue").sum(), cm1=(pl.col("revenue") - pl.col("energy_cost") - pl.col("wear_cost")).sum(),
            cm1_ss=(pl.col("revenue") - pl.col("e_ss") - pl.col("w_ss")).sum(), list_sum=pl.col("list_price_inr").sum())
       .with_columns(cm1_ps=pl.col("cm1") / pl.col("swaps"), cm1_ss_ps=pl.col("cm1_ss") / pl.col("swaps"),
                     disc=1 - pl.col("rev") / pl.col("list_sum"))
       .collect(engine="streaming")
       .join(pf.select("partner_id", "partner_name"), on="partner_id", how="left")
       .sort("cm1_ss_ps"))
show("Partner steady-state cm1 per swap, Nov-24..Jun-25 (null partner = independents)", pss)
pss.write_csv(TAB / f"{QID}_partner_steady_state.csv")
zdr = pss.filter(pl.col("partner_id") == ZIPDROP_ID).row(0, named=True)
ffr = pss.filter(pl.col("partner_id") == "FP-01").row(0, named=True)
zd_list_ps = zdr["list_sum"] / zdr["swaps"]
ZD_SS_PM = zdr["cm1_ss_ps"] * ZD_PM
ZD_TO15_PM = (0.28 - 0.15) * zd_list_ps * ZD_PM
ZD_TO12_PM = (0.28 - 0.12) * zd_list_ps * ZD_PM
zd_band = {r["band"]: r["n"] / 6 for r in zd.group_by("band").agg(pl.col("n").sum()).iter_rows(named=True)}
ZD_BILL_PM = zd_band.get("PEAK", 0) * 15 - zd_band.get("OFFPEAK", 0) * 8
print(f"ZipDrop steady-state cm1 {zdr['cm1_ss_ps']:+.2f}/swap (actual {zdr['cm1_ps']:+.2f}; realised discount {zdr['disc']:.1%}) -> "
      f"exclusive on current terms Rs{ZD_SS_PM/LAKH:+.2f} L/month cm1; discount 28%->15% +Rs{ZD_TO15_PM/LAKH:.2f} L/month, "
      f"->12% +Rs{ZD_TO12_PM/LAKH:.2f} L/month; billable surcharge (network-wide bands) +Rs{ZD_BILL_PM/LAKH:.2f} L/month")
print(f"FeastFly steady-state cm1 {ffr['cm1_ss_ps']:+.2f}/swap; realised discount {ffr['disc']:.1%}")
BE_DISC = 1 - (zdr["rev"] / zdr["swaps"] - zdr["cm1_ss_ps"]) / zd_list_ps
print(f"ZipDrop break-even discount at steady-state cm1 = 0: {BE_DISC:.1%}")

w3 = (pe.filter(pl.col("vehicle_class") == "3W").join(nkavg.lazy(), on="vehicle_class")
      .select(pl.len().alias("swaps"), pl.col("revenue").sum().alias("rev"), pl.col("list_price_inr").sum().alias("lst"),
              pl.col("e_nk").first(), pl.col("w_nk").first()).collect().row(0, named=True))
ratio3 = w3["rev"] / w3["lst"]
cm1_3w_ss = w3["rev"] / w3["swaps"] - w3["e_nk"] - w3["w_nk"]
be_list_3w = (w3["e_nk"] + w3["w_nk"]) / ratio3
sw3_pm = mon.filter(pl.col("month").is_in(WIN["H1-25"]) & (pl.col("vehicle_class") == "3W"))["swaps"].sum() / 6
P3_PM = -cm1_3w_ss * sw3_pm
print(f"3W steady state: revenue {w3['rev']/w3['swaps']:.2f}, energy {w3['e_nk']:.2f}, wear {w3['w_nk']:.2f} -> cm1 {cm1_3w_ss:+.2f}/swap; "
      f"realised/list {ratio3:.3f}; break-even list Rs{be_list_3w:.0f}; {sw3_pm:,.0f} 3W swaps/month; pricing to break-even "
      f"+Rs{P3_PM/LAKH:.1f} L/month (if demand holds; 3W elasticity n.s. in q5)")

# %% [markdown]
# **5c. Stations (proposal a).** Fixed cost and swaps per station-month by wave (Apr-Jun 2025); the swaps a station needs
# to cover its fixed cost at the steady-state cm1 per swap (Q1-25 with Cellora-grade packs).

# %%
SS_CM1 = W["Q1-25"]["cm1_allcellora_ps"]
smq = (station_month().filter(pl.col("month").is_in(WIN["Q2-25"]))
       .group_by("expansion_wave").agg(stations=pl.col("station_id").n_unique(), fixed=pl.col("fixed_cost").sum(),
                                        swaps=pl.col("completed").sum(), sm=pl.len())
       .with_columns(fixed_pm=pl.col("fixed") / pl.col("sm"), swaps_pm=pl.col("swaps") / pl.col("sm"),
                     fixed_ps=pl.col("fixed") / pl.col("swaps"))
       .with_columns(breakeven_swaps_pm=pl.col("fixed_pm") / SS_CM1,
                     cm2_pm_at_ss=pl.col("swaps_pm") * SS_CM1 - pl.col("fixed_pm"))
       .sort("expansion_wave"))
show(f"Station economics by wave, Apr-Jun 2025 (steady-state cm1 Rs{SS_CM1:.2f}/swap)", smq)
smq.write_csv(TAB / f"{QID}_station_economics.csv")
w1 = smq.filter(pl.col("expansion_wave") == "Wave1_2024H2").row(0, named=True)
lg = smq.filter(pl.col("expansion_wave") == "Launch").row(0, named=True)
WAVE_FIXED_PM = 30 * w1["fixed_pm"]
WAVE_CM2_PM = 30 * w1["cm2_pm_at_ss"]
print(f"Another wave of 30 Wave-1-like sites: +Rs{WAVE_FIXED_PM/LAKH:.2f} L/month fixed cost; at Wave-1 utilisation and "
      f"steady-state cm1 the wave contributes Rs{WAVE_CM2_PM/LAKH:+.2f} L/month cm2 if every swap is incremental, "
      f"-Rs{WAVE_FIXED_PM/LAKH:.2f} L/month if swaps are only redistributed")

# %% [markdown]
# **5d. Batteries (proposal b).** Fleet expansion of +10% 2W packs vs the evidence that pack count does not move stockouts
# (q4: +1,385 packs, +33.5% of the 2W fleet, changed stockouts by -0.00 pp). Fade is calendar-driven (q4), so idle extra
# packs depreciate without adding swaps.

# %%
n2w_ids = int(bt.filter(pl.col("vehicle_class") == "2W").height)
ADD = round(0.10 * n2w_ids)
cost2w = float(nk2.cost_mean)
EXP_CAPEX = ADD * cost2w
life_m = float(nk2.life_days_median) / 30.44
EXP_DEP_PM = EXP_CAPEX / life_m
print(f"+10% 2W packs = {ADD:,} packs x Rs{cost2w:,.0f} = Rs{EXP_CAPEX/CRORE:.2f} Cr; calendar life {life_m:.1f} months -> "
      f"Rs{EXP_DEP_PM/LAKH:.2f} L/month depreciation with no measured stockout benefit")
KY_GAP_PM = W["Q1-25"]["kyron_gap_ps"] * h25["swaps_pm"]
print(f"Quality-gated replacement (Cellora-grade vs Kyron-grade at ~25% of issues): +Rs{W['Q1-25']['kyron_gap_ps']:.2f}/swap, "
      f"Rs{KY_GAP_PM/LAKH:.1f} L/month at H1-25 volume")

# %% [markdown]
# ## 6. Cited results from the verified analyses (printed so the synthesis quotes only printed numbers)

# %%
CITED = {
    "q2: hot-city Gen1 station-days stockout rate, peak cabinet >=45C vs <40C": "12.00% vs 2.34%",
    "q2: completed queue wait, lightest vs busiest station-hours (s)": "244.0 vs 243.8",
    "q2: Gen1 vs Gen3 risk ratio, hot cities, Apr-Jun 2025": "3.05 (2.97-3.12)",
    "q3: Gen1 charge time at <40C vs 50C cabinet (min)": "88 -> ~168 (Gen3 42 -> 44)",
    "q3: heat x generation lifts station-month R2": "0.552 -> 0.945",
    "q3: within-Gen1 load effect on failure rate (pp per log-unit)": "-0.03 (CI -0.70 to +0.65, p=0.94)",
    "q3: hotspot attempts Apr-Jun 2024 -> 2025 at flat 16.7% -> 16.6% rate": "+50.9%",
    "q3: new-slot DiD on legacy failure rate at median intensity (pp)": "-0.16",
    "q3: fixed cost per swap Wave-1 vs legacy, Apr-Jun 2025 (Rs)": "14.73 vs 11.33",
    "q3: 3 unserved-zone sites vs 16 served-zone fuel pumps (swaps/slot-day)": "11.56 vs 3.05",
    "q4: Kyron bad-lot fade vs Cellora/Amptek, per day": "3.8x (3.26x per issue)",
    "q4: adding 1,385 packs (+33.5% 2W fleet): change in stockouts / charged packs per station-hour": "-0.00 pp / +0.004",
    "q4: Kyron vs Cellora/Amptek median km per returned 2W pack, Q2-25": "39.5 vs 53.4 (-26%)",
    "q4: June-25 Kyron 2W issues below 70% SoH": "99.94% (65,504 of 65,543)",
    "q4: warranty/claim basis, lost-life value (2W + 3W)": "Rs3.37-3.56 Cr",
    "q5: pilot revenue per independent swap": "+Rs7.90 (+11.7%)",
    "q5: pilot effect on peak stockouts (pp)": "+0.05 (CI -0.09 to +0.18)",
    "q5: ZipDrop amendment revenue effect per 2W swap / bought volume": "-Rs10.40 / none (+56.5% vs +53.2%)",
    "q6: same-season D31-60 retention, Jan-Mar 2024 vs 2025 cohorts": "86.3% vs 86.3% (p=0.98)",
    "q6: 2+ vs 0-1 early failures, adjusted OR": "1.67 (1.43-1.94)",
    "q6: D31-60 non-retention with 2+ vs 0-1 early failures (n=2,246 with 2+)": "22.6% vs 13.1%",
    "q6: share of summer-2024 retention dip explained by early failures": "40-50%",
    "q6: what-if early failures capped at 1 (non-retention pp, overall / summer-entry)": "-1.45 / -2.72",
    "q6: pilot DiD on independent new-rider non-retention": "OR 1.28 (0.91-1.79), p=0.15",
    "q7: wear charged on post-EOL swaps": "Rs12.9M",
    "q7: test stations excluded (paid swaps / revenue)": "59,134 / Rs3.93M",
    "q7: 1-Jul-24 failure drop in hot cities vs cool cities (pts)": "-8 to -15 vs within +/-0.8 (heat, not price)",
    "q8: battery-health share of tickets (central; official low_range)": "49.5% (15.1%)",
    "q1: 2W price per km H1-24 -> H1-25": "Rs0.819 -> Rs1.165 (+42.3%)",
    "q1: km per 2W swap Jan-24 -> Jun-25": "70.5 -> 48.0",
    "q1: cm1 bridge Q1-24 -> Q1-25 (Rs/swap)": "Kyron -16.04, price +5.12, kWh +2.80, ZipDrop -1.70, pilot +1.68, mix +0.46",
    "q1: cm1 bridge Jul-Aug-24 -> Oct-24..Mar-25 (Rs/swap)": "Kyron -16.00, ZipDrop -1.45, pilot +1.69, kWh +1.31",
    "q2: Apr-Jun failure rate by city (Jaipur / Delhi NCR / Hyderabad / Mumbai)": "13.60% / 12.20% / 10.36% / 4.03%",
    "q2: avoidable failures per Gen1 site per summer (Delhi / Jaipur / Hyderabad / Pune)": "~835 / ~810 / ~644 / ~208",
    "q2: 3W vs 2W stockout risk ratio within station-month": "2.08 (2.06-2.11)",
    "q2: completed queue wait mean, every month (min)": "4.05-4.07",
    "q3: competitor entry effect on completed swaps": "-3.9% (CI -10.7% to +3.3%), MDE ~10%",
    "q5: cm1 per swap by segment, current terms (quick commerce / cargo 3W / independents)": "-17.51 / -22.01 / +1.43",
    "q5: ZipDrop steady-state cm2 after working capital": "-Rs16.00/swap = -Rs80.7 L/yr",
    "q5: ZipDrop 2W break-even discount at steady-state cm1": "24.6%",
    "q6: competitor exposure, attempt-level exit OR / landmark Cox HR": "1.42 / 1.20",
    "q7: system-error rate in every connectivity tier": "0.29-0.30%",
    "q8: stock-out + queue share of tickets": "39.5%",
    "q4: Kyron share of 2W packs issued from Oct-24": "~24-25%",
    "q1: monthly active riders H1-25 vs H1-24 / real stations Jun-25": "x2.18 / 149",
    "q1: Q2-25 attempts per slot-day, new vs Launch stations": "4.8 vs 5.0",
    "q2: Gen1 vs Gen3 RR in Mumbai / Pune, Apr-Jun 2025": "1.00 / 1.72",
    "q2: O/E of all 36 hot-city Gen1 sites vs their city x month x class rate": "0.95-1.05 (no rogue stations)",
    "q2: Jul-Mar failure rate range across cities": "3.85-4.00%",
    "q3: excess stockouts per hot hour within Gen1 station": "+0.81 pp (0.78-0.85)",
    "q3: tickets per extra failed attempt": "~1 per 9.9",
    "q4: wear per swap, Kyron 2W vs Cellora 2W": "Rs103.84 vs Rs31.85",
    "q4: battery value destroyed by the bad lots, capped vs per-swap model": "Rs2.94 Cr vs Rs4.31 Cr",
    "q4: H1-25 cm1 with depreciation capped at purchase price": "+Rs1.64/swap",
    "q4: early-warning back-test, bad lots' fade ratio by week 8": "2.47-4.50x (others <=1.37x)",
    "q4: sub-EOL swaps in Jun-25 / extra swaps per rider-day in Kyron-heavy cities": "66,473 / +9.8%",
    "q5: pilot peak-share shift / city-permutation p (floor of the design)": "-2.49 pp / 0.067",
    "q5: partner cost-to-serve gaps vs peers (Rs/swap)": "-0.71 to +0.80",
    "q6: within-rider exit OR after stockout/abandon / subgroups adverse": "1.525 / 15 of 15",
    "q6: competitor-exposed eligible riders / 3W OR": "640 / 1.18 (1.01-1.37)",
    "q6: SoH OR per +10 pts / Kyron share OR / peak share OR": "1.07 / 0.90 / 1.06",
    "q6: pilot placebo on partner riders / churn model AUC": "OR 1.20 / 0.60-0.68",
    "q6: share of each summer dip closed by removing early failures (q6_01)": "28-61%",
    "q7: cm1/swap May/Jun-25 without wear past EOL / tariff_code share of pilot pricing hidden": "+12.4 / +18.0; 54.0%",
    "q7: hot-city temperature fall across 1-Jul-24": "11-17C",
    "q8: battery-complaint floor / Kyron dose ratio / hidden vs official multiple": "36.7% / 1.74 / 2.4-3.3x",
    "q8: pilot billing-complaint IRR vs Mumbai": "0.94 (0.78-1.12)",
    "q3: competitor DiD treated stations": "8",
    "q2: load at hot-city Gen3 vs Gen1 sites, Apr-Jun (attempts per station-day)": "84.1 vs 81.8",
}
for k_, v_ in CITED.items():
    print(f"CITED | {k_}: {v_}")

# %% [markdown]
# ## 7. Budget scorecard (all options in the same units)
# Units: Apr-Jun network failure rate (pp, 2025 volumes); new-rider D31-60 non-retention (pp; negative = better);
# cm1 in Rs lakh per month at H1-2025 run-rate (summer-only effects averaged over 12 months); one-off cost.

# %%
gen1_rev_annual_pm = (AV2 * REV_PS_HG1 / 12, AV3 * REV_PS_HG1 / 12)
rev_ret_pm = (ret_g2 * sw_per_ret * h25["rev_ps"] / 12, ret_g3 * sw_per_ret * h25["rev_ps"] / 12)
score = [
    dict(option="(a) More stations (another 30-site wave)", verdict="REJECT; demand-led sites only",
         fail_pp="~0 (-0.16 pp at existing sites; hotspot load effect -0.03 pp)", ret_pp="~0 (wave n.s.)",
         cm1_lakh_pm=f"{WAVE_CM2_PM/LAKH:+.1f} to {-WAVE_FIXED_PM/LAKH:+.1f} (cm2)", oneoff="site capex not in data",
         why="Hotspot rates are set by heat x Gen1, not load: +51% attempts at a flat 16.6%. Only demand-led sites pay."),
    dict(option="(b) More batteries (+10% 2W packs)", verdict="REJECT; replace instead",
         fail_pp="~0 (+33.5% packs moved stockouts -0.00 pp)", ret_pp="~0 (SoH not a churn driver)",
         cm1_lakh_pm=f"{-EXP_DEP_PM/LAKH:+.1f} (depreciation)", oneoff=f"Rs{EXP_CAPEX/CRORE:.1f} Cr",
         why="Hourly stock does not predict stockouts; hot-day stockouts occur with full cabinets."),
    dict(option="(c) Network-wide peak/off-peak pricing", verdict="SUPPORT with changes",
         fail_pp="~0 (+0.05 pp, CI -0.09 to +0.18)", ret_pp="n.s. (OR 1.28, p=0.15)",
         cm1_lakh_pm=f"{ROLL_PM/LAKH:+.1f} (+{BILL_ALL_PM/LAKH:.1f} if partners billed)", oneoff="~0",
         why="A revenue lever, not a congestion lever: +Rs7.90 per independent swap, timing barely moves."),
    dict(option="(d) Long-term exclusive with ZipDrop (largest by swaps)", verdict="REJECT on current terms",
         fail_pp=f"locks {ZD_PEAK_SHARE_NET:.0%} of peak swaps", ret_pp="~0 (OR 1.06)",
         cm1_lakh_pm=f"{ZD_SS_PM/LAKH:+.1f} (steady state)", oneoff="-",
         why=f"28% discount, no billable surcharge: cm1 {zdr['cm1_ss_ps']:+.1f}/swap even with healthy packs; amendment bought no volume."),
    dict(option="ALT 1: Gen1 thermal fix at 36 hot-city sites", verdict="FUND as pilot, then scale",
         fail_pp=f"{(rate_after2-NET_AJ25)*100:+.1f} to {(rate_after3-NET_AJ25)*100:+.1f} pp ({NET_AJ25:.2%} -> {rate_after3:.2%})",
         ret_pp=f"{S25['nonret_pp_gen2_central']:+.2f} to {S25['nonret_pp_gen3_central']:+.2f} (summer entrants)",
         cm1_lakh_pm=f"+{(gen1_rev_annual_pm[0]+rev_ret_pm[0])/LAKH:.1f} to +{(gen1_rev_annual_pm[1]+rev_ret_pm[1])/LAKH:.1f} (revenue)",
         oneoff="capex not in data: pilot sets ceiling",
         why=f"{hg25['sh_excess']:.0%} of Apr-Jun-25 excess failures; same-city Gen3 sites at similar load fail 3x less."),
    dict(option="ALT 2: EOL lockout + quality-gated pack replacement", verdict="FUND (replace, not expand)",
         fail_pp="~0", ret_pp="~0", cm1_lakh_pm=f"+{KY_GAP_PM/LAKH:.0f} vs Kyron-grade packs",
         oneoff=f"Rs{bad_capex/CRORE:.1f} Cr now; Rs{nk2.cost_total/CRORE:.1f} Cr 2W renewal Jan-May 2026",
         why=f"Kyron lots fade 3.8x/day; 122K sub-EOL swaps; non-Kyron 2W fleet reaches 70% ~{nk2.eol_p50:%b-%Y}."),
    dict(option="ALT 3: Contract and price re-terms", verdict="DO (no capex)",
         fail_pp="~0", ret_pp="monitor", cm1_lakh_pm=f"+{(ZD_TO15_PM+BILL_ALL_PM+P3_PM)/LAKH:.1f}",
         oneoff="~0",
         why=f"ZipDrop to 15% (+{ZD_TO15_PM/LAKH:.1f} L), billable surcharge (+{BILL_ALL_PM/LAKH:.1f} L), 3W list to ~Rs{be_list_3w:.0f} (+{P3_PM/LAKH:.1f} L)."),
    dict(option="ALT 4: New-rider failure recovery (A/B)", verdict="TEST (low cost)",
         fail_pp="~0", ret_pp="up to -1.45 (cap at 1 failure)", cm1_lakh_pm="small (volume)", oneoff="credits only",
         why="2+ early failures: 22.6% vs 13.1% not retained; stockout/abandon exits OR 1.5-1.7."),
]
score_df = pd.DataFrame(score)
show("Budget scorecard", score_df)
score_df.to_csv(TAB / f"{QID}_budget_scorecard.csv", index=False)
print(f"Gen1 fix revenue (12-month average): lost swaps recovered Rs{gen1_rev_annual_pm[0]/LAKH:.2f}-{gen1_rev_annual_pm[1]/LAKH:.2f} L/month; "
      f"retained new riders (second month only) Rs{rev_ret_pm[0]/LAKH:.2f}-{rev_ret_pm[1]/LAKH:.2f} L/month")
SUMMER_DIP_PP = 4.58   # cited, q6: Apr-Jun 2024 vs Jan-Mar 2024 cohorts
print(f"Gen1 fix value per site per summer (revenue, 2025 volume): Rs{AV2*REV_PS_HG1/36/1e3:.1f}K-{AV3*REV_PS_HG1/36/1e3:.1f}K; "
      f"share of the {SUMMER_DIP_PP} pp summer retention dip closed: {-S25['nonret_pp_gen2_central']/SUMMER_DIP_PP:.0%}-"
      f"{-S25['nonret_pp_gen3_central']/SUMMER_DIP_PP:.0%}; Gen1 sites' Apr-Jun attempts 2024 -> 2025: "
      f"{seg.filter((pl.col('year') == 2024) & (pl.col('seg') == 'Hot city Gen1'))['att'].item():,} -> {hg25['att']:,}")
cm2_path = dict(start=W["Q1-25"]["cm2_ps"], kyron=W["Q1-25"]["kyron_gap_ps"], rollout=ROLL_PM / h25["swaps_pm"],
                partner=(ZD_TO15_PM + BILL_ALL_PM) / h25["swaps_pm"], three_w=P3_PM / h25["swaps_pm"])
cm2_path["end"] = sum(cm2_path.values())
COMM_PM = ROLL_PM + BILL_ALL_PM + ZD_TO15_PM + P3_PM
print(f"Commercial levers total: rollout {ROLL_PM/LAKH:.2f} + billable surcharge {BILL_ALL_PM/LAKH:.2f} + ZipDrop 15% "
      f"{ZD_TO15_PM/LAKH:.2f} + 3W {P3_PM/LAKH:.2f} = Rs{COMM_PM/LAKH:.1f} L/month cm1; re-terms excl. rollout "
      f"Rs{(COMM_PM-ROLL_PM)/LAKH:.1f} L/month")
print("cm2 per swap path (Rs): " + ", ".join(f"{k_} {v_:+.2f}" for k_, v_ in cm2_path.items()))

# %% [markdown]
# ## 8. Key numbers for the report, deck and video

# %%
KEY = []


def key(metric, value, period, source):
    KEY.append(dict(metric=metric, value=value, period=period, source_analysis=source))
    print(f"KEY | {metric} = {value} | {period} | {source}")


key("Completed swaps growth (x)", round(mult["swaps"], 2), "H1-25 vs H1-24", "q1, q9 recomputed")
key("Revenue growth (x)", round(mult["revenue_pm"], 2), "H1-25 vs H1-24", "q1, q9 recomputed")
key("Service-failure rate (%)", f"{h24['fail_rate']*100:.2f} -> {h25['fail_rate']*100:.2f}", "H1-24 -> H1-25", "q1, q9 recomputed")
key("Naive failures vs swaps growth (x)", f"{naive_fail_x:.2f} vs {naive_swap_x:.2f}", "Jan-24 -> Jun-25 (winter vs summer)", "q1, q9 recomputed")
key("Apr-Jun failure rate, same 90 Launch stations (%)", f"{W['Q2-24']['launch_fail_rate']*100:.2f} -> {W['Q2-25']['launch_fail_rate']*100:.2f}", "Apr-Jun 2024 -> 2025", "q1/q3, q9 recomputed")
key("Apr-Jun failure rate, whole network (%)", f"{W['Q2-24']['fail_rate']*100:.2f} -> {W['Q2-25']['fail_rate']*100:.2f}", "Apr-Jun 2024 -> 2025 (dilution by new sites)", "q1/q3, q9 recomputed")
key("Failed attempts, Apr-Jun", f"{W['Q2-24']['failures']:,} -> {W['Q2-25']['failures']:,}", "Apr-Jun 2024 -> 2025", "q3, q9 recomputed")
key("Off-season (Jul-Mar) failure-rate baseline (%)", round(offseason * 100, 2), "Jul-24..Mar-25 and Jul-Mar months", "q2, q9 recomputed")
key("36 hot-city Gen1 stations: share of attempts / failures / excess failures (%)",
    f"{hg25['sh_att']*100:.1f} / {hg25['sh_fail']*100:.1f} / {hg25['sh_excess']*100:.1f}", "Apr-Jun 2025", "q2/q3, q9 recomputed")
key("Hot-city failure rate Gen1 vs Gen3 (%)",
    f"{hg25['rate']*100:.1f} vs {seg.filter((pl.col('year') == 2025) & (pl.col('seg') == 'Hot city Gen3'))['rate'].item()*100:.1f}", "Apr-Jun 2025", "q2, q9 recomputed")
key("City-day failure rate below 34C vs at 36-38C vs 38-42C (%)",
    f"{HEAT['<34C']['rate']*100:.1f} / {HEAT['36-38C']['rate']*100:.1f} / {HEAT['38-42C']['rate']*100:.1f}", "Jan-24..Jun-25", "q2, q9 recomputed")
alert_below42 = heat.filter(pl.col("tbin") != ">=42C").select((pl.col("alert_share") * pl.col("days")).sum() / pl.col("days").sum()).item()
key("Share of city-days with an official heat alert, below 42C vs at/above 42C (%)",
    f"{alert_below42*100:.1f} vs {HEAT['>=42C']['alert_share']*100:.0f}", "Jan-24..Jun-25", "q2, q9 recomputed")
key("Gen1 charge time, cabinet <40C vs 50C (min)", "88 -> ~168", "Jan-24..Jun-25", "q3 (cited)")
key("Stockout rate on hot vs cool days at hot-city Gen1 stations (%)", "12.0 vs 2.3", "Apr-Jun, station-days", "q2 (cited)")
key("Hot-day 19-22h 2W stockout rate across charged-stock buckets 0 / 1-2 / 3-5 / 6+ (%)",
    " / ".join(f"{r*100:.1f}" for r in hot_eve.sort("stock")["rate"].to_list()), "Apr-Jun, hot-city Gen1", "q2, q9 recomputed")
key("Gen1 fix: failed attempts avoided per summer (Gen2-like to Gen3-like)", f"{AV2:,.0f} to {AV3:,.0f}", "Apr-Jun 2025 volume", "q1/q2/q3, q9 recomputed")
key("Gen1 fix: network Apr-Jun failure rate (%)", f"{NET_AJ25*100:.2f} -> {rate_after2*100:.2f} to {rate_after3*100:.2f}", "Apr-Jun 2025 volume", "q9")
key("Gen1 fix: summer-entry new-rider non-retention (pp)",
    f"{S25['nonret_pp_gen2_central']:+.2f} to {S25['nonret_pp_gen3_central']:+.2f}", "Apr-Jun 2025 entrants", "q9 (q6 effect size)")
key("Same-season new-rider D31-60 retention (%)", "86.3 vs 86.3 (p=0.98)", "Jan-Mar 2024 vs Jan-Mar 2025 cohorts", "q6 (cited)")
key("Non-retention with 2+ vs 0-1 early failures (%)", f"22.6 vs 13.1 (q9 re-derivation {P2*100:.1f} vs {P0*100:.1f})",
    "cohorts Jan-24..Apr-25", "q6 (cited), q9 recomputed")
key("Adjusted odds ratio, 2+ vs 0-1 early failures", "1.67 (1.43-1.94)", "cohorts Jan-24..Apr-25", "q6 (cited)")
key("Share of summer retention dip explained by early failures (%)", "40-50", "Apr-Jun 2024 vs Jan-Mar 2024", "q6 (cited)")
key("cm1 per swap, same season (Rs)", f"{W['Q1-24']['cm1_ps']:+.2f} -> {W['Q1-25']['cm1_ps']:+.2f}", "Q1-24 -> Q1-25", "q1, q9 recomputed")
key("cm1 per swap H1-25 under three wear conventions (Rs)",
    f"{h25['cm1_ps']:+.2f} / {h25['cm1_celwear_ps']:+.2f} / {h25['cm1_cap_ps']:+.2f}", "H1-25 (per-cycle / post-EOL at Cellora wear / wear stops at EOL)", "q1/q4/q7, q9 recomputed")
key("Kyron-grade vs Cellora-grade wear gap (Rs per swap; Rs lakh per month)",
    f"{W['Q1-25']['kyron_gap_ps']:.2f}; {KY_GAP_PM/LAKH:.1f}", "Q1-25 per-swap gap (clean window) x H1-25 volume", "q1, q9 recomputed")
key("Gen1 fix: revenue recovered per summer if each failure is a lost swap (Rs M)",
    f"{AV2*REV_PS_HG1/1e6:.2f} to {AV3*REV_PS_HG1/1e6:.2f}", "Apr-Jun 2025 volume (upper bound)", "q9")
key("Kyron bad-lot fade vs other lots (per day)", "3.8x", "Sep-24..Jun-25", "q4 (cited)")
key("Swaps that issued a pack below 70% SoH", f"{h25['post_eol_swaps']:,}", "Apr-Jun 2025", "q1/q4/q7, q9 recomputed")
key("Median km per returned 2W pack, Kyron vs others", "39.5 vs 53.4", "Q2-25", "q4 (cited)")
key("Non-Kyron 2W fleet: median projected date at 70% SoH", f"{nk2.eol_p50:%b-%Y} (trend method {trend_date:%b-%Y})", "projection from 30-Jun-25", "q1/q4, q9 recomputed")
key("Pack capex: Kyron bad lots now / non-Kyron 2W renewal Jan-May 2026 / non-Kyron 3W renewal 2027 (Rs Cr)",
    f"{bad_capex/CRORE:.2f} / {nk2.cost_total/CRORE:.2f} / {nk3.cost_total/CRORE:.2f}", "purchase cost, batteries master", "q9")
key("Pricing rollout cm1 uplift (Rs lakh per month)", round(ROLL_PM / LAKH, 1), "four non-pilot cities, H1-25 volume", "q5 effect, q9 volumes")
key("Pilot effect on peak stockouts (pp)", "+0.05 (CI -0.09 to +0.18)", "Oct-24..Mar-25", "q5 (cited)")
key("ZipDrop share of network swaps (%)", round(ZD_PM / h25["swaps_pm"] * 100, 1), "H1-25", "q5, q9 recomputed")
key("ZipDrop steady-state cm1 per swap (Rs)", round(zdr["cm1_ss_ps"], 2), "Nov-24..Jun-25 terms", "q5, q9 recomputed")
key("FeastFly steady-state cm1 per swap (Rs)", round(ffr["cm1_ss_ps"], 2), "Nov-24..Jun-25 terms", "q5, q9 recomputed")
key("3W steady-state cm1 per swap (Rs) / break-even list price (Rs)", f"{cm1_3w_ss:+.2f} / {be_list_3w:.0f}", "Nov-24..Jun-25", "q5, q9 recomputed")
key("Wave-1 vs legacy fixed cost per swap (Rs)", f"{w1['fixed_ps']:.2f} vs {lg['fixed_ps']:.2f}", "Apr-Jun 2025", "q3, q9 recomputed")
key("Battery-health share of support tickets (%)", "49.5 (official 15.1)", "Jan-24..Jun-25", "q8 (cited)")
key("cm2 per swap: Q1-25 actual -> with recommended levers (Rs)", f"{cm2_path['start']:+.2f} -> {cm2_path['end']:+.2f}", "Q1-25 unit costs, H1-25 volume", "q9")
kdf = pd.DataFrame(KEY)
kdf.to_csv(TAB / "key_numbers.csv", index=False)
print(f"\nkey_numbers.csv: {len(kdf)} rows")

# %% [markdown]
# ## 9. Figure 1 - causal chain (drivers -> mechanisms -> outcomes)

# %%
C_PRI, C_SEC, C_MECH, C_OUT, C_NOT = PALETTE["accent"], PALETTE["warn"], "#E8EEF4", PALETTE["primary"], "#EFEFEF"


def box(ax, x, y, w, h, text, fc, tc="white", fs=10.5, weight="normal", ec="none"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.012", fc=fc, ec=ec, lw=1.0,
                                transform=ax.transAxes, clip_on=False))
    ax.text(x + 0.012, y + h / 2, text, ha="left", va="center", fontsize=fs, color=tc, weight=weight,
            transform=ax.transAxes, linespacing=1.3)
    return (x, y, w, h)


def link(ax, b1, b2, label=None, color="#4A4A4A", lw=1.8, ls="-", start="r", end="l", f1=0.5, f2=0.5,
         lab_dx=0.0, lab_dy=0.0, fs=9.5, ha="center"):
    """Arrow from side `start` of box b1 to side `end` of b2; f1/f2 = position along that side (0-1)."""
    pts = {"r": lambda b, f: (b[0] + b[2], b[1] + b[3] * f), "l": lambda b, f: (b[0], b[1] + b[3] * f),
           "b": lambda b, f: (b[0] + b[2] * f, b[1]), "t": lambda b, f: (b[0] + b[2] * f, b[1] + b[3])}
    (x1, y1), (x2, y2) = pts[start](b1, f1), pts[end](b2, f2)
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, shrinkA=1, shrinkB=1, mutation_scale=15))
    if label:
        ax.text((x1 + x2) / 2 + lab_dx, (y1 + y2) / 2 + lab_dy, label, fontsize=fs, color=color, ha=ha, va="center",
                transform=ax.transAxes, bbox=dict(fc="white", ec="none", pad=1.0), weight="bold")


fig = plt.figure(figsize=(20, 11.5))
ax = fig.add_axes([0.01, 0.02, 0.98, 0.84]); ax.axis("off")
fig.text(0.01, 0.965, "Two equipment problems, not growth, drive most of the failures, churn and margin loss",
         fontsize=19, weight="bold", ha="left")
fig.text(0.01, 0.925, "Heat-sensitive Gen1 chargers at 36 hot-city stations and one defective battery batch (Kyron lots KY-2407/08/09). "
         "Red = primary driver, amber = secondary, grey = tested and rejected.\nVoltRelay, Jan 2024 - Jun 2025; same-season comparisons; "
         "margins under our usage-based battery-depreciation assumption (cm1 = revenue - energy - battery wear).",
         fontsize=11.5, color="#444444", ha="left", va="center")
for xh, txt in ((0.0, "ROOT CAUSE"), (0.335, "MECHANISM (measured)"), (0.715, "OUTCOME LEADERSHIP SEES")):
    ax.text(xh, 1.0, txt, fontsize=12, weight="bold", color="#5A6B7B", transform=ax.transAxes, va="bottom")
XD, WD, XM, WM, XO, WO = 0.0, 0.25, 0.335, 0.29, 0.715, 0.285
D1 = box(ax, XD, 0.80, WD, 0.17, "PRIMARY\nSummer heat (city max >=36C) x Gen1\nchargers at 36 Launch stations in\nDelhi NCR, Jaipur, Hyderabad", C_PRI, weight="bold")
D2 = box(ax, XD, 0.585, WD, 0.17, "PRIMARY\nDefective battery lots KY-2407/08/09\n(all 1,385 Kyron 2W packs, Sep-24) and\nno end-of-life lockout at the cabinet", C_PRI, weight="bold")
D3 = box(ax, XD, 0.405, WD, 0.14, "SECONDARY\nContract and price terms: ZipDrop 28%\ndiscount, surcharge not billable;\n3W list price below cost", C_SEC, tc="#222222", weight="bold")
D4 = box(ax, XD, 0.255, WD, 0.115, "SECONDARY\nNeed-blind expansion: Wave-1 fuel-pump\nand residential sites in served zones", C_SEC, tc="#222222", weight="bold")
D5 = box(ax, XD, 0.105, WD, 0.115, "SECONDARY\nWhole-fleet calendar ageing\n(non-Kyron 2W packs at ~80% SoH)", C_SEC, tc="#222222", weight="bold")
M1 = box(ax, XM, 0.80, WM, 0.17, "Gen1 charge time 88 -> ~168 min at a 50C\ncabinet (Gen3 42 -> 44). Hot Gen1 days stock out\n"
         "5x as often (12.0% vs 2.3%). Station load does\nnot matter (-0.03 pp, p=0.94); stock level\nin the hour does not either.", C_MECH, tc="#1B1B1B")
M2 = box(ax, XM, 0.585, WM, 0.17, "SoH fades 3.8x faster per day -> wear Rs104 vs\nRs32 per 2W swap. "
         f"{h25['post_eol_swaps']/1e3:.0f}K swaps issued packs below\n70% SoH (Jun-25: 99.9% of Kyron 2W issues),\n"
         "delivering 26% fewer km (39.5 vs 53.4).", C_MECH, tc="#1B1B1B")
M3 = box(ax, XM, 0.405, WM, 0.14, "ZipDrop -Rs10.4 per 2W swap, bought no volume;\n"
         f"3W cm1 {cm1_3w_ss:+.1f}/swap even with healthy packs;\nquick-commerce segment cm1 -Rs17.5/swap", C_MECH, tc="#1B1B1B")
M4 = box(ax, XM, 0.255, WM, 0.115, f"Wave-1 fixed cost Rs{w1['fixed_ps']:.1f}/swap vs Rs{lg['fixed_ps']:.1f}\nlegacy; "
         "utilisation follows unmet zone demand", C_MECH, tc="#1B1B1B")
M5 = box(ax, XM, 0.105, WM, 0.115, "km per 2W swap 70.5 -> 48.0 (Jan-24 -> Jun-25);\nrider price per km +42% year on year", C_MECH, tc="#1B1B1B")
O1 = box(ax, XO, 0.80, WO, 0.17, "SERVICE FAILURES\n"
         f"{hg25['sh_excess']:.0%} of Apr-Jun-25 excess failures at the 36 sites\n(on {hg25['sh_att']:.0%} of attempts). Like-for-like rate flat\n"
         f"({W['Q2-24']['launch_fail_rate']:.2%} -> {W['Q2-25']['launch_fail_rate']:.2%}); failed attempts "
         f"+{W['Q2-25']['failures']/W['Q2-24']['failures']-1:.0%}", C_OUT, weight="bold")
O2 = box(ax, XO, 0.585, WO, 0.15, "NEW-RIDER CHURN\nNo structural decline (86.3% vs 86.3% same\nseason). Summer-entry dip: 40-50% explained\nby failures in a rider's first 14 days", C_OUT, weight="bold")
O3 = box(ax, XO, 0.30, WO, 0.235, "MARGIN EROSION\n"
         f"cm1/swap {W['Q1-24']['cm1_ps']:+.2f} -> {W['Q1-25']['cm1_ps']:+.2f} (Q1-24 -> Q1-25).\n"
         f"Kyron wear alone {-W['Q1-25']['kyron_gap_ps']:+.1f}/swap. Offsets: price\nrise +5.1, pilot +1.7; lower fixed cost\nper swap from utilisation (cm2)", C_OUT, weight="bold")
O4 = box(ax, XO, 0.105, WO, 0.15, "COMPLAINTS\nBattery health ~50% of tickets (official\n'low_range' 15%), up with Kyron (dose test\n1.74x); stock-outs + queues 40%", C_OUT, weight="bold")
link(ax, D1, M1, "R2 0.55 -> 0.95", lab_dy=0.03)
link(ax, M1, O1, f"{hg25['sh_fail']:.0%} of failures", lab_dy=0.03)
link(ax, O1, O2, "2+ early failures: 22.6% vs 13.1%\nnot retained (OR 1.67)", start="b", end="t", f1=0.3, f2=0.3,
     lab_dx=0.012, fs=9.5, ha="left")
link(ax, D2, M2)
link(ax, M2, O3, f"{-W['Q1-25']['kyron_gap_ps']:+.1f}/swap\n(Rs{KY_GAP_PM/LAKH:.0f} L/month)", f1=0.25, f2=0.85, lab_dy=0.012)
link(ax, D3, M3)
link(ax, M3, O3, "-Rs1.5 to -1.7/swap\n(ZipDrop amendment)", f1=0.5, f2=0.55, lab_dy=0.0)
link(ax, D4, M4)
link(ax, M4, O3, "cm2 only", color="#7A7A7A", lw=1.4, f1=0.5, f2=0.2, lab_dy=-0.012)
link(ax, D5, M5)
link(ax, M5, O4, "reported range\n54 -> 46 km", color="#7A7A7A", lw=1.4, f1=0.5, f2=0.45, lab_dy=0.0)
box(ax, 0.0, -0.035, 1.0, 0.105,
    "TESTED, NOT DRIVERS:  station load and queue capacity (completed wait 4.06 min at every load)  |  number of packs (+33.5% packs: "
    "stockouts -0.00 pp)  |  connectivity and firmware\n(system errors 0.30% in every tier)  |  battery SoH and peak pricing as churn drivers "
    "(n.s.)  |  competitor entry (-3.9%, n.s.)  |  the 1-Jul-24 price rise as a failure cause (that drop was the end of the heat)",
    C_NOT, tc="#444444", fs=10.5)
save_fig(fig, f"{QID}_01_causal_chain")

# %% [markdown]
# ## 10. Figure 2 - budget scorecard

# %%
VERDICT_COL = {"REJECT": PALETTE["accent"], "SUPPORT": PALETTE["good"], "FUND": PALETTE["good"], "DO": PALETTE["good"],
               "TEST": PALETTE["primary"]}
GOOD, BAD, NEU = "#DCEFE3", "#F8DAD3", "#F2F2F2"


def tone(s):
    s = str(s).strip()
    if s.startswith("~0") or s in ("-", "monitor", "small (volume)", "~0 (wave n.s.)") or s.startswith("n.s") or s.startswith("locks") or s.startswith("site capex"):
        return NEU if not s.startswith("locks") else BAD
    if s.startswith("+") or s.startswith("up to -") or (s.startswith("-") and "pp" in s and "ALT" not in s):
        return GOOD
    return NEU


cols = [("option", "Option", 0.165), ("verdict", "Verdict", 0.115), ("fail_pp", "Apr-Jun failure rate", 0.13),
        ("ret_pp", "New-rider churn, pp", 0.115), ("cm1_lakh_pm", "cm1, Rs lakh/month", 0.12),
        ("oneoff", "One-off cost", 0.115), ("why", "Why (evidence)", 0.24)]
fig = plt.figure(figsize=(22, 11.2))
ax = fig.add_axes([0.005, 0.02, 0.99, 0.845]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.text(0.01, 0.965, "Budget: fix the 36 Gen1 sites and buy better batteries; do not buy capacity, and do not lock in ZipDrop's terms",
         fontsize=18.5, weight="bold", ha="left")
fig.text(0.01, 0.905, f"Each option in the same units: Apr-Jun network failure rate (2025 volumes, {NET_AJ25:.2%} today); new-rider D31-60 "
         "non-retention (negative = fewer riders lost); cm1 in Rs lakh per month at the H1-2025 run-rate\n"
         f"({h25['swaps_pm']/1e3:.0f}K swaps/month; summer-only gains averaged over 12 months). Green = better, red = worse, grey = no measurable effect. "
         "Margins under our usage-based battery-depreciation assumption.", fontsize=11.5, color="#444444", ha="left", va="center")
y = 0.985
xs = np.cumsum([0] + [c[2] for c in cols])
for (k_, lab, w_), x0 in zip(cols, xs):
    ax.text(x0 + 0.004, y, lab, fontsize=11, weight="bold", color="#3A4A5A", va="top")
y -= 0.045
row_h = 0.103
for i, r in score_df.iterrows():
    if i == 4:
        ax.add_patch(Rectangle((0, y - 0.004), 1, 0.003, color="#1F4E79", transform=ax.transAxes))
        ax.text(0.004, y - 0.012, "RECOMMENDED ALTERNATIVE ALLOCATION", fontsize=11, weight="bold", color=PALETTE["primary"], va="top")
        y -= 0.04
    top = y
    for (k_, lab, w_), x0 in zip(cols, xs):
        val = str(r[k_])
        if k_ == "verdict":
            vc = next((c for kk, c in VERDICT_COL.items() if val.startswith(kk)), PALETTE["warn"])
            ax.add_patch(FancyBboxPatch((x0 + 0.004, top - row_h + 0.016), w_ - 0.01, row_h - 0.03,
                                        boxstyle="round,pad=0,rounding_size=0.01", fc=vc, ec="none", transform=ax.transAxes))
            ax.text(x0 + w_ / 2, top - row_h / 2, textwrap.fill(val, 17), fontsize=10.5, weight="bold", color="white",
                    ha="center", va="center")
        elif k_ in ("fail_pp", "ret_pp", "cm1_lakh_pm", "oneoff"):
            fc = tone(val) if k_ != "oneoff" else NEU
            if k_ == "cm1_lakh_pm" and val.startswith("-"):
                fc = BAD
            ax.add_patch(Rectangle((x0 + 0.003, top - row_h + 0.012), w_ - 0.006, row_h - 0.022, fc=fc, ec="none", transform=ax.transAxes))
            ax.text(x0 + 0.008, top - row_h / 2, textwrap.fill(val, 23 if w_ >= 0.125 else 21), fontsize=10.5, va="center", color="#1B1B1B")
        else:
            wrapw = 27 if k_ == "option" else 46
            ax.text(x0 + 0.006, top - row_h / 2, textwrap.fill(val, wrapw), fontsize=10.5 if k_ == "why" else 11.5,
                    weight="bold" if k_ == "option" else "normal", va="center", color="#1B1B1B")
    y -= row_h
ax.text(0.004, y - 0.005, f"Path to positive cm2 per swap (Q1-25 unit costs, H1-25 volume): {cm2_path['start']:+.1f} today "
        f"+ {cm2_path['kyron']:.1f} Cellora-grade packs + {cm2_path['rollout']:.1f} pricing rollout + {cm2_path['partner']:.1f} partner terms "
        f"+ {cm2_path['three_w']:.1f} 3W price = {cm2_path['end']:+.1f}.  Station and Gen1-upgrade capex are not in the data: "
        "the Gen1 pilot sets the spending ceiling.", fontsize=11, color="#333333", va="top")
save_fig(fig, f"{QID}_02_budget_decision")
print("figures saved:", f"{QID}_01_causal_chain.png", f"{QID}_02_budget_decision.png")
