# %% [markdown]
# ### Two closing visuals for the budget discussion
# 1. **Path to a positive cm2 per swap**: the margin effect of the recommended levers, in rupees per swap (values printed by
#    the synthesis script above, at Q1-2025 unit costs and H1-2025 volume).
# 2. **Fleet end-of-life timeline**: when each battery pack is projected to cross 70% state of health, from its own calendar
#    fade. This is why the battery budget is about *replacement*, not expansion.

# %%
import re, sys, datetime as dt
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
try:
    from common import batteries, set_style, save_fig, PALETTE, TAB, EOL_SOH
except ImportError:
    sys.path.insert(0, str(Path.cwd() / "src")); from common import batteries, set_style, save_fig, PALETTE, TAB, EOL_SOH

set_style()
log = (TAB / "q9_run_log.txt").read_text(encoding="utf-8")
m = re.search(r"cm2 per swap path \(Rs\): start ([-+\d.]+), kyron ([-+\d.]+), rollout ([-+\d.]+), partner ([-+\d.]+), three_w ([-+\d.]+), end ([-+\d.]+)", log)
start, kyron, rollout, partner, three_w, end = map(float, m.groups())
print(f"path: start {start} | kyron {kyron} | rollout {rollout} | partner {partner} | 3W {three_w} | end {end}")

steps = [("Q1-2025\ncm2 per swap", start, "total"),
         ("Cellora-grade packs\n(no Kyron-grade batch)", kyron, "step"),
         ("Peak/off-peak pricing\nrolled out", rollout, "step"),
         ("Partner terms\n(ZipDrop ~15%, billable\nsurcharge)", partner, "step"),
         ("3W list price\n~Rs126", three_w, "step"),
         ("cm2 per swap\nwith the levers", end, "total")]
fig, ax = plt.subplots(figsize=(12, 6))
level = 0.0
for i, (lab, val, kind) in enumerate(steps):
    if kind == "total":
        color = PALETTE["accent"] if val < 0 else PALETTE["good"]
        ax.bar(i, val, color=color, width=0.6)
        ax.text(i, val + (0.8 if val >= 0 else -0.8), f"{'+' if val > 0 else '−'}₹{abs(val):.1f}", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=13, fontweight="bold")
        level = val
    else:
        ax.bar(i, val, bottom=level, color=PALETTE["primary"], width=0.6)
        ax.text(i, level + val + 0.6, f"+₹{val:.1f}", ha="center", va="bottom", fontsize=12, fontweight="bold", color=PALETTE["primary"])
        ax.plot([i - 0.7, i - 0.3], [level, level], color="#999", lw=0.8, ls=":")
        level += val
ax.axhline(0, color="#333", lw=1)
ax.set_xticks(range(len(steps)), [s[0] for s in steps], fontsize=10)
ax.set_ylabel("Contribution after station costs (cm2), ₹ per swap")
ax.set_ylim(min(start, 0) - 4, max(end, 0) + 6)
fig.suptitle(f"From −₹{abs(start):.1f} to +₹{end:.1f} per swap: fix battery quality and contract terms before buying capacity",
             x=0.01, ha="left", fontsize=14, fontweight="bold")
ax.set_title("cm2 = revenue − energy − battery wear − station rent & maintenance; Q1-2025 unit costs, H1-2025 volume; "
             "under our usage-based battery-depreciation assumption", fontsize=9.5, color="#555", loc="left", fontweight="normal")
save_fig(fig, "q10_01_path_to_positive_cm2")

# %%
bt = batteries()
asof = dt.datetime(2025, 6, 30)
b = (bt.with_columns(((pl.lit(asof) - pl.col("first_seen")).dt.total_days()).alias("days_in_service"))
     .with_columns(((pl.col("initial_soh_pct") - pl.col("current_soh_pct")) / pl.col("days_in_service")).alias("fade_pd"))
     .with_columns(pl.when(pl.col("current_soh_pct") <= EOL_SOH).then(0.0)
                   .otherwise((pl.col("current_soh_pct") - EOL_SOH) / pl.col("fade_pd")).alias("days_to_eol"))
     .with_columns(pl.when(pl.col("manufacturing_lot").is_in(["KY-2407", "KY-2408", "KY-2409"])).then(pl.lit("Kyron lots KY-2407/08/09"))
                   .otherwise(pl.concat_str([pl.lit("Cellora/Amptek/other "), pl.col("vehicle_class")])).alias("grp")))
b = b.with_columns((pl.lit(asof) + pl.duration(days=pl.col("days_to_eol").round(0).cast(pl.Int64))).dt.truncate("1mo").alias("eol_month"))
summary = (b.group_by("grp").agg(pl.len().alias("packs"), (pl.col("purchase_cost_inr").sum() / 1e7).round(2).alias("cost_cr"),
                                 pl.col("eol_month").quantile(0.5, "nearest").alias("median_eol"),
                                 (pl.col("current_soh_pct") <= EOL_SOH).sum().alias("already_below_70")).sort("grp"))
print(summary)
monthly = b.group_by("grp", "eol_month").len().sort("eol_month")
fig, ax = plt.subplots(figsize=(12, 5.5))
colors = {"Kyron lots KY-2407/08/09": PALETTE["accent"], "Cellora/Amptek/other 2W": PALETTE["primary"], "Cellora/Amptek/other 3W": PALETTE["good"]}
months = sorted(set(monthly["eol_month"].to_list()))
bottom = np.zeros(len(months))
for g, col in colors.items():
    s = monthly.filter(pl.col("grp") == g)
    lookup = dict(zip(s["eol_month"].to_list(), s["len"].to_list()))
    vals = np.array([lookup.get(mm, 0) for mm in months])
    ax.bar(months, vals, bottom=bottom, width=25, color=col, label=g)
    bottom += vals
ax.axvspan(dt.datetime(2026, 1, 1), dt.datetime(2026, 5, 31), color=PALETTE["warn"], alpha=0.15, lw=0)
r2w = summary.filter(pl.col("grp") == "Cellora/Amptek/other 2W")
ax.annotate(f"Renewal window Jan–May 2026\n{int(r2w['packs'][0]):,} 2W packs, ≈₹{float(r2w['cost_cr'][0]):.1f} Cr",
            xy=(dt.datetime(2026, 5, 25), ax.get_ylim()[1] * 0.72), xytext=(dt.datetime(2026, 7, 15), ax.get_ylim()[1] * 0.72),
            ha="left", va="center", fontsize=10.5, color="#8a5a00", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#8a5a00", lw=1.2))
ax.axvline(asof, color="#333", ls="--", lw=1)
ax.text(asof, ax.get_ylim()[1] * 0.6, " data ends\n 30 Jun 2025", fontsize=9, color="#333")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.set_ylabel("Packs reaching 70% SoH (projected), per month")
ax.legend(loc="upper right")
fig.suptitle("The battery budget is a replacement wave, not an expansion: the original 2W fleet reaches end-of-life in early 2026",
             x=0.01, ha="left", fontsize=13.5, fontweight="bold")
ax.set_title("Each pack's projected date at 70% SoH from its own calendar fade (initial SoH − SoH on 30-Jun-25, per day in service). "
             "Kyron's defective lots already crossed it in Apr–Jun 2025.", fontsize=9.5, color="#555", loc="left", fontweight="normal")
save_fig(fig, "q10_02_fleet_eol_timeline")
print("extra visuals done")
