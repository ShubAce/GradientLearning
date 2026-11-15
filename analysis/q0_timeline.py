# %% [markdown]
# ## What changed during the 18 months?
# Eight business changes and external shocks overlap with the growth curve. Any analysis that compares "before" and
# "after" must know when each one happened, otherwise effects get attributed to the wrong cause (e.g. a summer heat spike
# read as a pricing effect).

# %%
import sys, datetime as dt
from pathlib import Path
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
try:
    from common import events, set_style, save_fig, PALETTE
except ImportError:
    sys.path.insert(0, str(Path.cwd() / "src")); from common import events, set_style, save_fig, PALETTE

set_style()
m = (events(["month", "is_completed", "is_service_failure"]).group_by("month")
     .agg(pl.col("is_completed").sum().alias("swaps"), (pl.col("is_service_failure").mean() * 100).alias("fail_pct"))
     .sort("month").collect())
print(m)

EVENTS = [  # (date, label, lane)
    (dt.date(2024, 7, 1), "Base price +8%\n(2W ₹60→65, 3W ₹100→110)", 0),
    (dt.date(2024, 8, 7), "Wave 1: +30 stations\n(Aug–Oct 2024)", 1),
    (dt.date(2024, 9, 10), "New supplier Kyron:\n1,385 2W packs in 3 days", 2),
    (dt.date(2024, 10, 1), "Peak/off-peak pilot\n(Bengaluru, Pune)", 3),
    (dt.date(2024, 11, 1), "ZipDrop discount\n12% → 28%", 4),
    (dt.date(2025, 1, 15), "Competitor opens near\n10 stations", 0),
    (dt.date(2025, 2, 1), "Wave 2: +30 stations\n(Feb–Apr 2025)", 1),
    (dt.date(2025, 3, 10), "Firmware clock bug\n(10 Mar–14 Apr)", 2),
    (dt.date(2025, 6, 15), "All Kyron packs\nretired (EOL)", 3),
]

fig, ax = plt.subplots(figsize=(14, 6.4))
x = [d + dt.timedelta(days=14) for d in m["month"].to_list()]
for yr in (2024, 2025):
    ax.axvspan(dt.date(yr, 4, 1), dt.date(yr, 6, 30), color=PALETTE["warn"], alpha=0.10, lw=0)
    ax.text(dt.date(yr, 5, 15), 508, f"Summer {yr}", ha="center", va="top", fontsize=9.5, color="#b7791f", fontweight="bold")
ax.bar(x, [v / 1000 for v in m["swaps"]], width=24, color=PALETTE["light"], edgecolor="none", label="Completed swaps per month (K)")
ax.set_ylabel("Completed swaps per month (thousands)")
ax.set_ylim(0, 520)
ax2 = ax.twinx()
ax2.plot(x, m["fail_pct"].to_list(), color=PALETTE["accent"], lw=2.4, marker="o", ms=4, label="Service-failure rate (%)")
ax2.set_ylabel("Service-failure rate, % of attempts", color=PALETTE["accent"])
ax2.set_ylim(0, 26); ax2.grid(False); ax2.spines["right"].set_visible(True)
for d, label, lane in EVENTS:
    y = 470 - lane * 44
    ax.plot([d, d], [0, y], color="#5b6b7b", lw=0.9, ls=(0, (3, 3)))
    ax.plot(d, y, marker="o", color=PALETTE["primary"], ms=5)
    ax.text(d + dt.timedelta(days=3), y, label, fontsize=8.6, va="center", ha="left", color="#1d2733",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#D9E2EC", lw=0.8))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.set_xlim(dt.date(2023, 12, 20), dt.date(2025, 7, 25))
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(0.0, 0.93), ncol=1, frameon=True, facecolor="white", edgecolor="#D9E2EC")
fig.suptitle("18 months, 9 overlapping changes: volume tripled while failures spiked every summer", x=0.01, ha="left", fontsize=14, fontweight="bold")
save_fig(fig, "q0_02_timeline")
print("timeline drawn")
