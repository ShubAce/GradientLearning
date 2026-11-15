"""Shared paths, constants, loaders and plotting style for the VoltRelay analysis.

Every analysis script imports from here so that definitions (cleaning, cost model, KPIs, colours) are identical
across all questions. Clean tables are produced by src/clean.py.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Hackathon Data Set _ Gradient"
PARQ = ROOT / "data" / "parquet"
CLEAN = ROOT / "data" / "clean"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
FIND = ROOT / "outputs" / "findings"
for _p in (CLEAN, FIG, TAB, FIND):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- business calendar / known interventions
WINDOW_START = dt.date(2024, 1, 1)
WINDOW_END = dt.date(2025, 6, 30)
BASE_PRICE_CHANGE = dt.date(2024, 7, 1)        # 2W list 60 -> 65, 3W list 100 -> 110 (all tariffs)
PILOT_START = dt.date(2024, 10, 1)             # peak / off-peak pricing pilot
PILOT_CITIES = ["Bengaluru", "Pune"]           # treated cities; other four are controls
ZIPDROP_ID = "FP-03"                           # ZipDrop: discount 12% -> 28% on 2024-11-01, peak surcharge NOT billable
ZIPDROP_AMEND = dt.date(2024, 11, 1)
COMPETITOR_ENTRY = dt.date(2025, 1, 15)        # 10 stations got a competitor within 1.5 km
FW_BUG_START = dt.datetime(2025, 3, 10, 0, 0, 0)
FW_BUG_END = dt.datetime(2025, 4, 15, 0, 0, 0)  # exclusive; logged ts in [start, end) on v3.2.0 are shifted
FW_BUG_SHIFT = "5h30m"
EOL_SOH = 70.0                                 # all retired packs have SoH < 70; all active packs >= 70
CITIES = ["Bengaluru", "Delhi NCR", "Hyderabad", "Pune", "Mumbai", "Jaipur"]
CITY_CODE = {"BLR": "Bengaluru", "DEL": "Delhi NCR", "HYD": "Hyderabad", "PUN": "Pune", "MUM": "Mumbai", "JAI": "Jaipur"}
EVENT_TYPES = ["swap_completed", "failed_no_charged_battery", "abandoned_queue", "cancelled_by_rider",
               "failed_system_error"]
FAIL_TYPES = ["failed_no_charged_battery", "abandoned_queue", "failed_system_error"]  # service failures
# Indian seasons (IMD-style): Winter Dec-Feb, Summer Mar-May, Monsoon Jun-Sep, Post-monsoon Oct-Nov
SEASON_OF_MONTH = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer", 5: "Summer",
                   6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon", 10: "Post-monsoon", 11: "Post-monsoon"}

# ---------------------------------------------------------------- loaders (lazy, projection-friendly)

def events(cols: list[str] | None = None) -> pl.LazyFrame:
    """Clean swap events (test stations removed, timestamps fixed, outliers flagged, costs attached)."""
    lf = pl.scan_parquet(CLEAN / "events.parquet")
    return lf.select(cols) if cols else lf


def hourly(cols: list[str] | None = None) -> pl.LazyFrame:
    lf = pl.scan_parquet(CLEAN / "station_hourly.parquet")
    return lf.select(cols) if cols else lf


def stations() -> pl.DataFrame:
    return pl.read_parquet(CLEAN / "stations.parquet")


def riders() -> pl.DataFrame:
    return pl.read_parquet(CLEAN / "riders.parquet")


def batteries() -> pl.DataFrame:
    return pl.read_parquet(CLEAN / "batteries.parquet")


def tickets() -> pl.DataFrame:
    return pl.read_parquet(CLEAN / "tickets.parquet")


def city_daily() -> pl.DataFrame:
    return pl.read_parquet(CLEAN / "city_daily.parquet")


def partners() -> pl.DataFrame:
    return pl.read_parquet(CLEAN / "fleet_partners.parquet")


def station_month() -> pl.DataFrame:
    return pl.read_parquet(CLEAN / "station_month.parquet")


# ---------------------------------------------------------------- plotting style
PALETTE = {
    "primary": "#1F4E79",   # deep blue  - main series
    "accent": "#E4572E",    # warm red   - problems / failures
    "good": "#2E8B57",      # green      - improvements
    "warn": "#F2A541",      # amber      - caution
    "muted": "#8C8C8C",     # grey       - context
    "light": "#D9E2EC",
}
SERIES = ["#1F4E79", "#E4572E", "#2E8B57", "#F2A541", "#6C4F9E", "#17A2B8", "#8C8C8C", "#B5651D"]
CITY_COLORS = dict(zip(CITIES, SERIES))
SUPPLIER_COLORS = {"Cellora": "#1F4E79", "Amptek": "#2E8B57", "Kyron": "#E4572E"}


def set_style():
    matplotlib.use("Agg")
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10.5, "axes.labelcolor": "#333333",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#888888", "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.8,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "legend.frameon": False, "legend.fontsize": 9.5,
        "xtick.color": "#555555", "ytick.color": "#555555",
    })


def save_fig(fig, name: str, subtitle: str | None = None, source: str | None = "Source: VoltRelay operational data, Jan 2024 - Jun 2025"):
    """Save a figure to outputs/figures/<name>.png with an optional source footnote."""
    if source:
        fig.text(0.01, -0.02, source, fontsize=8, color="#777777", ha="left", va="top")
    path = FIG / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    try:  # inside a notebook, also display a light (palette-quantized) copy inline; no effect in plain scripts
        from IPython import get_ipython
        if get_ipython() is not None:
            import io as _io
            from PIL import Image as _PIL
            from IPython.display import Image as _Img, display as _disp
            _im = _PIL.open(path).convert("RGB"); _im.thumbnail((1300, 1300))
            _im = _im.quantize(colors=128, method=_PIL.Quantize.MEDIANCUT, dither=_PIL.Dither.NONE)
            _buf = _io.BytesIO(); _im.save(_buf, format="PNG", optimize=True)
            _disp(_Img(data=_buf.getvalue(), width=900))
    except Exception:
        pass
    return path


def fmt_inr(x: float) -> str:
    """Indian-style compact currency: 1.2 Cr, 3.4 L, 5.6K."""
    a = abs(x)
    if a >= 1e7:
        return f"₹{x/1e7:.2f} Cr"
    if a >= 1e5:
        return f"₹{x/1e5:.1f} L"
    if a >= 1e3:
        return f"₹{x/1e3:.1f}K"
    return f"₹{x:.0f}"
