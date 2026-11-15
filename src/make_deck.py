"""Build the 3-minute video deck (16:9) with narration in speaker notes, then export PDF + slide PNGs via PowerPoint."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_deck import build

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "working" / "old_deck_v1" / "VoltRelay_Video_Deck.pptx"  # superseded by deck/build.js

SLIDES = [
    {"type": "title",
     "kicker": "TEAM PRAISE · GRADIENT LEARNINGS DATA ANALYTICS HACKATHON",
     "title": "VoltRelay grew 2.4×. Two of its three alarms are misread; the third is real and fixable.",
     "subtitle": "What is really driving service failures, rider churn and margin erosion, and where the next budget should go.   Shubham Suman · Durgesh Kumar",
     "kpis": [("2.43×", "completed swaps, H1-25 vs H1-24"), ("−₹16", "per swap: one bad battery batch"),
              ("80%", "of excess failures at 36 hot-city stations"), ("Flat", "new-rider retention, same season (86.3% vs 86.3%)")],
     "notes": ("[0:00-0:17] VoltRelay more than doubled its swaps in a year. Yet leadership sees three alarms: rising failures, "
               "falling new-rider retention and eroding margins. We are Team Praise, and we analysed 3.8 million swap attempts "
               "to find what is really driving them.")},
    {"type": "chart", "kicker": "The business problem", "title": "Nine overlapping changes in 18 months, plus two Indian summers",
     "image": "q0_02_timeline.png",
     "bullets": ["Volume **tripled**; revenue **3.2×**", "Failures spike **every April–June**",
                 "Two expansion waves, a price rise, a pricing pilot, a **new battery supplier**, a renegotiated fleet contract",
                 "So every comparison must be **like for like**"],
     "notes": ("[0:17-0:35] In 18 months, nine things changed at once: two expansion waves, a price rise, a peak-pricing pilot, "
               "a new battery supplier and a renegotiated fleet contract, on top of two Indian summers. So every number has to "
               "be compared like for like.")},
    {"type": "text", "kicker": "Our approach", "title": "Audit the data, build a per-swap P&L, compare like for like",
     "rows": [("Audit the data", "**26 checks, 13 undocumented issues**: a firmware clock bug shifted **139,490 swaps** by 5h30m; 'test' stations held **59,134 paid swaps**; peak pricing hidden inside the PARTNER tariff"),
              ("Per-swap P&L", "revenue − energy − **battery wear** − station rent & maintenance, for every completed swap"),
              ("Like-for-like evidence", "same season & same station; **difference-in-differences** for pricing; **survival models + SHAP** for churn"),
              ("Independently verified", "every analysis re-derived by a second analyst with separate code before we drew conclusions")],
     "notes": ("[0:35-0:55] First we audited the data: 26 checks, including a firmware clock bug that shifted 139,000 swaps by "
               "five and a half hours, and test stations holding 59,000 paid swaps. Then we built a per-swap P&L and used "
               "same-season comparisons, difference-in-differences and survival models.")},
    {"type": "chart", "kicker": "Insight 1 · Margin", "title": "One bad battery batch erased the price-rise gain: −₹16 on every swap",
     "image": "q1_05_cm1_bridge_waterfall.png",
     "bullets": ["July price rise: margin to **+₹6.7/swap**",
                 "Sep-24: **1,385 Kyron packs** in 3 days; lots KY-2407/08/09 lose health **3.3× faster**, same price",
                 "Wear **−₹16/swap ≈ ₹40 lakh a month** → margin **−₹7.5**",
                 "Rest of the fleet reaches end-of-life in **early-to-mid 2026**"],
     "notes": ("[0:55-1:25] Insight one: the margin alarm is real, and it has one main cause. The July price rise lifted margin "
               "to plus 6.7 rupees per swap. Then in September, 1,385 packs from a new supplier, Kyron, went into service. Three "
               "of its lots lose health more than three times faster than any other pack, at the same price. That wear costs 16 "
               "rupees on every swap, about 40 lakh rupees a month, and turned margin negative.")},
    {"type": "chart", "kicker": "Insight 2 · Service failures", "title": "Failures are a hot-city, hot-season, old-charger problem",
     "image": "q2_04_concentration_lorenz.png",
     "bullets": ["Like for like, **no deterioration** (same-station RR 0.986)",
                 "Apr–Jun: **36 Gen1 stations** in Delhi, Jaipur, Hyderabad = **80% of excess failures**",
                 "Above 40 °C Gen1 charging **88 → 193 min**; stockouts **5×**",
                 "Load doesn't move their rate → **more stations won't fix it**"],
     "notes": ("[1:25-1:55] Insight two: like for like, failures did not get worse; the same stations fail at the same rate as "
               "last year. But they are extremely concentrated. From April to June, 36 old Gen1 stations in Delhi, Jaipur and "
               "Hyderabad produce 80 percent of excess failures. Above 40 degrees their chargers take twice as long and stockouts "
               "jump five-fold, and extra load does not change their rate.")},
    {"type": "chart", "kicker": "Insight 3 · Retention", "title": "Retention did not fall structurally; early failures drive the summer dip",
     "image": "q6_03_driver_effects.png",
     "bullets": ["Same season: **86.3% vs 86.3%** retained (Jan–Mar 24 vs 25)",
                 "**Primary:** 2+ stockouts/abandons in first 14 days → **22.6% vs 13.1%** leave (OR 1.67)",
                 "Holds in **15 of 15** subgroups",
                 "**Not drivers:** battery health, peak price, queue time"],
     "notes": ("[1:55-2:20] Insight three: retention has not fallen structurally. This January's new riders stayed exactly like "
               "last January's, 86.3 percent. The visible dip is a summer effect, and its primary driver is early failures: a new "
               "rider who hits two stockouts in the first two weeks is 1.7 times as likely to leave. Battery health and price are "
               "not drivers.")},
    {"type": "chart", "kicker": "Insight 4 · Pricing & partners", "title": "Pricing raises revenue, not capacity; the largest partner is the biggest loss",
     "image": "q5_04_partner_value_bubble.png",
     "bullets": ["Peak pilot: **+₹7.90 per swap (+11.7%)**, but only **2.5 pts** of demand moved and peak stockouts **unchanged**",
                 "ZipDrop: **15.5%** of swaps, **−₹69 lakh** since its discount went 12% → 28%, **no extra volume**",
                 "Break-even discount **~22–25%**; FeastFly's terms (10%, surcharge billable) earn **+₹12/swap**"],
     "notes": ("[2:20-2:35] Insight four: pricing is a revenue lever, not a congestion fix. The peak pilot added 7.90 rupees per "
               "swap but moved only 2.5 points of demand. And ZipDrop, the largest partner, is the biggest loss; its deeper "
               "discount bought no extra volume.")},
    {"type": "text", "kicker": "Recommendations · the budget decision", "title": "Fund fixes, not more capacity",
     "rows": [("✗  More stations / batteries", "hotspot failure rates ignore load; hot-day stockouts happen with full cabinets"),
              ("1  Battery quality", "replace Kyron packs, **gate new suppliers**, quarantine sub-70% packs, plan the **2026** fleet renewal: ~**₹40–48 lakh/month**"),
              ("2  Cool / upgrade 36 Gen1", "pilot, then scale before April 2026: **19,000–27,000 fewer failures** a summer"),
              ("3  Protect new riders", "failure-recovery trigger after an early stockout (A/B test)"),
              ("4  Pricing & partners", "roll out peak pricing for revenue (**~₹1.1 Cr/yr**); **no ZipDrop exclusive** on current terms")],
     "notes": ("[2:35-3:00] So for the budget: don't fund more stations or more batteries for capacity. Instead, one: replace the "
               "Kyron packs, gate new suppliers, and plan the fleet renewal due in early 2026. Two: cool or upgrade the 36 "
               "hot-city Gen1 stations before April, up to 27,000 fewer failures a summer. Three: protect new riders after a "
               "failure. Four: roll out peak pricing for revenue, about 1.1 crore a year, and don't sign ZipDrop's exclusive on "
               "current terms. Thank you.")},
]


def export(pptx: Path):
    pdf = pptx.with_suffix(".pdf")
    png_dir = pptx.parent / "deck_png"
    ps = (f"$ppt = New-Object -ComObject PowerPoint.Application; $p = $ppt.Presentations.Open('{pptx}', $true, $false, $false); "
          f"$p.SaveAs('{pdf}', 32); $p.Export('{png_dir}', 'PNG', 1600, 900); $p.Close(); $ppt.Quit(); 'ok'")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=300)
    print(r.stdout.strip(), r.stderr.strip()[-400:])


if __name__ == "__main__":
    build(SLIDES, OUT)
    script = ROOT / "working" / "old_deck_v1" / "Video_Script_3min.md"  # superseded by deck/finalize.py
    words = sum(len(s["notes"].split("]", 1)[-1].split()) for s in SLIDES)
    lines = ["# 3-minute video script - Team Praise (VoltRelay)", "",
             f"Target pace ~145 words/min · {words} words · record with PowerPoint: Slide Show → Record → Export → Create a Video.", ""]
    for i, s in enumerate(SLIDES, 1):
        lines += [f"## Slide {i}: {s['title']}", "", s["notes"], ""]
    script.write_text("\n".join(lines), encoding="utf-8")
    print("script words:", words, "->", script)
    if "--no-export" not in sys.argv:
        export(OUT)
