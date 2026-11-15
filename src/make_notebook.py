"""Driver: assemble + execute the Colab notebook in the final section order.

python src/make_notebook.py            # build and execute (proves it runs top to bottom)
python src/make_notebook.py --no-exec  # build only
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_notebook import build, ROOT

EXEC_SUMMARY = (ROOT / "docs" / "notebook_text" / "notebook_exec_summary.md")
CONCLUSION = (ROOT / "docs" / "notebook_text" / "notebook_conclusion.md")

TITLE = """# ⚡ VoltRelay Energy: what is really driving service failures, rider churn and margin erosion?
**Team Praise** · Shubham Suman & Durgesh Kumar · Gradient Learnings Data Analytics Hackathon · 15 November 2026

This notebook is the complete, reproducible analysis behind our report and video. It downloads the organisers' data,
cleans it with an audited pipeline, answers the six core questions plus five deep dives, and ends with our
recommendations for VoltRelay's next operating budget. **Runtime:** about 3 minutes of computation (≈10-15 minutes on a standard Colab CPU runtime, most of it downloading the 0.85 GB of data).

**How to read it:** every section opens with the question and the answer in plain words, then shows the evidence.
Charts are titled with their insight. Numbers quoted in the text are printed by the code cell above or below them.
"""

TOC = """## Contents
0. Setup & data acquisition
1. Data understanding (inventory, profile, referential integrity)
2. Data cleaning (9 documented + 13 undocumented issues, each detected, quantified and decided)
3. Data-quality audit & anomalies (impact of every cleaning decision)
4. Problem framing (what changed in 18 months; hypothesis tree)
5. **Q1** Network performance over time
6. **Q2** Service failures & customer experience
7. **Q3** Station & geographic patterns (+ expansion-wave effectiveness, competitor entry)
8. **Q4** Battery & equipment performance
9. **Q5** Pricing & partner economics (+ fleet-partner value)
10. **Q6** Root-cause analysis of new-rider retention
11. Deep dive: support-ticket text (English + Hinglish)
12. Synthesis: primary vs secondary drivers, the budget decision, recommendations
13. Conclusions and recommendations
"""

SECTIONS = [
    ("md", TITLE),
    ("md", EXEC_SUMMARY.read_text(encoding="utf-8") if EXEC_SUMMARY.exists() else "## Executive summary\n_(added at build time)_"),
    ("md", TOC),
    ("md", "## 0. Setup & data acquisition"),
    ("setup",),
    ("script", "analysis/q0_data_prep.py"),
    ("cleanup",),
    ("script", "analysis/q7_data_quality_anomalies.py", "## 3. Data-quality audit & anomalies"),
    ("cleanup",),
    ("md", "## 4. Problem framing"),
    ("script", "analysis/q0_timeline.py"),
    ("script", "analysis/q0_issue_tree.py"),
    ("cleanup",),
    ("script", "analysis/q1_network_performance.py", "## 5. Q1 · Network performance over time"),
    ("cleanup",),
    ("script", "analysis/q2_service_failures.py", "## 6. Q2 · Service failures & customer experience"),
    ("cleanup",),
    ("script", "analysis/q3_stations_geography.py", "## 7. Q3 · Station & geographic patterns"),
    ("cleanup",),
    ("script", "analysis/q4_battery_equipment.py", "## 8. Q4 · Battery & equipment performance"),
    ("cleanup",),
    ("script", "analysis/q5_pricing_partners.py", "## 9. Q5 · Pricing & partner economics"),
    ("cleanup",),
    ("script", "analysis/q6_retention.py", "## 10. Q6 · Root-cause analysis of new-rider retention"),
    ("cleanup",),
    ("script", "analysis/q8_ticket_text.py", "## 11. Deep dive · Support-ticket text"),
    ("cleanup",),
    ("script", "analysis/q9_synthesis.py", "## 12. Synthesis · drivers, budget decision, recommendations"),
    ("script", "analysis/q10_extra_visuals.py"),
    ("md", CONCLUSION.read_text(encoding="utf-8") if CONCLUSION.exists() else "## Conclusions\n_(added at build time)_"),
]

if __name__ == "__main__":
    build(SECTIONS, execute="--no-exec" not in sys.argv)
