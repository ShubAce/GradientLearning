"""Assemble deliverables/report/VoltRelay_Analysis_Report.md from section files, fill placeholders, render PDF."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_report import build

ROOT = Path(__file__).resolve().parents[1]
SEC = ROOT / "working" / "report" / "sections"
OUT = ROOT / "working" / "report" / "VoltRelay_Analysis_Report.md"
FINAL_PDF = ROOT / "deliverables" / "2_Analysis_Report" / "VoltRelay_Analysis_Report.pdf"

ORDER = ["00_cover.md", "00_exec_summary.md", "01_problem.md", "02_approach.md", "03a_q1.md", "03b_q2_q3.md",
         "03c_q4.md", "03d_q5_q6_q8.md", "04_findings.md", "05_recommendations.md", "06_limitations.md", "07_appendix.md"]


def cost_numbers():
    log = json.loads((ROOT / "outputs" / "tables" / "data_quality_log.json").read_text(encoding="utf-8"))
    cm = next(s for s in log["steps"] if s["step"] == "cost_model")
    f = lambda k: f"₹{cm[k]:.2f}" if cm[k] >= 0 else f"−₹{abs(cm[k]):.2f}"
    return {"{{REV}}": f("revenue_per_swap"), "{{ENERGY}}": f("energy_cost_per_swap"), "{{WEAR}}": f("wear_cost_per_swap"),
            "{{CM1}}": f("cm1_per_swap"), "{{FIXED}}": f("fixed_cost_per_swap"), "{{CM2}}": f("cm2_per_swap")}


def scorecard_table(levels=("high", "medium", "low", "pass")):
    p = ROOT / "outputs" / "tables" / "q7_data_quality_scorecard.csv"
    if not p.exists():
        return "_Data-quality scorecard: see notebook section 3._"
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    def kind_label(k):
        k = (k or "").lower()
        return "**New**" if "undoc" in k else "Documented" if "doc" in k else "Analysis trap" if "trap" in k else "Check passed" if "check" in k or "pass" in k else k
    sev = {"high": "🔴 High", "medium": "🟠 Med", "low": "⚪ Low", "pass": "🟢 Pass"}
    out = ["| ID | Issue | Type | Severity | Rows affected | Decision | Impact if untreated |", "|---|---|---|---|---:|---|---|"]
    for r in rows:
        if (r.get("severity", "") or "").lower() not in levels:
            continue
        try:
            n = f"{int(float(r['rows_affected'])):,} {r.get('unit', '')}"
        except (ValueError, TypeError):
            n = r.get("rows_affected", "")
        cells = [r["id"], r["issue"], kind_label(r.get("kind")), sev.get(r.get("severity", "").lower(), r.get("severity", "")),
                 n, r["decision"], r.get("impact_short") or r.get("impact_on_kpis", "")]
        out.append("| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in cells) + " |")
    return "\n".join(out)


def assemble():
    parts = []
    for name in ORDER:
        p = SEC / name
        if p.exists():
            parts.append(p.read_text(encoding="utf-8").strip())
        else:
            print("missing section:", name)
    text = "title: VoltRelay Energy - Analysis Report (Team Praise)\n\n" + "\n\n".join(parts) + "\n"
    for k, v in cost_numbers().items():
        text = text.replace(k, v)
    text = text.replace("{{DQ_SCORECARD}}", scorecard_table(("high", "medium")))
    text = text.replace("{{DQ_SCORECARD_REST}}", scorecard_table(("low", "pass", "passed", "check")))
    OUT.write_text(text, encoding="utf-8")
    print("assembled", OUT, len(text), "chars")
    return OUT


if __name__ == "__main__":
    out = build(assemble(), pdf="--no-pdf" not in sys.argv)
    if str(out).endswith(".pdf"):
        import shutil
        FINAL_PDF.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out), FINAL_PDF)
        print("final PDF ->", FINAL_PDF)
