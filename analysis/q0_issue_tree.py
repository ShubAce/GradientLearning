# %% [markdown]
# ## Problem framing: hypothesis tree
# Leadership's question is decomposed into the three outcomes that moved the wrong way, and each outcome into testable
# hypotheses tied to a specific data source and method. Status colours are filled in from the verified findings.

# %%
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
try:
    from common import set_style, save_fig, PALETTE
except ImportError:
    sys.path.insert(0, str(Path.cwd() / "src")); from common import set_style, save_fig, PALETTE

set_style()
# status: None = to test, "yes" = supported (primary), "part" = supported (secondary / partial), "no" = not supported
TREE = [
    ("Service failures &\nqueues rose", [
        ("H1  Summer heat slows charging -> stockouts", "telemetry x events", None),
        ("H2  Failures concentrate at a few busy stations / hours", "station-hour Lorenz", None),
        ("H3  Expansion added slots where easy, not where needed", "zone supply vs demand, DiD", None),
        ("H4  3W capacity is the tighter constraint", "3W slots vs 3W demand", None),
    ]),
    ("New-rider retention\nfell", [
        ("H5  Bad first-two-week experiences drive churn", "cohorts, logit, SHAP, KM", None),
        ("H6  Degraded batteries -> short range -> frustration", "pack SoH -> km per swap", None),
        ("H7  Higher prices (base price, pilot) push riders out", "DiD by segment", None),
        ("H8  Competitor entry poaches riders", "event study (10 stations)", None),
    ]),
    ("Per-swap margin\neroded", [
        ("H9  New supplier's packs wear out faster (battery cost)", "per-pack SoH slopes, cost model", None),
        ("H10 Largest fleet contract renegotiated too deep", "partner P&L, before/after", None),
        ("H11 Under-utilised new stations add fixed cost", "station-month fixed cost / swap", None),
        ("H12 Pricing changes did not pay for themselves", "ITS + DiD on revenue & cm", None),
    ]),
]
COL = {None: "#D9E2EC", "yes": "#E4572E", "part": "#F2A541", "sup": "#9EC1E0", "no": "#E3E7EB"}
LBL = {None: "", "yes": "PRIMARY", "part": "SECONDARY", "sup": "SUPPORTED", "no": "NOT SUPPORTED"}
ROOT_Q = "Why is VoltRelay\ngrowing ~3x while\nservice, retention\nand margin\nget worse?"


def draw(tree, name="q0_01_issue_tree", title="From one leadership question to 12 testable hypotheses"):
    fig, ax = plt.subplots(figsize=(14, 7.6))
    ax.set_xlim(0, 14.2); ax.set_ylim(0.35, 8.25); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=10, bold=False, color="#1d2733", ha="center"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="none"))
        ax.text(x + (w / 2 if ha == "center" else 0.12), y + h / 2, text, ha=ha, va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=color)

    box(0.06, 3.55, 2.5, 1.6, ROOT_Q, PALETTE["primary"], fs=10.5, bold=True, color="white")
    ys = [5.55, 3.0, 0.45]
    for (outcome, hyps), y0 in zip(tree, ys):
        box(2.95, y0 + 0.62, 2.25, 1.1, outcome, "#0f2a44", fs=10.2, bold=True, color="white")
        ax.annotate("", xy=(2.95, y0 + 1.17), xytext=(2.55, 4.35), arrowprops=dict(arrowstyle="-|>", color="#8C8C8C", lw=1.2))
        for i, (h, method, status) in enumerate(hyps):
            yy = y0 + 1.80 - i * 0.6
            box(5.55, yy, 6.05, 0.5, h, COL[status], fs=9.2, ha="left", color="white" if status == "yes" else "#1d2733")
            ax.text(11.75, yy + 0.25, method, fontsize=8.6, color="#5b6b7b", va="center")
            if status:
                ax.text(11.52, yy + 0.25, LBL[status], fontsize=7.2, color="white" if status == "yes" else "#1d2733",
                        va="center", ha="right", fontweight="bold")
            ax.plot([5.2, 5.55], [y0 + 1.17, yy + 0.25], color="#B8C4CF", lw=0.9)
    ax.text(5.55, 8.05, "Hypothesis", fontsize=10, fontweight="bold", color="#5b6b7b")
    ax.text(11.75, 8.05, "Data & method", fontsize=10, fontweight="bold", color="#5b6b7b")
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold")
    return save_fig(fig, name)


draw(TREE)

# %% [markdown]
# ### The same tree after testing
# Status from the verified analyses (Q1-Q8): red = primary driver, amber = secondary / contributing, blue = supported
# description, grey = not supported by the evidence.

# %%
RESULT = {"H1": "yes", "H2": "sup", "H3": "part", "H4": "part", "H5": "yes", "H6": "no", "H7": "no", "H8": "part",
          "H9": "yes", "H10": "part", "H11": "part", "H12": "no"}
tree_res = [(o, [(h, m, RESULT[h.split()[0]]) for h, m, _ in hs]) for o, hs in TREE]
draw(tree_res, name="q0_03_hypothesis_results", title="What survived testing: three primary drivers, five secondary, three rejected")
print("issue tree drawn")
