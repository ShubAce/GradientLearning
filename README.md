# VoltRelay Energy: what is really driving service failures, rider churn and margin erosion

**Team Praise** · Shubham Suman, Durgesh Kumar · Gradient Learnings Data Analytics Hackathon · 15 November 2026

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShubAce/GradientLearning/blob/main/deliverables/1_Colab_Notebook/VoltRelay_Analysis.ipynb)

## Deliverables

|     | Deliverable                                                   | Location                                                                                                                       |
| --- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 1   | Google Colab notebook (runs top to bottom, all outputs saved) | [`deliverables/1_Colab_Notebook/VoltRelay_Analysis.ipynb`](deliverables/1_Colab_Notebook/VoltRelay_Analysis.ipynb)             |
| 2   | Analysis report (PDF, 31 pages)                               | [`deliverables/2_Analysis_Report/VoltRelay_Analysis_Report.pdf`](deliverables/2_Analysis_Report/VoltRelay_Analysis_Report.pdf) |
| 3   | 3-minute video: slide deck and video                          | [`deliverables/3_Video/`](deliverables/3_Video/)                                                                               |

## The answer in one paragraph

VoltRelay's growth is real: completed swaps grew **2.43×** and revenue **2.63×** year on year. Two of leadership's three
alarms are misread. Like-for-like, service failures did not get worse (the same stations failed 9.52% → 9.55% of summer
attempts), and new-rider retention did not fall (86.3% vs 86.3% for same-season cohorts): both move with the summer heat.
The third alarm, margin erosion, is real, and it has one main cause: **a defective battery batch** (Kyron lots
KY-2407/08/09) that loses health 3.8× faster at the same price and adds **about ₹16 of wear to the average swap** (₹16.8 at
Q1-2025 costs). Most of the damage
traces back to two equipment problems, not to growth, so the budget should fix those rather than buy more stations or batteries.

## Key findings

- **Service failures** are a hot-city, hot-season, old-charger problem. 36 Gen1 stations in Delhi NCR, Jaipur and Hyderabad
  produce 77–80% of excess summer failures. Above about 41 °C in the cabinet, Gen1 charge time climbs steeply and nearly doubles
  by 50 °C (88 → ~168 min); extra load does not change the failure rate.
- **New-rider churn**: the primary driver is two or more failed swap attempts in a rider's first 14 days
  (22.6% vs 13.1% not active in days 31–60; adjusted odds ratio 1.67; consistent in 15 of 15 segments). Battery health and price are not drivers.
- **Margin**: Kyron battery wear took −₹16 per swap and erased the July-2024 price rise. 122,204 swaps handed riders packs
  already below end-of-life, and the rest of the fleet reaches end-of-life in early-to-mid 2026.
- **Pricing and partners**: the peak/off-peak pilot raised revenue per independent swap by ₹7.90 (+11.7%) but moved only
  2.5 points of demand. The largest partner (ZipDrop, 28% discount) loses money on every swap even with healthy packs.
- **Budget decision**: reject more stations and more batteries as capacity fixes; support a network-wide pricing rollout as a
  revenue measure; reject a long-term ZipDrop exclusive on current terms. Fund battery quality, a Gen1 thermal fix and
  new-rider failure recovery instead. The battery fix plus pricing and partner re-terms take margin after station costs from
  **−₹18.5 to +₹6.8 a swap**; the Gen1 fix and new-rider recovery cut failures and churn.

## How we worked

- **Data quality first**: 26 checks, 13 issues the documentation does not mention, each detected, quantified and decided
  (for example, a firmware clock bug that shifted 139,490 swaps by 5h30m, validated independently by the pricing pilot's own
  tariff clock; "test" stations holding 59,134 paid swaps; peak pricing hidden inside the PARTNER tariff code).
- **A per-swap P&L** for every completed swap: revenue − energy − battery wear − station rent and maintenance.
- **Like-for-like evidence**: same-season and same-station comparisons, difference-in-differences with permutation tests
  for the pricing pilot, survival models, logistic regression and gradient boosting with SHAP for churn, dose-response tests.
- **Independent verification**: every analysis was re-derived by a second analyst with separate code
  (`analysis/verify/`, `outputs/findings/*_verification.md`) and corrected before conclusions were drawn.

## Repository map

| Folder          | Contents                                                                                                                                         |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `deliverables/` | The three submission items (see above)                                                                                                           |
| `analysis/`     | One script per question, in jupytext percent format (q0 data prep, q1–q8 analyses, q9 synthesis); `verify/` holds the independent re-derivations |
| `src/`          | Shared definitions (`common.py`), the audited cleaning pipeline (`clean.py`), notebook and report builders                                       |
| `outputs/`      | Every figure, table and written finding, including the verification reports                                                                      |
| `docs/`         | Problem statement and the shared analysis brief                                                                                                  |

## Reproduce

**Colab (simplest):** open the notebook with the badge above and choose _Runtime → Run all_. It downloads the organisers'
data from their public Google Drive folder and runs end to end (~10–15 minutes including the download).

**Locally (Python 3.12):**

```bash
pip install polars pyarrow pandas numpy scipy statsmodels scikit-learn lightgbm shap lifelines matplotlib seaborn nbformat nbconvert gdown
# put the 8 CSVs from the organisers' Drive folder into "Hackathon Data Set _ Gradient/"
python src/convert_to_parquet.py && python src/clean.py   # raw CSV -> typed Parquet -> audited clean tables
python analysis/q1_network_performance.py                 # ...each analysis script also runs standalone
python src/make_notebook.py                               # assemble and execute the notebook
python src/make_report.py                                 # build the PDF report
```

The dataset itself is not in this repository (836 MB; the clean event table alone is 228 MB).
