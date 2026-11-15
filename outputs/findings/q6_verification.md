# Q6 verification: root cause of new-rider retention

**Verifier:** an independent re-derivation from `data/clean`, using only the `src/common.py` loaders. None of the analyst's code was used.
**Script:** `analysis/verify/q6_verify.py` (percent format, about 22 s to run). Every number below is printed by that script unless it is marked *(analyst table)*.

## Bottom line
The core story holds:
* **No structural decline.** Same-season cohorts are flat on D31-60, and also on two checks the analyst did not run, R14 and D1-7, which let May and June be compared year on year.
* **Main driver.** Early stockouts and abandoned queues are the strongest measured and actionable driver in all three designs.
* **Raw numbers reproduce almost exactly.** This covers cohort rates, the dose table, attempt-level counts and odds ratios (ORs), the landmark Cox model, the segment ORs and the what-if.

Four points must be fixed before judges see them:
1. **Hero figure q6_01 and finding 9 imply that early failures explain the whole summer dip.** The dashed "flat" line is a pure-model counterfactual set against observed data. Removing the early-failure effect from the observed data takes out only about a third of the May-24 dip (79.1% becomes 81.7%, not about 85.7%). Across methods, early failures explain about 40-50% of the summer-2024 gap.
2. **"One failure is tolerated, the second is not" is a Simpson's paradox caused by exposure.** With early activity controlled (the analyst's own model), the 1st failure already raises the odds of non-retention by 24%.
3. **The 3W OR of 1.52 comes from controlling for km per swap.** 3W packs give 92 km per swap against 61 km for 2W. Without that control the OR is 1.16 [1.00-1.36].
4. **The "H1-2024 vs Oct-24 to Apr-25: +1.85 pp, p=0.009" comparison is itself a seasonal-mix artefact.** 54.5% of H1-2024 riders are summer-entry, against 14.9% in the later period.

## Verdict table
| # | Claim (analyst) | Verdict | Recomputed |
|---|---|---|---|
| 1 | No structural decline: Jan-Mar 2024 vs Jan-Mar 2025 D31-60 retention 86.3% vs 86.3%, p=0.98. 13,797 eligible riders, 85.4% retained. | confirmed | 86.3% (n=1,750) vs 86.3% (n=3,046), +0.02 pp, p=0.981. 13,797 eligible riders; 85.4% retained (2,017 = 14.62% not retained). |
| 2 | Apr-24 vs Apr-25 82.4% vs 81.7% (p=0.69). Jan-Mar 24 vs Apr-25 -4.6 pp (p=0.0009). Summer dip 86.3% to 81.7% (p=0.0001). Naive metric 89.7% to 0.0%. Log-rank p=0.41 / 0.96. | confirmed | 82.4% (660) vs 81.7% (1,108), p=0.693. -4.61 pp, p=0.0009. -4.58 pp, p=0.0001. Naive 89.7% (Jan-24), 77.5% (May-25), 0.0% (Jun-25). Log-rank 2024Q1 vs 2025Q1 p=0.408; 2024Q2 vs 2025Q2 p=0.965. S90 72.5 / 66.2 / 73.0 / 70.5%. |
| 3 | H1-2024 cohorts 83.8% vs Oct-24 to Apr-25 85.6% (+1.85 pp, p=0.009), so later cohorts retain better. | **corrected** | Numbers reproduce, but this is a seasonal-mix artefact: 54.5% vs 14.9% summer-entry riders. Season-matched Jan-Apr 2024 vs Jan-Apr 2025: 85.2% (n=2,410) vs 85.1% (n=4,154), -0.15 pp, p=0.87. |
| 4 | Headline: the apparent decline is a summer dip *driven by* early service failures. | **corrected** | Early (day 0-13) failures explain 40-52% of the observed summer-2024 gap: 1.81 of 4.58 pp in the analyst's decomposition; 43% by logit shrinkage; 46-52% by direct standardisation; 51% with an activity control. Including failures up to day 29 explains 66% among riders active in days 14-29. Apr-25: 29-30%. |
| 5 | Headline: 2+ stockouts or abandoned queues raise non-retention from 13.1% to 22.6%, adjusted OR 1.67 [1.37-2.02]. | **corrected** | 22.6% (n=2,246) is for 2+ *service failures* (system errors included). For 2+ stockouts or abandoned queues it is 13.1% to **23.5% (n=2,067)**. The OR 1.67 [1.37-2.02] is for exactly 2 vs 0 failures. The 2+ vs 0-1 OR is **1.66 [1.43-1.93]** (cohort-month and city fixed effects (FE)); the analyst's own "All" segment model gives 1.68 [1.49-1.90]. |
| 6 | Cohort retention tracks early failures: r=-0.89 over 16 cohorts. Early failure rate 7.6 / 11.1 / 8.6% (Apr-Jun-24) and 6.8% (Apr-25) vs about 4%. | confirmed | r=-0.895 (p=2.9e-6). Rates 7.62 / 11.09 / 8.58 / 6.78%; other months 3.7-4.8%. Note: this is an ecological correlation driven mostly by season. Within quarter-of-year, r=-0.54 (p=0.03). |
| 7 | Dose table 13.3% / 12.8% / 22.6% / 22.6% (n 7,350 / 4,201 / 1,497 / 749). Logit OR 2 failures 1.67, 3+ 1.41. Per stockout 1.17, per abandoned queue 1.16, system error 0.74. With activity control, 2 failures 2.47. | confirmed | Identical table. Lean re-specification: 1.66 [1.37-2.00] and 1.43 [1.15-1.77]; per stockout 1.17 [1.09-1.26]; per abandoned queue 1.16 [1.06-1.28]; system error 0.74 [0.58-0.96]; with log attempts 2.50 [2.06-3.04]. |
| 8 | "One failure is tolerated, the second is not" / "risk jumps at the 2nd failure" (threshold). | **corrected** | With the activity control, **1 failure OR 1.24 [1.09-1.41]** (analyst table: 1.235 [1.08-1.41]). Within early-activity bands, non-retention rises with the 1st failure: 10-14 attempts 9.1% to 11.5% (then 22.2%); 15-19 attempts 8.6% to 11.0% (then 17.9%). The raw 0-vs-1 inversion arises because 400 of the 472 riders with 1-5 attempts (65.7% not retained) have 0 failures. A failure-*rate* model is monotone: OR 1.00 / 1.21 / 2.10 / 2.99 for 0-5% / 5-10% / 10-20% / >20% of attempts failed. |
| 9 | Attempt level: 2,350,544 attempts and 8,117 exits. Exits per 1,000 attempts 3.34 / 5.56 / 6.10 / 3.15 / 2.79. Adjusted OR stockout 1.50, abandoned queue 1.66, system error 0.93. Prior failures in 7 days 1.29 / 1.75 / 2.06. | confirmed | Exact match: 2,350,544 / 8,117; 3.34 / 5.56 / 6.10 / 3.15 / 2.79. ORs 1.50 [1.36-1.66], 1.66 [1.46-1.89], 0.93, 0.83; prior failures 1.29 / 1.75 / 2.06; tenure 0.81 / 0.63. A true within-rider check (Mantel-Haenszel) of exit after a stockout or abandoned queue gives 1.53 [1.41-1.66] with each rider as a stratum, and 1.25 [1.15-1.37] with rider x month strata. |
| 10 | Landmark Cox: n=13,447, 6,394 events, HR 1.24 (2 failures) and 1.22 (3+). 180-day survival 56.8 / 48.4 / 48.3%, log-rank p=6.3e-13. 16/16 segments above 1, 15 significant, Jaipur null. | confirmed | n=13,447, 6,394 events. HR 1.24 [1.14-1.35] and 1.22 [1.10-1.37]. S180 56.8 / 54.6 / 48.4 / 48.3%; p=6.3e-13. Segments 16/16 above 1, 15/16 significant, Jaipur 1.01 [0.73-1.40]. (One of the 16 is the pooled "All", so the count is 14 of 15 distinct subgroups.) |
| 11 | Decomposition: summer-2024 observed +4.58 pp, model +2.18 pp, failures +1.81 pp (83% of model, 40% of observed). Apr-25: failures +1.31 of +2.11 pp. | confirmed | Analyst table reproduces. Independent non-parametric standardisation gives +2.11 to +2.37 pp (46-52%), so 40% is conservative. Apr-25: +1.35 to +1.38 pp. |
| 12 | With failures held at Jan-Mar 2024 levels, modelled cohort retention is flat: 85.2-86.3% (sd 0.27 pp) vs observed 79.1-88.5% (sd 2.52 pp). | **corrected** | This compares a model counterfactual with observed data, but the model itself reproduces only sd 0.92 pp. Observed retention with early failures removed (observed + counterfactual - predicted) is **81.7-88.6%, sd 1.85 pp**: May-24 81.7%, Apr-24 83.8%, Apr-25 82.9%. It is not flat. |
| 13 | Secondary: 3W logit OR 1.52 [1.08-2.16] and Cox HR 1.18. | **corrected** | 1.52 reproduces only with km per swap (`km10`) in the model. 3W riders average 92 km per swap vs 61 km for 2W, which reflects pack size. Without `km10`: **OR 1.16 [1.00-1.36], p=0.051**. Lean model: 1.17 [1.00-1.36]. Raw: 17.2% vs 14.2%. Cox without `km10`: HR 1.10 [1.02-1.18]. The 3W share is stable at 14-15%. |
| 14 | Competitor: exits per attempt OR 1.42 [1.26-1.61], Cox HR 1.20, D31-60 OR 1.05 n.s., 640 exposed. First attempt failed: OR 1.28, Cox HR 1.02. | confirmed | 1.42 [1.26-1.61]; Cox 1.21 (p=0.03); D31-60 OR 1.06 [0.81-1.39]; 640 riders (14.7% vs 15.0%). First attempt failed: 1.28 [1.04-1.58], Cox 1.02. Note: in the analyst's activity-controlled model, first attempt failed becomes n.s. (1.085 [0.88-1.34]). |
| 15 | Not supported: Kyron share OR 0.90, exit OR 0.99, pilot DiD 1.28 (p=0.15), ZipDrop DiD 1.19 (p=0.24), base-price DiD 1.37 (p=0.042, fails Bonferroni). | confirmed | Kyron share 0.92 [0.83-1.02]; exit OR 0.99. Pilot 1.28 [0.91-1.79], p=0.153; ZipDrop 1.19 [0.89-1.61], p=0.244; base price 1.37 [1.01-1.85], p=0.042. State of health (SoH) of packs received: OR 1.08 [0.71-1.66] per +10 points, so no adverse effect either. |
| 16 | Proxies: declared shift 41.3-42.9% at 17-21h. Activity 1-5 attempts 65.7% (n=472), 6-9 29.4%, 10+ 11-12%. GBM area under the curve (AUC) 0.60 / 0.68. | confirmed | 41.3-42.9%; 65.7% (472), 29.4% (1,061), 11.1-11.6%; AUC 0.601 / 0.689 (own LightGBM). The vehicle-model LR test and the SHAP values were not re-derived. |
| 17 | Recommendations: cap at 1 failure gives -1.45 pp overall and -2.72 pp for summer cohorts; 1,061 riders/month, about 15 riders, 27.1 swaps per retained rider, about 418 swaps. | confirmed | 14.62% to 13.16% (-1.46 pp); summer 16.01% to 13.28% (-2.73 pp); 1,061 per month; 27.1 swaps. |
| 18 | Recommendation 2 volume: 3,022 riders reach a 2nd failure within 14 days (median day 8). 449 / 1,119 / 1,730 stockout-or-abandon events per month. | **corrected** (minor) | 3,022 counts any service failure, system errors included. The trigger is defined on stockouts and abandoned queues, which gives **2,821 riders** (median day 8). 449 / 1,119 / 1,730 confirmed. |
| 19 | 2,169 of 7,437 churners (29.2%) came back. | confirmed | 2,169 / 7,437 = 29.2%. |

## Issues that need fixing
1. **Hero figure q6_01 and finding 9 over-claim (misleading chart).** The green line is the pure-model counterfactual (`ret_cf_fail`), but the pooled model only reproduces cohort retention between 82.9% and 86.1% (sd 0.92 pp). So the gap between the dashed line and the observed line is not caused by failures.
   * **Fix:** plot `ret + (ret_cf_fail - ret_pred)`, the observed retention with early failures removed. The analyst's table gives May-24 81.7% (observed 79.1%), Apr-24 83.8%, Jun-24 85.3% and Apr-25 82.9%; the range is 81.7-88.6% with sd 1.85 pp, against 2.52 pp observed.
   * Retitle and recaption along the lines of "early failures explain about a third to a half of each summer dip".
2. **Headline: "driven by early service failures".** Replace it with a quantified statement:
   * Early failures explain about 40-50% of the summer-2024 dip: 1.81 of 4.58 pp in the decomposition; 43% by logit shrinkage; 46-52% by direct standardisation.
   * Counting failures up to day 29 explains about 66% among riders still active after day 14.
   * The rest is unexplained.
3. **Threshold claim.** Remove "one failure is tolerated, the second is not" (finding 4) and "threshold at the 2nd failure" (driver table).
   * Report the activity-adjusted gradient instead: 1 failure OR 1.24 [1.08-1.41], 2 failures 2.47 [2.03-3.02], 3+ 2.34 [1.87-2.92].
   * Optionally add the failure-rate dose-response: 1.00 / 1.21 / 2.10 / 2.99.
   * Retitle q6_04, because its left panel shows 2 = 3+ and 0 > 1 (Simpson's paradox). "Keeps rising" is true only for the attempt-level panel.
   * Recommendation 2 can keep the 2nd-failure trigger as a choice of who to target for cost reasons. Its rationale changes, and the A/B test should add a 1st-failure arm.
4. **3W OR.** Drop `km10` from the rider-level and Cox models, or normalise km per swap within vehicle class. Km per swap mostly reflects pack size (3W 92 km vs 2W 61 km) and is inversely related to swap frequency.
   * Report 3W as OR 1.16 [1.00-1.36] (p=0.051) and Cox HR about 1.10 [1.02-1.18]: a weak, borderline secondary factor.
   * Update finding 6, the driver table, Fig q6_03 and recommendation 6 ("3W onboarding (OR 1.52)").
5. **H1-2024 vs Oct-24 to Apr-25 (+1.85 pp, p=0.009).** Remove it, or label it as season-mixed. Use the season-matched Jan-Apr 2024 vs Jan-Apr 2025 comparison instead: 85.2% vs 85.1%, -0.15 pp, p=0.87.
6. **Headline numbers.** Use one exposure definition consistently:
   * Either "2+ service failures: 13.1% to 22.6% (n=2,246), OR (2+ vs 0-1) 1.66 [1.43-1.93]",
   * or "2+ stockouts or abandoned queues: 13.1% to 23.5% (n=2,067)".
   * Do not attach the exactly-2-vs-0 OR (1.67 [1.37-2.02]) to a "2+" claim.

## Minor issues (no fix required, but worth tidying)
* **"Near within-rider" wording.** The attempt-level GLM pools riders and has no rider fixed effect. The true within-rider Mantel-Haenszel OR is 1.53 [1.41-1.66], or 1.25 [1.15-1.37] with rider x month strata, so the trigger effect is real but smaller once matched closely. Say so.
* **Segment count.** "16 of 16 segments" includes the pooled "All" row. The accurate wording is 15 of 15 subgroups above 1 and 14 significant (Jaipur null).
* **Jaipur null.** Jaipur riders face the most early failures (0.96 per rider vs 0.53-0.89 elsewhere). Raw non-retention is 21.1% vs 14.9% (2+ stockouts or abandoned queues), but the effect vanishes within cohort-quarter (OR 1.02). Worth one line, for example "saturation: failures are everywhere, so they do not separate riders".
* **First attempt failed.** This secondary driver loses significance with the activity control (OR 1.085 [0.88-1.34]). By the analyst's own argument that the activity-adjusted model is the less biased one, it is not robust.
* **Activity control.** Early activity is partly post-treatment (a failure may reduce later swaps), so 2.47 is not unambiguously "the" effect. The failure-rate specification avoids this problem.
* **First row of Fig q6_03.** It mixes three exposures under one label, "2+ in first 14 days": exactly 2 vs 0 (logit and Cox) and 2 prior failures in 7 days (attempt level).
* **Survival convention.** The 3 riders with no completed swap are treated as censored survivors. This is immaterial: the Cox HR moves from 1.240 to 1.238.
* **Spec sensitivity of segment terms.** In the analyst's activity-controlled model, ZipDrop moves to OR 1.57 [1.32-1.86] and independent riders to 0.62-0.68. These are spec-sensitive segment effects that the findings do not mention.

## Figure review (all 8 exist and are legible at slide size)
| Figure | Verdict |
|---|---|
| q6_01 cohort retention trend (HERO) | **Misleading.** The dashed counterfactual implies failures explain the whole dip (see issue 1). The annotation box also hides the naive-metric line from Jul-24 to Mar-25 (cosmetic). The numbers on the observed line, CIs and failure bars are correct. |
| q6_02 KM by quarterly cohort | Correct. The title matches the data (2024Q2 vs 2025Q2 p=0.96; S90 values reproduce). |
| q6_03 driver effects (HERO) | Mostly correct. The 3W logit point (1.52) is inflated (issue 4). The first row mixes exposures. The independent-rider diamond (adverse, significant) is labelled NOT SUPPORTED; that is explained in the caveats but confusing on a slide. |
| q6_04 dose-response | **Title over-claims.** "Risk jumps at the 2nd failure and keeps rising": the left panel shows 2 = 3+ and the 0-vs-1 comparison is confounded by exposure (issue 3). The right panel is correct. |
| q6_05 KM by early failures (HERO) | Correct (57 / 55 / 48 / 48% at 180 days; HR 1.24; p=6.3e-13). Grammar nit: "1 early failures". |
| q6_06 decomposition | Correct. The legend shows observed vs model gaps, and "most of the *explained* dip" is accurate. The "-0.40" label overlaps the y-tick label (cosmetic). |
| q6_07 SHAP | Mostly fine. Km per swap (0.177) nearly ties with stockouts (0.181). It is an engagement proxy: riders in the highest-km quartile make 11.9 early attempts vs 16.3 in the lowest. Label it as such or drop it. |
| q6_08 segment stability | Correct values. The "16 of 16" title includes the pooled row. |

## Missed angles
1. **Same-season May and June checks on shorter outcomes.** These strengthen "no decline": R14 for May-24 vs May-25 is 89.4% vs 91.2% (p=0.19), and D1-7 is flat in every month (97.7-99.1%; no 2024-vs-2025 difference with p<0.15). D31-60 cannot compare May or June, so this closes the main gap in the trend evidence.
2. **Battery degradation as a natural test.** Mean SoH of packs new riders received fell from 98.7 (Jan-24 cohort) to 82.0 (Apr-25), and km per swap fell from 74.7 to 56.4. Same-season retention stayed flat, and the SoH OR is 1.08 (n.s.). This is a strong Q4 cross-link that firmly supports "battery is not a retention lever".
3. **Failure-rate (exposure-free) dose-response.** It gives a clean monotone gradient (1.00 / 1.21 / 2.10 / 2.99) and is the best basis for the trigger threshold.
4. **Failures after day 14.** Including failures up to day 29 lifts the explained share of the summer-2024 gap from 54% to 66% (riders active in days 14-29). This supports extending the CX trigger to the full first 30 days, as recommendation 2 already proposes.
5. **Economic size of the prize.** About 15 riders and about 418 swaps per cohort-month is small against about 300K swaps per month. Express it cumulatively over the rider lifetime, with revenue per swap. Note that under our usage-based battery-depreciation assumption, cm1 per swap is negative in 2025 (Q1/Q5), so retention gains add volume, not margin, until unit economics are fixed.

## Checks that passed
* **Base and timestamps.** Clean events were used: test stations are removed and `event_ts` is the corrected timestamp. `t0` equals `riders.first_attempt_ts` for all 16,341 riders.
* **Eligibility.** The 60-day follow-up rule (t0 < 2025-05-01) is right.
* **Outcome denominator.** The outcome uses completed swaps, and failure rates use attempts, which is the right denominator.
* **Look-ahead.** No look-ahead: features come from days 0-13 and the outcome from days 30-59, and the prior-failure window looks backward only.
* **Censoring.** Censoring is handled, with administrative censoring 30 days before data end.
* **Data and cost model.** No telemetry is used and no cost-model levels are claimed.
* **Multiple testing.** The Bonferroni note on the 4 DiDs is appropriate.
