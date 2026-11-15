# Q1 verification: Network performance over time

Verifier script: `analysis/verify/q1_verify.py`. It is fresh code. Flags are rebuilt from `event_type`, station aggregates are rebuilt from events rather than `station_month`, and the cm1 bridge uses a different (2-segment) decomposition. It runs in about 1 s. Output: `outputs/tables/q1_verify_run_log.txt` and `outputs/tables/q1_verify_numbers.csv` (358 numbers).

## Verdict summary
- **Arithmetic: excellent.** Every number I recomputed matches the analyst to rounding. This covers the KPI panel, growth multiples, mix decomposition, Launch panel tests, charger-generation cells, deciles, queue wait, bridge components, counterfactual and sensitivity table. Different code and a different bridge segmentation gave the same results (for example Kyron −₹15.82 per swap and kWh +₹2.53 vs +₹2.55).
- **One material problem: the cost model keeps charging wear after packs pass end of life (EOL).** It charges wear on every issue, including packs already below the 70% EOL. As a result, Kyron 2W packs were charged **1.30x their purchase cost** over their life (401.7 cycles issued vs 308.9 cycles to EOL).
  - **122,204 Kyron swaps in Apr–Jun 2025** used packs already below 70% State of Health (SoH). That is 6.1% of Kyron issues in Apr, 70.2% in May and 90.1% in Jun.
  - These swaps add **₹7.58 per swap to H1-25 wear (₹2.15M a month)** and **₹14.39 per swap to Q2-25 wear (₹4.30M a month)**.
  - This inflates the headline H1-25 cm1 level and the Kyron magnitude. It reverses the same-season Q2 margin bridge (figure q1_06). It also breaks the claim that "the direction holds under every wear assumption".
  - The core story **survives** in windows with no post-EOL issues:
    - Q1-24 → Q1-25: cm1 per swap −₹7.59 (Kyron −₹16.04).
    - Jul-Aug-24 → Oct-24..Mar-25: −₹14.28 (Kyron −₹16.00).
- **Wording over-claims:** Finding 3 says "no station got better" (49 of 90 Launch stations did not get worse). Finding 6 presents the survivorship mechanism as established.
- **needs_fix = true.**

## Verdict table

| # | Claim | Analyst | Recomputed | Verdict |
|---|---|---|---|---|
| H | Headline: H1 swaps 2.43x, revenue 2.63x, failure rate 7.04→6.06%, cm1 +6.75 (Jul-Aug-24) → −6.39 (H1-25), Kyron −₹15.8 per swap ≈ ₹4.5M a month | as stated | 2.4287x, 2.6267x, 7.04→6.06%, +6.749 → −6.392, Kyron −15.82 (−₹4.49M a month). **But** ₹7.58 per swap of H1-25 wear is charged on packs past EOL. H1-25 cm1 is **−₹1.13** if those swaps are costed at Cellora wear, **+₹1.19** if wear stops at EOL. Kyron change Jul-Aug-24 → H1-25 is **−₹10.50 / −₹8.19** per swap (≈ ₹3.0M / ₹2.3M a month). | **corrected** |
| 1 | Metrics diverge: H1 attempts 2.40x, swaps 2.43x, revenue 2.63x, riders 2.18x, failures 2.07x. Jan-24 → Jun-25: swaps 2.93x, revenue 3.18x (₹6.20M → ₹19.69M), stations 90→149. Index 260/282/242. cm1 per swap +0.64 → +6.75 → −6.39; cm2 per swap −17.43 → −7.15 → −17.87; total cm2 −₹2.03M → −₹5.07M a month | as stated | 2.403 / 2.429 / 2.627 / 2.182 / 2.067. 2.929x, 3.177x, 6.199 → 19.691M, 90→149. 259.7 / 281.6 / 242.1. cm1 0.636 / 6.749 / −6.392; cm2 −17.43 / −7.15 / −17.87; −2.035M → −5.068M | **confirmed** (the H1-25 cm levels carry the post-EOL caveat; "turned negative" holds for Oct-24..Mar-25: −₹7.53) |
| 2 | "Failures rose faster than swaps" holds only for naive winter-vs-summer | naive 6.26x vs 2.93x; Q1 2.43 vs 2.48; Q2 1.94 vs 2.38; H1 2.07 vs 2.43; Jun 1.88 vs 2.32; May 11.93% / 9.54%; stockouts 7.82%; abandoned 3.81% | 6.257 vs 2.929; 2.427 vs 2.485; 1.945 vs 2.381; 2.067 vs 2.429; 1.884 vs 2.317; 11.93 / 9.54; 7.82; 3.81. Jul–Mar range 3.82–4.01%; system errors 0.27–0.32%; cancellations 0.39–0.46% | **confirmed** |
| 3 | Same-season improvement is a mix effect | Q2 −1.61 pp (CI −3.02 to −0.25) = within +0.02 + mix −1.62. New stations 36.6% of attempts at 5.08% vs 9.55%. Launch 9.52 → 9.55 (CI −0.22 to +0.26), median −0.12 pp, 41/90 worse, Wilcoxon p = 0.045, +48% attempts, May 11.9% both years | −1.61 = +0.019 − 1.625. 36.56% at 5.081% vs 9.555%. Launch +0.030 pp (CI −0.21 to +0.26), median −0.12, 41/90, p = 0.0449, +48.4%, May 11.93 / 11.89. Unpaired CI −2.98 to −0.23; **paired station-cluster CI −2.09 to −1.14 pp** | **confirmed** (the heading "no station got better" over-claims; the CI is conservative) |
| 4 | New stations fail less because none are Gen1, not because they are lightly loaded | 4.8 vs 5.0 attempts per slot-day. Gen1 12.64→12.41, Gen2 5.25→5.23, new Gen2 5.87, new Gen3 4.69. Gen1 = 34% of stations / 38% of attempts / 60% of failures (46,376 of 77,445). Hot-city Gen1: 36 stations, 53% of failures, 15.2% rate | 4.825 vs 5.043. 12.64→12.41, 5.25→5.23, 5.87, 4.69. 34.0 / 38.2 / 59.9% (46,376 of 77,445). 36 stations, 52.6%, 15.20%. Same-city Gen2 Launch vs new: Delhi 8.46 vs 8.42, Hyderabad 7.21 vs 7.23, Pune 5.13 vs 5.21. Extra load check: Delhi Gen1 runs *lower* load than Gen2 (5.29 vs 6.38 attempts per slot-day) yet fails 2x; its load rose 68% YoY with a flat rate (16.35 → 16.64%) | **confirmed** (nuance: 37 of 60 new stations are Gen3, which beats Gen2 by about 3 pp in hot cities) |
| 5 | No hidden deterioration at the busiest stations | D9 12.5 vs 4.5; D10 10.7 vs 4.8. Busiest quintile 11.69 → 11.42 with +66% attempts. All quintiles within ±0.5 pp. Spearman ρ = −0.01 (p = 0.95). Decile ρ = 0.71. Low deciles 53–73% new | 12.50 vs 4.50; 10.73 vs 4.80. 11.69 → 11.42, +65.6%. Max abs change 0.49. ρ = −0.007 (p = 0.951). 0.709. 53–73% | **confirmed** |
| 6 | Queue wait is blind (flat) because of survivorship | mean 4.06, p90 7.07 every month; Q2 4.061 → 4.060 | 4.049–4.070 mean, 7.05–7.10 p90; 4.0608 → 4.0604. **But** the wait is also flat by hour (4.05–4.09) and across the most and least stressed cells (May-25 hot-city Gen1 4.056 vs other cells 4.05–4.09). Abandoned-queue waits are flat too (11.6–11.8 min every month; stockouts about 0.49 min) | **confirmed** (numbers). The survivorship mechanism is not demonstrated; the field shows no load signal at all |
| 7 | cm1 bridge: price +₹5.16 (leg 1); Kyron −₹15.82 (−₹4.49M a month), ZipDrop −₹1.67, pilot +₹1.71, kWh +₹2.55 (leg 2). Q2-24 → Q2-25 −₹6.51 with Kyron −₹15.61. Fixed cost 16.81 → 11.76; cm2 −15.86 → −17.32 | as stated | Leg 1 price +5.15, kWh +0.71. Leg 2 Kyron −15.82 (−4.49M), discount −1.69 (all partners; ZipDrop direct −1.667), pilot +1.71, kWh +2.53. Q2 −6.51, Kyron −15.61, fixed 16.81 → 11.76, cm2 −15.86 → −17.32. **But** ₹14.39 per swap of Q2-25 wear is post-EOL. Q2-25 cm1 is **+₹8.83** (cap) or **+₹4.44** (post-EOL at Cellora wear), so the same-season Q2 cm1 **improved** by +₹7.88 / +₹3.49. Clean replacement Q1-24 → Q1-25: +0.27 → −7.32 (−₹7.59) = Kyron −16.04 (−₹4.31M a month), price +5.11, kWh +2.78, pilot +1.68, discounts −1.65, mix +0.44. Fixed cost 19.52 → 11.16, so cm2 −19.25 → −18.48 | **corrected** |
| 8 | Falling energy per swap reflects pack ageing | kWh 2.069 → 1.571 (−24.1%); SoH 98.9 → 75.4; SoC 9.31 → 9.26; km 70.5 → 48.0 (−32%) | 2.0687 → 1.5710 (−24.1%); 98.86 → 75.36; 9.311 → 9.264; 70.51 → 47.97 (−32.0%). Within H1-25 by SoH band: kWh 1.53 (60–65%) → 1.78 (85–90%); km 36.9 → 57.0 | **confirmed** (moderate: association) |
| 9 | Counterfactual Kyron at Cellora wear: H1-25 cm1 +10.24 vs −6.39 (+₹16.63, ₹4.72M a month); cm2 −1.24; Jun-25 +11.41; Kyron 2W excess ₹4.48M a month (62,290 swaps a month), 3W ₹0.23M | as stated | +10.238, −1.242, +11.411, ₹4.716M a month; 2W ₹4.484M (62,291 a month), 3W ₹0.231M. **But** the "actual" includes post-EOL charges. Uplift vs a consistent actual is **+₹9.05 per swap (₹2.57M a month, cap)** to **+₹11.36 (₹3.22M a month, post-EOL at Cellora wear)**. Warranty basis: one-off lifetime shortfall **₹30.7M** (of ₹44.3M purchase cost) | **corrected** |
| 10 | Direction of the decline holds under every wear assumption; level and sign do not | Jul-Aug→H1-25: −13.14 / −10.04 / −9.19 / −6.88. H1→H1: −7.03 / −3.96 / −3.11 / −0.83. H1-25 cm1: −6.39 / +4.21 / +7.14 / +15.04. cm2: −17.87 / −7.27 / −4.34 / +3.56 | All 16 numbers reproduce exactly. **But** the scenario set omits the cap on post-EOL wear. Under that cap: H1→H1 **+0.55**, Q2→Q2 **+7.88** (direction reverses); Jul-Aug→H1-25 −5.56 (holds). Robust version: Q1-24→Q1-25 −7.59 / −4.52 / −3.67 / −1.38 (no post-EOL issue, so the cap changes nothing) | **corrected** |
| R1 | Rec: replace Kyron with Cellora-grade packs; +₹16.6 per swap ≈ ₹4.7M a month; Kyron 2W excess wear ₹4.48M a month for the warranty claim | as stated | Valid as a *forward* per-swap cost gap (₹103.9 vs ₹31.9 lifetime wear per 2W swap at about 25% share). Not valid as the H1-25 margin gap or as the claim basis; use ₹30.7M one-off. Early gating works: Kyron issued SoH fell 98.2 → 92.4% from Sep to Nov-24 vs Cellora 91.7 → 89.8% | **corrected** (framing) |
| R2 | Rec: upgrade 36 hot-city Gen1 stations; 20,177 fewer failures (26%); 7.92 → 5.86% | as stated | 20,177; 5.856%. With a same-city Gen3 target: 28,493 avoided; 5.01% (association; Gen3 sites are all new) | **confirmed** |
| R4 | Rec: ZipDrop amendment −₹1.67 per swap, ₹0.47M a month, 15.4% of swaps (23.4% of partner-billed) | as stated | 12.00% → 28.00% exactly; −1.667; ₹0.473M; 15.42%; 23.42% | **confirmed** |

## Methodology review (ranked)
1. **(Material) Wear charged after EOL.** The brief's per-cycle formula multiplies purchase cost by loss per cycle over (initial SoH − 70) and applies it on every issue. Nothing stops the charge once a pack crosses 70%, so lifetime wear for Kyron 2W = **1.30x** purchase cost. That is inconsistent with the stated assumption (70% EOL, no salvage), under which total depreciation equals the purchase cost.
   - Only Kyron is affected (0 Cellora/Amptek issues below 70%).
   - It is concentrated in May–Jun 2025: Jun-25 post-EOL wear is ₹23.08 per network swap.
   - Every H1-25 and Q2-25 margin level and the Kyron magnitude depend on it.
   - The sensitivity scenarios (EOL 60%, salvage) do not address it.
2. **Window choice.** H1-25 and Q2-25 are the only windows contaminated by item 1. Q1-25 and Oct-24..Mar-25 have zero post-EOL issues, and both are cleaner evidence for the same story.
3. **Over-claim (Finding 3 heading).** "No station got better" is false at station level: 49 of 90 did not get worse, and the median change was −0.12 pp. Correct claim: the 90-station panel as a whole was flat (+0.03 pp).
4. **Over-claim (Finding 6 mechanism).** Survivorship is asserted, not shown. The completed wait is identical across hour of day and stress level. Waits of abandoned attempts are also invariant. That points to a field with no load signal, not to truncation. The recommendation (use attempt-based rates) still stands.
5. **Conservative CI.** The network Q2 bootstrap resamples 2024 and 2025 stations independently. The 90 Launch stations are paired across years, and a paired cluster bootstrap gives −2.09 to −1.14 pp (vs −3.02 to −0.25). The conclusion is unchanged.
6. **Clarity.** "Kyron share 2.8–2.9% to Aug-24" is entirely Kyron **3W**: the Kyron share of 2W issues was 0% until Sep-24. Kyron 3W reference wear is ₹107.2 vs ₹76.2 for Cellora 3W (53 bad 3W packs), which contradicts the brief's "Kyron 3W behaves like the others". The analyst's ₹0.23M a month is consistent with this.
7. **Checks passed.**
   - Seasonality handled (same-season everywhere).
   - Simpson's paradox and mix decomposed exactly.
   - Denominators correct (failures per attempt, economics per completed swap).
   - Cancellations excluded from failures, as in the brief.
   - Corrected `month` used; no telemetry used.
   - Multiple tests interpreted with practical significance.
   - The attempt count from `station_month` matches the event scan exactly.

## Figure review
All 8 PNGs exist, are legible at slide size, and their annotations match the recomputed data.
- **q1_01 (hero):** correct. The bottom-panel variable cost per swap for May–Jun-25 includes post-EOL wear (the gap would close to about +₹12–18 per swap under the cap). Add a footnote.
- **q1_05 (hero):** numbers correct. The Kyron bar (−₹15.8, −₹4.5M a month) is inflated by post-EOL charges. Split it, or rebase leg 2 to Oct-24..Mar-25 (Kyron −₹16.00, ending cm1 −₹7.53).
- **q1_06: misleading as titled.** "Kyron wear outweighed the price rise" in Q2 a year apart reverses once post-EOL wear is capped (Q2-25 cm1 +₹8.83). Replace it with the Q1-24 → Q1-25 bridge.
- **q1_07:** the end-of-line labels (+₹16.4 / +₹11.4 / +₹8.5 / +₹5.6 / −₹5.0) are **Jun-25** values, but the subtitle quotes H1-25 levels. Label them "Jun-25". Add a post-EOL-cap line.
- **q1_02:** correct. The queue-wait panel note asserts survivorship; soften it.
- **q1_03, q1_04, q1_08:** correct. In q1_08 the right x-label is clipped ("...both years" loses its closing parenthesis); this is cosmetic.

## Missed angles
1. **Sub-EOL packs were issued to riders.**
   - 122,204 Kyron issues below 70% SoH in Apr–Jun-25.
   - This includes **32,903 2W issues after the recorded 2025-06-15 retirement date**, with mean issued SoH 61.7% and the last issue on 2025-06-30.
   - In H1-25, packs at 55–65% SoH gave 35–39 km per swap, vs 57 km at 85–90%.
   - This is a Battery Ops control failure and a master-data inconsistency (links to Q4 and Q7).
2. **Price per km.** Riders paid ₹0.79 per km in Jan-24 and ₹1.27 per km in Jun-25 (2W, **+60%**); H1-24 → H1-25 **+42%**. That is far larger than the ₹5 list-price rise and is a candidate driver for Q6 retention.
3. **Whole-fleet ageing.**
   - At 30 Jun 2025, non-Kyron 2W packs average 80.0% SoH (53% at or below 80%, none at or below 75%). The SoH of returned non-Kyron 2W packs falls about 1.4 points a month (86.5% Feb-25 → 80.8% Jun-25).
   - Extrapolating linearly, the bulk of the 4,133 packs reaches 70% around early 2026. The tight SoH spread means they would all need replacing at about the same time, which is relevant to the "more batteries" budget proposal.
4. **A failure is mostly a lost swap, not a delayed one.** Of Q2-25 service failures, 5.3% were followed by the same rider's completed swap within 60 min, 11.5% within 3 h and 59.5% within 24 h. Upper bound on lost revenue: ₹5.03M in Q2-25.
5. **Early detection supports Rec 1.** Kyron issued SoH fell 5.7 points in two months (Sep → Nov-24) vs 1.9 points for Cellora. A 2–3-month gate would have flagged it by Nov-24.
6. **Upgrade target.** Gen3 beats Gen2 by about 3 pp in hot cities (Delhi 5.30 vs 8.44, Hyderabad 4.66 vs 7.22, Jaipur 5.25 vs 8.60). With a same-city Gen3 target, 28,493 failures are avoidable (network 5.01%) vs 20,177 with a Gen2 target. Gen3 is confounded with new sites.
7. **cm2 nuance.** Same season in Q1, cm2 per swap was roughly flat (−19.25 → −18.48), because lower fixed cost per swap (+₹8.36) offset the cm1 fall. Total cm2 losses still grew with volume.

## Fix instructions
See the structured output (`fix_instructions`). In short:
- Add a post-EOL flag and variants (cap, and post-EOL swaps at Cellora wear).
- Anchor the headline and bridges on Q1-24 → Q1-25 and Jul-Aug-24 → Oct-24..Mar-25.
- Replace or annotate q1_06 and q1_05.
- Add a fifth scenario to the q1_07 sensitivity and label its end values as Jun-25.
- Reframe Rec 1's impact and the warranty basis.
- Fix the wording in Findings 3 and 6.
- Add the sub-EOL issuance finding.
