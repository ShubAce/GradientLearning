# Q5 verification - Pricing & partner economics (adversarial review)

Verifier script: `analysis/verify/q5_verify.py`. It was written from scratch, runs in about 10 s and re-derives everything from `data/clean` via `src/common.py`.

Estimators were deliberately different from the analyst's:
- Two-way fixed effects by weighted **alternating projections**, with cluster-robust SEs.
- Swap- or attempt-weighted **city-aggregate 2×2 DiDs**.
- A **balanced station panel**.
- **Mechanical price × mix** checks.

**Bottom line.** The core pricing results reproduce to the second decimal: revenue +₹7.90, peak share −2.49 pp, peak stockouts +0.05 pp, Partner Y/N split, Kyron trap, base-price elasticities. The partner P&L also reproduces exactly. **Five things need fixing before judging:**
1. Two hero-figure statements are wrong or contradict the data.
2. The ZipDrop amendment DiD is contaminated by the pilot.
3. The "evening = real congestion" generalisation is an aggregation artefact.
4. Budget (d) is framed on backward-looking Kyron wear.
5. One recommendation (drop the 12-13h surcharge) silently gives up about 40% of the revenue it promotes.

## 1. Verdict table

| # | Claim (analyst) | Verdict | Recomputed (verifier) |
|---|---|---|---|
| H | Headline: +₹7.90 (+11.7%); −2.5 pp (4%) peak shift; stockouts +0.05 pp; ZipDrop "least valuable at −₹20.6 cm1/swap (−₹69.3 L, rank 12/12)" | **corrected (wording)** | All numbers reproduce. But rank 12/12 is on **total** cm1. **Per swap ZipDrop is 10th of 12**: CargoTuk −₹22.25 and HaulKing −₹21.42 are lower. "Least valuable at −₹20.6 per swap" mixes the two bases. |
| 1 | Pilot revenue: IND +₹7.90 (CI 7.84-7.95) on ₹67.45; Partner Y +₹5.43; Partner N +₹0.22 | **confirmed** | FE +7.897 [7.842, 7.952]; city 2×2 +7.93; pre 67.45. Mechanical: 15 × 59.3% − 8 × 12.5% = ₹7.90 (₹8.49 at pre timing). PY +5.432 [5.154, 5.710] on 65.02; PN +0.218 [−0.282, 0.718]. Permutation p 0.067. |
| 2 | Peak share −2.49 pp (CI −3.11 to −1.86) on 61.9%; off-peak +2.50 pp on 9.9% | **confirmed** | FE −2.488 [−3.113, −1.862]; 2×2 −2.53; off-peak +2.495 [2.095, 2.895]. Split: **−0.98 pp from 12-13h** (base 21.3%) and **−1.51 pp from 19-22h** (base 40.6%). Gains: +1.10 pp at 14-16h and +1.40 pp at 23-05h. The monthly gap fades to −1.24/−1.26 pp in May/Jun-25. |
| 3 | Congestion unchanged: peak stockout +0.05 pp (CI −0.09 to +0.18), failure +0.07, abandonment +0.04 | **confirmed** (but see Fig 1 issue) | FE +0.049 [−0.085, +0.183]; city 2×2 +0.066; balanced 90-station 2×2 +0.076. **Mumbai-only (Kyron-matched) control −0.069 [−0.249, +0.111].** Evening 19-22h only +0.016 [−0.137, +0.170]. Failure +0.074; abandonment +0.038. Summer-2025 artefact confirmed: balanced gaps May/Jun-24 −5.95/−7.47 vs May/Jun-25 −6.93/−7.30. |
| 4 | Timing follows who pays: PY billed +₹15/−₹8 and shifted −2.25 pp; PN billed ₹0 and shifted −0.38 pp | **confirmed** | Surcharge (amount − list + discount) in pilot cities post: PY +14.997 at peak, −8.001 off-peak; PN 0.00. PY −2.250 [−2.753, −1.748], off-peak +1.73. PN −0.379 [−1.107, +0.348]. IND 2W −2.57; IND 3W −1.04 [−3.81, +1.72]. |
| 5 | Base price rise passed into revenue; 2W elasticity +0.02 (log) / −0.24 (levels, p=0.024); continuation +0.01 pp | **confirmed** | Revenue/swap: IND 2W 60→65, 3W 100→110, partner 2W 53.53→57.97. IND 2W cm1 9.59→15.15. 2W log +0.18% (ε +0.02); levels −2.0% (ε −0.24, CI −0.45 to −0.03); n = 36,627. Continuation 2×2 +0.03 pp on 91.8%. *Nit:* 3W log ε = +0.55 (n.s.), so "\|ε\| < 0.5 in every specification" is not literally true. |
| 6 | Kyron trap: Kyron 37.5-38.3% in DEL/HYD/JAI vs ~10% elsewhere; naive cm1 DiD +19.08 vs +7.87 adjusted (2.4x); volume DDD +4.85% (n.s.), common shock −10% | **confirmed** | Kyron shares identical (10.2/10.3/10.4 vs 37.5/38.2/38.3%). cm1 DiD +19.08 naive / +7.87 Kyron-adjusted / +7.43 Mumbai-only (2.42x). Log swaps: IND −6.32, PN −8.22, PY −7.39. DDD +4.85 [−1.14, +10.84]; common shock −10.05. |
| 7 | ZipDrop largest by swaps and "least valuable on **every** margin measure"; FeastFly larger by revenue and riders | **corrected** | Totals confirmed: 336,422 swaps (15.5% of network, 23.4% of partner swaps); revenue ₹1.64 Cr vs FeastFly ₹2.01 Cr; riders 1,992 vs 2,271. Per swap: 48.64 / 18.16 / 51.08 / 11.10; cm1 −20.60; cm2 after WC −32.42. cm1 −₹69.3 L (12/12); cm2 after WC −₹1.09 Cr (12/12); full window −₹75.1 L / −₹1.49 Cr (12/12). **Per-swap ranks are 10/12** on cm1/swap (CargoTuk −22.25, HaulKing −21.42), on cm2 after WC/swap (−35.05, −34.90 vs −32.42) and on steady-state cm1/swap (−14.70, −13.69 vs −4.18). Worst per swap among **2W** partners only. |
| 8 | Cost to serve is the same; contract price explains the gap (ZipDrop revenue gap −14.70, cost gap +0.10, cm1 gap −15.42) | **confirmed** | Within city × month × class × band cells: revenue −14.70, variable cost +0.72, fixed −0.61 (net +0.11), cm1 −15.42. Segment cm1/swap checks: quick commerce −17.51, cargo 3W −22.01. |
| 9 | Amendment cost "about ₹11 per swap" (DiD revenue −11.62, cm1 −11.03; rider FE −11.66 [−11.77, −11.56]); ₹4.5 L/month; no extra volume | **corrected** | The all-city DiD replicates (−11.62 / −11.03; rider FE −11.66 / −11.19). However, the comparison group's revenue rose +₹1.22/swap (58.75 → 59.97) because **billable partners began paying the pilot surcharge in Bengaluru/Pune from Oct-24**. In the 4 non-pilot cities the rider-FE DiD is **revenue −₹10.40 [−10.41, −10.40]**, which is exactly 16 pp × ₹65, and **cm1 −₹9.77 [−10.45, −9.09]**. The analyst's CI excludes the clean value. ₹4.55 L/month and ₹54.5 L/yr (mechanical) are correct. Volume +56.5% vs +53.2% and share 24.9% → 25.3% are confirmed. The pre-trend was already +55.8% vs +50.7%, so there is no acceleration. |
| 10 | 3W loses money even with healthy packs: cm1 −22.49; steady-state −14.90; break-even list ≈ ₹126; 2W steady-state +12.23 | **confirmed** | 101.31 / 39.55 / 84.25 / −22.49. Non-Kyron energy 39.98 + wear 76.23 = 116.21; steady-state cm1 −14.90; break-even list 126.2. 2W steady-state +12.23. (Depends on the no-salvage wear assumption, as stated.) |
| 11 | "Real congestion sits in 19-22h; 12-13h is least congested" (failure 6.17% vs 4.95%) | **corrected (interpretation)** | The network numbers reproduce: failure 6.17% vs 4.95%; stockout 3.92% vs 3.08%; queue wait 243.3-244.2 s in every band; 3,814,982 attempts. **The evening excess is a hot-city × hot-season pattern.** Evening-minus-midday failure is +2.6 to +4.8 pp in Delhi NCR, Hyderabad and Jaipur in Summer/Monsoon. It is within ±0.4 pp in Bengaluru, Mumbai and Pune in every season except Pune summer (+1.1), and in Winter/Post-monsoon in all cities. All-month figures: Bengaluru +0.10, Pune +0.33, Mumbai −0.18 vs DEL +2.34, HYD +1.88, JAI +2.74 pp. |
| 12c | Budget (c): +₹9.5 L/month (CI 9.2-9.7); cm1 +9.3 L; stockouts +57/month; ₹4.5 L/month uncollected from ZipDrop + Swiggle Go | **confirmed, minor correction** | DiD-based ₹9.48 L/month replicates. Re-pricing the control cities' **own** timing mix mechanically gives ₹9.08 L (−4%), and the CI ignores that extrapolation. The uncollected surcharge is **gross**: net of the −₹8 off-peak credit the same tariff gives billable partners, it is **₹4.13 L/month** (29,898 × 15 − 4,400 × 8). |
| 12d | Budget (d): ~504,633 swaps/yr; cm1 −₹1.04 Cr/yr; cm2 after WC −₹1.64 Cr/yr; steady-state cm2 after WC −₹81 L/yr; break-even discount 24.6% / 6.1%; 19.7% of peak swaps | **confirmed (arithmetic), framing needs fix** | All numbers reproduce (504,633; −1.04 Cr; −1.64 Cr; 72.6% at peak; 19.7% of peak swaps; ZipDrop 2W steady-state variable cost ₹49.03 → 24.6% / 6.1%). However, −₹1.04 Cr/yr cm1 includes wear on **Kyron packs retired 2025-06-15**. The forward-looking steady-state cm1 is **−₹4.18/swap ≈ −₹21 L/yr**, about 5x smaller. |

## 2. Issues that need fixing

1. **Hero Fig `q5_04` title is false.** It says ZipDrop has the "deepest loss per swap", but the same chart plots CargoTuk (−22.2) and HaulKing (−21.4) below ZipDrop (−20.6). The headline and finding 7 ("least valuable on every margin measure") repeat the error. ZipDrop is 12/12 on *total* cm1 and cm2. Per swap it is 10/12 overall and worst only among 2W partners.
2. **Hero Fig `q5_01` stockout panel contradicts its own annotation.**
   - The event study plots all six Oct-24..Mar-25 coefficients as negative (−0.08 to −0.53 pp; Nov, Feb and Mar CIs below zero), so it reads as "stockouts fell". The annotation (and the correct DiD) says +0.05 pp.
   - The cause is the unbalanced station panel: Wave1/Wave2 stations enter, and station FE are estimated across all 18 months including the summer spikes.
   - On the 90 stations present in all 18 months, the Oct-Mar gaps are **+0.02, 0.00, +0.01, +0.15, −0.06, +0.08 pp** vs Sep-24.
3. **The ZipDrop amendment DiD has a concurrent-treatment confound.**
   - Other partners' revenue rose ₹1.22/swap from the pilot surcharge (billable partners in Bengaluru/Pune from Oct-24), so the DiD over-states the amendment's cost by ~12%. Fig `q5_06` annotations ("DiD −11.6", "DiD −11.0") disagree with its own title (₹10.4), which is the correct figure.
   - Clean estimates: **revenue −₹10.40 / 2W swap; cm1 −₹9.77 (CI −10.45 to −9.09)**.
4. **"Real congestion sits in 19-22h" is an aggregation artefact, and recommendation 1 rests on it.**
   - The excess exists only in DEL/HYD/JAI in Summer/Monsoon (the heat/thermal-derating story in q2/q3). It does not exist in the pilot cities, where 19-22h stockouts did not move either (+0.02 pp).
   - Dropping 12-13h from the surcharge gives up **≈₹3.0 of the ₹7.90 per independent swap** (20.3% midday share × ₹15). That is **≈₹3.9 L/month of the ₹9.5 L/month** rollout uplift, with no demonstrated congestion gain. Rec 1 currently promises both the full ₹9.5 L and the band change.
5. **Budget (d) leads with backward-looking costs.**
   - "Avoid locking in −₹1.04 Cr/yr cm1" uses Nov-24..Jun-25 wear, which includes Kyron packs that are already retired. A long-term exclusive should be judged on steady state: **cm1 −₹4.18/swap ≈ −₹21 L/yr; cm2 after WC −₹16.0/swap ≈ −₹81 L/yr**.
   - cm2 adds allocated station rent and maintenance, which an exclusive does not avoid. Use cm1 plus an explicit peak-capacity cost for the decision; the 6.1% cm2 break-even is not a marginal-decision threshold.
6. **Uncollected surcharge** (rec 2 / Fig `q5_07`): ₹4.5 L/month is gross. Net of the off-peak credit it is ₹4.1 L/month.

## 3. Methodology review

| Trap | Status |
|---|---|
| Uncorrected timestamps | **Handled.** Corrected `hour` used throughout. Validated: in the v3.2.0 bug window the PEAK tariff code matches the corrected-hour band on **100%** of 14,121 pilot-city independent swaps, vs **32.7%** on the raw hour. |
| Tariff schedule | **Handled and right.** PEAK = 12-13h + 19-22h (+₹15.00, 2W and 3W); OFFPEAK = 23-05h + 14-16h (−₹8.00); the brief's "12:00-22:59" is wrong. |
| Confounding (Kyron) | **Handled well** for pilot margins and volume (Kyron share, Mumbai-only, DDD). Not applied to the congestion DiD, but my Mumbai-only check (−0.07 pp, n.s.) shows it does not matter. |
| Concurrent treatment | **Missed** in the ZipDrop amendment DiD: the pilot surcharge moves the comparison group (issue 3). |
| Seasonality | **Handled** (Oct-Mar main window; summer-2025 artefact shown with pre-pilot summer gaps). |
| Simpson / aggregation | **Handled** for partner cost-to-serve (within-cell gaps). **Missed** for the hour-band congestion claim (issue 4). |
| Denominators | **Correct:** attempts for congestion rates, completed swaps for money. |
| Inference | **Good:** city-permutation p = 0.067 is the design floor and is reported honestly. Caveat: the rollout CI (₹9.2-9.7 L) is rider-cluster sampling error only and ignores transfer to cities with a different timing mix (mechanical ₹9.08 L). |
| Causal over-claiming | Mostly careful ("association", "not identified" for volume). "Bought no extra volume" rests on an assumed parallel growth path; pre-trends support it (+55.8% vs +50.7% before, +56.5% vs +53.2% after). |
| Multiple testing | Many segment and spec cuts, but the headline effects are p < 0.001 at rider level and consistent across all specs. Acceptable. |
| Survivorship / censoring | Rider-month outcomes are conditional on activity (intensive margin), which is acknowledged. Continuation uses transitions inside the window, so no right-censoring problem. |
| Cost-model consistency | Uses the shared revenue/energy/wear/fixed definitions. The 12% p.a. working-capital rate is an added assumption, so label it. cm2 includes allocated fixed cost, which is not a marginal cost (issue 5). |
| Telemetry blanks | Not used. |

## 4. Figure review

| Figure | Exists / legible | Verdict |
|---|---|---|
| `q5_01_pilot_event_study` (hero) | Yes / yes | Panels 1-2 match the data. **Panel 3 misleading**: unbalanced-panel event study shows a post-pilot decline that the DiD and the balanced panel do not (issue 2). |
| `q5_02_pilot_segments` | Yes / yes | Correct; all estimates match. |
| `q5_03_base_price_change` | Yes / yes | Correct (annotation partly overlaps the CI band; cosmetic). |
| `q5_04_partner_value_bubble` (hero) | Yes / yes | **Title wrong** ("deepest loss per swap"; issue 1). The data points are correct. |
| `q5_05_partner_unit_economics` | Yes / yes | Correct; it also shows CargoTuk/HaulKing below ZipDrop on cm1 and cm2, contradicting Fig 4's title. |
| `q5_06_zipdrop_amendment` (hero) | Yes / yes | Title (₹10.4) is correct; **annotations DiD −11.6 / −11.0 are inflated** (issue 3). |
| `q5_07_budget_pricing_partner` | Yes / yes | Numbers match. "Not collected ₹4.5 L" is gross (net ₹4.1 L). Panel (d) shows no steady-state cm1 bar (−₹21 L/yr), so the reader sees only the Kyron-inflated −₹104 L (issue 5). |

## 5. Missed angles worth adding

1. **Where the timing shift came from.** About 40% of the peak-share drop came out of the uncongested 12-13h band (−0.98 pp) and about 60% out of 19-22h (−1.51 pp). Riders moved to 14-16h (+1.10 pp) and 23-05h (+1.40 pp). The response also fades by May-Jun 2025 (−1.25 pp vs −2.2 pp in Oct-Mar), so the long-run congestion lever is even weaker.
2. **The pilot was run where there is no evening congestion.** Bengaluru and Pune have no evening excess. The cities that need congestion relief (DEL/HYD/JAI in Summer/Monsoon) were controls. A rollout there in summer would be the first real test of congestion pricing; pair it with q2/q3 cooling and capacity fixes.
3. **"Largest partner" is ambiguous for budget (d).**
   - By revenue and riders it is **FeastFly**. Its steady-state cm1 is **+₹12.22/swap (+₹57.5 L/yr)** and steady-state cm2 after WC is about **+₹0.03/swap**, so an exclusive with FeastFly on current terms is roughly break-even, unlike ZipDrop.
   - ZipDrop was #1 by revenue only before the amendment (Jan-Oct 2024: ₹1.22 Cr vs ₹1.19 Cr).
4. **ZipDrop's real cost is peak capacity, not unit margin.** At steady state its cm1 gap is small (−₹4.18/swap; cutting the discount below 24.6% makes it cm1-positive). The stronger argument is that it holds 19.7% of all peak-hour swaps (72.6% of its own swaps at peak) with a non-billable surcharge. Quantify the peak-capacity opportunity cost (q2/q3 links).
5. **Competitor entry (Jan-2025, 10 stations)** is a risk to the network-wide price rise. Check whether independents near competitor stations in the pilot cities responded more strongly before extrapolating ₹7.90/swap to all cities.
