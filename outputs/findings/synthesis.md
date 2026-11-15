# Synthesis: what drives VoltRelay's performance, and what the budget should fund

Script: `analysis/q9_synthesis.py` (runs in about 5 s; full printed output in `outputs/tables/q9_run_log.txt`). Tables: `outputs/tables/q9_*.csv`, `outputs/tables/key_numbers.csv` (40 quotable numbers). Figures: `q9_01_causal_chain.png`, `q9_02_budget_decision.png`.
Sources: the eight verified analyses (q1-q8). Where a verification note changed a claim, the revised claim is used. The script re-derives the headline numbers from the clean data with fresh code, and prints every number cited from the other analyses.
All margin levels are **under our usage-based battery-depreciation assumption** (no salvage, 70% SoH end of life). Trends and differences are the robust part.

## Executive summary

Swaps grew 2.43x and revenue 2.63x (H1-25 vs H1-24). Two of leadership's three alarms are mostly measurement artefacts.
- **Failures:** the same 90 stations failed 9.52% of Apr-Jun attempts in 2024 and 9.55% in 2025. Counts still nearly doubled (+94%) as volume grew where failures happen.
- **Retention** did not fall (86.3% vs 86.3%, same season).
- **Margin** erosion is real: cm1 per swap fell from +Rs0.27 to −Rs7.32 (Q1-24 → Q1-25).

Two equipment problems explain most of it:
1. **Heat × Gen1 chargers** at 36 hot-city stations produce 77% of the extra summer failures. Failures in a new rider's first two weeks explain 40–50% of the summer retention dip.
2. **One defective Kyron battery batch** fades 3.8x faster, costs Rs16.8 per swap (Rs48 lakh/month) and was issued below end of life 122,204 times.

Budget:
- **Reject** more stations and more batteries: load and pack count do not move failures.
- **Support** the pricing rollout with changes (+Rs9.5 lakh/month).
- **Reject** a ZipDrop exclusive on current terms.
- **Fund instead:** a Gen1 thermal fix (summer failure rate 7.9% → 5.1–6.0%), quality-gated pack replacement (Rs4.95 Cr now, Rs13.2 Cr by mid-2026) and contract re-terms (+Rs14.5 lakh/month).

## The storyline

1. **Growth is real, and it is broad.**
   - H1-25 vs H1-24: attempts x2.40, swaps x2.43, revenue x2.63, monthly active riders x2.18.
   - In Jun-25 the network ran 303K swaps and Rs19.7M of revenue a month, on 149 stations (q1; q9 recomputed). See `q1_01_kpi_divergence`.
2. **The failure alarm compares winter with summer.**
   - Jan-24 → Jun-25, failures grew x6.26 vs swaps x2.93 (4.00% → 8.16%).
   - Same season, failures grew x2.07 vs swaps x2.43 (H1: 7.04% → 6.06%).
   - Like for like, nothing improved and nothing deteriorated. The 90 Launch stations ran 9.52% → 9.55% in Apr-Jun, while the network average fell to 7.92% only because new stations, none of them Gen1, dilute it.
   - Failed attempts nearly doubled (39,820 → 77,445, +94.5%) because the stations that fail grew their volume. Their Apr-Jun attempts went from 172,800 to 267,903 (q1, q3; q9 recomputed). See `q1_04`, `q3_05`.
3. **Failures are a heat × equipment problem, concentrated in place and time.**
   - Outside Apr-Jun every city, generation, weekday and hour sits on a flat 3.91% baseline.
   - In Apr-Jun 2025 the 36 hot-city Gen1 stations (Delhi NCR 13, Hyderabad 13, Jaipur 10) handled 27.4% of attempts, 52.6% of failures and **77.1% of the failures above baseline** (80.1% over both summers).
   - Failures switch on above a threshold. The city-day failure rate is 3.9% below 34°C, 10.0% at 36–38°C and 14.1% at 38–42°C.
   - The mechanism is Gen1 thermal derating: charge time goes from 88 to about 168 min at a 50°C cabinet. Hot Gen1 station-days stock out 5x as often (12.0% vs 2.3%).
   - In the same hot cities and months, at similar load, Gen3 sites fail 5.1% and Gen2 8.0%, against Gen1's 15.2% (q2, q3, q7; q9 recomputed). See `q2_04`, `q3_01`, `q2_05`.
4. **Load, capacity and inventory are not the cause.**
   - Hotspot attempts rose 51% at an unchanged 16.6% failure rate. Within Gen1 stations the load effect is −0.03 pp (p = 0.94).
   - Completed queue wait is 4.06 min at every load.
   - Adding 1,385 packs (+33.5% of the 2W fleet) changed stockouts by −0.00 pp.
   - On hot evenings, 2W stockouts run 16.7–17.6% whatever the cabinet's charged stock (0 to 6+ packs) (q2, q3, q4; q9 re-test).
5. **Failures drive the churn that does exist.**
   - New-rider retention is flat year on year, same season. The visible "decline" is a summer-entry dip plus censoring of recent cohorts.
   - Riders with 2+ service failures in their first 14 days are not retained 22.6% of the time, against 13.1% (adjusted OR 1.67 [1.43–1.94], 15 of 15 subgroups).
   - Early failures explain 40–50% of the summer dip. Battery health and peak pricing show no retention effect (q6). See `q6_01`.
6. **Margin erosion is a battery-quality problem first.**
   - Kyron lots KY-2407/08/09 lose health 3.8x faster per day at the same purchase price. Wear is Rs104 vs Rs32 per 2W swap, which costs **Rs16.8 per swap at a 25% share of issues** (Rs47.7 lakh/month at H1-25 volume).
   - That more than erased the July-2024 price rise (+Rs5.1/swap) and the pilot (+Rs1.7/swap).
   - Battery Ops then kept issuing dead packs. 122,204 swaps handed out a pack below 70% SoH, including 99.9% of June Kyron 2W issues, and those packs delivered 26% fewer km. The cost model charged Rs12.9M of wear on them, so the H1-25 cm1 level is convention-dependent: −6.39 / −1.12 / +1.19 per swap (q1, q4, q7; q9 recomputed). See `q1_05`, `q4_01`.
7. **The secondary drivers are contractual and structural.**
   - ZipDrop's 28% discount, with a peak surcharge it does not pay, costs −Rs10.4 per ZipDrop 2W swap and bought no extra volume.
   - 3W swaps lose Rs14.9 each even with healthy packs.
   - Wave-1 fuel-pump and residential sites in already-served zones cost Rs14.7 per swap in fixed cost, against Rs11.3 at legacy sites.
   - The whole fleet is ageing on the calendar: km per 2W swap fell from 70.5 to 48.0, and the rider's price per km rose 42%.
   - Complaints are mostly about batteries (49.5% of tickets once mis-filed "other" and "app_issue" tickets are read). Official categories show 15% (q3, q5, q8). See `q5_04`, `q8_03`.
8. **What this means for the budget.** VoltRelay is constrained by equipment quality and contract terms, not by capacity. Money should go to fixes, and commercial levers need no capital. The figure `q9_01_causal_chain` shows the chain end to end.

## Contradictions between analyses, resolved

| # | Point of disagreement | Where | Resolution and evidence |
|---|---|---|---|
| C1 | Pre-charging or higher inventory as a heat lever | q3 and q6 recommend it; q2 rejects it | **Rejected.** At the 36 sites in Apr-Jun (q9 re-test), 2W stockouts on hot days at 19–22h are 17.6% / 16.9% / 16.9% / 16.7% at 0 / 1–2 / 3–5 / 6+ charged packs; in other hours they are 8.5–8.9% at every stock level. The pooled zero-stock hours are not worse (9.2% vs 10.0%). +33.5% packs moved stockouts −0.00 pp (q4). The levers are charger thermal performance and redirecting demand. |
| C2 | Heat trigger: official heat-alert days, or a temperature forecast? | q3 vs q2 | **Forecast city max ≥36°C.** Failure rate is 3.9% below 34°C, 4.8% at 34–36°C, 10.0% at 36–38°C and 14.1% at 38–42°C. The official alert fires on 0.0% of days below 42°C and 99% at or above 42°C, after failures have plateaued (q9). |
| C3 | When does the non-Kyron fleet reach end of life? | q1 says ~Feb-2026; q4 says ~832 days (~Apr-2026) | **Both, as a range.** Per-pack calendar fade puts the median 2W pack at 70% SoH in Apr-2026 (p10 Feb-2026, p90 May-2026), with 41% before Apr-2026. The recent issued-SoH trend (−1.37 pts/month) gives Jan-2026. Plan renewal for Jan–May 2026 (q9). |
| C4 | Kyron margin effect: +Rs16.8/swap (q1) vs "no cm1 gain from replacing" (q4) | q1 vs q4 | **Different questions.** q1 compares buying Kyron-grade with buying Cellora-grade packs, a forward procurement choice measured in the clean Q1-25 window. q4 compares replacing packs that are already fully depreciated with running them on (book view). Packs past EOL must go anyway (range −26%, safety). The grade you buy is worth +Rs16.8/swap. |
| C5 | Margin level in H1-25 / Q2-25 | q1 −6.39 (per-cycle); q4 +1.64 (capped); q7 May/Jun +12.4/+18.0 | **Convention-dependent.** −6.39 / −1.12 / +1.19 per swap for H1-25 (q9). Trends are anchored in post-EOL-free windows (Q1-24 → Q1-25: −7.59). The Cellora-wear convention is the economic comparator. |
| C6 | "Failures rose faster" vs "the failure rate fell" | leadership and naive reading vs q1/q2/q3 | **Each is half true.** Like for like the rate is flat (9.52% → 9.55%). The network rate fell only by dilution (9.52% → 7.92%), and the count rose 94.5%. Report the fixed-panel rate together with counts. |
| C7 | Price per km as a retention driver | q1 suggests it; q6 tests it | **Not supported.** Km per swap, SoH and Kyron share are not adverse in the rider models (SoH OR 1.07 n.s.; Kyron 0.90). Same-season retention stayed flat while km per swap fell. It is a value-for-money and complaints issue (q8). |
| C8 | Did the 1-Jul-2024 failure drop come from the price rise? | possible misreading | **No, it was the end of the heat.** Only the hot cities dropped (−8 to −15 pts) as they cooled 11–17°C. Cool cities were flat under the same price rise (q7). |
| C9 | Share of excess failures at the 36 sites: 80.4% (q2) vs 77.1% (q3) | q2 vs q3 | **Same finding, different windows.** 77.1% in Apr-Jun 2025; 80.1% over both summers (q9, flat 3.91% baseline). |
| C10 | Size of the Gen1 fix: 20,177/28,493 (q1), 27,158 (q2), 19,718/27,751 (q3) | q1/q2/q3 | **Consistent within ±5%.** q9's same-city, same-month estimate is 19,212 (Gen2-like) to 27,266 (Gen3-like) failed attempts avoided per Apr-Jun at 2025 volume. |
| C11 | ZipDrop break-even discount: 24.6% | q5 | **That is the 2W threshold.** At ZipDrop's 2W/3W mix it is 21.9% (q9). Either way the 28% discount is above break-even and 15% is safely below it. |
| C12 | The brief's "low-utilisation new stations dilute the average" | brief vs q1/q3 | **The mechanism is refuted.** New stations are not lighter-loaded (4.8 vs 5.0 attempts per slot-day in Q2-25). They fail less because none of them is Gen1. |

Refinements to the hypothesis scorecard (`q0_03`):
- **H2 (concentration)** holds, but at hot-city Gen1 sites in Apr-Jun, not at "busy" sites.
- **H3 (expansion)** is need-blind, as hypothesised, but it is a cost driver, not a failure driver.
- **H12 (pricing did not pay)** is rejected: pricing paid.

## Primary vs secondary drivers

| Outcome | Class | Driver | Evidence (period) |
|---|---|---|---|
| Service failures | **PRIMARY** | Summer heat (city max ≥36°C) × Gen1 chargers at 36 hot-city Launch stations | <ul><li>77.1% of Apr-Jun-25 excess failures on 27.4% of attempts.</li><li>Hot-city Gen1 vs Gen3 RR 3.05 (Mumbai 1.00).</li><li>Heat × generation lifts R² from 0.55 to 0.95.</li><li>Gen1 charge time 88 → 168 min at 50°C.</li><li>+0.81 pp stockouts per hot hour within station.</li></ul> |
| | Secondary | Gen2 heat sensitivity | Hot-city Gen2 8.0% vs Gen3 5.1% (Apr-Jun-25) |
| | Secondary | 3W riders | Stockout RR 2.08 within station-month. 3W slot metadata is unreliable. |
| | Secondary | Hot-day evenings (19–22h) | 16.7–17.6% vs 8.5–8.9% in other hours on hot days |
| | Not drivers | Load and utilisation; queue capacity; battery count and hourly stock; connectivity and firmware; location and host type; competitor; pricing pilot | <ul><li>Load: −0.03 pp, p = 0.94.</li><li>Queue wait: 4.06 min at every load.</li><li>Pack count: −0.00 pp.</li><li>System errors: 0.30% in every tier.</li><li>Pricing pilot: +0.05 pp, n.s.</li></ul> |
| New-rider churn | **PRIMARY** | Stockouts and abandoned queues in the first 14 days | <ul><li>2+ failures: 22.6% vs 13.1% not retained; OR 1.67 [1.43–1.94].</li><li>Within-rider exit OR 1.53.</li><li>15/15 subgroups.</li><li>Explains 40–50% of the summer dip.</li></ul> |
| | Secondary | Competitor within 1.5 km | Exit OR 1.42, Cox HR 1.20. Localised (640 riders); no 60-day effect. |
| | Secondary (weak) | 3W riders | OR 1.18 [1.01–1.37]; n.s. with the activity control |
| | Not drivers | Battery SoH and Kyron; peak pricing; queue wait; tickets; ZipDrop affiliation; station wave and generation | SoH OR 1.07; Kyron 0.90; peak OR 1.06; pilot DiD p = 0.15 |
| Margin erosion | **PRIMARY** | Kyron lots KY-2407/08/09: 3.8x faster fade per day at the same price | <ul><li>−Rs16.8/swap (Q1-25), Rs47.7 lakh/month.</li><li>Bridge: Kyron −16.0, more than the price rise (+5.1).</li><li>Value destroyed Rs2.94 Cr (capped) to Rs4.31 Cr.</li></ul> |
| | Secondary | Partner contract terms | <ul><li>ZipDrop amendment −Rs10.4/ZipDrop swap (−Rs1.5 to −1.7 network).</li><li>Quick commerce cm1 −Rs17.5/swap.</li><li>Non-billable surcharge Rs6.3 lakh/month uncollected.</li></ul> |
| | Secondary | 3W priced below cost | Steady-state cm1 −Rs14.9/swap; break-even list Rs126 |
| | Secondary | Need-blind Wave-1 siting | Fixed cost Rs14.7 vs Rs11.3/swap; Rs1.68 lakh/month excess |
| | Secondary | Sub-EOL issuing | Rs12.9M wear charged; 26% fewer km per affected swap |
| | Offsets | <ul><li>Base price</li><li>Pilot</li><li>Utilisation</li><li>Lower energy per swap (a fleet-health warning, not a saving)</li></ul> | <ul><li>Base price +Rs5.1/swap.</li><li>Pilot +Rs1.7/swap.</li><li>Fixed cost per swap Rs19.5 → Rs11.2 (Q1).</li></ul> |
| Complaints | **PRIMARY** | Battery health | 49.5% of tickets (floor 36.7%; official 15.1%); Kyron dose test 1.74x |
| | Secondary | Stock-outs and queues | 39.5% of tickets; about 1 ticket per 10 extra failed attempts |

## Where performance varies most

- **Stations and equipment:**
  - In Apr-Jun 2025, hot-city Gen1 ran 15.2% failures, Gen2 8.0% and Gen3 5.1%. Other cities ran 4.6%.
  - There are no rogue stations: all 36 Gen1 sites fail at their city-month rate (O/E 0.95–1.05; q2, q7).
- **Geography:**
  - Apr-Jun rates were Jaipur 13.6%, Delhi NCR 12.2%, Hyderabad 10.4% and Mumbai 4.0%. From Jul to Mar every city sits at 3.85–4.00%.
  - Pune's 4 Gen1 sites are the next tier (Gen1 vs Gen3 RR 1.72).
- **Time:** Apr-Jun (city max ≥36°C) and 19–22h on hot days.
- **Battery cohort:** lots KY-2407/08/09 fade 3.8x per day. Later Kyron 3W lots fade like Cellora (1.00x), so the defect is in those lots, not across the supplier.
- **Partner segment (steady-state cm1/swap, Nov-24..Jun-25):** independents +18.2, FeastFly +12.1, ZipDrop −4.1, CargoTuk −14.7, HaulKing −13.7. The contract decides the margin: cost to serve is within ±Rs1 across partners.
- **Vehicle class:** 3W riders fail about 2x as often, and 3W cm1 is −Rs14.9/swap even with healthy packs.

## Budget decision

### The four proposals
Common units:
- Apr-Jun network failure rate at 2025 volume (7.92% today).
- New-rider D31-60 non-retention in pp (negative = fewer riders lost).
- cm1 in Rs lakh per month at the H1-2025 run-rate (284K swaps/month; summer-only gains averaged over 12 months).
- One-off cost.

See `q9_02_budget_decision`.

| Proposal | Verdict | Evidence | Impact (common units) | Modify to / instead |
|---|---|---|---|---|
| **(a) More stations** | **Reject** as a fix for failures; fund demand-led sites only | <ul><li>Hotspot rates do not respond to load: +51% attempts at a flat 16.6%; −0.03 pp per log-unit, p = 0.94.</li><li>New slots lowered legacy failure rates by only −0.16 pp at median intensity.</li><li>Both waves were need-blind.</li><li>Utilisation follows unmet zone demand: 3 unserved-zone sites run 11.6 swaps/slot-day vs 3.05 for served-zone fuel pumps.</li></ul> | <ul><li>Another 30-site Wave-1-like wave: failures ~0; retention ~0.</li><li>+Rs7.4 lakh/month fixed cost.</li><li>cm2 −Rs2.6 lakh/month if every swap is new, −Rs7.4 lakh/month if swaps only move between sites.</li><li>At steady-state cm1 (Rs9.5/swap) no wave covers its fixed cost: break-even needs 2,370–2,585 swaps per site-month vs 1,668–2,103 actual.</li></ul> | <ul><li>A few Gen3 or cooled sites in unserved or under-slotted zones only (HYD-West, JAI-Central, DEL-Central, PUN-Kharadi).</li><li>Review the 16 new sites above Rs20 fixed cost per swap.</li></ul> |
| **(b) More batteries** | **Reject** expansion; **replace** instead | <ul><li>+1,385 packs (+33.5%) changed stockouts by −0.00 pp (charged stock +0.004 per station-hour).</li><li>Hot-evening stockouts are the same at every stock level.</li><li>Battery SoH does not affect churn.</li></ul> | <ul><li>+10% 2W packs (552): Rs1.77 Cr capex.</li><li>−Rs6.5 lakh/month calendar depreciation.</li><li>Failures and retention ~0.</li></ul> | <ul><li>Replace the 1,461 bad-lot packs now (Rs4.95 Cr).</li><li>Renew 4,133 2W packs in Jan–May 2026 (Rs13.2 Cr).</li><li>Gate suppliers on early fade (+Rs16.8/swap vs Kyron-grade = Rs47.7 lakh/month).</li><li>Claim Rs3.37–3.56 Cr.</li></ul> |
| **(c) Network-wide pricing rollout** | **Support with changes** | <ul><li>Pilot: +Rs7.90 per independent swap (+11.7%) and +Rs5.43 per billable-partner swap.</li><li>Peak share only −2.5 pp; stockouts +0.05 pp (CI −0.09 to +0.18).</li><li>Demand is inelastic.</li><li>No billing-complaint effect (IRR 0.94).</li><li>Retention n.s. (OR 1.28, p = 0.15; partner placebo 1.20).</li></ul> | <ul><li>+Rs9.5 lakh/month cm1 (Rs3.3 per network swap).</li><li>+Rs6.3 lakh/month more if every partner pays the surcharge.</li><li>Failures ~0; retention n.s. (watch with a guardrail).</li></ul> | <ul><li>Keep the current bands.</li><li>Billable surcharge in every partner contract.</li><li>Staged rollout with holdout stations.</li><li>A summer evening surcharge test in one hot city.</li><li>A 3W price test.</li></ul> |
| **(d) Long-term exclusive with the largest fleet partner** | **Reject on current terms** | <ul><li>ZipDrop (largest by swaps, 15.4%) has a 28% discount and does not pay the surcharge.</li><li>Steady-state cm1 is −Rs4.1/swap even with healthy packs.</li><li>The amendment bought no volume (+56.5% vs +53.2%).</li><li>73% of its swaps are at peak, 20% of all network peak swaps.</li><li>The largest by revenue and riders (FeastFly) earns +Rs12.1/swap.</li></ul> | <ul><li>Locks in −Rs1.8 lakh/month of cm1 (cm2 after working capital −Rs80.7 lakh/yr).</li><li>Retention ~0 (OR 1.06).</li></ul> | <ul><li>Renegotiate ZipDrop to ≤15% with a billable surcharge (+Rs3.8 lakh + Rs4.4 lakh/month).</li><li>Use FeastFly-type terms as the template.</li><li>Short tenor with cost indexation.</li></ul> |

### Recommended alternative allocation

| Line | Spend | What it buys (common units) |
|---|---|---|
| **1. Battery replacement now** (1,385 Kyron 2W + 76 3W bad-lot packs) and an end-of-life lockout | Rs4.95 Cr (largely offset by a Rs3.37–3.56 Cr lost-life claim) | <ul><li>Sub-EOL issues go from about 66K/month (Jun-25) to 0.</li><li>Riders get normal range back: a Kyron pack delivered a median 39.5 km vs 53.4 km for others, and a new pack about 70 km.</li><li>Safety exposure removed.</li></ul> |
| **2. 2W fleet renewal, Jan–May 2026**, quality-gated | Rs13.2 Cr for 4,133 packs | <ul><li>This is non-discretionary: the median pack reaches 70% SoH in Apr-2026.</li><li>Buying Cellora-grade rather than Kyron-grade decides +Rs16.8/swap (Rs47.7 lakh/month).</li><li>Renew in tranches, most-faded first, to avoid a cliff.</li></ul> |
| **3. Gen1 thermal fix at 36 hot-city sites** (pilot, then scale; Pune's 4 next) | <ul><li>Not in the data; the pilot sets the ceiling.</li><li>Direct revenue is worth about Rs33–48K per site per summer at 2025 volume.</li></ul> | <ul><li>Per summer: −19.2K to −27.3K failed attempts (25–35% of all Apr-Jun failures).</li><li>Network Apr-Jun failure rate 7.92% → 5.95–5.13%; at the sites, −7.4 to −10.4 pp.</li><li>About 1.9–2.8K fewer tickets.</li><li>Summer-entry new-rider non-retention −0.6 to −1.0 pp (14–21% of the summer dip).</li><li>Up to Rs1.2–1.7M revenue per summer.</li></ul> |
| **4. Commercial re-terms** (no capex) | ~0 | <ul><li>+Rs23.9 lakh/month cm1 in total:</li><li>rollout +9.5</li><li>billable surcharge +6.3</li><li>ZipDrop at 15% +3.8</li><li>3W at ~Rs126 list +4.3, if demand holds</li></ul> |
| **5. CX and ops** (low cost) | Credits and app work | <ul><li>Heat playbook: 36°C trigger and redirection at 19–22h.</li><li>New-rider failure-recovery A/B: up to −1.45 pp overall non-retention if early failures were capped at 1 (upper bound).</li></ul> |
| **Not funded** | – | <ul><li>A new capacity wave: saves +Rs7.4 lakh/month of fixed cost.</li><li>+10% packs: saves Rs1.77 Cr capex and Rs6.5 lakh/month depreciation.</li><li>The ZipDrop exclusive.</li></ul> |

**Path to a positive cm2 per swap** (Q1-25 unit costs, H1-25 volume):

| Step | Rs per swap |
|---|---|
| Q1-25 actual cm2 | −18.5 |
| Cellora-grade packs | +16.8 |
| Pricing rollout | +3.3 |
| Partner terms | +3.6 |
| 3W price | +1.5 |
| **cm2 per swap after the levers** | **+6.8** |

At steady-state cm1 alone (+Rs9.5/swap), no station type covers its fixed cost (Rs22.5–24.6K per site-month). Stations pay only once unit economics are fixed. That is another reason not to expand first.

**Cash timing.** About Rs18 Cr of packs is due within roughly 12 months (Rs4.95 Cr now, Rs13.2 Cr in Jan–May 2026). The 729 non-Kyron 3W packs (Rs4.95 Cr) follow around mid-2027. The cost model already accrues this as wear (about Rs32 per 2W swap), but the cash leaves in a lump. It belongs at the top of the budget, ahead of any expansion.

**Assumptions:**
- Rs/month figures use the H1-2025 run-rate. Per-swap effects come from the cleanest window available: Q1-25 for Kyron (no post-EOL issues), Nov-24..Jun-25 for partners, and q5's difference-in-differences for pricing.
- Steady state replaces Kyron-issued energy and wear with non-Kyron class averages.
- Failed attempts are treated as lost swaps (an upper bound).
- The Gen1 counterfactual applies same-city, same-month Gen2/Gen3 rates. That is an association, confounded with site age.
- The retention conversion uses q6's adjusted OR 1.67 (CI 1.43–1.94), which gives 7.1 pp [4.7–9.7] per rider moved from 2+ to 0–1 early failures.
- Fleet end of life is a linear projection of each pack's own calendar fade.
- Capex for stations, retrofits and chargers is not in the data.

## Prioritised recommendations

| # | Action | Owner | Quantified impact | Evidence | KPI and target | Timing |
|---|---|---|---|---|---|---|
| 1 | **End-of-life lockout and replacement of the bad battery lots.** <ul><li>Block issuing any pack below 70% SoH at the cabinet, with 1:1 replacement first, city by city (Delhi NCR, Hyderabad, Jaipur first).</li><li>Replace the 1,461 KY-2407/08/09 packs with Cellora/Amptek-grade packs.</li><li>File a lost-life claim.</li></ul> | COO / Head of Battery Ops; Procurement; Platform Eng | <ul><li>Sub-EOL swaps go from about 66K/month to 0.</li><li>About +14 km per affected swap (39.5 → 53.4 median).</li><li>Claim Rs3.37–3.56 Cr against Rs4.95 Cr capex.</li></ul> | q4, q7, q1 | <ul><li>Issues below 70% SoH = 0.</li><li>Median km per 2W swap ≥ 53.</li></ul> | Immediately (before Jan-2026) |
| 2 | **Quality-gated 2W fleet renewal.** <ul><li>Renew 4,133 non-Kyron 2W packs in tranches, most-faded first.</li><li>Lot acceptance rule: flag any lot whose median per-day fade over its first 6–8 weeks exceeds 1.5x the incumbent fleet's.</li><li>Stage rollouts by city.</li></ul> | CFO / Procurement / Quality | <ul><li>Rs13.2 Cr, non-discretionary.</li><li>The gate protects +Rs16.8/swap (Rs47.7 lakh/month) against a Kyron-like batch.</li><li>The rule flagged KY lots at 2.5–4.5x by week 8.</li></ul> | q4, q1, q9 (EOL Feb–May 2026) | <ul><li>2W wear ≤ Rs33/swap.</li><li>Lot fade ratio ≤ 1.5x at week 8.</li><li>No pack issued below 70%.</li></ul> | Tranches Jan–May 2026 |
| 3 | **Gen1 thermal fix at the 36 hot-city stations.** <ul><li>Start with the largest Delhi NCR and Jaipur sites, then Hyderabad; Pune's 4 next.</li><li>Cheapest option first: shading, ventilation, cabinet cooling to keep cabinets below about 40°C. Swap to Gen3 only where that fails.</li><li>Run as a matched pilot before scaling.</li></ul> | Head of Network Ops / Station Engineering | <ul><li>−19.2K to −27.3K failed attempts per Apr-Jun.</li><li>Network Apr-Jun failure rate 7.92% → 5.1–6.0%.</li><li>About 1.9–2.8K fewer tickets.</li><li>Summer-entry non-retention −0.6 to −1.0 pp.</li><li>Rs1.2–1.7M revenue per summer.</li></ul> | q2, q3, q7, q9 | <ul><li>Apr-Jun 2026 failure rate at the 36 sites ≤ 8% (from 15.2%).</li><li>Share of station-hours with cabinet ≥45°C cut by half.</li><li>Network Apr-Jun rate ≤ 6%.</li></ul> | <ul><li>Pilot installed Feb-2026.</li><li>Go/no-go on Mar–Apr cabinet telemetry.</li><li>All 36 before May-2026.</li></ul> |
| 4 | **Roll out peak/off-peak pricing network-wide**, keeping the current bands (PEAK 12–13h and 19–22h). <ul><li>Make the surcharge billable in every partner contract.</li><li>Stage the rollout with holdout stations.</li><li>Test a summer evening surcharge in one hot city.</li></ul> | CCO / Pricing lead | <ul><li>+Rs9.5 lakh/month cm1.</li><li>+Rs6.3 lakh/month from billable partners.</li><li>Failures ~0.</li></ul> | q5, q8, q6 | <ul><li>Revenue per independent swap +Rs7–8 in rollout cities.</li><li>Peak stockouts unchanged.</li><li>Guardrail: independent new-rider D31-60 retention within 1 pp of holdouts.</li></ul> | Staged from Jan-2026; partner billing at each renewal |
| 5 | **No long-term exclusive. Re-term ZipDrop and price 3W to cost.** <ul><li>ZipDrop discount ≤15% with a billable surcharge and volume tiers tied to off-peak volume, using FeastFly-type terms as the template.</li><li>Test a 3W list price of about Rs126.</li></ul> | CEO / CFO / Key accounts | <ul><li>Avoids locking in −Rs1.8 lakh/month.</li><li>+Rs3.8 lakh/month (ZipDrop at 15%).</li><li>+Rs4.3 lakh/month (3W, if demand holds).</li></ul> | q5, q6 | <ul><li>ZipDrop steady-state cm1 ≥ 0.</li><li>3W cm1 ≥ 0.</li><li>100% of partner swaps surcharge-billable.</li></ul> | Next renewal (Q1-2026) |
| 6 | **Heat playbook.** <ul><li>Trigger on a forecast city max ≥36°C, not the official alert.</li><li>At 19–22h, redirect riders in the app from hot-city Gen1 sites to nearby Gen3 sites and show live charged-pack availability.</li><li>Do not fund pre-charging or higher inventory targets.</li></ul> | City Ops / Product | <ul><li>Not sizable from observational data.</li><li>The addressable pool is hot-day 19–22h stockouts at the 36 sites (16.7–17.6% vs 8.5–8.9% in other hours).</li></ul> | q2, q9 (C1, C2) | Hot-day 19–22h stockouts at Gen1 sites −25% vs matched controls | Live by Mar-2026 |
| 7 | **New-rider failure-recovery A/B.** On a stockout or abandoned queue in the first 30 days: apology and credit, a reserved charged pack, and a route to a stocked site. Arms: 1st failure vs 2nd failure vs control. | Head of CX | Up to −1.45 pp overall non-retention (−2.72 pp summer-entry) if early failures were capped at 1; this is an upper bound. | q6 | D31-60 retention of riders with 2+ early failures: +3 pp vs control (gap today 22.6% vs 13.1%) | Launch Jan-2026; read before summer |
| 8 | **Stop capacity expansion and fix the measurement.** <ul><li>No new station wave. Open sites only in unserved zones, Gen3 or cooled in hot cities.</li><li>Review the 16 sites above Rs20 fixed cost per swap.</li><li>Fix the cost model so wear stops at EOL.</li><li>Report same-season fixed-panel failure rates with counts, D31-60 cohort retention, the effective tariff from the price residual, and a battery-complaint category.</li></ul> | CFO / Network Planning / Analytics | <ul><li>Avoids +Rs7.4 lakh/month of fixed cost.</li><li>Rs1.68 lakh/month of Wave-1 excess is addressable.</li><li>Ends the misreads: naive failure claim, Rs12.9M post-EOL wear, 54% of pilot pricing hidden in tariff_code.</li></ul> | q3, q1, q7, q8 | <ul><li>New-site swaps/slot-day ≥ 4.5 (legacy).</li><li>Fixed cost ≤ Rs11.3/swap.</li><li>KPI pack in use.</li></ul> | Before budget sign-off |

## Gaps (completeness critic)

**Core questions not fully answered:**
- **Thermal mechanism at hour level is unidentified.** On hot days, stockouts happen even with full cabinets (16.7% at 6+ charged packs). The heat effect is a day-level association, which looks like a property of the simulation. The Gen1 fix therefore rests on the strong same-city Gen3 contrast, not on a measured hour-level mechanism.
- **The Gen1 → Gen3 effect is confounded with site age.** Every Gen3 site is new, and no station was upgraded within the window. The 19–27K estimate is an upper-end association; only the pilot can confirm it.
- **Retention:**
  - About half of the summer dip is unexplained, and churn models reach AUC 0.60–0.68.
  - Churn of existing (pre-2024) riders and long-run rider value were not analysed.
  - The effect of a failure-recovery offer is unknown until it is A/B tested.
- **Battery:**
  - The +9.8% jump in swaps per rider-day in Kyron-heavy cities has no identified mechanism.
  - Pack IDs do not follow physical packs, so pack turnaround and the optimal fleet size cannot be measured. "How many batteries" is answered only indirectly.
  - For 2W, lot, supplier and climate cannot be separated.
- **Pricing:**
  - With only 6 cities, the permutation p cannot go below 0.067.
  - The rollout effect is transferred from mild-summer cities to hot ones.
  - Volume and retention responses are not identified (CIs are wide).
  - 3W price elasticity is unknown.
- **Partners:** how ZipDrop would react to re-terms is unknown. At steady state its cm1 is negative, so losing it costs little cm1, but its 15% of swaps would raise fixed cost per swap.

**Claims that are weakly evidenced:**
- "Each failure is a lost swap": Rs1.2–1.7M per summer is an upper bound.
- Fleet end-of-life timing is a linear projection; seasonal fade could move it by ±2 months.
- Competitor effect: the MDE is about 10%, on only 8 treated stations.
- The impact of the heat playbook and redirection is not sized.

**Missing data for the budget:** capex for stations, retrofits and charger swaps; cost per ticket; rider lifetime value. Because of this, the Gen1 fix's return can only be bounded by revenue (Rs33–48K per site per summer).

**Missing visuals:**
- A "path to positive cm2" waterfall (it is a text line in `q9_02` today).
- A fleet end-of-life timeline with the renewal tranches.
- A map or small multiples of the 36 sites with their retrofit priority.
- A simple heat-trigger chart (failure rate vs daily max temperature, with the official-alert line) for the video.

**Residual data issues:**
- The firmware edge bug (about 147 events on 9-Mar) is not fixed in clean.py.
- slots_3w is unreliable, and so is station_month.swaps_per_slot_day, which depends on it.
- Battery master dates are unreliable.
- tariff_code hides partner pilot pricing; the analysts used the price residual instead.

## Figures

| File | Use | Caption |
|---|---|---|
| `q9_01_causal_chain.png` | **HERO (synthesis)** | Two equipment problems, not growth, drive most of the failures, churn and margin loss. Root causes → measured mechanisms → outcomes, with the evidence on each link; tested non-drivers in grey. |
| `q9_02_budget_decision.png` | **HERO (budget)** | Fix the 36 Gen1 sites and buy better batteries; do not buy capacity or lock in ZipDrop's terms. Each option is scored in the same units, with the path to a positive cm2. |
| `q1_01_kpi_divergence.png` | Deck: performance | Swaps and revenue nearly tripled while margin per swap turned negative; failure rates follow the season. |
| `q2_04_concentration_lorenz.png` | Deck: failures | In Apr-Jun the 36 hot-city Gen1 stations handle a third of attempts but produce about 80% of excess failures. |
| `q3_01_thermal_derating.png` | Deck: mechanism | Gen1 charge time nearly doubles by a 50°C cabinet; Gen3 barely moves. |
| `q6_01_cohort_retention_trend.png` | Deck: retention | Retention dips only for summer-entry cohorts; early failures explain 28–61% of each dip. |
| `q1_05_cm1_bridge_waterfall.png` | Deck: margin | Kyron battery wear erased the July-2024 price gain. |
| `q4_01_soh_vs_cycles.png` | Deck: battery | Kyron lots lose health 3.8x faster per day; still issued at 61% SoH. |
| `q5_04_partner_value_bubble.png` | Deck: partners | The largest partner by swaps is the least valuable. |
| `q8_03_hidden_battery_complaints_kyron.png` | Deck: CX | Read the text and battery complaints are 2.4–3.3x the official count. |

## Method notes

- **Re-derivation.** q9 rebuilds from `data/clean`: the monthly panel by vehicle class (revenue, energy, wear, fixed cost, post-EOL flag `soh_out_pct < 70`, Kyron issues) and Apr-Jun station segments. The headline numbers of q1, q2 and q3 reproduce exactly (2.43x, 9.52% → 9.55%, 39,820 → 77,445, 52.6% / 77.1%, cm1 +0.27 → −7.32, Kyron +16.82/swap). Small definitional differences:
  - Non-retention 22.7% vs 13.3% here, vs q6's 22.6% vs 13.1%: day 0 = the first clean attempt.
  - ZipDrop and FeastFly steady-state cm1 −4.12 / +12.12 here, vs q5's −4.18 / +12.22: window averages.
- **Wear conventions.** Three are used:
  - base: the shared per-cycle model;
  - cap: no wear charged on issues below 70% SoH;
  - Cellora-wear: post-EOL issues charged at the Cellora class wear of Rs31.85 (2W) and Rs76.19 (3W).
- **Inventory re-test (C1).**
  - Unit: station-hours at the 36 sites in Apr-Jun with `telemetry_status == 'ok'`, a non-null charged-2W minimum and at least one 2W attempt. Blanks are never zero-filled.
  - A hot day is one whose peak cabinet temperature reaches ≥45°C.
  - Stockout rates are compared within day type × window, by charged-stock bucket.
- **Retention conversion (section 4 of the script).** For each rider, binomial thinning of early failures at hot-city Gen1 sites in Apr-Jun, using the avoided share from the city-month counterfactual (Gen2-like 47.2%, Gen3-like 67.0%). The change in the share of riders with 2+ early failures is multiplied by the pp effect implied by q6's OR. Summer-2025 entrants are restricted to first attempts by 16-Jun so that each has a full 14-day window.
- **Fleet end of life.** Per-pack fade per day = (initial SoH − SoH on 30-Jun-25) / days since first seen; days to 70% = (current SoH − 70) / fade. The cross-check is a linear trend of issued non-Kyron 2W SoH over Jan–Jun 2025.
- **Cited numbers.** Numbers not recomputed here are printed in the script's CITED block with their source analysis (q1–q8, all revised after verification).
