title: VoltRelay Energy - Analysis Report (Team Praise)

:::cover
event: Gradient Learnings · Data Analytics Hackathon · 15 November 2026
title: VoltRelay Energy: what is really driving service failures, rider churn and margin erosion
subtitle: An evidence-based analysis of 3.8 million swap attempts across six Indian cities (January 2024 – June 2025), and where the next operating budget should go
band: 2.43× | completed swaps, H1-25 vs H1-24 ;; −₹16 | per swap: battery wear from one bad supplier batch ;; 80% | of excess failures from 36 hot-city Gen1 stations ;; 1.67× | odds of a new rider leaving after 2+ early failures
author: Team Praise — Shubham Suman · Durgesh Kumar
date: Analysis report · submitted 15 November 2026
:::

# Executive summary

:::callout
**The answer in one paragraph.** VoltRelay's growth is real: completed swaps **2.43×** and revenue **2.63×** year on year.
Two of leadership's three alarms are **misread**. Like-for-like, service failures did not get worse, and new-rider
retention did not fall: both move with the summer heat. The third alarm, **margin erosion, is real**, and it has one main cause:
**a defective battery batch** that costs **≈₹16.8 per swap on average**. Most of what is going wrong traces back to **two equipment
problems, not to growth**: heat-sensitive Gen1 chargers at 36 hot-city stations, and the battery batch. The budget should
**fix those**, not buy more stations or batteries.
:::

:::kpis
2.43× | Completed swaps | H1-25 vs H1-24 (revenue 2.63×)
good:Flat | New-rider retention | 86.3% vs 86.3%, same season (Jan–Mar 2024 vs 2025)
warn:80% | of excess failures | from 36 Gen1 stations in 3 hot cities, Apr–Jun
bad:−₹7.32 | cm1 per swap, Q1-25 | from +₹0.27 in Q1-24 (same season)
:::

**What is really driving the outcomes (primary drivers).**
1. **Heat × old chargers → service failures.** Outside April–June every station fails at the same ≈3.9%. In summer, Gen1
   cabinets in Delhi NCR, Jaipur and Hyderabad overheat and charge 2× slower (88 → ~168 min at 50 °C). These 36 stations
   produce **77–80% of excess failures**. At the same stations the failure rate did not change year on year (9.52% → 9.55%);
   the network average "improved" only because new stations have better chargers.
2. **Early failures → new riders leave.** A new rider who hits 2+ failed swap attempts in the first 14 days fails to
   return **22.6% vs 13.1%** of the time (adjusted OR 1.67, consistent in 15 of 15 subgroups; stockouts and abandoned queues
   alone: 23.5%, OR 1.76). This explains 40–50% of the
   summer retention dip; same-season retention is flat.
3. **A defective battery batch → margin erosion.** Kyron lots KY-2407/08/09 (1,461 packs) lose health **3.8× faster** at the
   same price. Their wear erased the July-2024 price gain (cm1 +₹6.75 → −₹7.53 per swap). The batch was never pulled:
   **122,204 swaps** handed riders packs below the 70% end-of-life threshold, which deliver 26% less range.

**Secondary contributors:** ZipDrop's 28% discount (−₹10.4 per ZipDrop swap, no extra volume), 3W prices set below cost
(break-even ≈ ₹126 vs ₹110 list), need-blind expansion (Wave-1 fixed cost ₹14.7 vs ₹11.3 per swap) and whole-fleet ageing (range
per swap 70.5 → 48 km).
**Tested and rejected as drivers:** station load and queue capacity, number of packs in cabinets, battery health or price as
causes of churn, competitor entry, connectivity and firmware.

**The budget decision.**

| Proposal | Verdict | Why |
|---|---|---|
| (a) More stations | **Reject** for service; demand-led sites only | Hotspot failure rates do not respond to load (+51% attempts at a flat 16.6%) |
| (b) More batteries | **Reject**; replace instead | +33.5% more packs changed stockouts by 0.00 pp; hot-day stockouts happen with full cabinets |
| (c) Network-wide peak pricing | **Support**, as a revenue measure | +₹7.90 per independent swap ≈ **₹9.5 lakh a month**; it does not ease congestion |
| (d) Exclusive with ZipDrop | **Reject** on current terms | cm1 −₹4.1 per swap even with healthy packs; would lock in 20% of peak swaps without the surcharge |

**What to fund instead** (with expected impact):
1. **Gen1 thermal fix at the 36 hot-city stations**, piloted, then scaled before April 2026: **19,000–27,000 fewer failed
   attempts per summer**, network summer failure rate 7.9% → 5.1–6.0%.
2. **End-of-life lockout plus quality-gated pack replacement**: ≈**₹48 lakh a month** of wear avoided versus Kyron-grade packs, a
   **₹3.4–3.6 crore** warranty claim, and a planned renewal of the original fleet (≈₹13 crore, January–May 2026).
3. **Contract and price re-terms** at no capex: ZipDrop back to ~15%, a billable peak surcharge for all partners, and 3W re-priced
   (≈ **+₹14.5 lakh a month**).
4. **New-rider failure recovery** (A/B test): up to **−1.45 pp** non-retention.

:::callout
**Path to a positive margin.** At Q1-2025 unit costs, cm2 per swap goes from **−₹18.5 today to +₹6.8**: +16.8 from
Cellora-grade packs, +3.3 from the pricing rollout, +3.6 from partner terms and +1.5 from 3W pricing.
:::

# 1. Problem understanding

## 1.1 The business
VoltRelay Energy runs unmanned battery-swap cabinets for electric two- and three-wheelers in six Indian cities
(Bengaluru, Delhi NCR, Hyderabad, Pune, Mumbai, Jaipur). Its customers are gig and logistics riders (food delivery, quick
commerce, e-commerce logistics, cargo, bike-taxi) for whom time off the road is lost income: a rider swaps a depleted
pack for a charged one in about two minutes, instead of charging for hours. Independent riders pay per swap or with a
prepaid pack; riders affiliated with one of 12 fleet partners are billed to their employer at a contracted discount.

## 1.2 The paradox leadership faces
Between January 2024 and June 2025 completed swaps roughly tripled and monthly revenue more than tripled, yet leadership
sees three things moving the wrong way: **service failures**, **new-rider retention** and **per-swap profitability**.
At the same time the company changed almost everything at once: two station-expansion waves, a base-price increase,
a peak/off-peak pricing pilot in two cities, a new battery supplier and a renegotiated contract with its largest fleet
partner. External shocks were layered on top: two Indian summers, a competitor opening near ten stations and a firmware
clock bug.

:::fig q0_02_timeline.png | **Figure 1.1** Nine overlapping changes in 18 months. Any before/after comparison has to separate these, especially the summer heat that recurs in April–June of both years.

The decision at stake is the **next operating budget**. Internal teams have proposed four uses for it: more stations, more
batteries, a network-wide pricing rollout, and a long-term exclusive with the largest fleet partner. Leadership needs to
know what is actually driving the three outcomes before choosing.

## 1.3 How we framed the question
We decomposed leadership's question into the three outcomes that deteriorated and turned each into testable hypotheses,
each tied to a data source and a method (Figure 1.2). The six core questions of the brief map onto this tree: network
performance over time (Q1) measures the outcomes; service failures (Q2), stations and geography (Q3), batteries (Q4),
pricing and partners (Q5) and retention (Q6) test the branches.

:::fig q0_01_issue_tree.png | **Figure 1.2** Hypothesis tree. Each hypothesis was tested against the data; Section 5 reports which ones survived and whether each is a primary or a secondary driver.

## 1.4 What a good answer looks like
| Outcome | KPI we track | Definition |
|---|---|---|
| Growth | Completed swaps, revenue | `swap_completed` events; `amount_charged_inr` (test stations excluded) |
| Service quality | Service-failure rate; queue wait | (stockouts + abandoned-in-queue + system errors) / attempts; p90 wait of completed swaps |
| Retention | New-rider return rate | share of riders who sign up in the window and swap again in days 31–60 after their first attempt |
| Profitability | Contribution margin per swap | **cm1** = revenue − energy − battery wear; **cm2** = cm1 − station rent & maintenance |

Two principles guide the analysis. First, **same-season comparisons**: failure rates swing from about 4% to 12% between
winter and summer, so comparing January 2024 with June 2025 manufactures a trend. Second, **association is not
causation**: where the data allows we use designs that get closer to cause (difference-in-differences for the pricing
pilot, dose-response and within-rider comparisons for batteries, pre-outcome features for churn) and we label the strength
of each claim.

# 2. Data and analytical approach

## 2.1 The data
Eight related tables cover every swap attempt, hourly cabinet telemetry, the battery ledger, riders, fleet contracts,
support tickets and daily city context for **January 2024 – June 2025**. Before any join we tested the documented
relationships: all nine foreign keys match **100%**, issued and returned packs always match the rider's vehicle class,
and the station-hour telemetry grid has no missing hours.

| Table | Rows | Grain | Used for |
|---|---:|---|---|
| swap_events | 3,877,013 | one swap attempt | volume, failures, queues, pricing, battery condition, revenue |
| station_hourly_status | 1,487,712 | station × hour | charged stock, cabinet temperature, charge time, outages |
| riders | 20,000 | rider | cohorts, segments, acquisition channel, plan type |
| batteries | 6,500 | battery pack | supplier, lot, purchase cost, state of health |
| support_tickets | 44,000 | ticket | complaints, resolution, CSAT, free-text comments |
| stations | 152 | station | city, zone, charger generation, wave, rent & maintenance, tariff |
| city_daily_context | 3,282 | city × day | heat, rain, festivals, outages, competitor promos |
| fleet_partners | 12 | B2B customer | discount, amendments, peak-surcharge billing, payment terms |

## 2.2 Cleaning: every issue detected, quantified and decided
We treated the organisers' list of data-quality issues as hypotheses to test, not instructions to follow. A separate audit
re-checked every cleaning decision and searched for problems the documentation does not mention. Result: **26 checks,
13 undocumented issues, 6 of them high-severity** (they would distort a headline KPI or invalidate an analysis if untreated).

:::kpis
bad:139,490 | swaps re-timed | firmware v3.2.0 logged local time as UTC (−5h30m), 10 Mar–14 Apr 2025
bad:62,031 | test-station attempts excluded | documented as "a few zero-value rows"; actually 59,134 paid swaps (₹39 lakh)
warn:233,264 | pilot-priced swaps hidden | peak/off-peak charges sit inside the PARTNER tariff code
good:0 | near-duplicates | the documented offline-sync pattern does not exist under any rule we tested
:::

**Three decisions we verified independently rather than assumed.**
- **The timestamp fix is confirmed by a second clock.** The pricing pilot bills peak hours by the true local time. On the
  raw timestamps only 11.6% of affected PEAK-billed swaps fall inside peak hours; after our +5h30m correction, 100% do. The
  same fix makes all 30 apparent "events before sign-up/commissioning" disappear: they were evening swaps the bug had dated
  to the previous day.
- **Telemetry blanks were kept as missing, not zero.** Blanks occur only at poor-connectivity stations; zero-filling them
  would have invented 9,384 "empty cabinet" hours and inflated stockout-hours 4.4× at those stations.
- **The 1-July-2024 failure drop is heat, not the price rise.** Failures fell from 9.6% to 3.7% on the day the base price rose,
  but only in the three hot cities, where temperatures fell 11–17 °C. Cool cities saw the same price rise and no change.

The full scorecard follows; the notebook prints each detection test and its impact.

| ID | Issue | Type | Severity | Rows affected | Decision | Impact if untreated |
|---|---|---|---|---:|---|---|
| DQ01 | Battery IDs not physically tracked (packs 'teleport' between cities) | **New** | 🔴 High | 239,049 pack re-issues | Per-pack SoH valid; per-pack location / rotation / turnaround NOT valid | 239,049 impossible cross-city re-issues ≤60 min |
| DQ02 | Peak/off-peak adjustments hidden inside PARTNER tariff code | **New** | 🔴 High | 233,264 swaps | Derive pilot exposure from the residual, not tariff_code | tariff_code hides 54% of pilot-priced swaps (233,264; ₹1.89M) |
| DQ03 | Firmware v3.2.0 timestamp bug (10-Mar..14-Apr-2025) | Documented | 🔴 High | 139,490 attempts | Shift +5h30m (clean.py) | 00-05h share 18.6%→3.5%; tariff clock 11.6%→100% |
| DQ04 | Packs below the 70% EOL SoH still issued (incl. 'retired' packs after 15-Jun-2025) | **New** | 🔴 High | 122,204 swaps | Keep (real swaps); flag; treat retirement record as administrative; size wear-past-EOL as a cm1 sensitivity | 122,204 sub-EOL swaps; ₹12.9M wear charged past EOL |
| DQ05 | Test stations STN-TST-01/02 hold full paid activity | Documented | 🔴 High | 62,031 attempts | Exclude from all KPIs (kept in test_station_events.parquet) | Bengaluru revenue +8.2%, network +1.7% (₹3.9M) |
| DQ06 | Telemetry blanks (telemetry_status = missing) | Documented | 🔴 High | 9,384 station-hours | Keep null; null-aware means; report coverage | zero-fill → stockout-hours x1.5 (poor tier x4.4) |
| DQ07 | Packs used before recorded commission date | **New** | 🟠 Med | 5,154 packs | Age from first_seen; ignore commission_date | pack age from commission_date wrong for 79% of packs |
| DQ08 | CSAT mostly missing, not at random | Documented | 🟠 Med | 28,871 tickets | Report IPW/post-stratified CSAT for resolved tickets + bound for unresolved | naive 3.56 vs IPW 3.56; scenario with unresolved 3.26–3.50 |
| DQ09 | Season labels (Summer = Mar–May) misaligned with the Apr–Jun heat regime | Analysis trap | 🟠 Med | 898,816 attempts | Control seasonality with city-month temperature (or city x month effects), not SEASON_OF_MONTH | failures track month-level heat (>~34–36 °C), not season labels |
| DQ10 | Ticket category 'other' hides vehicle/battery complaints | Documented | 🟠 Med | 9,484 tickets | Re-classify via comment text (ticket-text analysis) | 22% of tickets in an uninformative 'other' bucket |
| DQ11 | PREPAID / PROMO_FREE tariff codes absent; prepaid riders pay standard tariffs | **New** | 🟠 Med | 372,083 swaps | Treat plan_type as a label; no prepaid discount modelled | prepaid pricing not observable |
| DQ12 | 3W swaps at stations with zero 3W slots | **New** | 🟠 Med | 237,310 attempts | Flag; do not use slots_3w / inventory_target_3w for 3W capacity | 237K 3W attempts at 0-slot stations; capacity metrics unreliable |
| DQ13 | Price fields populated on non-completed attempts | Documented | 🟠 Med | 228,307 attempts | Revenue = amount_charged on completed swaps only | list-based revenue would be +6.6% (₹15.0M) |
| DQ14 | riders.home_city inconsistent spellings | Documented | 🟠 Med | 585 riders | Map 21 spellings → 6 cities | city rider counts +2.6% to +3.7% |
| DQ15 | km_since_last_swap outliers (odometer resets) | Documented | 🟠 Med | 15,287 readings | Set to null (<0 or >200 km), flag km_invalid | 2W mean range 58.5→58.1 km; sd 18→11 |

## 2.3 A per-swap P&L
The brief asks about *contribution margin per swap*, which the data does not contain. We built it from first principles
for every completed swap and use the same definition everywhere:

| Component | Definition | Network average |
|---|---|---:|
| Revenue | `amount_charged_inr` (non-completed attempts earn nothing) | ₹63.90 |
| − Energy | kWh drawn to recharge the returned pack × the station's grid tariff | ₹19.49 |
| − Battery wear | pack purchase cost × SoH points lost per cycle ÷ (initial SoH − 70% end-of-life); the per-cycle loss is estimated from each pack's own SoH trajectory | ₹48.47 |
| **= cm1** | variable contribution margin | −₹4.06 |
| − Station fixed cost | rent + maintenance, pro-rated by active days, spread over that station-month's swaps | ₹13.20 |
| **= cm2** | contribution after station operating costs | −₹17.26 |

The 70% end-of-life threshold is taken from the data itself: every retired pack is below 70% SoH and every active pack is
at or above it. Battery wear is the largest and most assumption-sensitive cost (it assumes no salvage value), so we
report trends and differences rather than leaning on absolute levels, and show a sensitivity table in Section 3.1.

## 2.4 Methods
| Question | Main methods | Why this method |
|---|---|---|
| Q1 Performance over time | monthly KPI panel, same-season (YoY) comparisons, fixed station panel, cm1 bridge | separates seasonality and network mix from real change |
| Q2 Service failures | hour × month heatmaps, Lorenz / concentration curves, telemetry joins (temperature → charge time → stock → stockout) | locates *where* and *why* failures happen |
| Q3 Stations & geography | multivariable station-month models with city and month fixed effects; zone supply-vs-need; difference-in-differences for competitor entry and new-station openings | separates charger generation, location, city and utilisation effects |
| Q4 Batteries | per-pack degradation slopes; range-vs-SoH dose-response; within-rider swap-frequency panel | links pack health to rider experience and station load |
| Q5 Pricing & partners | interrupted time series (base price); **difference-in-differences** with event study (peak pilot, 2 treated vs 4 control cities); partner-level P&L | isolates price effects from seasonality and growth |
| Q6 Retention | cohort curves; logistic regression with cohort fixed effects; gradient boosting + SHAP; Kaplan–Meier / Cox; decomposition of the retention decline | ranks drivers and tests consistency, dose-response and timing |
| Text (deep dive) | rule-based taxonomy of the 89 English/Hinglish comment templates; inverse-probability-weighted CSAT | recovers mis-filed battery complaints; corrects MNAR bias |

All code is in the accompanying Colab notebook, which downloads the organisers' data and runs top to bottom.

# 3. Key insights

## 3.1 Network performance: growth is real, the failure alarm is a seasonal artefact, the margin alarm is real

:::callout
**Answer.** The metrics do **not** tell the same story. Year on year (H1-2025 vs H1-2024) completed swaps grew **2.43×**
and revenue **2.63×**, while the same-season failure rate *fell* (7.04% → 6.06%). But variable margin per swap collapsed:
from **+₹6.75** after the July-2024 price rise to **−₹7.53** (October-2024 to March-2025). The new supplier's battery wear
explains more than the whole fall: **−₹16.0 per swap, about ₹40–43 lakh a month.**
:::

:::kpis
2.43× | Completed swaps | H1-25 vs H1-24
2.63× | Revenue | H1-25 vs H1-24
good:6.06% | Failure rate, H1-25 | vs 7.04% in H1-24 (same season)
bad:−₹7.5 | cm1 per swap | Oct-24 to Mar-25, vs +₹6.75 in Jul–Aug-24
:::

:::fig q1_01_kpi_divergence.png | **Figure 3.1a** Scale (top), reliability (middle) and unit economics (bottom). Failures follow the season; the 2025 "improvement" is new stations diluting the average; the July price rise opened a positive margin gap that battery wear closed from September 2024.

**"Service failures rose faster than swaps" depends on which months you compare.** January-2024 (winter) against
June-2025 (peak season) gives failures 6.26× vs swaps 2.93×. Every same-season comparison reverses it:

| Comparison | Failures grew | Swaps grew | Failure rate |
|---|---:|---:|---|
| Q1-24 → Q1-25 | 2.43× | 2.48× | 3.98% → 3.89% |
| Q2-24 → Q2-25 | 1.94× | 2.38× | 9.52% → 7.92% |
| H1-24 → H1-25 | 2.07× | 2.43× | 7.04% → 6.06% |

The same-season improvement itself is a **mix effect**: the 90 original stations did not improve (Q2 9.52% → 9.55% on
48% more attempts), while new stations, none of which have Gen1 chargers, fail at 5.1%. New stations are *not* lightly
loaded (4.8 vs 5.0 attempts per slot-day); they are better equipped (Section 3.3).

**Why margin per swap eroded: an exact bridge.**

:::fig q1_05_cm1_bridge_waterfall.png | **Figure 3.1b** cm1 per swap bridge. The July-2024 price rise added +₹5.2; from September the Kyron packs' wear took −₹16.0; the pilot (+₹1.7) and cheaper energy per swap (+₹1.3) only partly offset it, and the ZipDrop renegotiation took −₹1.4.

- **Primary: Kyron battery wear, −₹16.0 per swap** (same-season Q1-24 → Q1-25: −₹16.04, −₹43 lakh a month). Kyron went from 0%
  of 2W packs issued before September 2024 to 24–25% from October, and costs 3.26× the wear of other 2W packs.
- **Secondary: ZipDrop's discount 12% → 28%**, −₹1.45 to −₹1.70 per network swap.
- **Offsets:** base price +₹5.1, pilot surcharge +₹1.7, lower energy per swap +₹1.3 to +₹2.8, and a falling station fixed
  cost per swap (₹19.5 → ₹11.2 as utilisation rose), which keeps **cm2** roughly flat (−₹19.3 → −₹18.5) even as cm1 fell.
- **Growth is scaling a negative unit margin:** total cm2 moved from −₹21 lakh a month (Q1-24) to −₹50 lakh (Q1-25) under
  our depreciation assumption.

**Hidden warning signs in the "good" numbers.**
- Energy cost per swap *fell* 24% because ageing packs hold less energy, not because charging got more efficient. A 2W swap
  now delivers 48 km instead of 70.5, so riders pay **₹1.17 per km vs ₹0.82** a year earlier (+42%).
- In April–June 2025, **122,204 swaps issued Kyron packs already below the 70% end-of-life threshold**, including 32,903 issued
  after the packs' recorded retirement date (mean health 61.7%). Riders on 55–65% packs got 35–37 km per swap vs 57 km on
  healthy ones.
- The 4,133 non-Kyron 2W packs average 80% health and age in step: **the rest of the fleet reaches end-of-life in early-to-mid
  2026 (January–April, depending on the projection method), almost all at once.**

:::warn
**How robust is the margin story?** Battery wear is the most assumption-sensitive cost. The fall in cm1 from August 2024 to
March 2025 holds under every assumption we tested (end-of-life at 70% or 60%, 0% or 20% salvage value); the same-season Q1
decline ranges from −₹7.6 to −₹1.4 per swap. Absolute levels do depend on the assumption (cm1 is negative only under the
strictest one), and from April 2025 our per-cycle model also charges wear on packs issued past end-of-life, so we anchor
trends in windows before April 2025 and quote later levels under three treatments.
:::

## 3.2 Service failures: a hot-city, hot-season, old-charger problem

:::callout
**Answer.** Service quality is **not** spread evenly. Outside April–June every city, station, weekday and hour sits on the
same **≈3.9%** failure baseline once vehicle mix is accounted for. The excess comes from one place: **36 Gen1 stations in
Delhi NCR, Jaipur and Hyderabad during April–June**. They handle **32%** of summer attempts but produce **80%** of the
**62,558** excess failures. Like-for-like, failure rates did **not** worsen year on year (same-station risk ratio **0.986**).
:::

:::fig q2_04_concentration_lorenz.png | **Figure 3.2a** Concentration. Excess failures (above the off-season baseline) are far more concentrated than attempts (Gini 0.68 vs 0.29). The 36 hot-city Gen1 stations fail at 15.3% in April–June, against 5.1% for Gen3 stations in the same cities.

**What fails, and how often.** 211,849 of 3,814,982 attempts failed (5.55%): stockouts are 62.7% of failures,
abandoned queues 31.9% and system errors 5.4%. System errors are a flat noise floor (0.29–0.30%) across connectivity tiers,
firmware versions and sync modes, so they are not a lever.

**The mechanism is heat acting on old chargers.**
- Cabinets run **6.6 °C above the air** (r = 0.99). Above ~41 °C in the cabinet, **Gen1 charge time climbs steeply, from 88
  minutes to about 168 at 50 °C** (193 on average at 50 °C and above); Gen2 goes from 64 to 76 minutes and Gen3 barely moves (42 → 44).
- On days when a Gen1 cabinet peaks at 45 °C or more, the same station-days see **12.0% stockouts vs 2.3%** (5×; station-day
  GLM odds ratio 4.46 with city, month and weekday fixed effects).
- Failures start rising at a **36 °C** daily maximum (8.0% vs 3.9%) and plateau at ~14–15% from 39 °C. The official heat
  alert only fires at ~42 °C, six degrees too late to be useful as an operating trigger.
- Hot cities spend far more summer hours above 40 °C in the cabinet: Delhi NCR 26.7%, Jaipur 26.5%, Hyderabad 21.8% vs
  Mumbai 5.3%.

:::fig q2_05_heat_mechanism.png | **Figure 3.2b** The heat chain. Panels 1–3 are hourly telemetry facts; panel 4 is a day-level association; panel 5 shows that within a hot day, stockouts do *not* follow the hours when the cabinet is empty.

**A trap we avoided.** Pooled data suggest that "hours with an empty cabinet cause stockouts" (2.25× crude ratio). Compared
like-for-like (same station-day and hour window) the ratio is **0.99–1.00**: on hot days stockouts run at 8.6% at every hour
of the day and jump to **16.8% at 19:00–22:00 even when packs are in the cabinet**. Heat acts on the whole day, so **adding
batteries to cabinets is not evidenced to help**; charger generation and cabinet temperature are the levers.

**Where else it varies.**
- **3W riders fail 2.0× as often as 2W at the same station and month, all year** (cool-season stockouts 5.5% vs 2.0%).
  83 stations listed with zero 3W slots still serve 57% of 3W attempts, so 3W slot metadata does not describe 3W service.
- **Queue wait is not a congestion signal.** Completed swaps wait a median 3.5 minutes regardless of load. Riders who abandon
  leave after a median 10.8 minutes; the abandonment hazard climbs from 0.3% per minute at minute 5 to 4.6% at minute 10.
- **A failure is a lost sale, not a delay.** After a failure, riders complete a swap within an hour no more often than after
  a success (4.5% vs 4.6%), so every avoided failure is roughly one more paid swap.

:::warn
**Why the network average misleads.** In May the network failure rate fell from 11.9% (2024) to 9.5% (2025) purely
through mix: legacy stations still ran 11.9%, while new stations ran 5–6%. Failed attempts in April–June almost doubled
(39,820 → 77,445). Leadership should track same-station change and the hotspot rate, not the network average.
:::

## 3.3 Stations and geography: charger generation matters; site type and expansion timing do not

:::callout
**Answer.** Once heat × charger generation is modelled, station-level failure differences are almost fully explained
(R² 0.55 → **0.95**). Location type, host type and connectivity barely matter. Both expansion waves were **need-blind**:
new slots followed the existing footprint rather than the failure hotspots, and the hotspots' failure rate did not respond
to extra load anyway.
:::

:::fig2 q3_01_thermal_derating.png | **Figure 3.3a** Thermal derating. Gen1 turnaround nearly doubles by a 50 °C cabinet (88 → ~168 min, +9.5 min per °C above 41 °C); Gen2 slows modestly; Gen3 barely moves. || q3_03_expansion_placement_vs_need.png | **Figure 3.3b** Placement vs need. The highest-need third of zones held 49% of failures but received 36% of Wave-1 slots (and 55% vs 36% for Wave 2 on same-season need).

**Battery turnaround is a charger-generation story.** Gen1's penalty is conditional on heat: with no hot hours it charges
like other generations; at 20% hot hours its failure gap over Gen2 is **+3.8 points**. 70.6% of Gen1 stations sit in Delhi
NCR, Jaipur or Hyderabad, which is why raw city rankings (Jaipur 12.1% vs Mumbai 4.0% in April–June 2025) mostly reflect
where the Gen1 chargers are. Complaints follow: Gen1 stations log 18.6 tickets per 1,000 attempts vs 12.4 at Gen2.

**Load does not set the hotspot rate.** The 23 worst legacy stations absorbed **51% more attempts** (3.51 → 5.30 per slot-day)
at an unchanged **16.6%** failure rate; within Gen1 stations, load moves the failure rate by −0.03 points (p = 0.94). Adding
stations nearby could not have fixed these rates. This is the cleanest evidence against "more stations" as a service fix.

**Expansion economics.** 27 of Wave 1's 30 stations are highway fuel pumps or residential sites, 26 of them in zones that
already had stations; they run **3.5 swaps per slot-day at ₹15.6 fixed cost per swap** against ₹11.3 at legacy stations
(excess ≈ ₹1.7 lakh a month). What pays is **unmet demand**: the three stations opened in formerly unserved zones run
**11.6 swaps per slot-day** at ₹4.3 per swap; across the 60 new stations, utilisation rises by +5.2 swaps per slot-day per
log-unit of resident riders per zone slot (p < 0.001), and site type stops mattering once demand is controlled.

**Competitor entry.** The competitor that opened near 10 stations in January 2025 caused no statistically detectable loss
(swaps −3.9%, 95% CI −10.7% to +3.3%; riders stayed on the network). The test can rule out a large effect (>~10%), not a
small one, so those stations stay on a watch-list.

:::callout
**So what.** Retrofit or replace the 36 hot-city Gen1 cabinets before April 2026, prioritised by city × volume (Delhi NCR
≈835 and Jaipur ≈810 avoidable failures per site per summer). At 2025 volume this avoids **19,000–27,000 failed attempts per
summer** (Gen2-like to Gen3-like performance), cutting the network's April–June failure rate from 7.9% to 5.1–6.0% and
recovering up to **₹12–17 lakh** of revenue per summer. Site future stations by unmet
demand (resident riders per slot), not by ease of access.
:::

## 3.4 Batteries and equipment: one defective batch, and a fleet that is ageing all at once

:::callout
**Answer.** Battery health drives delivered range almost one-for-one: **each point of state of health (SoH) lost costs about
1% of range** (≈0.7 km per 2W swap, ≈1.0 km per 3W swap). One cohort stands out completely: **Kyron lots KY-2407/08/09**
(1,461 packs, including every Kyron 2W pack) lose health **3.8× faster per day** than every other lot, **at the same
purchase price and initial health**. Kyron's later 3W lots age normally, so this is a **lot defect, not a supplier-wide one**.
:::

:::fig q4_01_soh_vs_cycles.png | **Figure 3.4a** Mean SoH of issued packs by days in service. Kyron's defective lots fall below the 70% end-of-life threshold after ~245 days and were still being issued at 61% on 30 June 2025; Cellora and Amptek are at ~80% after 539 days and are projected to reach 70% around 832 days.

**How we know it is the lot, not the climate or the firmware.**
- In 3W, where exposure can be held equal (48.8% vs 47.6% of issues in the hot cities), the three lots fade 2.4× faster
  than Cellora while Kyron's nine later lots fade at **1.00×**. For 2W, lot, supplier and climate cannot be separated
  (every Kyron 2W pack is from these lots and 83.5% of their issues were in hot cities).
- Battery-management firmware has no effect (lot fixed effects: p = 0.33 for 2W, 0.24 for 3W), as the data dictionary hinted.
- Fade follows **calendar time, not issue count**: packs issued more often had not lost more health by the same date
  (elasticity −0.03). Battery wear therefore behaves like time-based depreciation.

**What it cost.**
- **₹2.94 crore of battery value destroyed** by June 2025 (depreciation capped at purchase price; ₹4.31 crore under the
  uncapped per-swap model). A Kyron pack recovers **42.5% of its price** before end-of-life; a Cellora pack a projected 139.5%.
- **Riders got 26% less range**: a returned Kyron 2W pack had delivered a median 39.5 km vs 53.4 km in Q2-2025.
- **Packs past end-of-life kept circulating.** The battery master "retired" every Kyron pack on 15 June 2025, but the
  cabinets kept issuing all of them: **99.94% of June Kyron 2W issues were below 70% SoH**, 66,473 sub-end-of-life swaps in
  June alone (21.9% of all swaps).

:::fig q4_06_kyron_economics.png | **Figure 3.4b** Same price, one-third the life. Excess wear by cohort, the lost-life value that underpins a warranty claim (₹3.4–3.6 crore), and payback per pack.

**Two findings that bear directly on the budget decision.**
- **More packs did not mean fewer stockouts.** When 1,385 packs arrived in September 2024 (+33.5% of the 2W fleet), charged
  inventory per station-hour changed by +0.004 and stockouts by 0.00 points in the cities that received them. At the same
  90 stations a year later, hot-city June stockouts were 10.08% → 9.93%. Stockouts are a charging-speed problem (Section 3.2),
  not a pack-count problem.
- **The defect was detectable within 6–8 weeks.** Over their first eight weeks the three lots already faded 2.5–4.5× faster
  than incumbents, while every other lot stayed below 1.4×. A lot-level acceptance rule would have flagged them by early
  November 2024, before ~₹2.3 crore of further value was lost.

**One unexplained signal.** Riders in the Kyron-heavy cities (Delhi NCR, Hyderabad, Jaipur) started swapping **9.8% more
often for the same daily distance** in the very week Kyron arrived (within-rider DiD; placebos and a 3W negative control are
null). The gap appeared while the packs were still healthier than the fleet and never tracked their fade, so its mechanism
is unidentified; we report it as an association and recommend tracking it after the packs are replaced.

:::callout
**So what.** Block issue of any pack below 70% SoH in cabinet firmware, pull and replace the three lots city by city
(≈₹4.4 crore for 2W + ₹0.5 crore for 3W, justified by service and safety rather than margin), and file a warranty claim
for the **₹3.4–3.6 crore** lost-life value. Introduce a lot-level early-warning rule on per-day fade over the first 6–8 weeks.
Budget now for the **replacement wave of the original fleet**, which reaches end-of-life in early-to-mid 2026.
:::

## 3.5 Pricing and partner economics: a revenue lever, not a congestion lever

:::callout
**Answer.** Both price moves raised revenue without visibly hurting usage. The July-2024 base-price rise (2W ₹60→65,
3W ₹100→110) passed straight into revenue (demand elasticity ≈ 0 to −0.24). The peak/off-peak pilot added **₹7.90 (+11.7%)
per independent swap** but moved only **2.5 points** of swaps out of peak and left peak stockouts unchanged (+0.05 pp).
The **largest fleet partner by swaps, ZipDrop, is the least valuable**: its 2024 renegotiation cost ₹54.5 lakh a year and
bought no extra volume.
:::

:::fig q5_01_pilot_event_study.png | **Figure 3.5a** Difference-in-differences event study of the pilot (Bengaluru + Pune vs the four other cities). Revenue steps up exactly when the pilot starts; the timing shift is small and fades; peak stockouts do not move. The summer dips in the right panel exist in 2024 too: they reflect milder summers in the pilot cities, not the pilot.

**What the pilot actually did.**
- **The real schedule is not "12:00–22:59".** In the data, PEAK (+₹15) applies at 12:00–13:59 and 19:00–22:59; OFFPEAK
  (−₹8, not recorded in `discount_inr`) at 23:00–05:59 and 14:00–16:59. Partners whose contracts make the surcharge
  billable pay it; ZipDrop and Swiggle Go do not.
- **Revenue:** +₹7.90 per independent swap (95% CI 7.84–7.95), exactly the mechanical value of the tariff at riders' new
  timing; billable partners +₹5.43 per swap.
- **Timing:** −2.49 points of independents' swaps moved out of peak (CI −3.11 to −1.86), 40% of it from the *uncongested*
  midday window. The shift faded to about −1.3 points by May–June 2025. With only six cities the permutation p-value is
  0.067, the smallest this design can produce.
- **Congestion:** no change in peak stockouts, abandonment or failures, robust to a Mumbai-only control and a balanced
  station panel. The evening congestion the pilot might target exists only in Delhi NCR, Jaipur and Hyderabad in
  summer and monsoon (evening failures +2.6 to +4.8 points above midday); the pilot cities barely have it.

**Partner economics: the contract decides the margin.** Compared within the same city, month, vehicle class and tariff
band, every partner costs the same to serve (cost gaps within ±₹0.80 per swap); the discount explains essentially all of
the margin gap.

:::fig2 q5_04_partner_value_bubble.png | **Figure 3.5b** Partner value. ZipDrop has the biggest volume (15.5% of network swaps) and the biggest total loss (cm1 −₹69.3 lakh, Nov-24 to Jun-25). || q5_06_zipdrop_amendment.png | **Figure 3.5c** ZipDrop's discount went from 12% to 28%: revenue −₹10.40 per 2W swap (DiD), with no extra volume (+56% vs +53% for other partners).

- **ZipDrop** (quick commerce; 28% discount; surcharge not billable; 72.6% of its swaps at peak) earns ₹48.6 per swap and
  carries **19.7% of all peak-hour swaps**. Even at steady state, with Kyron packs gone, its cm1 is −₹4.18 per swap
  (≈ −₹21 lakh a year). The break-even discount is **24.6%**.
- **FeastFly**, the largest partner by revenue and riders (10% discount, surcharge billable), earns **+₹12.2 cm1 per swap
  at steady state (+₹57.5 lakh a year)**. Its terms are the template for any anchor-partner deal.
- **3W swaps lose money even with healthy batteries**: energy + wear ≈ ₹116 per swap against a ₹110 list price. The cm1
  break-even 3W list price is about **₹126**.

:::callout
**So what.** Roll peak/off-peak pricing out network-wide **as a revenue measure** (+₹9.5 lakh a month in the four
remaining cities, ≈ ₹1.14 Cr a year, with costs unchanged), not as a congestion fix. Make the surcharge billable in every
partner contract at renewal (+₹4.1 lakh a month). **Do not sign a long-term exclusive with ZipDrop on current terms**;
renegotiate to a discount of 12–15% with a billable surcharge, using FeastFly's terms as the model. Re-price 3W.
:::

## 3.6 Why new riders do not come back

:::callout
**Answer.** New-rider retention **did not fall structurally**: riders who started in January–March 2025 retained exactly
as well as those who started in January–March 2024 (**86.3% vs 86.3%** still swapping in days 31–60, p = 0.98). What
leadership sees is a **summer dip** plus a **censoring artefact** in the naive metric. The **primary, actionable driver is
service failures in a rider's first two weeks**: riders with 2+ failed swap attempts in days 0–13 fail to return
**22.6%** of the time vs **13.1%** (adjusted odds ratio **1.67**, 95% CI 1.43–1.94). Counting only stockouts and abandoned
queues, it is 23.5% (OR 1.76); system errors on their own do not matter.
:::

:::fig q6_01_cohort_retention_trend.png | **Figure 3.6a** Cohort retention (completed swap in days 31–60). Dips appear only for summer-entry cohorts (May-24 79.1%, Apr-25 81.7%), exactly when new riders' early failure rates spike (bottom panel). The dotted naive metric "collapses" only because recent cohorts have not been observed for 60 days yet.

**Why we are confident about the primary driver.** It passes every test we set before looking at results:
- **Three independent designs agree**: rider-level logit (2 vs 0 early failures OR 1.67), landmark Cox survival model
  (HR 1.24), and an attempt-level exit model (OR 1.75 after two failures in seven days; within-rider OR 1.53).
- **Dose-response**: once early activity is controlled (a Simpson's-paradox trap: riders who leave early make few attempts
  and so collect few failures), the first failure raises the odds 1.23× and the second 2.47×.
- **Consistency**: the effect is adverse in **15 of 15** rider subgroups (significant in 14).
- **It explains the trend**: early failures account for **40–50%** of the summer-2024 dip (four methods), 66% counting
  failures up to day 29.

:::fig q6_03_driver_effects.png | **Figure 3.6b** Drivers of not returning, with 95% CIs from three designs. Primary: early stockouts and abandoned queues. Secondary: competitor nearby (localised), 3W riders and a failed first attempt (both weak). Not supported: Kyron packs, peak pricing, queue time itself, tickets, plan type, channel, KYC, age, vehicle model.

**What does *not* drive churn (and why that matters).**
- **Degraded batteries:** packs received by later cohorts were far worse (mean SoH 98.7 → 82.0, km per swap 74.7 → 56.4,
  Kyron share 3.7% → 23.5%), yet same-season retention stayed flat. Kyron is a **cost** problem, not a retention problem.
- **Price:** the peak pilot shows no retention effect (DiD p = 0.15); the base-price effect on independents (p = 0.042) does not
  survive a multiple-testing correction.
- **Proxies:** vehicle model is a fleet-affiliation proxy (adds nothing once partner is known, p = 0.71); declared shift says
  nothing about when riders actually ride (41–43% of early attempts fall at 17:00–21:00 whatever shift they declared).

:::callout
**So what.** Protect new riders in summer: a failure-recovery trigger in their first 30 days (apology + credit + a reserved
charged pack + redirection after the 2nd failure, run as an A/B test), backed by the summer heat fixes in Section 3.2.
Capping early failures at one would cut summer-cohort non-retention from 16.1% to 13.4%. Replace the naive retention KPI
with D31–60 cohort retention compared same-season.
:::

## 3.7 Deep dive: what the support-ticket text adds

:::callout
**Answer.** Reading the 89 English/Hinglish comment templates against the event log changes the complaint picture:
**battery health is the largest complaint driver: about half of all tickets (49.5%; floor 36.7%) vs 15.1% in the
official `low_range` category.** Two filing patterns hide it: the `other` bin (21.6% of tickets) consists entirely of
vehicle-phrased battery complaints ("scooter weak lag raha hai aajkal"), and `app_issue` tickets carry the same battery
fingerprint with no app failures behind them.
:::

:::fig q8_03_hidden_battery_complaints_kyron.png | **Figure 3.7a** Battery complaints per 1,000 2W swaps: official category vs text-corrected. They step up with the Kyron rollout (confirmed by a city dose test outside summer: 1.74× in the Kyron-heavy cities vs placebos 0.69–0.87) and keep rising as the fleet ages.

- **The text carries a number the category cannot:** 1,349 tickets say "dying at X km". Riders on Kyron packs report a median
  45 km vs 54 km for other packs, and reported range tracks pack health at 0.73 km per SoH point within the month.
- **Agent priority carries no information** (independent of the issue; no effect on resolution time), and battery complaints
  are "resolved" in the same 13 hours as everything else although no agent can fix a degraded pack.
- **CSAT is missing not at random but the bias is small**: slow tickets answer less (28% vs 44%) but score the same, so
  inverse-probability weighting moves CSAT by only +0.002. The real blind spot is coverage: the 14% of tickets that are
  unresolved or duplicates are never surveyed.
- **No signal where one might expect it**: no comment template mentions a competitor or switching, and the pilot shows no
  billing-complaint effect against a like-for-like control city (IRR 0.94).

# 4. Business findings: what is driving what

:::fig q9_01_causal_chain.png | **Figure 4.1** The causal picture. Two equipment problems (red) explain most of the failures, churn and margin loss; contract terms, need-blind expansion and whole-fleet ageing (amber) contribute; the grey bar lists what we tested and ruled out.

**Primary vs secondary drivers.** We called a factor *primary* only if it showed a large effect, a dose-response or
mechanism, consistency across segments and periods, and explained a material share of the change leadership is worried
about. Factors that pass some tests, or act only in one segment, are *secondary*.

| Outcome | Primary driver | Secondary / contributing | Tested, not a driver |
|---|---|---|---|
| Service failures | Summer heat × Gen1 chargers at 36 hot-city stations (R² 0.55 → 0.95; 77–80% of excess failures) | 3W riders (2× failure rate all year); Gen2 heat sensitivity | Station load, queue capacity, number of packs, connectivity, firmware, expansion wave once charger generation is known |
| New-rider churn | 2+ failed swap attempts in the first 14 days (OR 1.67; 40–50% of the summer dip) | Competitor within 1.5 km (localised, Cox HR 1.20); 3W riders; a failed first attempt (weak) | Battery health, peak pricing, queue time, tickets, plan type, channel, KYC, age, vehicle model |
| Margin erosion | Defective battery lots KY-2407/08/09 (−₹16.8 per swap of wear vs Cellora-grade) | ZipDrop discount (−₹1.5 to −1.7 per network swap); 3W priced below cost; Wave-1 fixed-cost drag | Energy cost (fell), station fixed cost per swap (fell with utilisation), the pricing changes (they added revenue) |

:::fig q0_03_hypothesis_results.png | **Figure 4.2** The hypothesis tree from Section 1, after testing: three primary drivers, five secondary, one descriptive pattern confirmed, three rejected.

**Where performance varies most.**

| Dimension | Best | Worst | Gap |
|---|---|---|---|
| Season | July–March: 3.9% failures | April–June: 55% of failures on 37% of attempts | ~2.5× |
| City (April–June) | Mumbai 4.0%, Bengaluru 4.5% | Jaipur 13.6%, Delhi NCR 12.2% | ~3× |
| Charger generation, hot cities (April–June 2025) | Gen3 5.1% | Gen1 15.2% | 3× |
| Battery lot | Cellora / Amptek / later Kyron lots | Kyron KY-2407/08/09 | 3.8× faster health loss |
| Vehicle class | 2W: 5.0% failures; steady-state cm1 +₹12 per swap | 3W: 9.95% failures; cm1 −₹14.9 per swap | 2× failures; ₹27 margin gap |
| Partner (steady-state cm1 per swap) | Independents +₹18.2; FeastFly +₹12.1 | ZipDrop −₹4.1; 3W cargo partners −₹14 to −15 | contract-driven (cost to serve within ±₹1) |
| Hour of day (hot days, hot-city Gen1) | 00:00–18:00: 8.6% stockouts | 19:00–22:00: 16.8% stockouts | 2× |

**Three implications for how VoltRelay manages the business.**
1. **The headline KPIs mislead.** The network failure average falls as new stations dilute it, retention "falls" because
   recent cohorts have not been observed long enough, and energy cost per swap "improves" because ageing packs hold less
   energy. Leadership should track same-station, same-season failure rates, D31–60 cohort retention and price per km.
2. **The fixes are targeted, not network-wide.** 36 stations, three battery lots, one partner contract and the 3W price list
   account for most of the damage.
3. **Operations data is not reaching the field.** Packs marked retired kept circulating, and battery complaints are filed as
   "other" or "app issue". Closing these loops costs little and prevents the next battery batch from repeating the damage.

# 5. The budget decision and recommendations

## 5.1 The four proposals, judged on the same scale

:::fig q9_02_budget_decision.png | **Figure 5.1** Each option in the same units: April–June failure rate, new-rider churn and cm1 in ₹ lakh a month at H1-2025 volume. The four original proposals (top) against the recommended alternative allocation (bottom).

- **(a) More stations: reject as a service fix; fund demand-led sites only.** The worst stations kept a 16.6% summer
  failure rate while absorbing 51% more attempts, and within Gen1 stations load moves the failure rate by −0.03 points
  (p = 0.94), so new capacity nearby cannot fix them. New sites pay only where demand is unmet: the three stations opened in
  formerly unserved zones run 11.6 swaps per slot-day, against 3.1 for fuel-pump sites added to already-served zones.
- **(b) More batteries: reject; replace instead.** Adding 1,385 packs (+33.5% of the 2W fleet) changed charged stock by
  +0.004 packs per station-hour and stockouts by 0.00 points. On hot days stockouts run at ~17% in the evening whatever the
  cabinet holds. The battery money should go to **replacing** the defective lots and renewing the ageing fleet: about
  **₹18 crore of packs falls due within ~12 months** (₹4.95 crore now, ₹13.2 crore in January–May 2026), which belongs at the
  top of the budget, ahead of any expansion.
:::fig q10_02_fleet_eol_timeline.png | **Figure 5.2** Projected month in which each pack crosses 70% state of health, from its own calendar fade. The defective Kyron lots already crossed it; the original 2W fleet follows in January–May 2026, a ≈₹13.2 crore renewal that must be quality-gated.

- **(c) Network-wide peak/off-peak pricing: support, with changes.** Pricing is a revenue lever (+₹7.90 per independent
  swap, ≈ ₹9.5 lakh a month across the four remaining cities, ≈ ₹1.1 crore a year) with no detectable effect on congestion or
  churn. Keep the current bands, make the surcharge billable in every partner contract (+₹6.3 lakh a month), and test a
  summer evening surcharge in one hot city before relying on price for congestion.
- **(d) Long-term exclusive with ZipDrop: reject on current terms.** ZipDrop is the largest partner by swaps (15.4%) and
  the biggest loss. Even with healthy packs its cm1 is −₹4.1 per swap at a 28% discount (break-even ≈ 22–25%), and an exclusive
  would lock in 20% of all peak-hour swaps without the surcharge. FeastFly's terms (10% discount, billable surcharge, +₹12.1 per
  swap) are the template for any anchor-partner deal.

## 5.2 What to do instead: prioritised recommendations

| # | Action | Owner | Expected impact | Success KPI | When |
|---|---|---|---|---|---|
| 1 | **Battery quality reset**: block issue of packs below 70% SoH in cabinet firmware; pull and replace lots KY-2407/08/09; file a warranty claim; gate every new lot on its first 6–8 weeks of fade | Head of Battery Ops + Procurement | ≈ **₹48 lakh/month** of wear avoided vs Kyron-grade packs; **₹3.4–3.6 Cr** claim; +26% km per swap on affected swaps; zero sub-EOL swaps (122K in Q2-25) | % of issues below 70% SoH = 0; lot fade ratio < 1.5× | Now |
| 2 | **Plan the fleet renewal**: the original 2W fleet reaches end-of-life between January and April 2026 | CFO + Battery Ops | ≈ ₹13.2 Cr capex (Jan–May 2026), quality-gated; avoids a repeat of the sub-EOL period | Fleet mean SoH > 75%; km per swap | Order by Q4-2025 |
| 3 | **Gen1 thermal fix at 36 hot-city stations** (cooling, shading or Gen3 swap): pilot at matched sites in March, then scale, prioritising Delhi NCR and Jaipur | Head of Network Ops | **19,000–27,000 fewer failed attempts per summer**; network Apr–Jun failure rate 7.9% → 5.1–6.0%; summer-entrant churn −0.6 to −1.0 pp | Same-station Apr–Jun failure rate at the 36 sites | Before April 2026 |
| 4 | **Summer heat playbook** triggered by a forecast maximum of **36 °C** (not the 42 °C official alert): steer riders in the app to cooler or Gen3 stations at 19:00–22:00 and protect afternoon charging | City Ops | Targets the 2× evening stockout spike on hot days (16.8% vs 8.6%) | Hot-day 19–22h stockout rate | April–June each year |
| 5 | **New-rider failure recovery** (A/B test): after an early stockout or abandoned queue, apology + credit + a reserved charged pack + redirection | Head of CX | Up to **−1.45 pp** new-rider non-retention overall (−2.7 pp for summer entrants) | D31–60 cohort retention, same season | Pilot next quarter |
| 6 | **Contract and price re-terms**: ZipDrop discount back to ~15% with volume tiers on off-peak growth; billable peak surcharge in all contracts; 3W list price towards ~₹126 | CCO + Pricing | ≈ **+₹14.5 lakh/month** cm1 (ZipDrop +₹3.8 L, surcharge +₹6.3 L, 3W +₹4.3 L) | cm1 per swap by partner | At each renewal |
| 7 | **Roll out peak/off-peak pricing** to the four remaining cities, current bands | Pricing | ≈ **+₹9.5 lakh/month** revenue at unchanged cost | Revenue per swap; peak stockouts (guardrail) | Next quarter |
| 8 | **Fix the instruments**: same-station and same-season KPIs, D31–60 retention, a "battery/vehicle performance" ticket category, effective tariff derived from the price residual, wear that stops at end-of-life in the cost model | Analytics / BI | Prevents misreading the next summer; battery complaints become visible (≈50% of tickets, not 15%) | KPI definitions adopted | Before summer 2026 |

:::fig q10_01_path_to_positive_cm2.png | **Figure 5.3** Path to a positive contribution per swap after station costs, at Q1-2025 unit costs and H1-2025 volume. Battery quality and contract terms close the gap; no capacity is needed.

:::callout
**Sequencing.** Items 1, 6, 7 and 8 need little or no capex and start immediately. Items 2 and 3 are the capital decisions.
Item 3 should be piloted first, because charger generation is confounded with site age in this data. Together, items 1, 6 and
7 take cm2 per swap from **−₹18.5 to about +₹6.8** at Q1-2025 unit costs, before any service gains.
:::

*Note on the battery figure.* The ₹16.8 per swap (≈ ₹48 lakh a month) compares Cellora-grade with Kyron-grade packs going
forward, which is why the lot gate and quality-checked renewal matter. Replacing packs that are already fully written off
does not by itself raise the current month's cm1 (Section 3.4), so the case for pulling them **now** rests on service (26%
more range) and safety, not on margin.

# 6. Limitations and what we would test next

**What the data cannot tell us.**
- **Association, not proof.** The data are observational. Where possible we used designs that get closer to cause
  (difference-in-differences with permutation tests for the pilot, within-station and within-rider comparisons, dose-response,
  pre-outcome features for churn). The charger-generation effect is confounded with site age, because every Gen3 charger
  is at a new site, so it needs a controlled upgrade pilot before a full rollout.
- **Battery wear drives margin *levels*.** Our usage-based depreciation (no salvage, 70% end-of-life) makes cm1 negative.
  Under 60% end-of-life or 20% salvage it turns positive. Every trend we report holds under all of these assumptions. From April 2025 the
  per-cycle model also charges wear on packs issued past end-of-life; we quote those months under three treatments, and the
  model should be changed so that wear stops at end-of-life.
- **Six cities limit pricing inference.** With two treated cities the smallest attainable permutation p-value is 0.067.
  The pilot effect is consistent across nine months and every specification, but a station-level randomised test would be
  far more conclusive.
- **Churn is only partly explained.** Early failures explain 40–50% of the summer retention dip; the models' AUC is
  0.60–0.68, so most individual churn comes from things the data do not record (earnings, vehicle ownership, competing
  jobs). The competitor test can rule out only effects larger than ~10%.
- **Data signatures of the simulation.** Battery IDs are not physically tracked (per-pack health is valid, per-pack
  location is not). `batteries.csv` commission dates are unreliable. Queue wait never responds to load. On hot days,
  stockouts happen even with full cabinets. We designed around these rather than over-reading them.

**What we would test next (cheap, fast, decisive).**
1. **Gen1 cooling/upgrade pilot** at matched hot-city Gen1 stations vs controls, March–June 2026: measured on stockouts and
   abandonment.
2. **New-rider failure-recovery A/B test** (trigger on the 1st vs 2nd early failure vs control): measured on D31–60 retention.
3. **Station-level randomised pricing test** in one hot city (summer evening surcharge): measured on 19:00–22:00 stockouts.
4. **Supplier acceptance gate**: health-loss-per-cycle tracking for every new pack cohort in its first 2–3 months.

# Appendix: reproducibility and definitions

**Reproducibility.** Every number in this report is printed by code in the accompanying Google Colab notebook
(`VoltRelay_Analysis.ipynb`). The notebook downloads the organisers' eight CSV files from the shared Google Drive folder,
converts them to typed Parquet, runs the audited cleaning pipeline, and then runs one section per question. It runs top to
bottom on a standard Colab CPU runtime. Each analysis was **independently re-derived by a second analyst with separate
code**; every refuted or corrected number was fixed before this report was written.

**Key definitions.**
- **Attempt**: one row of `swap_events`. **Completed swap**: `event_type = swap_completed`.
- **Service failure**: stockout (`failed_no_charged_battery`) + abandoned in queue + system error. Rider cancellations are
  reported separately.
- **Excess failures**: observed failures minus attempts × the July–March baseline rate of the vehicle class (2W 3.3%, 3W 8.6%).
- **Hot cities**: Delhi NCR, Jaipur, Hyderabad (26.7%, 26.5% and 21.8% of summer cabinet-hours above 40 °C).
- **New-rider retention (D31–60)**: share of riders who signed up from January 2024, with at least 60 days of follow-up,
  who complete a swap in days 31–60 after their first attempt.
- **cm1** = revenue − energy − battery wear; **cm2** = cm1 − station rent and maintenance (pro-rated). Battery wear =
  pack cost × health lost per cycle ÷ (initial health − 70%).
- **Steady state**: economics with Kyron packs replaced by non-Kyron averages (all Kyron packs reached end-of-life by
  June 2025).
- **Hours**: all hour-of-day analysis uses firmware-corrected local time.

**Data-quality scorecard (continued): low-severity issues and passed checks.** The high- and medium-severity issues are in
Section 2.2.

| ID | Issue | Type | Severity | Rows affected | Decision | Impact if untreated |
|---|---|---|---|---:|---|---|
| DQ16 | payment_mode constant ('partner_invoice' on 100%) | **New** | ⚪ Low | 3,877,013 attempts | Drop field; use plan_type/tariff | field unusable |
| DQ17 | station_firmware is static per station (not 'firmware at event time') | **New** | ⚪ Low | 1,083,456 attempts | Use only as station attribute; no pre/post firmware comparisons | firmware field cannot date firmware changes |
| DQ18 | Riders without KYC verification swapping | **New** | ⚪ Low | 149,851 attempts | Keep; flag for compliance | compliance flag, no KPI effect |
| DQ19 | Telemetry 'partial' status rows | **New** | ⚪ Low | 36,747 station-hours | Treat as usable | no blanks — usable |
| DQ20 | Tickets outside window / at test stations / before sign-up | **New** | ⚪ Low | 661 tickets | Exclude from in-window ticket KPIs (flags in_window, is_test) | 661 tickets to exclude/flag |
| DQ21 | Unexplained ±₹5/±₹10 billing deviations | **New** | ⚪ Low | 7,194 swaps | Keep amount_charged as revenue; flag for billing audit | net ₹8K; disputes 0.8% vs 0.9% |
| DQ22 | SoC/SoH readings >100% (sensor drift) | Documented | ⚪ Low | 9,684 readings | Clip at 100, flag reading_clipped | mean SoH shift 0.0002 pts (negligible) |
| DQ23 | queue_wait_sec hard caps (1,800 s completed / 2,400 s abandoned) | **New** | ⚪ Low | 177 attempts | Keep; treat as right-censored; report medians/p90 | negligible (censoring only) |
| DQ24 | Near-duplicate offline-sync retries | Documented | ⚪ Low | 0 attempts | None removed (rule applied, 0 hits) | 0 duplicates; looser rules ≤0.17% & look like chance |
| DQ25 | Energy-to-recharge vs SoC/SoH/capacity consistency | Check passed | 🟢 Pass | 719 swaps | No action (energy cost per swap trustworthy) | consistent (r = 0.98); Gen1 uses 8% more energy than Gen3 |
| DQ26 | Integrity checks: events before sign-up / commissioning, after decommissioning, pack-vs-vehicle class | Check passed | 🟢 Pass | 0 attempts | No action | 0 violations (4 checks) |
