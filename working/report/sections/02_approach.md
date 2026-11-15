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

{{DQ_SCORECARD}}

## 2.3 A per-swap P&L
The brief asks about *contribution margin per swap*, which the data does not contain. We built it from first principles
for every completed swap and use the same definition everywhere:

| Component | Definition | Network average |
|---|---|---:|
| Revenue | `amount_charged_inr` (non-completed attempts earn nothing) | {{REV}} |
| − Energy | kWh drawn to recharge the returned pack × the station's grid tariff | {{ENERGY}} |
| − Battery wear | pack purchase cost × SoH points lost per cycle ÷ (initial SoH − 70% end-of-life); the per-cycle loss is estimated from each pack's own SoH trajectory | {{WEAR}} |
| **= cm1** | variable contribution margin | {{CM1}} |
| − Station fixed cost | rent + maintenance, pro-rated by active days, spread over that station-month's swaps | {{FIXED}} |
| **= cm2** | contribution after station operating costs | {{CM2}} |

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
