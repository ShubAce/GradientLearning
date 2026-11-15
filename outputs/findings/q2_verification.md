# Q2 verification - Service failures and customer experience

Verifier script: `analysis/verify/q2_verify.py`. It is independent code. It uses only the `src/common.py` loaders and re-derives outcome flags from `event_type`; it does not use the analyst's script or tables. Run it with `PYTHONIOENCODING=utf-8 python analysis/verify/q2_verify.py` (under 1 minute). Every number below is printed by that script.

## Verdict summary
- **Almost every number reproduces exactly**: the taxonomy, seasonality, city and generation rates, MH risk ratios, the GLM, queue and Kaplan-Meier values, Gini values, 3W, year on year, tickets, the firmware check and the Rec 1 arithmetic.
- **One claim is refuted.** The analyst says empty cabinets cause the stockouts and that a 2-3 hour "carry-over" explains the evening peak. Neither holds at hour level once hour of day or station-day is held fixed. The day-level heat association is solid and confirmed. This also removes the stated rationale for Rec 2.
- **Two claims need correcting.**
  - Rec 4 says 9,181 3W stockouts "per 9 cool months". That figure actually covers 12 cool months.
  - The "chronic hotspots" are not special stations. They are simply the largest hot-city Gen1 sites.
- **One framing fix.** The headline and hero figure compare 80% of excess with "11.6% of attempts". That denominator includes Jul-Mar attempts, which have zero excess by construction.
- **needs_fix = true.**

## Verdict table
| # | Claim (analyst) | Recomputed | Verdict |
|---|---|---|---|
| H | Jul-Mar baseline 3.91%; 62,558 excess failures = 29.5% of 211,849; 80.4% of them at 36 hot-city Gen1 stations in Apr-Jun, which carry 11.6% of attempts; Gen1 charge time 88 to 193 min; same-station RR 0.986 | Identical | **Confirmed** (numbers). Framing fix: see Issue 3 |
| 1 | 3,814,982 attempts; 211,849 failures (5.55%). Stockouts 132,786 (3.48%, 62.7% of failures); abandoned 67,663 (1.77%, 31.9%); system errors 11,400 (0.30%, 5.4%); cancelled 16,458 (0.43%). System-error rate 0.294-0.301% by tier, 0.299% on both firmware versions, 0.292% vs 0.300% by sync mode. 2W 5.02% of 3,399,568; 3W 9.95% of 415,414 | Identical | **Confirmed** |
| 2 | Jul-Mar 3.91% (monthly 3.82-4.01%); 18 city x generation cells 3.54-4.09%; Apr-Jun = 36.6% of attempts and 55.4% of failures; May-24 11.9%, May-25 9.5%; worst cell May-24 19:00 16.0%, best 2.8%; May 19-22h 13.2% vs 8.4%; 19-22h holds 38.2% of attempts and 42.4% of failures | Identical. The best cell (Aug-24 00:00) has only 933 attempts. The weekday range 5.48-5.62% is for the **full period**; for Jul-Mar it is 3.89-3.93% | **Confirmed** (relabel the weekday period) |
| 3 | Apr-Jun: Jaipur 13.60%, Delhi 12.20%, Hyderabad 10.36%, Pune 5.55%, Bengaluru 4.47%, Mumbai 4.03%. Gen1 vs Gen3 MH RR 2.54 (2.49-2.60); Gen2 vs Gen3 1.35 (1.32-1.38). Hot cities: Gen1 15.2%, Gen2 8.0%, Gen3 5.1%. Wave RRs 1.00 (0.98-1.02) and 0.97 (0.93-1.02) | Identical. The pooled 2.54 averages very different cities: Delhi 3.16, Jaipur 3.17, Hyderabad 2.80 (hot-city pooled 3.05, CI 2.97-3.12) vs Pune 1.72, Bengaluru 1.19, Mumbai 1.00. The 15.2% Gen1 rate is 2025-only; across both summers it is 15.31% | **Confirmed** (report by city) |
| 4a | Cabinet = ambient + 6.6C (r = 0.992). Gen1 charge time 88/156/193 min; Gen2 64/72/76; Gen3 42/43/44. Gen1 hours with no charged 2W pack 0.58%/37.3%/83.6%. Gen1 station-days with peak cabinet >=45C: stockout 12.00% (5,245 days) vs 2.34%; Gen3 3.35% vs 2.34%. GLM ORs 4.46 (4.22-4.72), 2.11 (2.00-2.21), 1.32 (1.20-1.46). Telemetry coverage 96.8% ok / 2.5% partial / 0.64% missing; 99.33% of attempts covered. Hours at >=45C: Jaipur 23.8%, Delhi 22.2%, Hyderabad 12.5% | Identical (Gen2 OR 2.10, CI 2.00-2.21) | **Confirmed** |
| 4b | "Charge time empties cabinets, which leads to stockouts and abandonment." Hours with zero stock: 2W stockout 6.88% and abandonment 3.37%, vs 2.94% and 1.50% with >=5 packs. "Zero stock 3h earlier predicts 11.62% vs 7.27% (carry-over); this is why summer evenings are worst." Finding 5 adds: "Abandonment tracks charged inventory" | The pooled numbers reproduce (11.40% on a true hourly lag), but every like-for-like test is null. See Issue 1 for the figures | **Refuted** |
| 5 | City-day failure rate: <34C 3.92% (2,620 days), 34-38C 7.45% (305), >=38C 14.25% (357). Within city-month, >=36C RR 2.20 (2.14-2.25). Heat alert 1.01 (0.99-1.04); grid outage 1.02 (1.01-1.03), for system errors 1.00 (0.96-1.03); rain 1.02 (1.00-1.04) | Identical. `heat_alert` is a fixed temperature rule: it is set on all 148 days above 42C, 2 of 87 days at 40-42C, and no day at or below 40C. Failures climb from 35C (5.4%) through 36C (8.0%) and 37C (11.7%), then plateau at about 14-15% from 38C | **Confirmed** |
| 6 | Completed wait: median 3.5 min, mean 4.06, p90 7.1. Abandoned median 10.8 min (3.1x); 85.5% of abandoners waited past the completed p90. KM: 99.9% / 90.5% / 60.2% still waiting at 5/10/15 min; median patience 16.6 min. Hazard 0.3%, 2.2%, 4.6%, 10.2% per minute. Mean completed wait 244.0 s (243.4-244.5) vs 243.8 s (242.7-244.9); station means 241-247 s; abandonment 1.75-1.94% across demand bins | Identical | **Confirmed** |
| 7 | Gini: attempts 0.294, failures 0.383, excess 0.683. Top 10%: 18.6% / 24.7% / 42.9%. Hot-city Gen1 in Apr-Jun: 11.6% of attempts, 31.8% of failures, 80.4% of excess. All hot-city Apr-Jun: 19.0% of attempts, 92.4% of excess. Top 15 = Delhi 7, Jaipur 6, Hyderabad 2, all Gen1 Launch; Apr-Jun median 16.8% / 16.5% vs 3.88%; 4.35 vs 3.81 swaps per slot-day. Station-hour Gini 0.863 vs 0.857 / 0.842 | Identical (top-15 share of net excess 43.1%). Robustness: with a city x class baseline the 80.4% becomes 80.5%; by year it is 85.5% (2024) and 77.4% (2025). **But** each of the 36 hot-city Gen1 stations fails at its city's rate: observed/expected 0.95-1.05 (see Issue 4) | Numbers **confirmed**; "chronic hotspot" reading **corrected** |
| 8 | 3W vs 2W within station-month (2,037 strata): stockout RR 2.08 (2.06-2.11), abandonment 2.04 (2.01-2.08), failure 2.01 (1.99-2.03). Jul-Mar stockouts 5.47% vs 1.97% (RR 2.78); Apr-Jun RR 1.61. 83 zero-3W-slot stations take 57.1% of 3W attempts (212,621 completed), stockouts 6.41% vs 6.44%. Zero charged 3W packs: 1.57% of hours, 12.2% vs 6.3% | Identical. The Apr-Jun 1.61 is the within-station-month MH ratio; the crude ratio is 1.57. The Spearman of -0.002 was not re-derived | **Confirmed** |
| 9 | Fixed 90 Launch stations: attempts x1.65, failures x1.57. Same station-month RR 0.986 (0.975-0.996); Apr-Jun 0.989 (0.977-1.001). May 11.9% both years. Hot-city Gen1 May: 52.6 to 82.5 attempts per station-day, 19.1% then 19.0%. Busiest 15: RR 0.969 (0.949-0.988). May-25: network 9.5%, Launch 11.9%, Wave1 6.3%, Wave2 5.0% | Identical. Adding vehicle class to the strata gives RR 0.991 (0.981-1.001) | **Confirmed** |
| 10 | Station Spearman 0.843 / 0.670 / 0.833 (n=150). About 8 no_battery tickets per 100 stockouts (7.8 / 8.2 / 7.9). Tickets per 1K attempts 5.58 (Jan-24) to 10.49 (Jan-25); 13.29 in May-24. Jan-Mar category growth | Identical, but only for **station-linked** tickets (2,227 tickets have no station). With all in-window tickets: 6.04 to 11.06 per 1K, and 8.2 / 8.6 / 8.3 per 100 stockouts | **Confirmed** (state the denominator) |
| 11 | 139,490 corrected events: 38.0% in 19-22h after correction vs 2.3% raw (failures 41.0% vs 2.2%). Jul-Mar evening 3.69% vs 4.05%; 3W share 6.6% vs 13.4%; 2W 3.33% vs 3.34% | Identical. The shift is exactly 330 min, and no v3.2.0 event in the bug window is left uncorrected | **Confirmed** |
| R1 | About 22,577 fewer failures per summer (45,154 over two summers = 21.3% of failures); top 15 about 12,211 per summer | Identical arithmetic. By year: 17,996 (2024) and 27,158 (2025). An MH counterfactual for 2025 gives 27,356. Hot-city Gen3 and Gen1 sites carry similar load in Apr-Jun 2025 (84.1 vs 81.8 attempts per station-day) | **Confirmed** (quote the 2025-volume figure) |
| R2 | Rationale: "Empty cabinets lead stockouts by 2-3 hours (11.62% vs 7.27%)" | See 4b | **Refuted** |
| R4 | "Removes about 9,181 3W stockouts per 9 cool months" | 9,181 = 261,935 3W attempts x 3.50 pp across **all 12 cool months** in the data (Jan-Mar 2024 plus Jul 2024-Mar 2025). The latest 9-month cool season (Jul-24 to Mar-25) gives **7,778**; pro-rata, 9 months gives 6,885 | **Corrected** |

## Issues (most severe first)
1. **The hour-level mechanism is refuted (finding 4 last link, finding 5 "abandonment tracks inventory", the Rec 2 rationale, the fig 05 title).** The pooled link between zero stock and stockouts is confounded by hour of day and by day-level heat. Held like-for-like, the hourly stock level explains nothing:
   - **Same station-day.** Zero-stock hours have *fewer* 2W stockouts than stocked hours: MH RR 0.72 (0.69-0.74); abandonment 0.73. Splitting by evening vs other hours as well gives 0.99 (0.95-1.03); abandonment 0.93.
   - **Same city x generation x month x hour.** RR 1.00 (0.95-1.04); abandonment 0.96 (0.90-1.02).
   - **Carry-over.** Zero stock 0, 1, 2, 3, 4 or 6 hours earlier gives RR 1.00, 1.03, 1.06, 1.04, 1.03 and 1.06 within the same strata. The pooled lag gradient (6.88% at 0h rising to 13.34% at 6h) is pure hour-of-day confounding: afternoon empty hours are followed by the evening peak.
   - **Hot Gen1 days (peak cabinet >=45C, Apr-Jun).**
     - 2W stockouts run at **8.6% at every hour from 0 to 18h**, whether few cabinets are empty or many: 8.64% at 12-17h, when 34.7% of hours are empty.
     - They then jump to **16.9% at 19-22h, when only 1.3% of hours are empty**.
     - At 19-22h the stockout rate is 17.8%, 16.9%, 17.0% and 16.9% with 0, 1-2, 3-5 and 6 or more charged packs.
     - At 0-8h, with full cabinets (mean minimum 8.0 packs), the same stations run 8.29% stockouts on hot days vs 3.53% on other days.
   - **Fix.** Keep links 1-3 (heat, hot cabinet, slower charging, afternoon empty cabinets) as telemetry facts. Keep the day-level association (station-day GLM; 12.0% vs 2.3%). Drop "empty cabinets cause the stockouts", the "carry-over" and "this is why evenings are worst". Say instead: "on hot days Gen1 stations fail more at every hour, and twice as much again in the 19-22h peak; hourly cabinet stock does not predict which hours fail." Present this test as a trap handled; it earns rigour points.
2. **The Rec 4 impact number is mislabelled.** 9,181 covers 12 cool months, not 9. Use about **7,800 per cool season** (Jul-24 to Mar-25; 7,778), or "9,181 over the 12 cool months in the data".
3. **The headline and q2_04 title frame concentration unfairly.** "80.4% of excess at a segment with 11.6% of attempts" divides by all attempts. Jul-Mar holds 63% of attempts and has zero excess by construction.
   - Within Apr-Jun, hot-city Gen1 stations take **31.6% of attempts** for 80.4% of excess (about 2.5x, not 7x).
   - The 36 stations are the network's busiest third: **34.4% of all attempts and 47.7% of all failures** over 18 months. "Which carry only 11.6% of attempts" makes them sound small.
   - Reword to: "In Apr-Jun, the 36 hot-city Gen1 stations handle 32% of attempts but produce 80% of the excess failures."
4. **The "chronic hotspots" are not special stations.**
   - Against each station's own city x month x class rate, all 36 hot-city Gen1 stations sit at observed/expected 0.95-1.05. There is no extra variation between them (chi2 43.6 on 33 df, p = 0.10).
   - The top 15 have median O/E 1.00, vs 0.99 for the other 21.
   - They rank top only because they are bigger (median 14,750 vs 10,522 Apr-Jun attempts) and sit in hotter Jaipur and Delhi: Jaipur 6 of 10, Delhi 7 of 13, Hyderabad 2 of 13.
   - "They fail in both summers" is true of all 36.
   - **Fix.** Call them "the 15 largest hot-city Gen1 sites (by absolute excess)", not "chronic hotspots". Say the other 21 fail at the same rate, and prioritise by city (Jaipur and Delhi, RR about 3.2) times volume.
5. **The pooled generation RR hides the heat interaction.** The 2.54 mixes hot cities (Delhi 3.16, Jaipur 3.17, Hyderabad 2.80; pooled 3.05) with Pune 1.72, Bengaluru 1.19 and Mumbai 1.00. Report the hot vs cool split: it is direct evidence for the heat x generation story and the better input for Rec 1.
6. **"Every station sits on the same baseline" needs a qualifier: "after vehicle-class mix".** Raw Jul-Mar station rates range 2.80-5.30% (chi2 262 on 140 df, p = 2e-9). After class standardisation the excess spread is chi2 172 on 140 df (p = 0.035), with O/E 0.70-1.34.
7. **The heat alert is a fixed rule at about 42C.** State this and the playbook trigger becomes sharper: alerts fire about 6C after failures start climbing at 35-36C, and after the plateau at about 38C.
8. **Cosmetic issues.**
   - The weekday range is full-period.
   - The threshold is "~35C" in fig 03 but "~36C" in the text.
   - The hot-city Gen1 rate appears as 15.2% (2025) and 15.31% (both summers) without labels.
   - Fig 03 says "confined to" hot cities, yet Pune has a summer excess: Gen1 vs Gen3 RR 1.72, and "Apr-Jun other cities" holds 7.6% of excess (4,782).
   - Gen3 bars in fig 03 are 2025-only, while Gen1/Gen2 pool both summers.
   - The q2_07 CI reads "0.98-1.00", but the actual 0.975-0.996 excludes 1.
   - The q2_04 annotation mixes three different top-10% station sets.

## Figure review
| Figure | Exists / legible | Matches data? | Verdict |
|---|---|---|---|
| q2_01_queue_wait_patience | yes / yes | 3.5 / 10.8 min, KM 91% / 60% / 31%, median 16.6, flat 4.06-4.07 min | OK |
| q2_02_hour_month_heatmap (hero) | yes / yes | Peak May-24 19:00 16.0%; Jul-Mar flat | OK. "Flat at every hour" holds after class mix |
| q2_03_city_generation_heat | yes / yes | Bars and threshold curve match | Minor: "confined" overstates (Pune); Gen3 bars are 2025-only; 35 vs 36C |
| q2_04_concentration_lorenz (hero) | yes / yes | Gini and shares match | **Fix framing**: 12% denominator (Issue 3); annotation mixes three different top-10% sets |
| q2_05_heat_mechanism (hero) | yes / yes | Each panel's numbers match | **Misleading chain**: the title and 1-to-4 sequence imply empty cabinets cause stockouts, which is refuted at hour level (Issue 1). Panel 4 is a day-level association; retitle and caption accordingly |
| q2_06_vehicle_class_3w | yes / yes | 2.1x / 2.0x; 57% at zero-slot stations | OK |
| q2_07_same_station_yoy | yes / yes | Rates and RR match | OK (show the CI as 0.975-0.996) |
| q2_08_tickets_vs_failures | yes / yes | rho 0.83; monthly lines | OK |

## Methodology review
- **Confounding / Simpson's paradox.** Generation, heat, 3W and year on year are well stratified (MH within city-month and station-month; GLM with fixed effects and clustered SE). The hourly stock-to-stockout link was **not** stratified. The pooled association reverses within station-day (RR 0.72), a Simpson's pattern the analysis missed.
- **Seasonality.** Same-season comparisons are used throughout. Apr-Jun as the "hot" window is data-driven and differs from SEASON_OF_MONTH (Summer = Mar-May); say so.
- **Survivorship.** The fixed Launch panel and fixed busiest-15 set are good. The single decommissioned station (STN-DEL-051, Gen2) is negligible.
- **Censoring.** KM is used correctly, with independent-censoring and cap (30 / 40 min) caveats.
- **Denominators.** Attempts are used consistently. Ticket rates use station-linked tickets only; state this.
- **Timestamps.** The corrected hour is used everywhere. Verified: shift exactly 330 min, no v3.2.0 event in the window left uncorrected.
- **Telemetry nulls.** Missing rows are excluded and means are null-aware. However, the lag analysis shifts by row within station, so it misaligns hours where missing rows were dropped. Use a time-based self-join.
- **Causal over-claim.**
  - The hour-level mechanism and "this is why evenings are worst" (Issue 1).
  - "Chronic hotspots" (Issue 4).
  - The Gen3 counterfactual is cross-sectional. Load is comparable, but Rec 1 should still read as "if Gen3 performance transfers".
- **Multiple testing.** Many MH tests, but the effects are large; no concern.
- **Cost-model consistency.** Q2 does not use money, so there is no conflict with the brief.

## Missed angles
1. **Turn the hour-level null into a strength.** Showing that hourly stock does not predict failures within the same station-day is a trap handled. Heat acts as a day-level state at Gen1 stations, so the lever is charger throughput and generation, not hourly inventory moves. This also weakens "pre-charge / raise inventory targets" as an evidenced lever: on hot days, stockouts run at 8.3% at night with full cabinets.
2. **The hot-day evening doubling is the operational window.** On hot Gen1 days, 2W stockouts are 8.6% from 0-18h and 16.9% at 19-22h. Rider redirection to Gen3 or cooler sites in that window fits better than restocking.
3. **Failure cost: there is no catch-up behaviour.**
   - After a service failure, the same rider completes a swap within 60 min 4.5% of the time, vs 4.6% after a completed swap.
   - Within 24 h it is 55.7% vs 57.2%; the median gap is 22.7 h vs 22.3 h.
   - Each failure therefore looks like a lost transaction rather than a deferred one. Value it with the brief's cost model (revenue per swap; cm1 context).
4. **Q6 link.** New riders whose first attempt failed return within 30 days at the same rate as those who succeeded (99.8% vs 99.8%; n = 646 vs 13,084). The data show no short-run churn penalty from a failure; Q6 should test longer horizons before blaming failures for retention.
5. **The problem is diluting, but not shrinking.** Hot-city Gen1's share of summer excess fell from 85.5% (2024) to 77.4% (2025) as its share of summer traffic fell from 41.3% to 27.4%. Absolute avoidable failures still grew from 17,996 to 27,158.
6. **Pune.** Its Gen1 summer excess (Gen1 vs Gen3 RR 1.72) puts the 4 Pune Gen1 sites on the retrofit list. It also confounds the Q5 pricing DiD in Apr-Jun.
