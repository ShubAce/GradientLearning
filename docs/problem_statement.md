# VoltRelay Energy — Gradient Learnings Data Analytics Hackathon (15 Nov 2026)

Dataset (Google Drive folder): https://drive.google.com/drive/folders/1qsjGWrgmEvOaf2Is2fh7RgVEkKd1V3df?usp=sharing
Local raw copy: `Hackathon Data Set _ Gradient/*.csv` (uncompressed). Generation seed: 59500.
Synthetic, seeded discrete-event simulation: station capacity, battery degradation, thermal charging behaviour,
pricing response and rider churn all emerge from the same mechanics and are mutually consistent.

## Hackathon logistics
- One day: 15 Nov 2026. Submission deadline **15 Nov 2026, 11:59 PM IST**.
- Scoring: Problem understanding 15% | Data Cleaning & Analytical 20% | Insight Quality & Evidence 25% |
  LinkedIn content + engagement 15% | Actionable Recommendations & Visualization 25%.
- LinkedIn bonus: post updates tagging Gradient Learning (+6 to +10 points by engagement rank).

## Business background
VoltRelay Energy runs a battery-swapping network for electric 2W and 3W across six Indian cities: Bengaluru, Delhi NCR,
Hyderabad, Pune, Mumbai, Jaipur. Customers are gig/logistics riders (food delivery, quick commerce, e-com logistics,
cargo, bike-taxi). Unmanned swap cabinets at kirana stores, fuel pumps, mall parking. Swap takes ~2 minutes.
Riders pay per swap or via a prepaid pack; fleet-affiliated riders are billed to their employer at a contracted discount.

## The business challenge
~18 months of history (Jan 2024 – Jun 2025): every swap attempt, hourly station telemetry, battery lifecycle, support
tickets, fleet partner contracts, daily weather/calendar context. Over this period VoltRelay:
- expanded its station network in **two waves**,
- **changed its base pricing**,
- **piloted peak/off-peak pricing in two cities**,
- brought on a **new battery supplier**,
- **renegotiated its largest fleet contract**.
Completed swaps and revenue both grew substantially, but **service failures rose faster, new-rider retention fell, and
per-swap profitability eroded** — and no consolidated analysis has connected these threads. Leadership wants to know
what is actually driving these outcomes before committing its next operating budget.

## Objective
Determine how the network has performed, what factors are associated with service failures, rider churn and eroding
margins, where performance varies most (stations, geography, equipment, partner segment), and what VoltRelay should
prioritize. Evidence over assumptions. Causation not required; a well-reasoned, well-evidenced case is.

## Core questions (primary story)
1. **Network performance over time** — completed swaps, revenue, failure rates, contribution margin per swap. Do they
   tell the same story, or do some metrics move in different directions?
2. **Service failures & CX** — how do queue wait, failed attempts, abandoned swaps relate to station, hour of day,
   season, vehicle class? Evenly spread or concentrated?
3. **Station & geographic patterns** — charger generation, location type, commissioning timing, uneven city
   distribution vs. service performance, battery turnaround time, complaints.
4. **Battery & equipment** — battery health (SoH, charge cycles, supplier, manufacturing lot) vs. delivered range and
   swap frequency. Do any equipment/supplier cohorts stand out?
5. **Pricing & partner economics** — base price change, peak/off-peak pilot, individual fleet partner contract terms
   vs. swap timing, revenue, contribution margin. Same effect on every rider segment?
6. **Root cause of retention** — most important factors associated with new riders not returning. Distinguish
   **primary drivers** vs **secondary/contributing** factors. More than one factor likely contributes.

## Optional deep dives
- **Budget decision**: four proposals — (a) more stations, (b) more batteries, (c) network-wide pricing rollout,
  (d) long-term exclusive with largest fleet partner. Which does the evidence support; what instead/in addition?
- **Expansion wave effectiveness**: did the two waves improve service where riders needed it, or where sites were
  easiest to open? Network averages vs. busiest stations may disagree.
- **Fleet partner value**: contribution margin by partner after energy, battery wear, station costs. Is the largest
  partner the most valuable?
- **Support ticket text** (English + Hinglish): does free text add to category/resolution fields? Categories are
  not always accurate.
- **Anomalies**: stations, battery cohorts, time windows that are unusual (extreme failure, unusual degradation,
  firmware/connectivity irregularities).

## Deliverables
1. **Google Colab notebook**: data understanding, cleaning, EDA, analysis (core + chosen optional), visualizations,
   key findings. Must run top to bottom without errors.
2. **Analysis report**: problem understanding, analytical approach, key insights, supporting visualizations,
   business findings, actionable recommendations.
3. **3-minute video**: business problem → approach → most important insight(s) → recommendations, presented to
   VoltRelay ops & CX leadership.

Skill note: advanced techniques welcome where they genuinely fit (statistical testing, difference-in-differences for the
pricing pilot, cohort/retention curves, text analysis). Judges evaluate quality of reasoning and evidence, not a match
to an expected answer.

## Files
| File | Rows | Grain |
|---|---|---|
| swap_events.csv | 3,877,013 | 1 row per swap attempt (completed or not) |
| station_hourly_status.csv | 1,487,712 | 1 row per station per hour |
| riders.csv | 20,000 | 1 row per registered rider |
| batteries.csv | 6,500 | 1 row per battery pack |
| support_tickets.csv | 44,000 | 1 row per support ticket |
| stations.csv | 152 | 1 row per station |
| city_daily_context.csv | 3,282 | 1 row per city per day |
| fleet_partners.csv | 12 | 1 row per B2B fleet customer |

## Data dictionary (key points)
**swap_events**: event_id (PK); rider_id; station_id; event_ts; event_type ∈ {swap_completed, failed_no_charged_battery,
abandoned_queue, cancelled_by_rider, failed_system_error}; attempt_seq (always 1); queue_wait_sec (arrival → completion
or exit); battery_in_id (missing when rider never inserted a pack); battery_out_id (only for swap_completed);
soc_in_pct/soh_in_pct (missing unless a pack was inserted); soc_out_pct/soh_out_pct (missing unless completed);
km_since_last_swap (range proxy; contains odometer-reset outliers); tariff_code ∈ {STD, PEAK, OFFPEAK, PREPAID,
PARTNER, PROMO_FREE}; list_price_inr; discount_inr; amount_charged_inr (net revenue); payment_mode ∈ {upi, wallet,
prepaid, partner_invoice}; energy_to_recharge_kwh (missing unless completed; grid energy to recharge the returned pack);
station_firmware (firmware active at event time); sync_mode ∈ {realtime, offline_batch}.

**station_hourly_status**: station_id + hour_start (PK); charged_2w_avg / charged_2w_min; charged_3w_min (missing where
no 3W slots or telemetry missing); packs_charging; packs_quarantined (thermal/fault); chargers_online; ambient_temp_c;
cabinet_temp_c; avg_charge_minutes (turnaround); outage_minutes; grid_kwh; telemetry_status ∈ {ok, partial, missing}.

**riders**: rider_id; partner_id (blank = independent); vehicle_class 2W/3W; vehicle_model (**correlates with fleet
affiliation, not a mechanical cause on its own**); home_city (inconsistent spellings); home_zone; signup_date;
signup_channel ∈ {partner_onboarding, referral, app_store, field_agent}; plan_type ∈ {pay_as_you_go, prepaid_pack,
partner_billed}; declared_shift (self-reported, does not always match actual activity); age_band; kyc_verified.

**batteries**: battery_id; pack_type 2W_2.1kWh / 3W_4.8kWh; supplier ∈ {Cellora, Amptek, Kyron}; manufacturing_lot;
manufacture_date; commission_date; rated_capacity_kwh; purchase_cost_inr; initial_soh_pct; bms_firmware ("included for
completeness"); current_soh_pct (as of 30 Jun 2025); retired_date / retirement_reason (mostly blank).

**support_tickets**: ticket_id; rider_id; station_id (optional); battery_id (optional); created_ts; channel ∈ {app_chat,
call_center, whatsapp, partner_escalation}; category (agent-assigned, not verified); rider_comment (English/Hinglish);
priority low/medium/high; resolution_status resolved/unresolved/duplicate; resolution_hours (blank if unresolved);
csat_score 1–5 (mostly missing, NOT missing at random).

**stations**: station_id; city; zone; lat/long; location_type ∈ {commercial_hub, market, tech_park, residential,
transit_hub, highway_fuel_pump}; host_type ∈ {kirana, fuel_retailer, mall_parking, owned_lot}; commissioned_date;
decommissioned_date; expansion_wave ∈ {Launch, Wave1_2024H2, Wave2_2025H1}; charger_generation Gen1/Gen2/Gen3;
slots_2w / slots_3w; inventory_target_2w / _3w; monthly_rent_inr; monthly_maintenance_inr; grid_tariff_inr_kwh;
connectivity_tier good/fair/poor; firmware_version; firmware_updated_date; competitor_within_1_5km_since.

**city_daily_context**: city, date; max/min temp; rainfall_mm; heat_alert; flood_disruption; festival_or_event;
is_public_holiday; grid_outage_hours; competitor_promo_active; ecommerce_sale_event.

**fleet_partners**: partner_id; partner_name; partner_segment ∈ {quick_commerce, food_delivery, ecom_logistics,
cargo_3w, bike_taxi}; vehicle_class; contract_type pay_per_swap / volume_tiered; contract_start_date; discount_pct;
amendment_date / discount_pct_after_amendment; peak_surcharge_billable Y/N (pays pilot peak surcharge?);
payment_terms_days; cities_active (pipe-separated).

## Relationships
fleet_partners → riders → swap_events → stations; swap_events → batteries (battery_in_id, battery_out_id);
riders → support_tickets → stations / batteries (optional); stations → station_hourly_status;
stations.city + date → city_daily_context (matched by city and date of event_ts / hour_start).

## Documented data-quality issues (present on purpose — must be handled explicitly)
1. ~94% of attempts complete; pricing/battery fields populated only for certain event types — treat non-completed
   attempts deliberately in revenue/delivery analyses.
2. Missing telemetry in station_hourly_status where telemetry_status = partial/missing, concentrated at poor-connectivity
   stations — aggregate deliberately, never treat blanks as zero.
3. **Timestamp bug**: swap_events at stations on firmware **v3.2.0** between **2025-03-10 and 2025-04-14** have event_ts
   written ~**5h30m earlier** than true local time (timezone bug) — distorts hour-of-day analysis if not fixed.
4. Inconsistent riders.home_city spellings (variants of Bengaluru, Delhi NCR, Mumbai...) — standardize.
5. Near-duplicate swap_events (offline_batch sync retries, concentrated at poor-connectivity stations): same rider,
   station and batteries, timestamp seconds to a couple of minutes apart — decide whether/how to de-duplicate.
6. Outliers: km_since_last_swap negative or implausibly large (odometer resets); charge readings slightly > 100%
   (sensor drift).
7. csat_score mostly missing and MNAR — quickly-resolved tickets more likely to have a score → naive mean biased.
8. Ticket category agent-assigned, not verified — battery range/condition issues sometimes filed as "other" or
   "app_issue"; comments sometimes blame the vehicle rather than the battery.
9. Two inactive test stations (station_id starting **STN-TST**) with zero-value transactions at unusual hours —
   decide whether to exclude from network analysis.
