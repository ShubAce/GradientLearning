# Shared analysis brief — VoltRelay hackathon (read fully before starting)

Project root: `f:\Data Analytics Gradient Learning` (Windows; use Git-Bash or PowerShell; Python = Anaconda 3.12 `python`).
Problem statement + judging rubric: `docs/problem_statement.md` (read it). Deadline is tonight; work efficiently.
Judging: Problem understanding 15% | Data cleaning & analytical rigor 20% | Insight quality & evidence 25% |
Actionable recommendations & visualization 25% | LinkedIn 15%. Judges reward **evidence-based reasoning**, catching
traps (confounding, seasonality, Simpson's paradox, survivorship, censoring, MNAR), and **quantified, actionable** recommendations.

## Shared code and clean data (DO NOT modify src/common.py, src/clean.py or anything in data/)
`src/common.py` — import with `sys.path.insert(0, str(ROOT/'src')); from common import *`. Gives paths (ROOT, CLEAN, FIG,
TAB, FIND), constants (BASE_PRICE_CHANGE=2024-07-01, PILOT_START=2024-10-01, PILOT_CITIES=[Bengaluru, Pune],
ZIPDROP_ID='FP-03', ZIPDROP_AMEND=2024-11-01, COMPETITOR_ENTRY=2025-01-15, EOL_SOH=70, SEASON_OF_MONTH, CITIES),
lazy loaders `events(cols)`, `hourly(cols)` and eager loaders `stations() riders() batteries() tickets() city_daily()
partners() station_month()`, plus plotting helpers `set_style()`, `PALETTE`, `SERIES`, `CITY_COLORS`,
`SUPPLIER_COLORS`, `save_fig(fig, name)`, `fmt_inr(x)`.
`src/clean.py` produced `data/clean/*.parquet`; its audit log is `outputs/tables/data_quality_log.json`.

### data/clean/events.parquet (3,814,982 swap attempts, test stations removed) — key columns
event_id, rider_id, station_id, event_ts (**corrected** local time), event_ts_raw, ts_corrected (bool: firmware v3.2.0 bug
window 2025-03-10..2025-04-14 shifted +5h30m), date, month (Date, 1st of month), week, hour (corrected), dow (1=Mon),
season (Winter Dec-Feb, Summer Mar-May, Monsoon Jun-Sep, Post-monsoon Oct-Nov), event_type, is_completed, is_stockout
(failed_no_charged_battery), is_abandoned (abandoned_queue), is_system_error, is_cancelled, is_service_failure
(stockout|abandoned|system_error), queue_wait_sec, vehicle_class (2W/3W), city, zone, battery_in_id, battery_out_id,
soc_in_pct, soh_in_pct, soc_out_pct, soh_out_pct (clipped at 100), km_since_last_swap (null if invalid: <0 or >200 km),
km_invalid, reading_clipped, tariff_code (STD/PEAK/OFFPEAK/PARTNER), list_price_inr, discount_inr, amount_charged_inr,
energy_to_recharge_kwh, station_firmware, sync_mode, location_type, host_type, expansion_wave (Launch/Wave1_2024H2/
Wave2_2025H1), charger_generation (Gen1/2/3), connectivity_tier, grid_tariff_inr_kwh, commissioned_date,
competitor_since, competitor_active (bool, event after a competitor opened within 1.5 km), partner_id (null =
independent), rider_vehicle_class, plan_type (pay_as_you_go/prepaid_pack/partner_billed), signup_date, signup_channel,
rider_home_city, is_independent, supplier_out (supplier of issued pack), supplier_in (supplier of returned pack),
revenue, energy_cost, wear_cost, fixed_cost, cm1, cm2 (all 0 for non-completed attempts).

**Cost model (use it; do not invent another)** — per completed swap:
revenue = amount_charged_inr; energy_cost = energy_to_recharge_kwh x station grid tariff; wear_cost = issued pack's
purchase_cost x (SoH points lost per cycle, from that pack's own SoH trajectory) / (initial SoH − 70% EOL);
fixed_cost = station rent+maintenance (pro-rated by active days) / completed swaps at that station-month;
**cm1 = revenue − energy − wear (variable contribution margin); cm2 = cm1 − fixed_cost**. Levels depend on the wear
assumption (no salvage value, 70% EOL) — always discuss trends and differences, and where you quote levels, say
"under our usage-based battery-depreciation assumption". Wear by pack: Cellora/Amptek 2W ≈ ₹32/swap (≈1,005 cycles to
EOL), **Kyron 2W ≈ ₹104/swap (≈308 cycles)**, 3W ≈ ₹76–79/swap.

### Other clean tables
- station_month.parquet: station_id × month with active_days, fixed_cost, attempts, completed, stockouts, abandoned,
  system_errors, cancelled, q_wait_completed_mean/p90, revenue, energy_cost, wear_cost, cm1, cm2, unique_riders,
  fixed_cost_per_swap, swaps_per_slot_day + station attributes (city, wave, generation, location, host, connectivity, slots).
- station_hourly.parquet (1.46M rows, test stations removed): telemetry + city, charger_generation, expansion_wave,
  connectivity_tier, location_type, slots/inventory targets, date, hour, month, season. **Blanks are null — never treat
  as zero**; filter telemetry_status=='ok' or use null-aware means and report coverage.
- riders.parquet: + home_city_clean (6 cities), zone_city, is_independent, first/last attempt ts, first_swap_ts,
  last_swap_ts, n_attempts, n_swaps. 16,400 riders signed up in-window (>=2024-01-01); 3,600 before.
- batteries.parquet: + n_cycles (times issued), first_seen, soh_loss_per_cycle (pack's own slope), loss_per_cycle_used,
  wear_cost_per_cycle, expected_cycles_to_eol, used_before_commission, vehicle_class.
- tickets.parquet: + is_test, station_city, created_date, month, in_window, has_csat.
- city_daily.parquet (join on city + date), fleet_partners.parquet, stations.parquet (+ is_test).

## Facts already established (verify if central to your question, but do not waste time rediscovering)
- Volume: completed swaps Jan-24 103.5K → Jun-25 303.2K (2.9x); revenue ₹6.2M → ₹19.7M/month. Stations 90 real at
  Launch; Wave1 (2024H2) +30, Wave2 (2025H1) +30; STN-DEL-051 decommissioned 2025-05-20.
- Service-failure rate is strongly seasonal: ~3.9% baseline, spikes Apr–Jun (May-24 11.9%, May-25 9.5%).
  H1-2025 vs H1-2024: attempts 2.40x, swaps 2.43x, failures 2.07x, revenue 2.63x. The naive Jan-24 vs Jun-25
  comparison (failures 6.3x vs swaps 2.9x) is a **seasonal artefact** — check same-season and per-station comparisons;
  network averages may hide deterioration at the busiest stations (new low-utilisation stations dilute the average).
- Pricing: base price 2024-07-01 (2W list 60→65, 3W 100→110, all tariffs). Peak/off-peak pilot from 2024-10-01 in
  Bengaluru & Pune only: PEAK ≈ 12:00–22:59 charged list + ₹15 (discount_inr=0); OFFPEAK charged list − ₹8 (the ₹8 is
  NOT recorded in discount_inr). PARTNER tariff = list − contracted discount (discount_inr). Whether partner riders pay the
  peak surcharge (peak_surcharge_billable=Y) must be checked in the data. ZipDrop (FP-03, quick_commerce, 2,381 riders,
  ~23% of swaps) discount 12%→28% from 2024-11-01 and peak surcharge NOT billable. FeastFly FP-01 has the most riders.
- CM per swap (cm1): ≈₹0 (H1-24) → +₹6.8 (Jul–Aug-24, price rise) → −₹8.3 (Nov-24) → −₹5.0 (Jun-25). Wear cost per
  swap jumps ₹37.5 → ₹53 from Sep-24 as Kyron's share of issued packs rises 3% → 25%. Energy cost per swap falls
  ₹22 → ₹17 (degraded packs hold less energy); fixed cost per swap falls ₹20 → ₹12 (utilisation).
- Batteries: Kyron 2W (1,385 packs) first used ~Q3-2024, SoH falls ~3x faster than Cellora/Amptek (0.095 vs 0.029 pts per
  cycle); all 1,385 + 53 Kyron 3W retired on 2025-06-15 (end_of_life, SoH<70). Median km per pack drops with SoH
  (≈70 km at 98% SoH → 53 km at 82% → 39.5 km for Kyron 2W in Q2-25). Kyron 3W behaves like the others.
  batteries.csv commission_date/lot metadata contradicts usage (5,154 packs used before recorded commission) — use
  first_seen for age. Battery IDs in swaps are not physically tracked (same pack issued in two cities within minutes),
  but each ID's SoH trajectory is internally consistent → per-pack SoH analysis is valid, per-pack location is not.
  bms_firmware is "included for completeness" (likely no effect — test rather than assume).
- Data-quality facts: test stations STN-TST-01/02 hold 62,031 events incl. 59,134 paid swaps (₹3.93M) — contradicts the
  documentation's "small number of zero-value transactions"; excluded. Firmware bug fixed (139,490 events; v3.2.0 mean
  event hour 10.40 → 15.13 after fix vs 15.18 for v3.2.1). Documented offline-sync near-duplicates: 0 found under the
  documented rule (offline_batch rows are legitimate; 591K rows). payment_mode = 'partner_invoice' on 100% of rows
  (unusable). km outliers: 7,687 negative + 7,600 >200 km nulled. 9,680 SoC/SoH readings >100 clipped. CSAT missing
  65.6%, response rate 44% for fastest-resolved quartile vs 28% for slowest (MNAR). Ticket comments: only 89 distinct
  templated strings (English + Hinglish).
- riders.vehicle_model correlates with fleet affiliation (organisers warn it is not a mechanical cause); declared_shift
  is self-reported and may not match actual activity.

## Rules for your work
1. **Memory**: the machine has ~4 GB free RAM shared with ~8 concurrent analyses. Always `events([...needed cols])`
   (lazy, projected), filter/aggregate in polars before `.collect()`, never collect all 63 columns, never convert the full
   event table to pandas. Rider-level or station-hour aggregates are fine. Prefer `.collect(engine="streaming")` for big scans.
2. **Script**: put all your code in ONE runnable script `analysis/<id>_<slug>.py` written in jupytext "percent" format
   (`# %% [markdown]` cells with short what/why narrative, `# %%` code cells) so it can be dropped into the Colab notebook.
   It must run top-to-bottom with `PYTHONIOENCODING=utf-8 python analysis/<file>.py` in < 4 minutes, print the key
   numbers it relies on, and save every figure/table it references. Re-run it after every change; never report a number
   your script does not print.
3. **Figures**: matplotlib only (no kaleido for plotly). Call `set_style()`; use PALETTE/SERIES/CITY_COLORS/
   SUPPLIER_COLORS; save with `save_fig(fig, '<id>_<nn>_<slug>')` → outputs/figures/. Title = the insight in plain words
   (e.g. "Kyron packs lose health 3x faster than other suppliers"), subtitle/axis labels with units, annotate the key
   numbers, readable at slide size (≥ 10pt). 4–8 figures; mark 1–2 as hero figures (the ones that belong on a slide).
4. **Tables**: save supporting tables as CSV `outputs/tables/<id>_<slug>.csv`.
5. **Findings file**: write `outputs/findings/<id>.md` with sections: Headline (1 sentence + key number) · Key findings
   (numbered, each with exact numbers, denominators, periods and the figure that shows it) · Primary vs secondary drivers
   (where relevant) · Method notes (definitions, tests, models, why) · Caveats & traps handled · Recommendations
   (specific, owner-level, with quantified expected impact where the data allows) · Figures list with captions.
6. **Rigour**: same-season comparisons for anything seasonal; control for city/season/vehicle class/segment mix when
   comparing groups; report effect sizes with confidence intervals or significance tests where they fit; check for
   Simpson's paradox and survivorship; state when a pattern is association not causation. Do not over-claim.
7. Hypotheses in your brief are things to TEST, not conclusions to confirm. If the data says otherwise, say so clearly.
8. Do not edit files owned by other analyses (other ids' scripts/figures/findings).
