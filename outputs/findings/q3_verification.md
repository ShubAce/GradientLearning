# Q3 verification: station and geographic patterns, expansion waves, competitor entry

I checked q3 independently. My script is `analysis/verify/q3_verify.py`. It was written from scratch, re-derives everything from `data/clean` through the `src/common.py` loaders, runs in about 7 s and prints `CHECK <claim>: analyst X | ours Y` lines. Station-month counts were rebuilt from raw events and match `station_month.parquet` exactly (attempts, and failures = stockout + abandoned + system error).

## Bottom line
- **Almost every number reproduces exactly.** This covers 55 checks: the headline, thermal derating, heat → stockouts, the attribute models, tickets, zone placement, wave economics, DiD A/B, the competitor DiD and the counterfactual.
- **One sub-claim is wrong:** "Wave 2 leaned toward lower-failure zones" is refuted.
- **One recommendation is contradicted by the data:** "stop adding highway-fuel-pump sites".
- **The headline has a causal non-sequitur,** and one hero figure title is mislabelled.

**needs_fix = true.**

## Verdict table

| # | Claim (analyst) | Verdict | Recomputed |
|---|---|---|---|
| H1 | 36 hot-city Gen1 stations produced 52.6% of Apr-Jun 2025 failed attempts (40,723 of 77,445) from 27.4% of attempts | confirmed | 40,723 of 77,445 = 52.6%; 27.4% of attempts |
| H2 | Network 9.52% → 7.92%; legacy 9.52% → 9.55%; 23 hotspots 16.70% → 16.61% (Apr-Jun 24 → Apr-Jun 25) | confirmed | identical. Also: 36 Gen1 hot-city 15.48 → 15.20; busiest 9 stations 10.09 → 9.67; new 5.08; off-season 3.85 / 3.93 / 3.92 |
| H3 | "Both waves followed the existing footprint, **so** the rate fell only through dilution" | corrected | Dilution is confirmed, but the "so" does not hold. Failure rates at the hotspots do not depend on load. Hotspot attempts rose +50.9% (3.51 → 5.30 per slot-day) at an unchanged 16.7% / 16.6% rate. A within-station Gen1 model (station + month FE) gives log-utilisation −0.03 pp (p = 0.94). No siting pattern could have lowered these rates. |
| 1 | Thermal hinge: Gen1 88 min, +9.46/°C above 41.5 °C, 169 min at 50 °C (52%), R² 0.884; Gen2 73.6, Gen3 43.7 at 50 °C | confirmed | 0.25 °C grid: Gen1 88.0, +9.16/°C above 41.25 °C, 168.2 at 50 °C (52.3%), R² 0.885. The slope depends on the grid; the 50 °C value is robust (empirical mean at 49-51 °C is 165.6). Gen2 73.6 (0.96/°C), Gen3 43.7 (0.18/°C). |
| 1b | Summer Gen1 hot share 19.8%; 131.1 vs 88.0 min; 15:00 50.0%, 06:00 0%, winter 0%; cities DEL 26.7 … MUM 5.3; quarantines 1,315 vs 16 | confirmed | All identical. The quarantine figures are summed hourly counts, i.e. **pack-hours**, not distinct packs. |
| 2 | 8+ hot hours: Gen1 stockouts 12.2% vs 4.8% (failures 18.1 vs 7.5); 0 hot hours 2.35 vs 2.32; +0.81 pp per hot hour (0.78-0.85), n = 25,170 | confirmed | 12.18 vs 4.81; 18.12 vs 7.51; 2.35 vs 2.32; +0.813 (0.775-0.851), n = 25,170. Control stations average 8.9 hot hours on those same city-days, so this is Gen1 vs Gen2/3 *under the same heat*. |
| 3 | Station-month WLS: n 2,003; Gen1 +1.35, Wave2 −2.52, DEL +2.64, JAI +3.05; R² 0.552 → 0.945; slopes 0.359 / 0.154 / 0.069; Gen1 main −0.25; Wave2 +0.20 n.s.; city ≤ 0.29; loc/host/conn ≤ 0.23 n.s.; queue 243.3-244.2 s; sys-err 0.28-0.31% | confirmed | All identical (loc/host/conn max \|0.22\|, min p 0.13). Robustness: station FE gives slopes 0.360 / 0.160 / 0.084. Month FE + heat × generation alone give R² 0.942. |
| 4 | Tickets +1.005 per 1K swaps per +1 pp failure (0.961-1.049); mean 11.48; Gen1 21.38 vs 13.21 / 13.49; Gen1 +1.26; DEL/JAI/HYD +3.5 to +5.2 | confirmed (wording wrong) | Identical (DEL +4.59, JAI +5.16, HYD +3.50). **"One-for-one" is wrong.** The slope means about **1 extra ticket per 9.9 extra failed attempts**. |
| 5a | Hot cities: 67.8% of H1-24 failures, 49.0% of riders, 48.8% of legacy slots, 50.4% of new slots; top need tercile 49.1% of failures vs 35.6% of Wave-1 slots; Wave-1 ρ 0.15 (p 0.46); ρ with counts 0.40 / 0.43; 3 unserved zones (392 riders) got 1 + 2 stations | confirmed | All identical (158 slots in the top tercile). |
| 5b | "Wave 2 even leaned toward lower-failure-rate zones" (ρ −0.45, p 0.03) | **refuted** | This comes from measuring need on the off-season Jul-Dec 2024 rate. Zone rates there are flat noise: mean 3.91%, SD 0.16 pp (vs 2.66 pp in H1-24), and ρ with the H1-24 rate is −0.09. On the same-season Apr-Jun 2024 rate, ρ = 0.05 (p 0.82); on H1-24, ρ = 0.06 (p 0.76). Wave 2 was need-blind, like Wave 1: the highest-need tercile held 54.5% of Apr-Jun 24 failures and got 35.5% of Wave-2 slots. |
| 6a | 27/30 Wave-1 stations are fuel pumps (14) or residential (13); legacy failure share 1.5 / 5.7 / 47.3 / 37.5; Apr-Jun 25 swaps/slot-day and fixed cost/swap 4.54/₹11.33, 3.74/₹14.73, 5.51/₹10.71; Wave 1 82-85% of legacy; unserved zones 11.56 vs 4.56 (2.54x); ₹5.05 lakh/quarter; 40.0% of cost for 37.7% of swaps; rent ₹18,649 vs ₹17,510; 16 vs 10 stations > ₹20; CM1/CM2 by wave | confirmed (2 interpretation problems) | All identical. (i) The 1.5% / 5.7% failure shares only mirror slot shares (1.9% / 5.9%). Legacy failure *rates* by site type are equal: fuel pump 6.1%, residential 7.1%, commercial 7.0%, market 7.2%. (ii) **All 3 unserved-zone stations are highway fuel pumps** (n = 3: 21.57, 9.39, 8.73 swaps/slot-day). |
| 6b | Wave 2 at 116-131% of same-city legacy utilisation | corrected | Months 1-3 (the figure's n ≥ 20 rule): **116-123%**. The 131% is the partial commissioning month (n = 16). On attempts (demand, not stockout-capped completions): 111-116%. Without its 2 unserved-zone fuel pumps, Wave 2 runs 4.97 swaps/slot-day at **₹11.90/swap, above legacy ₹11.33**. |
| 7 | DiD A −0.26 pp per +1.0 intensity (−0.46 to −0.05), median intensity 0.60; DiD B +0.145 (−0.13 to +0.42), 74 vs 16 stations | confirmed | Identical (p 0.013; 5 control zones). Heat changed little year-on-year (legacy Δhot +0.30 pp), so the same-season comparison is like-for-like. |
| 8 | Competitor: 8 vs 21 stations, 34 weeks; swaps −3.9% (−10.7 to +3.3), p 0.28; permutation p 0.22; riders −1.3% (p 0.72); rider level 665 vs 1,921, −0.023, +0.5 pp; Bengaluru Wave 2 3.84 vs 5.89 | confirmed | −3.9% (−10.7 to +3.3), p 0.28; riders −1.3% (p 0.72); permutation p 0.24 (different seed, Monte-Carlo noise); rider level 665 vs 1,918, −0.033 (−0.193 to 0.128), still active +0.9 pp (p 0.47), same conclusion; 3.84 vs 5.89. Robust to VoltRelay's own Wave-2 openings in the treated zones (−3.7%, p 0.29). The event-study drift disappears with station-specific trends (+1.7%, p 0.58). |
| R1 | Counterfactual: Gen2-like −7.36 pp, 19,718 avoided (25.5%); Gen3-like −10.36 pp, 27,751 (35.8%) | confirmed | A naive same-city Apr-Jun 25 comparison agrees. Hot-city Launch Gen2 runs at 8.01%: −7.19 pp, 19,266 (24.9%). Hot-city Gen3 (19 stations) runs at 5.06%: −10.14 pp, 27,158 (35.1%). |
| R3 | "Stop adding highway-fuel-pump and residential sites" | **refuted (as a blanket rule)** | Fuel pumps in unserved zones run 11.56 swaps/slot-day vs 3.05 in served zones (n = 16). Wave-2 fuel pumps overall run 6.95 swaps/slot-day at ₹8.82/swap, the best Wave-2 site type. Across all 60 new stations, utilisation tracks zone demand (+5.17 swaps/slot-day per log-unit of resident riders per slot, p < 0.001); fuel pump vs market is −0.50 (p 0.60) once demand is controlled. Site type matters only inside already-served zones (fuel pump −1.78, residential −1.69 vs market, p ≤ 0.03). |

## Issues, ranked by severity

1. **The Wave-2 placement sub-claim is an artefact (refuted).** Wave-2 need is measured on the Jul-Dec 2024 zone failure rate, the off-season, when every zone sits at about 3.9%. The ρ = −0.45 therefore correlates new slots with noise. Use the same-season Apr-Jun 2024 rate (or H1-24): ρ = 0.05 (p 0.82).
2. **Recommendation 3 contradicts the analyst's own best evidence.** The 3 "previously unserved zone" stars, which the recommendation cites, are all highway fuel pumps. Location (unmet demand) is the driver, not site type. Wave 1 underperforms because 13 of its 14 fuel pumps and all 13 residential sites went into zones that were already served.
3. **The headline causal link is over-claimed.** "Followed the existing footprint, *so* … only dilution" implies better siting would have cut hotspot failure rates. The data show these rates do not respond to load: legacy +48% and hotspots +51% more attempts at a flat rate, and the within-Gen1 load coefficient is about 0. The problem is heat × Gen1 equipment, which supports Recommendation 1 even more strongly.
4. **The Fig 4 hero title mislabels Wave 2.** "Wave-2 market sites Rs 10.7" is the all-Wave-2 figure. Market sites alone are ₹10.16/swap (5.46 swaps/slot-day, n = 14). The ₹10.7 average is pulled down by 2 fuel pumps in unserved zones; without them Wave 2 costs ₹11.90 per swap, above legacy. "Wave-1 sites (fuel pumps, residential) Rs 14.7" is also the all-Wave-1 figure; fuel-pump plus residential alone is ₹15.59.
5. **Misleading supporting statistic for Wave-1 site types.** The 1.5% / 5.7% shares of legacy failures reflect how few such legacy sites existed, not low need. Failure rates by site type are equal, 6.1-7.2%.
6. **"Complaints track failures one-for-one" misstates the relationship.** It is about 1 ticket per 10 extra failed attempts. Reporting tickets per 1K *attempts* also removes the mechanical denominator effect: Gen1 18.63 vs Gen2 12.44 (still +50%).
7. **Wave-2 ramp range quoted as 116-131%** includes the partial commissioning month that the figure itself excludes; the correct range is 116-123%. Comparing completed swaps in summer also flatters Wave 2, because legacy completions are capped by stockouts (on attempts the range is 111-116%).
8. **Minor wording.**
   - Quarantine "packs" are pack-hours.
   - The Gen1 slope of 9.46/°C depends on the knot grid; quote "about 9-9.5 min/°C".
   - The Gen3 counterfactual is *not* outside heat support: 19 Gen3 stations sit in the hot cities. It is outside Launch-site support only.
   - The competitor title's "gradual erosion" is not statistically supported and is not robust to station-specific trends.
   - Recommendation 5's "deterioration" is true only in counts; quote the counts (missed angle 1).
   - Small n: the 11.56 swaps/slot-day rests on 3 stations, one of them at 21.57.

Checked and found sound:
- Timestamps: date, hour and week come from the corrected `event_ts` (clean.py lines 198-202); station-hourly is unaffected.
- Telemetry: ok rows only, with no null cabinet or charge values among them; blanks are never zero-filled.
- Denominators and seasonality: failure rates use attempts, and comparisons are same-season.
- Regression to the mean: the hotspots were chosen on the pre-period, so any bias would favour improvement, which makes the "no improvement" finding conservative.
- Cost model: follows the brief.

## Figures

| Figure | Exists / legible | Matches data | Issue |
|---|---|---|---|
| q3_01_thermal_derating (hero) | yes | yes (curves, 88 → 169 min, −48%, city shares, Gen1 counts) | Cosmetic only. City hot shares do not depend on generation (cabinet = ambient + 6.6 °C for every generation), so the right panel is not confounded. |
| q3_02_attributes_vs_failures | yes | yes (R² 0.55 → 0.95, slopes, CIs) | The right-panel title "Gen1 … gaps vanish" is misleading. Gen1's penalty is *conditional on heat* (−0.25 pp at 0% hot hours, +3.8 pp at 20%), and it is the core of the story. Suggested title: "Gen1's penalty appears only in heat; Wave-2 and city gaps vanish". |
| q3_03_expansion_placement_vs_need (hero) | yes | yes (zone rates, riders per slot, slots by wave; subtitle 49% / 36%, 68% / 50%) | None. It ranks zones by the H1-24 rate, so it does not depend on the refuted Wave-2 ρ. |
| q3_04_new_vs_legacy_utilisation_cost (hero) | yes | the scatter and ramp are correct | **The title mislabels "Wave-2 market sites Rs 10.7"** (issue 4). In the right panel the legend covers the "legacy … = 100" label. The JAI-148 callout should say it is a fuel pump in an unserved zone. |
| q3_05_network_vs_hotspots | yes | yes (9.5 → 7.9, 9.5 → 9.6, 16.7 → 16.6, 5.1) | None. Absolute counts would make the point stronger (missed angle 1). |
| q3_06_competitor_entry | yes | yes (−3.9%, CI, event-study bins, p values) | "Gradual … erosion" overstates a non-significant, trend-sensitive drift. Soften the wording. |

## Missed angles that would strengthen or change the story

1. **The rate fell but failures nearly doubled.** Apr-Jun failed attempts went from 39,820 to 77,445 network-wide (+94.5%). At the 23 hotspots they went from 18,465 to 27,710 (+50.1%), and at the 36 hot-city Gen1 stations from 26,746 to 40,723 (+52.3%). This is the honest "deterioration hidden by the average", and it belongs in the headline and in Recommendation 5.
2. **Excess-failure concentration is sharper than 52.6%.** Above the 3.91% off-season baseline, the 36 hot-city Gen1 stations produced **77.1% of Apr-Jun 2025 excess failures** (30,248 of 39,207).
3. **Load does not drive failure rates.** Hotspots absorbed +51% attempts at a flat rate, and the within-station Gen1 load effect is −0.03 pp (p 0.94). Adding stations cannot fix the hotspots; only Gen1 thermal remediation can. This is the cleanest evidence against budget proposal (a), "more stations", as a service fix.
4. **Heat exposure does not depend on equipment.** Cabinet temperature runs about 6.6 °C above ambient for Gen1, Gen2 and Gen3 alike, and summer hot shares within a city are the same across generations. The generation × heat interaction is therefore cleanly identified, not a mediator. It also gives an operational trigger for the heat playbook: forecast ambient of about 33.5 °C or more means cabinets above 40 °C.
5. **Siting rule backed by a model.** For new stations, zone demand (resident riders per slot) explains utilisation; site type matters only inside already-served zones. Use this to rewrite Recommendation 3.
6. **The counterfactual has an empirical anchor.** The plain same-city comparison (hot-city Gen2 at 8.01%, Gen3 at 5.06% vs Gen1 at 15.20%) reproduces the model's −7.4 / −10.4 pp. Quote both.
7. **Ticket mix supports the mechanism.** In Apr-Jun 2025, no_battery_available + long_queue make up 51.3% of Gen1 station tickets (3,574 of 6,964) vs 33.1% at Gen2 and 29.6% at Gen3. This links to q8 (ticket text) and q6 (retention exposure).
