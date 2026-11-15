# Q8 verification: support-ticket text analysis

**What was checked.** `outputs/findings/q8.md` and its script `analysis/q8_ticket_text.py`.
**Verification script.** `analysis/verify/q8_verify.py`. It is fresh code built on the `src/common.py` loaders, with its own keyword taxonomy, joins and placebo tests. It runs top to bottom in about 6 s.

**Overall verdict: needs fixes.**
- **What holds.** The descriptive core reproduces exactly: the counts, the taxonomy, the 3W and timing fingerprints, the monthly rates, the dose test, the reported km, CSAT, priority and resolution.
- **What does not hold.**
  - **The pricing finding is refuted.** The DiD "effect" comes from which cities were used as controls.
  - **The Kyron per-pack attribution is overstated about 4x.** It uses the wrong denominator and the effect is not specific to battery tickets.
  - **The "climbs as packs age" claim is not supported** by the test cited for it.
  - **One caveat about `battery_id` is wrong.** The analyst checked the wrong column.

## Verdict table

| # | Claim (analyst) | Verdict | Recomputed |
|---|---|---|---|
| H | Battery tickets are 15,912 to 21,629 of 43,380 (36.7% to 49.9%), 2.4 to 3.3x the low_range count of 6,551, above stock-outs (26.5%) | **Confirmed** (numbers), wording fix | 15,912 (2.429x, 36.68%); 21,629 (3.302x, 49.86%); stock-outs 26.48%. Stock-outs plus queues together are 39.50%, above the 36.7% floor. "#1" needs app_issue included. |
| 1 | Mis-filing is systematic: 11,278 tickets (26.0%) = other 9,361 + QR 1,917; 84.2% blame the vehicle; Hinglish 56.4%; typos 10.3%; 0 competitor mentions | **Confirmed** | 0 of 89 templates span more than one category. 11,278 (26.00%); 13,403/15,912 (84.23%); Hinglish 56.36%; typos 10.26%; 0 competitor or praise tickets under my own broader regex (competitor names, switching, "leave", "cheaper"). |
| 2a | 3W-rider share 1.5 / 1.0 / 1.7% vs 17.6 / 19.9 / 20.3%; 3W = 10.4% of swaps; app_chat 44.1% vs 45.4% | **Confirmed** | 1.45 / 0.97 / 1.71% vs 17.63 / 19.89 / 20.32%; 10.38%; 44.11% vs 45.41% |
| 2b | Same-season step-up (H1-25 / H1-24) 4.5x / 4.4x / 4.0x vs 0.94x / 0.85x / 0.35x | **Confirmed** | 4.49 / 4.43 / 4.05 vs 0.94 / 0.85 / 0.35 |
| 2c | Kyron O/E 1.39 / 1.36 / 1.36 vs queue 1.33, stock-out 1.28, billing 1.03; "unlike queue, stock-out and billing" | **Corrected** | Reproduced on the analyst's basis (1.38 / 1.35 / 1.35 / 1.32 / 1.27 / 1.05). But the expected share uses *issued* packs, while the ticket pack is the *inserted* pack. Kyron is 30.9% of inserted 2W packs but 24.0% of issued ones. **Correct O/E: 1.08 (1.03-1.14) / 1.06 (1.02-1.11) / 1.06 (1.00-1.12) vs queue 1.04 (0.97-1.11), stock-out 0.99 (0.95-1.04), billing 0.82.** Supplier does not clearly separate battery tickets from queue tickets. |
| 2d | Kyron share: other 37.4% vs low_range 37.9% (p=0.58); app_issue 37.4%; billing 19.5% | **Confirmed**, cherry-picked contrast | Same values. Stock-outs 35.7% and queues 36.4% are just as close, so billing is the only category that differs. |
| 2e | SoH gap within supplier-month −0.045 to −0.071; QR vs system-error ρ=−0.07 (p=0.40, 141 stations) | **Confirmed** | Gap per ticket pack at insertion: −0.07 to −0.01 (battery) vs −0.01 to +0.04 (others). ρ=−0.072, p=0.40, 141 stations. |
| 3a | Monthly rates per 1K 2W swaps: Jan-24 0.38/0.93/1.32 … Jun-25 3.24/7.87/10.58; other/low_range 1.33-1.60; Apr-Jun-25 39.9% vs stock-outs 28.4% | **Confirmed** | Exact. Apr-Jun-25 is 39.93% vs 28.40%, but stock-outs plus queues are 41.1%. |
| 3b | r=0.92 with Kyron share; after the plateau the rise "tracks pack ageing" (r=0.89 with share <80% SoH vs −0.56) | **Corrected** (over-claim) | Same r values, but the test is not specific. Stock-out+queue rates also give r=0.80 with the same <80% series. First differences give r=−0.67 (n=8). Complaints rose 6.27→7.98 from Oct-24 to Jan-25 while 0% of packs were below 80% SoH. |
| 3c | Non-summer dose test: 2.34→8.05 (3.44x) vs 2.20→4.37 (1.98x), ratio of ratios 1.74 | **Confirmed, and stronger than claimed** | Exact. It passes placebos: stock-out+queue 0.87, billing 0.69. Each battery category responds alone: low_range 1.75, other 1.71, app_issue 1.77. |
| 3d | Within every city Kyron packs draw 1.59-1.95x the battery tickets per swap; about 1,959 excess (15.4% of 12,757) | **Corrected** | Reproduced on the analyst's basis (1.59-1.95x; 1,959; 556,365 swaps). But the placebo shows the same pattern: stock-out+queue tickets give 1.25-1.71x within city on that basis. With the correct inserted-pack denominator: **1.07-1.51x (median 1.16x) vs placebo median 1.03x; excess about 510 (4.0%)**. The placebo-validated city estimate is about **1,113 excess tickets in Oct-Dec-24 in the 3 Kyron cities** (371/month, 42% of the 2,626 there; placebo −135). |
| 4 | "Dying at X km": 1,349 tickets (1 dropped, 1,330 from 2W riders); Kyron 45 vs 54 km (p=4.7e-47); r=0.67 (n=945); 0.57 km/pt; 58→46 vs 70.3→48.6 (r=0.96) | **Confirmed** | Exact; dropped value is −10. Within-month slope 0.73 km per SoH point (p=7.8e-79), so the link is not just a shared trend. |
| 5 | Billing rose 18-34% vs control after the pilot (IRR 1.29; trends 1.34; per rider 1.18; placebo 1.14); "surcharge" wording 34.2→34.4% vs 32.7% | **Refuted** (effect); wording part confirmed | All IRRs reproduce, but 3 of the 4 control cities are Kyron cities. **Against Mumbai (the only low-Kyron control): IRR 0.94 (0.78-1.12, p=0.46); per active rider 0.91.** Billing rate change Jan-Jun-24 → Oct-24..Jun-25: Bengaluru 0.61x, Pune 0.55x, Mumbai 0.61x vs Delhi 0.38x, Hyderabad 0.35x, Jaipur 0.38x. Inside the controls, Kyron cities vs Mumbai give IRR 0.67 (0.56-0.80). Independent riders (who pay PEAK) give IRR 1.10 (0.83-1.46, p=0.50). The gap opens in Aug-Sep-24, before the pilot. |
| 6 | Priority χ² p=0.56 (V 0.045); 13.4 vs 13.0 h (KW p=0.12); unresolved 12.1% vs 11.7%; "third time" ≥3 stock-outs 6.7% vs 7.9%; ≥1 94.5 / 94.0% vs 22.4%; lapse 8.1% of 38,001; 9.0 vs 8.3% (p=0.29); 10.2 vs 9.0% (p=0.15) | **Confirmed** (minor level caveat) | All exact (logit ORs not refit). The lapse level is inflated by right-censoring: 10.1% in Apr-25 and 13.7% in May-25 vs 6.0-7.3% in Dec-Mar. Tickets before Apr-25 give 6.6%. |
| 7 | CSAT: never surveyed 13.9%; response 39.9%; 43.7% vs 28.3%; scores 3.54 vs 3.61 (ρ 0.013, p 0.12); IPW +0.002; scenarios 3.26 / 2.96 / 3.37 / 3.25 | **Confirmed** | Exact. My independent post-stratified estimate (decile × category × channel) is 3.5595 vs naive 3.5572 (+0.002). |
| 8 | Unresolved 11.3-11.9% (p=0.85), median 13.0-13.4 h, p90 28.7-29.8 h; 6.32→17.74; H1 10.08 vs 14.68; ρ=0.85; city H1-25 3.18-7.09 and 6.86-10.81 | **Confirmed** | All exact. |
| C | Caveat: "battery_id is not a physical trace (0.03% = last issued pack)… population-level only" | **Refuted** (wrong column checked) | 99.94% of 30,330 linked tickets carry a pack this rider **inserted**. It is the rider's last insertion: median 4.8 h before the ticket, 97.7% within the previous 48 h. The analyst compared against `battery_out_id`. |

## Issues, ranked by severity

1. **Pricing DiD is confounded by the choice of control cities (refuted).**
   - The four control cities include the three cities that got 39% Kyron packs from Sep-24. Their billing tickets per swap fell much faster.
   - Kyron raised swaps per 2W rider: 25.6→30.6 a month in Kyron cities vs 24.2→26.4 elsewhere (Jul-Aug → Oct-Dec-24). That deflates per-swap rates there. A crowd-out effect may add to it.
   - The pilot cities move exactly like Mumbai. Fig 05 itself shows the gap opening in Aug-Sep-24, before the pilot.
   - So Finding 5, Recommendation 4 ("+25 to +47 tickets/month") and the Fig 05 title are not supported.
2. **Kyron per-pack attribution uses the wrong denominator and is not specific to battery tickets.**
   - Tickets reference the *inserted* pack, so expected shares must come from `supplier_in`, not `supplier_out`.
   - Even on the analyst's own basis, service tickets show the same within-city Kyron ratio.
   - The 1,959 excess (15.4%) becomes about 510 (4.0%), or about 340 net of the placebo on the analyst's basis.
   - Affected: Finding 2 (supplier fingerprint), Finding 3 (last bullet), Recommendation 2's impact, Fig 02 panel 3, Fig 07 panel 3.
3. **"Then climb as packs age" is an over-claim.**
   - It rests on a correlation of two trending series over 9 months. That correlation fails a placebo (r=0.80) and turns negative in first differences.
   - The complaint rise from Oct-24 to Jan-25 happened with 0% of packs below 80% SoH.
   - Affected: Fig 03 title and subtitle, Finding 3, and the primary-drivers section.
4. **The `battery_id` caveat is wrong.** It understates what the data can do (see the missed angles) and led to issue 2.
5. **Framing.**
   - (a) "Text-confirmed" is used for `other` tickets whose comments contain no battery or range word. The battery label comes from the fingerprints.
   - (b) "#1 driver": the 36.7% floor is below stock-outs plus queues together (39.5%; 41.1% in Apr-Jun-25). The ranking holds only with `app_issue` included. The evidence supports including it (missed angle 3).
6. **Minor.**
   - Right-censored lapse level (8.1% → 6.6%).
   - The km figures are truncated near 60 km (only about 40 tickets above 60), so medians describe complainers only.
   - Monthly r values of 0.92 and 0.96 are trend-driven. The 0.96 is backed by the within-month slope.

## Missed angles (each computed in `q8_verify.py`)

1. **Ticket-to-attempt link.** The ticket pack identifies the rider's last attempt before the ticket, which allows ticket-level validation:
   - 94.4% of `no_battery_available` tickets follow a failed_no_charged_battery attempt;
   - 75.4% of `long_queue` tickets follow an abandoned_queue attempt;
   - `low_range`, `other`, `app_issue` and billing tickets follow completed swaps (93-96%).
   - Arrival SoC is the same for every category (about 9.2-9.4%; 55% below 10%), so it is not a discriminator.
2. **App/QR text has no event-log trace.** Only 1.9-2.2% of app or QR tickets had a logged system error or cancellation in the prior 7 days. That equals every other category (1.9-2.3%). This is rider-level confirmation that `app_issue` tickets are not real app failures.
3. **`app_issue` is about 97% battery-like.** A two-group 3W-share mixing model (app_issue 1.71% vs battery 1.17% vs service 19.4%) gives that fraction. The best estimate of battery-health tickets is therefore about 49.5% (54% in Apr-Jun-25), not an "upper bound", and it exceeds stock-outs plus queues.
4. **Per-category dose test with placebos.** It is the strongest Kyron evidence and can replace the supplier O/E panel: low_range 1.75, other 1.71 and app_issue 1.77, vs stock-out+queue 0.87 and billing 0.69. It gives about 1,113 excess tickets in Oct-Dec-24 in the three Kyron cities.
5. **Per-swap denominators move with range loss.** Swaps per rider rose 19% vs 9%. Per-swap rates therefore understate the battery rise and depress other categories in Kyron cities. Add a per-active-rider robustness check, and repeat the billing DiD against Mumbai.

## Figure review

All 7 PNGs exist, are legible at slide size, and their numbers match my recomputation. Problems:

| Figure | Problem | Fix |
|---|---|---|
| **Fig 05** | Title "Billing complaints rose 18-34% … after peak pricing" is refuted. | Redraw with 3 lines (pilot cities, Mumbai, Kyron cities). Retitle around "no pilot effect". |
| **Fig 02, panel 3** | Kyron O/E uses the issued-pack expectation, which inflates it from about 1.06 to 1.36, and it does not discriminate. | Replace with the per-category dose test plus placebo bars, or recompute on inserted packs. |
| **Fig 07, panel 3** | "Kyron 1.9x; 1.6-2.0x within city" has the same denominator problem and no placebo. | Recompute per inserted-pack attempt (median 1.16x) and add a stock-out+queue line (1.03x), or drop the panel. |
| **Fig 03** | Title and subtitle claim "climb as packs age" (r=0.89); the lower panel itself shows the <80% SoH series at 0 through Jan-25. | Retitle; drop or qualify the r. |
| Fig 01 (minor) | The title asserts "battery complaints" while the column header says "(battery?)". | Say "with a battery fingerprint". |

Figs 04 and 06 are fine.

## What is solid and should stay

- The taxonomy: 89 templates, one category each, 0 competitor or praise mentions.
- The 3W and same-season fingerprints.
- The 2.4-3.3x volume correction and the monthly series.
- The non-summer Kyron dose test.
- The reported-km signal (Kyron 45 vs 54 km; r=0.67; holds within month).
- Priority and urgency carry no signal; "third time" is not borne out.
- Identical resolution across issues.
- CSAT: IPW +0.002, with the coverage blind spot and the scenarios.
- The "surcharge" wording does not track the surcharge.
