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
