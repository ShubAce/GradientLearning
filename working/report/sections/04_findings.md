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
