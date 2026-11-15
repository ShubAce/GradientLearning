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
