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

{{DQ_SCORECARD_REST}}
