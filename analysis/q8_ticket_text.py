# %% [markdown]
# # Q8 · Support-ticket text: does the free text add to the category and resolution fields?
#
# **Question.** All 44,000 support tickets carry a rider comment (English + Hinglish). Does that text tell us anything
# the agent-assigned `category`, `priority` and resolution fields do not? The data documentation warns that categories
# are agent-assigned and unverified, that battery range problems are sometimes filed as `other` or `app_issue`, and
# that comments sometimes blame the vehicle rather than the battery.
#
# **Approach.**
# 1. Only 89 distinct comment strings exist, so every template is labelled with transparent keyword rules (after
#    normalising three recurring typos) and reviewed one by one -> `outputs/tables/q8_comment_taxonomy.csv`.
# 2. Agent category vs text-derived issue (confusion matrix).
# 3. Validate labels against independent structured evidence that the text cannot influence: the rider's vehicle
#    class, timing relative to Kyron's arrival (Sep-2024), a city-level Kyron dose test with placebo categories, the
#    ticket's `battery_id` (the pack the rider inserted at a recent attempt) -> supplier, SoH and attempt outcome,
#    and the rider's own logged events.
# 4. Quantify what the text adds: hidden battery complaints, rider-reported range (km), pricing words around the
#    pilot, urgency/repeat claims, competitor mentions -> churn. Correct CSAT non-response with inverse-probability
#    weighting (IPW) and bound what IPW cannot fix.

# %%
import sys
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _find_root() -> Path:
    cands = [Path(__file__).resolve().parents[1]] if "__file__" in globals() else []
    cands += [Path.cwd(), Path.cwd().parent]
    for c in cands:
        if (c / "src" / "common.py").exists():
            return c
    raise FileNotFoundError("project root with src/common.py not found")


ROOT = _find_root()
sys.path.insert(0, str(ROOT / "src"))
from common import *  # noqa: E402,F401,F403

set_style()
warnings.filterwarnings("ignore")
T0 = time.time()
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(230)
pl.Config.set_fmt_str_lengths(60)
RNG = np.random.default_rng(59500)
INK, INK2 = "#222222", "#555555"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def show(title, df):
    print(f"\n=== {title} ===")
    print(df)


def headline(fig, title, subtitle=None):
    """Left-aligned insight title + grey subtitle; returns the `top` to pass to tight_layout(rect=...)."""
    H = fig.get_figheight()
    # fig.text (not suptitle) so tight_layout does not reserve the title space twice
    fig.text(0.012, 1 - 0.10 / H, title, ha="left", va="top", fontsize=14, fontweight="bold", color=INK)
    n_sub = 0
    if subtitle:
        n_sub = subtitle.count("\n") + 1
        fig.text(0.012, 1 - 0.46 / H, subtitle, ha="left", va="top", fontsize=10.5, color=INK2)
    return 1 - (0.55 + 0.21 * n_sub) / H


# %% [markdown]
# ## 1 · Scope
# Tickets outside Jan-2024..Jun-2025 (86) and tickets raised at the two internal test stations (535) are excluded,
# consistent with the network KPIs. Every ticket has a comment, so "tickets with a comment" = the scoped table.

# %%
tk_all = tickets()
tk = tk_all.filter(pl.col("in_window") & ~pl.col("is_test"))
N = tk.height
print(f"tickets raw {tk_all.height:,} | out-of-window {int((~tk_all['in_window']).sum()):,} | at test stations "
      f"{int(tk_all['is_test'].sum()):,} | scoped {N:,} | scoped with a comment {int(tk['rider_comment'].is_not_null().sum()):,} "
      f"| distinct comment strings (all tickets) {tk_all['rider_comment'].n_unique()}")

# %% [markdown]
# ## 2 · Comment taxonomy: keyword rules + template-level review
# Rules run on a normalised comment (lower case; typos `staion`->station, `battry`->battery, `rnage`->range).
# **First matching rule wins.** The order puts station-level stock-out phrases ("sab batteries discharge the" = *all
# batteries were discharged*) ahead of the generic battery/range rule, and range words ahead of vehicle words, so
# "vehicle not giving proper range" is a range complaint while "scooter weak lag raha hai" is vehicle-phrased.
# Competitor/switching and praise rules are included on purpose: a zero count is then a finding, not an omission.

# %%
TYPO_FIX = [(r"staion", "station"), (r"battry", "battery"), (r"rnage", "range")]
RULES = [  # (text_issue, regex) - evaluated in order on the normalised comment
    ("competitor_switching", r"competitor|switch(ing)? to|other (company|app|network)|dusri company|chhod|band kar|uninstall|leave"),
    ("other_praise", r"thank|great service|good service|accha|badhiya|shukriya|well done"),
    ("no_battery_stockout", r"no battery available|charged battery nahi|empty station|batteries discharge the|no charged"),
    ("long_queue", r"queue|lambi line|lamba|line thi|waited \d+ min"),
    ("billing_wrong_amount", r"galat|wrong amount|double charge|refund"),
    ("pricing_surcharge", r"surcharge|extra amount|charged extra|price|mehenga|costly"),
    ("station_hw_system", r"\bqr\b|scan|cabinet|door|slot|machine"),
    ("app_tech", r"\bapp\b|crash|login|otp"),
    ("battery_range_health", r"range|discharge fast|dying at -?\d+ ?km|battery"),
    ("vehicle_phrased", r"vehicle|scooter|bike|gaadi|motor"),
]
ISSUE_LABEL = {
    "battery_range_health": "Battery range / health",
    "vehicle_phrased": "Vehicle-phrased\n(battery-likely)",
    "no_battery_stockout": "No battery /\nstock-out",
    "long_queue": "Long queue",
    "billing_wrong_amount": "Billing:\nwrong amount",
    "pricing_surcharge": "Pricing:\nsurcharge / extra",
    "app_tech": "App / tech",
    "station_hw_system": "Station hardware\n(QR scan)",
    "competitor_switching": "Competitor /\nswitching",
    "other_praise": "Other /\npraise",
}
ISSUE_ORDER = list(ISSUE_LABEL)
HINDI = r"\b(hai|nahi|ho|gaya|raha|lag|lagti|mein|ki|bahut|bohot|kal|sab|laga|diya|bina|bataye|kata|galat|koi|mila|thi|tha|ke|liye|aaj|aajkal|pe|itna|lamba|lambi|dikkat|gaadi|se|ka)\b"


def normalise(s: str) -> str:
    s = s.lower().strip()
    for a, b in TYPO_FIX:
        s = re.sub(a, b, s)
    return s


def classify(norm: str):
    for lab, rx in RULES:
        if re.search(rx, norm):
            return lab, rx
    return "unmatched", ""


REVIEW = {  # template-level review of each rule outcome
    "battery_range_health": ("battery_range_health", "Range/discharge words or 'dying at X km'; correctly filed as low_range by agents."),
    "vehicle_phrased": ("battery_range_health", "Blames the vehicle with no battery or range word; filed as 'other'. Battery label rests on "
                        "the structured fingerprint in section 4 (2W-only, same-season step-up, responds to the Kyron city dose like low_range)."),
    "no_battery_stockout": ("no_battery_stockout", "Station had no charged pack; validated against the rider's logged stock-outs (section 7)."),
    "long_queue": ("long_queue", "Queue/wait wording."),
    "billing_wrong_amount": ("billing_pricing", "Wrong deduction."),
    "pricing_surcharge": ("billing_pricing", "Price level / surcharge wording; tested against the base-price change and the peak pilot (section 6)."),
    "station_hw_system": ("station_hw_system", "QR scan at cabinet: ambiguous between app and cabinet hardware; tested against station "
                          "system-error rates (section 9). Ticket population carries the battery fingerprint (section 4)."),
    "app_tech": ("app_tech", "Text agrees with app_issue, but the ticket population carries the battery fingerprint (section 4; ~97% "
                 "battery-like by 3W mix) and no logged app failure; counted in the central battery estimate, not in the wording floor."),
}

tpl = (tk_all.group_by("rider_comment").agg(pl.len().alias("n_tickets_all"), pl.col("category").n_unique().alias("n_categories"),
                                            pl.col("category").mode().first().alias("agent_category"))
       .join(tk.group_by("rider_comment").agg(pl.len().alias("n_tickets_scoped")), on="rider_comment", how="left")
       .with_columns(pl.col("n_tickets_scoped").fill_null(0))
       .sort(["n_tickets_all", "rider_comment"], descending=[True, False]))
rows = []
for i, r in enumerate(tpl.iter_rows(named=True), 1):
    norm = normalise(r["rider_comment"])
    lab, rx = classify(norm)
    km = re.search(r"dying at (-?\d+) ?km", norm)
    rev, note = REVIEW.get(lab, (lab, ""))
    rows.append(dict(
        template_id=f"T{i:02d}", rider_comment=r["rider_comment"], n_tickets_all=r["n_tickets_all"], n_tickets_scoped=r["n_tickets_scoped"],
        agent_category=r["agent_category"], categories_per_template=r["n_categories"], normalised_text=norm,
        language="Hinglish" if re.search(HINDI, norm) else "English",
        has_typo=bool(re.search(r"staion|battry|rnage", r["rider_comment"].lower())),
        text_issue=lab, rule_matched=rx,
        mentions_vehicle=bool(re.search(r"vehicle|scooter|bike|gaadi", norm)),
        mentions_battery_or_range=bool(re.search(r"battery|batteries|range|discharge|dying at", norm)),
        urgency_repeat=bool(re.search(r"again|third time|roz|har baar|phir se", norm)),
        urgency_business_impact=bool(re.search(r"\blate\b|delivery|order|customer", norm)),
        persistence_marker=bool(re.search(r"since morning|kal se|aajkal|\baaj\b|this week|\d+ min", norm)),
        km_reported=int(km.group(1)) if km else None,
        reviewed_true_issue=rev, review_note=note))
tax = pl.DataFrame(rows)
tax.write_csv(TAB / "q8_comment_taxonomy.csv")
assert tax.height == 89 and (tax["categories_per_template"] == 1).all() and (tax["text_issue"] != "unmatched").all()
print(f"templates: {tax.height} | each template filed under exactly one agent category: {bool((tax['categories_per_template'] == 1).all())}")
show("templates per text issue (all tickets)", tax.group_by("text_issue").agg(pl.len().alias("templates"), pl.col("n_tickets_scoped").sum().alias("tickets"))
     .sort("tickets", descending=True))
show("language / typo / vehicle-blame shares (scoped tickets)", tax.select(
    (pl.col("n_tickets_scoped").filter(pl.col("language") == "Hinglish").sum() / pl.col("n_tickets_scoped").sum()).alias("hinglish_share"),
    (pl.col("n_tickets_scoped").filter(pl.col("has_typo")).sum() / pl.col("n_tickets_scoped").sum()).alias("typo_share"),
    pl.col("n_tickets_scoped").filter(pl.col("text_issue") == "competitor_switching").sum().alias("competitor_mentions"),
    pl.col("n_tickets_scoped").filter(pl.col("text_issue") == "other_praise").sum().alias("praise_mentions")))
show("first 25 rows of the 89-template taxonomy", tax.select("template_id", "n_tickets_scoped", "agent_category", "language", "text_issue", "rider_comment").head(25))

# %% [markdown]
# ## 3 · Agent category vs text-derived issue
# Because every template is filed under exactly one category, disagreement is systematic (whole templates), not
# random agent error.

# %%
tk = tk.join(tax.select("rider_comment", "template_id", "text_issue", "reviewed_true_issue", "language", "mentions_vehicle",
                        "urgency_repeat", "urgency_business_impact", "km_reported"), on="rider_comment", how="left")
CATS = ["low_range", "other", "app_issue", "no_battery_available", "long_queue", "billing_dispute"]
NATURAL = {"low_range": {"battery_range_health"}, "no_battery_available": {"no_battery_stockout"}, "long_queue": {"long_queue"},
           "billing_dispute": {"billing_wrong_amount", "pricing_surcharge"}, "app_issue": {"app_tech"}, "other": {"other_praise"}}
cm = (tk.group_by("category", "text_issue").agg(pl.len().alias("n")).to_pandas()
      .pivot(index="category", columns="text_issue", values="n").reindex(index=CATS, columns=ISSUE_ORDER).fillna(0).astype(int))
cm_row = cm.div(cm.sum(axis=1), axis=0)
cm.to_csv(TAB / "q8_confusion_category_vs_text.csv")
agree = tk.with_columns(pl.struct("category", "text_issue").map_elements(lambda s: s["text_issue"] in NATURAL[s["category"]],
                                                                          return_dtype=pl.Boolean).alias("agree"))
n_dis = int((~agree["agree"]).sum())
n_other = int((tk["category"] == "other").sum())
n_qr = int((tk["text_issue"] == "station_hw_system").sum())
n_app = int((tk["category"] == "app_issue").sum())
n_lr = int((tk["category"] == "low_range").sum())
bat_text = tk.filter(pl.col("text_issue").is_in(["battery_range_health", "vehicle_phrased"]))
veh_blame_share = float(bat_text["mentions_vehicle"].mean())
with pd.option_context("display.width", 250, "display.max_columns", 20):
    print(cm)
print(f"\ntext disagrees with agent category: {n_dis:,}/{N:,} = {pct(n_dis / N)} | of which 'other' (all vehicle-phrased) "
      f"{n_other:,} = {pct(n_other / N)} of tickets; QR-scan inside app_issue {n_qr:,} = {pct(n_qr / N)}")
print(f"'other' bucket: {pct(cm_row.loc['other', 'vehicle_phrased'])} vehicle-phrased; app_issue: {pct(cm_row.loc['app_issue', 'app_tech'])} app text, "
      f"{pct(cm_row.loc['app_issue', 'station_hw_system'])} QR-scan text")
print(f"battery-group wording (range words or vehicle-phrased) that blames the vehicle: {int(bat_text['mentions_vehicle'].sum()):,} = "
      f"{pct(veh_blame_share)} of {bat_text.height:,}")
rank_off = tk.group_by("category").agg(pl.len().alias("n")).with_columns((pl.col("n") / N).alias("share")).sort("n", descending=True)
show("official ranking of complaint drivers (agent category)", rank_off)
Q2_25 = pl.col("month") >= pl.date(2025, 4, 1)
grp_txt = pl.col("text_issue").replace_strict({"battery_range_health": "battery: range wording + vehicle-phrased",
                                               "vehicle_phrased": "battery: range wording + vehicle-phrased",
                                               "billing_wrong_amount": "billing/pricing", "pricing_surcharge": "billing/pricing",
                                               "app_tech": "app/QR", "station_hw_system": "app/QR"}, default=None)
rank_txt = (tk.with_columns(pl.coalesce(grp_txt, pl.col("text_issue")).alias("g")).group_by("g")
            .agg(pl.len().alias("n"), Q2_25.sum().alias("n_q2_25")).with_columns((pl.col("n") / N).alias("share"))
            .sort("n", descending=True))
n_q2 = int(tk.filter(Q2_25).height)
rank_txt = rank_txt.with_columns((pl.col("n_q2_25") / n_q2).alias("share_q2_25"))
show(f"text-derived ranking (full window and Apr-Jun 2025, n={n_q2:,})", rank_txt)

# %% [markdown]
# ## 4 · Validation: do 'other' and 'app_issue' tickets look like battery complaints?
# Fingerprints that the comment text cannot influence:
# * **Vehicle class** - range loss is a 2W phenomenon (Kyron 2W packs degrade ~3x faster; Kyron 3W behaves normally).
#   3W riders make 10% of swaps and use app chat as much as 2W riders, so genuine app/queue/billing problems should
#   reach them too. A two-group mixing model on the 3W share estimates how much of `app_issue` is battery-like.
# * **Timing** - same-season step change (H1-2025 vs H1-2024) in tickets per 1K 2W swaps: Kyron's share of issued 2W
#   packs went from 0% (to Aug-24) to ~24-25% (Oct-24 onwards).
# * **Kyron dose by city, outside summer** - Delhi NCR, Hyderabad and Jaipur received ~39% Kyron packs vs ~8.5%
#   elsewhere. Change from Jul-Aug-24 (no Kyron) to Oct-Dec-24 (Kyron live, before summer), each category on its own,
#   with stock-out, queue and billing tickets as placebos.
# * **The ticket's own pack** - a ticket's `battery_id` is the pack the rider *inserted* at their last attempt before
#   the ticket (section 4b), so the expected Kyron share must come from inserted packs (`supplier_in`), not issued
#   packs (Kyron is a larger share of inserted than of issued 2W packs). The linked attempt also gives a ticket-level
#   check of the category, and the pack's SoH at insertion vs same supplier, city and month.

# %%
ev_c = events(["is_completed", "month", "city", "vehicle_class", "supplier_out", "soh_out_pct"]).filter(pl.col("is_completed"))
sw_cmvs = (ev_c.group_by("city", "month", "vehicle_class", "supplier_out")
           .agg(pl.len().alias("swaps"), pl.col("soh_out_pct").sum().alias("soh_sum"), pl.col("soh_out_pct").count().alias("soh_n"),
                (pl.col("soh_out_pct") < 80).sum().alias("lt80"))
           .collect(engine="streaming"))
sw_mvs = sw_cmvs.group_by("month", "vehicle_class", "supplier_out").agg(pl.col("swaps", "soh_sum", "soh_n", "lt80").sum())
sw_cmv = sw_cmvs.group_by("city", "month", "vehicle_class").agg(pl.col("swaps").sum())
KY_OUT = pl.col("swaps").filter(pl.col("supplier_out") == "Kyron").sum() / pl.col("swaps").sum()
# Kyron share of ISSUED 2W packs by city-month (kept only to show how much the wrong denominator inflates O/E)
ky_out_cm = sw_cmvs.filter(pl.col("vehicle_class") == "2W").group_by("city", "month").agg(KY_OUT.alias("p_ky_out_city"))
# packs riders INSERT at 2W attempts (any attempt type with a battery_in_id): the right reference for a ticket's battery_id.
# Kyron was not spread evenly across cities, so the expectation uses the ticket's city-month (Simpson-type trap).
ins_cms = (events(["vehicle_class", "battery_in_id", "supplier_in", "soh_in_pct", "city", "month"])
           .filter((pl.col("vehicle_class") == "2W") & pl.col("battery_in_id").is_not_null())
           .group_by("city", "month", "supplier_in").agg(pl.len().alias("att"), pl.col("soh_in_pct").mean().alias("ref_soh_in"))
           .collect(engine="streaming"))
KY_IN = pl.col("att").filter(pl.col("supplier_in") == "Kyron").sum() / pl.col("att").sum()
ky_in_cm = ins_cms.group_by("city", "month").agg(KY_IN.alias("p_ky_in_city"))
ky_in_m = ins_cms.group_by("month").agg(KY_IN.alias("p_ky_in"))
pack_month = (events(["is_completed", "battery_out_id", "month", "soh_out_pct"]).filter(pl.col("is_completed"))
              .group_by("battery_out_id", "month").agg(pl.col("soh_out_pct").mean().alias("pack_soh"))
              .collect(engine="streaming").rename({"battery_out_id": "battery_id"}))
m2w = (sw_mvs.filter(pl.col("vehicle_class") == "2W").group_by("month")
       .agg(pl.col("swaps").sum().alias("swaps_2w"), KY_OUT.alias("p_kyron"),
            (pl.col("soh_sum").sum() / pl.col("soh_n").sum()).alias("soh_2w"),
            (pl.col("lt80").sum() / pl.col("soh_n").sum()).alias("share_lt80"))
       .sort("month"))
msw = sw_mvs.group_by("month").agg(pl.col("swaps").sum().alias("swaps_all")).sort("month")
share_3w_swaps = float(sw_mvs.filter(pl.col("vehicle_class") == "3W")["swaps"].sum() / sw_mvs["swaps"].sum())
show("2W issued packs by month: Kyron share, mean SoH, share below 80% SoH", m2w)
_pi = ins_cms.filter(pl.col("month") >= pl.date(2024, 9, 1))
_po = sw_mvs.filter((pl.col("vehicle_class") == "2W") & (pl.col("month") >= pl.date(2024, 9, 1)))
p_in_post = float(_pi.filter(pl.col("supplier_in") == "Kyron")["att"].sum() / _pi["att"].sum())
p_out_post = float(_po.filter(pl.col("supplier_out") == "Kyron")["swaps"].sum() / _po["swaps"].sum())
print(f"Kyron share of 2W packs, Sep-24..Jun-25: {pct(p_in_post)} of packs riders INSERTED at attempts vs {pct(p_out_post)} of packs ISSUED "
      f"at completed swaps")

bat = batteries().select("battery_id", pl.col("supplier").alias("pack_supplier"), pl.col("vehicle_class").alias("pack_vc"))
rd = riders().select("rider_id", pl.col("vehicle_class").alias("rider_vc"), "home_city_clean", "partner_id", "plan_type", "first_swap_ts", "last_swap_ts")
tk = (tk.join(bat, on="battery_id", how="left").join(rd, on="rider_id", how="left")
      .join(pack_month, on=["battery_id", "month"], how="left")
      .with_columns(pl.coalesce("station_city", "home_city_clean").alias("city_any"), (pl.col("pack_supplier") == "Kyron").alias("is_kyron"))
      .join(m2w.select("month", "p_kyron"), on="month", how="left")
      .join(ky_in_m, on="month", how="left")
      .join(ky_out_cm.rename({"city": "city_any"}), on=["city_any", "month"], how="left")
      .join(ky_in_cm.rename({"city": "city_any"}), on=["city_any", "month"], how="left")
      .with_columns(pl.coalesce("p_ky_out_city", "p_kyron").alias("p_ky_out_city"), pl.coalesce("p_ky_in_city", "p_ky_in").alias("p_ky_in_city")))

# %% [markdown]
# ### 4b · What a ticket's `battery_id` points to, and what happened at that attempt
# For each ticket with a battery_id, take the attempts at which this rider inserted that pack and keep the one closest
# in time to the ticket; separately, check whether it is the rider's last inserted pack before the ticket. The linked
# attempt's outcome is a ticket-level check of the agent category (a stock-out ticket should follow a failed
# no-charged-battery attempt), and the pack's SoH at insertion is compared with same-supplier packs inserted in the
# same city and month. Finally, app/QR tickets are checked against the rider's own logged system errors and
# cancellations in the 7 days before the ticket.

# %%
pairs = tk.filter(pl.col("battery_id").is_not_null()).select("ticket_id", "rider_id", "battery_id", "created_ts")
ev_in = (events(["rider_id", "event_ts", "battery_in_id", "soh_in_pct", "event_type", "month", "city"])
         .filter(pl.col("battery_in_id").is_not_null())
         .join(pairs.lazy().select("rider_id", "battery_id").unique(), left_on=["rider_id", "battery_in_id"],
               right_on=["rider_id", "battery_id"], how="semi")
         .collect(engine="streaming").rename({"battery_in_id": "battery_id", "month": "ev_month", "city": "ev_city"}))
link = (pairs.join(ev_in, on=["rider_id", "battery_id"], how="inner")
        .with_columns(((pl.col("event_ts") - pl.col("created_ts")).dt.total_seconds() / 3600).alias("dt_h"))
        .with_columns(pl.col("dt_h").abs().alias("adt")).sort("adt").group_by("ticket_id").first()
        .select("ticket_id", "dt_h", pl.col("event_type").alias("link_event"), "soh_in_pct", "ev_month", "ev_city"))
ins_all = (events(["rider_id", "event_ts", "battery_in_id"]).filter(pl.col("battery_in_id").is_not_null())
           .join(pairs.lazy().select("rider_id").unique(), on="rider_id", how="semi").collect(engine="streaming").sort("event_ts")
           .rename({"event_ts": "last_in_ts", "battery_in_id": "last_in_id"}))
_la = pairs.sort("created_ts").join_asof(ins_all, left_on="created_ts", right_on="last_in_ts", by="rider_id", strategy="backward")
# the insertion before the last one (did the rider swap again between the ticket's attempt and the ticket?)
_lb = (_la.filter(pl.col("last_in_ts").is_not_null()).with_columns((pl.col("last_in_ts") - pl.duration(microseconds=1)).alias("prev_key")).sort("prev_key")
       .join_asof(ins_all.rename({"last_in_ts": "prev_in_ts", "last_in_id": "prev_in_id"}), left_on="prev_key", right_on="prev_in_ts", by="rider_id",
                  strategy="backward"))
share_last_in = float((_la["last_in_id"] == _la["battery_id"]).fill_null(False).mean())
share_prev_in = float(((_lb["prev_in_id"] == _lb["battery_id"]) & (_lb["last_in_id"] != _lb["battery_id"])).fill_null(False).sum() / _la.height)
del ins_all, _la, _lb, ev_in
n_pairs, n_link = pairs.height, link.height
dt_med, dt_48 = float(link["dt_h"].median()), float(link["dt_h"].is_between(-48, 0).mean())
print(f"tickets with a battery_id: {n_pairs:,} ({pct(n_pairs / N)}). battery_id is a pack this rider inserted: {n_link:,} = {pct(n_link / n_pairs, 2)}; "
      f"closest such insertion: median {dt_med:+.1f} h relative to the ticket, {pct(dt_48)} within the 48 h before it. It is the rider's last "
      f"inserted pack before the ticket in {pct(share_last_in)} and the one before that in {pct(share_prev_in)} (rider swapped again before "
      f"raising the ticket) -> the battery_id traces the attempt the ticket is about")
tk = (tk.join(link, on="ticket_id", how="left")
      .join(ins_cms.select(pl.col("supplier_in").alias("pack_supplier"), pl.col("city").alias("ev_city"), pl.col("month").alias("ev_month"), "ref_soh_in"),
            on=["pack_supplier", "ev_city", "ev_month"], how="left")
      .with_columns(pl.when(pl.col("dt_h").abs() <= 72).then(pl.col("soh_in_pct")).alias("soh_in_72"))
      .with_columns((pl.col("soh_in_72") - pl.col("ref_soh_in")).alias("soh_gap_in")))
# ticket-level validation: outcome of the linked attempt (insertion within the 48 h before the ticket)
tv = (tk.filter(pl.col("dt_h").is_between(-48, 0)).group_by("category")
      .agg(pl.len().alias("linked_48h"), (pl.col("link_event") == "swap_completed").mean().alias("completed"),
           (pl.col("link_event") == "failed_no_charged_battery").mean().alias("failed_no_charged_battery"),
           (pl.col("link_event") == "abandoned_queue").mean().alias("abandoned_queue"), pl.col("dt_h").median().alias("median_h_before"))
      .to_pandas().set_index("category").reindex(CATS))
tv.to_csv(TAB / "q8_ticket_linked_attempt_validation.csv")
with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
    print("\n=== outcome of the attempt each ticket links to (rider inserted the ticket pack within the 48 h before the ticket) ===\n", tv)
# rider-level app/QR check: did the rider log a system error or a cancellation in the 7 days before the ticket?
se_ev = (events(["rider_id", "event_ts", "event_type"]).filter(pl.col("event_type").is_in(["failed_system_error", "cancelled_by_rider"]))
         .collect(engine="streaming"))
_j9 = (tk.select("ticket_id", "rider_id", "created_ts").join(se_ev, on="rider_id")
       .filter((pl.col("event_ts") <= pl.col("created_ts")) & (pl.col("event_ts") > pl.col("created_ts") - pl.duration(days=7)))
       .group_by("ticket_id").agg((pl.col("event_type") == "failed_system_error").sum().alias("se7"),
                                  (pl.col("event_type") == "cancelled_by_rider").sum().alias("cx7")))
app_chk = (tk.select("ticket_id", "text_issue").join(_j9, on="ticket_id", how="left").with_columns(pl.col("se7", "cx7").fill_null(0))
           .group_by("text_issue").agg(pl.len().alias("n"), (pl.col("se7") >= 1).mean().alias("sys_error_7d"),
                                       (pl.col("cx7") >= 1).mean().alias("cancel_7d"))
           .sort("sys_error_7d"))
show("share of tickets whose rider logged a system error / a cancellation in the 7 days before the ticket, by text issue", app_chk)
del se_ev, _j9
ch = tk.filter(pl.col("rider_vc").is_not_null()).group_by("rider_vc").agg((pl.col("channel") == "app_chat").mean().alias("app_chat_share")).sort("rider_vc")
print("share of each vehicle class's tickets raised via app_chat: " + ", ".join(f"{r['rider_vc']} {pct(r['app_chat_share'])}" for r in ch.iter_rows(named=True)))
print(f"tickets with battery_id {pct(tk['battery_id'].is_not_null().mean())}; pack vehicle class = rider vehicle class in "
      f"{pct(tk.filter(pl.col('pack_vc').is_not_null()).select((pl.col('pack_vc') == pl.col('rider_vc')).mean()).item())} of linked tickets; "
      f"3W share of completed swaps {pct(share_3w_swaps)}")

H1_24 = (pl.col("month") >= pl.date(2024, 1, 1)) & (pl.col("month") <= pl.date(2024, 6, 1))
H1_25 = (pl.col("month") >= pl.date(2025, 1, 1)) & (pl.col("month") <= pl.date(2025, 6, 1))
POST_KY = pl.col("month") >= pl.date(2024, 9, 1)
sw2w_h1_24 = float(m2w.filter(H1_24)["swaps_2w"].sum())
sw2w_h1_25 = float(m2w.filter(H1_25)["swaps_2w"].sum())


def fingerprint(key: str) -> pd.DataFrame:
    out = []
    for g, d in tk.group_by(key):
        g = g[0]
        d2 = d.filter(pl.col("rider_vc") == "2W")
        r24 = d2.filter(H1_24).height / sw2w_h1_24 * 1000
        r25 = d2.filter(H1_25).height / sw2w_h1_25 * 1000
        lk = d.filter(POST_KY & (pl.col("pack_vc") == "2W"))
        O = int(lk["is_kyron"].sum())
        E, E_net, E_out, E_out_net = (float(lk[c].sum()) for c in ["p_ky_in_city", "p_ky_in", "p_ky_out_city", "p_kyron"])
        lo = stats.chi2.ppf(0.025, 2 * O) / 2 / E if O > 0 else 0.0  # exact Poisson CI on O, divided by the city-month expectation
        hi = stats.chi2.ppf(0.975, 2 * O + 2) / 2 / E
        gp = d.filter(pl.col("pack_vc") == "2W")["soh_gap_in"].drop_nulls().to_numpy()  # all months: gap is within supplier-city-month
        out.append(dict(group=g, tickets=d.height, share_3w_riders=float((d["rider_vc"] == "3W").mean()),
                        rate_h1_24=r24, rate_h1_25=r25, step_ratio=r25 / r24 if r24 > 0 else np.nan,
                        linked_2w_post=lk.height, kyron_share=O / max(lk.height, 1), kyron_expected=E / max(lk.height, 1),
                        kyron_OE=O / E if E > 0 else np.nan, OE_lo=lo, OE_hi=hi,
                        OE_inserted_network=O / E_net if E_net > 0 else np.nan,
                        OE_issued_city_superseded=O / E_out if E_out > 0 else np.nan, OE_issued_network_superseded=O / E_out_net if E_out_net > 0 else np.nan,
                        soh_at_insertion=float(d.filter(pl.col("pack_vc") == "2W")["soh_in_72"].mean()),
                        soh_gap_in=float(gp.mean()) if len(gp) else np.nan,
                        soh_gap_se=float(gp.std(ddof=1) / np.sqrt(len(gp))) if len(gp) > 1 else np.nan, soh_gap_n=len(gp)))
    return pd.DataFrame(out)


fp_cat = fingerprint("category").set_index("group").reindex(CATS)
fp_txt = fingerprint("text_issue").set_index("group").reindex([i for i in ISSUE_ORDER if i in tk["text_issue"].unique().to_list()])
# monthly correlation of 2W-rider ticket rate with Kyron share / mean issued SoH
mon = (tk.filter(pl.col("rider_vc") == "2W").group_by("month", "category").agg(pl.len().alias("n"))
       .join(m2w, on="month").with_columns((pl.col("n") / pl.col("swaps_2w") * 1000).alias("rate")).sort("month"))
for c in CATS:
    d = mon.filter(pl.col("category") == c)
    fp_cat.loc[c, "r_rate_vs_kyron_share"] = np.corrcoef(d["rate"], d["p_kyron"])[0, 1]
    fp_cat.loc[c, "r_rate_vs_mean_soh"] = np.corrcoef(d["rate"], d["soh_2w"])[0, 1]
fp_cat["battery_fingerprint"] = (fp_cat["share_3w_riders"] < 0.03) & (fp_cat["step_ratio"] > 2.5)

# Kyron dose by city, outside summer: Jul-Aug-24 (no Kyron) -> Oct-Dec-24 (Kyron live). Each category alone, 2W riders per
# 1K 2W swaps; stock-out, queue and billing are placebos (they should not respond to battery quality).
ky_city = (sw_cmvs.filter((pl.col("vehicle_class") == "2W") & H1_25).group_by("city").agg(KY_OUT.alias("kyron_share_2w"))
           .sort("kyron_share_2w", descending=True))
KY_CITIES = ky_city["city"].head(3).to_list()
ky_hi, ky_lo = float(ky_city.head(3)["kyron_share_2w"].mean()), float(ky_city.tail(3)["kyron_share_2w"].mean())
JA24 = pl.col("month").is_between(pl.date(2024, 7, 1), pl.date(2024, 8, 1))
OD24 = pl.col("month").is_between(pl.date(2024, 10, 1), pl.date(2024, 12, 1))
BATT, SVC = ["low_range", "other", "app_issue"], ["no_battery_available", "long_queue"]


def dose(cats, vc="2W"):
    t = tk.filter(pl.col("category").is_in(cats) & pl.col("station_city").is_not_null())
    s = sw_cmv
    if vc:
        t, s = t.filter(pl.col("rider_vc") == vc), s.filter(pl.col("vehicle_class") == vc)
    r = {}
    for nm, per in [("ja", JA24), ("od", OD24)]:
        n_ = t.filter(per).group_by(pl.col("station_city").is_in(KY_CITIES).alias("ky")).agg(pl.len().alias("n"))
        s_ = s.filter(per).group_by(pl.col("city").is_in(KY_CITIES).alias("ky")).agg(pl.col("swaps").sum().alias("sw"))
        j = n_.join(s_, on="ky")
        for ky in (True, False):
            row = j.filter(pl.col("ky") == ky)
            r[(nm, ky)] = (float(row["n"].item()), float(row["sw"].item()))
    rate = {k: v[0] / v[1] * 1000 for k, v in r.items()}
    cf = rate[("ja", True)] * rate[("od", False)] / rate[("ja", False)]  # Kyron cities had they moved like the other cities
    return dict(kyron_cities_JulAug24=rate[("ja", True)], kyron_cities_OctDec24=rate[("od", True)], other_cities_JulAug24=rate[("ja", False)],
                other_cities_OctDec24=rate[("od", False)], kyron_change=rate[("od", True)] / rate[("ja", True)],
                other_change=rate[("od", False)] / rate[("ja", False)],
                ratio_of_ratios=(rate[("od", True)] / rate[("ja", True)]) / (rate[("od", False)] / rate[("ja", False)]),
                tickets_kyron_cities_OctDec24=r[("od", True)][0], swaps_kyron_cities_OctDec24=r[("od", True)][1],
                excess_tickets_OctDec24=(rate[("od", True)] - cf) / 1000 * r[("od", True)][1])


DOSE_SETS = [(c, [c], "2W") for c in CATS] + [("battery group", BATT, "2W"), ("stock-out+queue", SVC, "2W"),
                                              ("stock-out+queue (all riders, all swaps)", SVC, None), ("billing (all riders, all swaps)", ["billing_dispute"], None)]
dose_tab = pd.DataFrame([dict(set=k, **dose(c, v)) for k, c, v in DOSE_SETS]).set_index("set")
dose_tab.to_csv(TAB / "q8_kyron_dose_test.csv")
fp_cat["kyron_dose_ratio_of_ratios"] = dose_tab.loc[CATS, "ratio_of_ratios"].values
fp_cat.to_csv(TAB / "q8_battery_fingerprint_by_category.csv")
fp_txt.to_csv(TAB / "q8_battery_fingerprint_by_text_issue.csv")
with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.3f}".format):
    print("\n=== battery fingerprint by agent category (Kyron O/E vs inserted packs; *_superseded = issued-pack basis) ===\n", fp_cat)
    print("\n=== battery fingerprint by text-derived issue ===\n", fp_txt)
    print(f"\n=== Kyron dose test, tickets per 1K swaps, Kyron cities {KY_CITIES} ({pct(ky_hi)} Kyron) vs others ({pct(ky_lo)}) ===\n",
          dose_tab[["kyron_cities_JulAug24", "kyron_cities_OctDec24", "kyron_change", "other_cities_JulAug24", "other_cities_OctDec24",
                    "other_change", "ratio_of_ratios", "excess_tickets_OctDec24"]])
dz_b, dz_s = dose_tab.loc["battery group"], dose_tab.loc["stock-out+queue"]
print(f"battery group: Kyron cities {dz_b['kyron_cities_JulAug24']:.2f} -> {dz_b['kyron_cities_OctDec24']:.2f} ({dz_b['kyron_change']:.2f}x) vs others "
      f"{dz_b['other_cities_JulAug24']:.2f} -> {dz_b['other_cities_OctDec24']:.2f} ({dz_b['other_change']:.2f}x), ratio of ratios {dz_b['ratio_of_ratios']:.2f}; "
      f"excess Oct-Dec-24 in the Kyron cities {dz_b['excess_tickets_OctDec24']:,.0f} tickets ({dz_b['excess_tickets_OctDec24'] / 3:,.0f}/month) = "
      f"{pct(dz_b['excess_tickets_OctDec24'] / dz_b['tickets_kyron_cities_OctDec24'])} of the {dz_b['tickets_kyron_cities_OctDec24']:,.0f} battery-group "
      f"tickets there; same calculation on stock-out+queue (placebo) {dz_s['excess_tickets_OctDec24']:,.0f}")

# Kyron share of the ticket's own (inserted) 2W pack, Sep-24..Jun-25, every category
mis = tk.filter(POST_KY & (pl.col("pack_vc") == "2W"))
b = mis.filter(pl.col("category") == "low_range")
mf_rows = []
for c in CATS:
    d = mis.filter(pl.col("category") == c)
    z_, p_ = proportions_ztest([int(d["is_kyron"].sum()), int(b["is_kyron"].sum())], [d.height, b.height]) if c != "low_range" else (np.nan, np.nan)
    mf_rows.append(dict(category=c, linked_2w_tickets=d.height, kyron_share=float(d["is_kyron"].mean()),
                        expected_kyron_inserted_city=float(d["p_ky_in_city"].mean()), p_vs_low_range=p_,
                        soh_at_insertion=fp_cat.loc[c, "soh_at_insertion"], soh_gap_in=fp_cat.loc[c, "soh_gap_in"], soh_gap_se=fp_cat.loc[c, "soh_gap_se"]))
mf = pd.DataFrame(mf_rows)
mf.to_csv(TAB / "q8_misfiled_vs_correct_battery_check.csv", index=False)
print("\n", mf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# 3W mixing model: app_issue as a mix of battery-like tickets (3W share of low_range+other) and service-like tickets (3W share of
# stock-out+queue+billing). Parametric bootstrap CI; sensitivity: genuine app tickets follow the swap mix (3W = 10.4%).
n3 = {r["category"]: (r["n"], r["n3w"]) for r in tk.group_by("category").agg(pl.len().alias("n"), (pl.col("rider_vc") == "3W").sum().alias("n3w"))
      .iter_rows(named=True)}
nb, kb = n3["low_range"][0] + n3["other"][0], n3["low_range"][1] + n3["other"][1]
ns, ks = sum(n3[c][0] for c in SVC + ["billing_dispute"]), sum(n3[c][1] for c in SVC + ["billing_dispute"])
na, ka = n3["app_issue"]
s_b, s_s, s_a = kb / nb, ks / ns, ka / na
f_app = (s_s - s_a) / (s_s - s_b)
f_app_swapmix = (share_3w_swaps - s_a) / (share_3w_swaps - s_b)
RNG_MIX = np.random.default_rng(7)  # separate stream so the CSAT bootstrap below is unchanged
_bb, _bs, _ba = (RNG_MIX.binomial(n_, p_, 4000) / n_ for n_, p_ in [(nb, s_b), (ns, s_s), (na, s_a)])
f_ci = np.percentile((_bs - _ba) / (_bs - _bb), [2.5, 97.5])
print(f"3W mixing model: 3W share battery-like (low_range+other) {pct(s_b, 2)}, service-like (stock-out+queue+billing) {pct(s_s, 1)}, "
      f"app_issue {pct(s_a, 2)} -> battery-like share of app_issue {pct(f_app, 1)} (95% CI {pct(f_ci[0], 1)}-{pct(f_ci[1], 1)}); "
      f"{pct(f_app_swapmix, 1)} if genuine app tickets followed the swap mix")

# %% [markdown]
# **Reading.** `other` and `app_issue` match `low_range` on the fingerprints that discriminate - ~1% 3W riders, a
# 4-4.5x same-season step-up, and a ~1.7x response to the Kyron city dose outside summer - while stock-out, queue and
# billing tickets do not (dose 0.69-0.87). The ticket's own pack is a weak signal: against the packs riders actually
# insert, Kyron is only 6-8% over-represented on battery tickets and 4% on queue tickets, so supplier mix of ticket
# packs does not separate battery from service complaints. Within supplier, city and month the pack's SoH at
# insertion is at most ~0.07 points lower: the link identifies the supplier cohort and time, not an individual bad pack.

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), sharey=True)
ylab = {"low_range": "low_range", "other": "other", "app_issue": "app_issue", "no_battery_available": "no_battery_available",
        "long_queue": "long_queue", "billing_dispute": "billing_dispute"}
yy = np.arange(len(CATS))[::-1]
colr = [PALETTE["accent"] if fp_cat.loc[c, "battery_fingerprint"] else PALETTE["primary"] for c in CATS]
ax = axes[0]
v = fp_cat["share_3w_riders"].values * 100
ax.barh(yy, v, color=colr, height=0.6)
ax.axvline(share_3w_swaps * 100, color=PALETTE["muted"], ls="--", lw=1.2)
ax.text(share_3w_swaps * 100 + 0.4, yy[0] + 0.05, f"3W share of\nall swaps {share_3w_swaps * 100:.1f}%", fontsize=9.5, color=INK2, va="center")
for y_, x_ in zip(yy, v):
    ax.text(x_ + 0.4, y_, f"{x_:.1f}%", va="center", fontsize=10, color=INK)
ax.set_yticks(yy, [ylab[c] for c in CATS])
ax.set_xlabel("% of tickets from 3W riders")
ax.set_title("1. Who complains: 2W-only?", fontsize=11.5)
ax.set_xlim(0, max(v) * 1.3)
ax = axes[1]
v = fp_cat["step_ratio"].values
ax.barh(yy, v, color=colr, height=0.6)
ax.axvline(1, color=PALETTE["muted"], ls="--", lw=1.2)
for y_, x_, c in zip(yy, v, CATS):
    ax.text(x_ + 0.1, y_, f"{x_:.1f}x  ({fp_cat.loc[c, 'rate_h1_24']:.2f} -> {fp_cat.loc[c, 'rate_h1_25']:.2f})", va="center", fontsize=9.5, color=INK)
ax.set_xlabel("H1-25 / H1-24 tickets per 1K 2W swaps\n(same season; rates in brackets)")
ax.set_title("2. When: step-up after Kyron (Sep-24)", fontsize=11.5)
ax.set_xlim(0, max(v) * 1.75)
ax = axes[2]
v = fp_cat["kyron_dose_ratio_of_ratios"].values
ax.barh(yy, v, color=colr, height=0.6)
ax.axvline(1, color=PALETTE["muted"], ls="--", lw=1.2)
for y_, x_, c in zip(yy, v, CATS):
    ax.text(x_ + 0.04, y_, f"{x_:.2f}  ({dose_tab.loc[c, 'kyron_change']:.1f}x vs {dose_tab.loc[c, 'other_change']:.1f}x)", va="center", fontsize=9.5, color=INK)
ax.set_xlabel("Jul-Aug -> Oct-Dec 24 change per 1K 2W swaps,\nKyron cities / other cities (brackets)")
ax.set_title("3. Dose: responds where Kyron went?", fontsize=11.5)
ax.set_xlim(0, max(v) * 1.75)
for a_ in axes:
    a_.grid(axis="y", visible=False)
top = headline(fig, "'Other' and 'app_issue' tickets carry the same battery fingerprint as 'low_range'",
               f"Red = battery fingerprint (<3% 3W riders and >2.5x same-season step-up). Panel 3, outside summer: {', '.join(KY_CITIES)} got "
               f"{ky_hi * 100:.0f}% Kyron 2W packs vs {ky_lo * 100:.1f}% elsewhere.\nThe ticket's own pack separates little: Kyron O/E vs the packs riders "
               f"insert is {fp_cat.loc[BATT, 'kyron_OE'].min():.2f}-{fp_cat.loc[BATT, 'kyron_OE'].max():.2f} for battery categories vs "
               f"{fp_cat.loc[SVC, 'kyron_OE'].min():.2f}-{fp_cat.loc[SVC, 'kyron_OE'].max():.2f} for stock-out/queue tickets.")
fig.tight_layout(rect=[0, 0, 1, top])
save_fig(fig, "q8_02_battery_fingerprint")

# %%
# HERO 1: agent category vs text-derived issue
fig, ax = plt.subplots(figsize=(15, 6.4))
mat = cm_row.values
im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        n_ = cm.values[i, j]
        txt = "0" if n_ == 0 else f"{n_:,}\n{mat[i, j] * 100:.0f}%"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9.5, color="white" if mat[i, j] > 0.55 else INK)
HM_LABEL = {"battery_range_health": "Battery\nrange /\nhealth", "vehicle_phrased": "Vehicle-\nphrased\n(no battery\nword)",
            "no_battery_stockout": "No battery\n/ stock-out", "long_queue": "Long\nqueue", "billing_wrong_amount": "Billing:\nwrong\namount",
            "pricing_surcharge": "Pricing:\nsurcharge\n/ extra", "app_tech": "App /\ntech", "station_hw_system": "Station\nhardware\n(QR scan)",
            "competitor_switching": "Competitor\n/ switching", "other_praise": "Other /\npraise"}
ax.set_xticks(range(len(ISSUE_ORDER)), [HM_LABEL[c] for c in ISSUE_ORDER], fontsize=10)
ax.set_yticks(range(len(CATS)), [f"{c}  (n={int(cm.loc[c].sum()):,})" for c in CATS], fontsize=10)
ax.set_xlabel("Issue derived from the rider's comment (keyword rules + template review)")
ax.set_ylabel("Agent-assigned category")
ax.grid(False)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.add_patch(plt.Rectangle((0.5, 0.5), 1, 1, fill=False, ec=PALETTE["accent"], lw=2.5))
fpx = len(ISSUE_ORDER) - 0.5 + 0.35
ax.text(fpx, -0.62, "Battery fingerprint?\n(3W share | H1 step)", fontsize=9.5, color=INK, ha="left", va="bottom", fontweight="bold")
for i, c in enumerate(CATS):
    yes = bool(fp_cat.loc[c, "battery_fingerprint"])
    ax.text(fpx, i, ("YES" if yes else "no") + f"\n3W {fp_cat.loc[c, 'share_3w_riders'] * 100:.1f}% | {fp_cat.loc[c, 'step_ratio']:.1f}x",
            fontsize=9, color=PALETTE["accent"] if yes else INK2, ha="left", va="center", fontweight="bold" if yes else "normal")
ax.set_xlim(-0.5, len(ISSUE_ORDER) - 0.5)
top = headline(fig, f"The 'other' bin is not 'other': all {n_other:,} tickets ({pct(n_other / N)}) are vehicle-phrased complaints with a battery fingerprint",
               f"Each of the 89 comment templates is filed under exactly one category, so mis-filing is systematic. Text disagrees with the category on "
               f"{pct(n_dis / N)} of {N:,} tickets.\nNo comment mentions a competitor, switching or praise. 'app_issue' text looks like app problems, "
               f"but its tickets carry the battery fingerprint (right; ~{f_app * 100:.0f}% battery-like by 3W mix).")
fig.subplots_adjust(left=0.19, right=0.86, top=top - 0.10, bottom=0.2)
save_fig(fig, "q8_01_category_vs_text_heatmap")

# %% [markdown]
# ## 5 · What the text adds (1): the true volume of battery-health complaints
# * **Floor** = `low_range` (range wording) + `other` (vehicle-phrased: no battery or range word; the battery label
#   rests on the section 4 fingerprint, not on the wording).
# * **Central** = floor + the battery-like share of `app_issue` from the 3W mixing model (~97%).
# * **Upper** = floor + all of `app_issue`.
# The comparison that matters is with station availability: stock-outs and queues combined.
# Rates are tickets from 2W riders per 1K completed 2W swaps.

# %%
mon_w = mon.to_pandas().pivot(index="month", columns="category", values="rate").reindex(columns=CATS).fillna(0)
m2w_pd = m2w.to_pandas().set_index("month")
tot_lr = n_lr
tot_lb = n_lr + n_other
tot_ub = tot_lb + n_app
tot_c = tot_lb + f_app * n_app
n_so, n_qu = int((tk["category"] == "no_battery_available").sum()), int((tk["category"] == "long_queue").sum())
print(f"battery-health complaints, full window: official low_range {tot_lr:,} ({pct(tot_lr / N)}) | floor (+ vehicle-phrased other) {tot_lb:,} = "
      f"{tot_lb / tot_lr:.2f}x ({pct(tot_lb / N)}) | central (+{pct(f_app, 0)} of app_issue) {tot_c:,.0f} ({pct(tot_c / N)}) | upper (+ all app_issue) "
      f"{tot_ub:,} = {tot_ub / tot_lr:.2f}x ({pct(tot_ub / N)})")
print(f"station availability, full window: stock-outs {pct(n_so / N)} + queues {pct(n_qu / N)} = {pct((n_so + n_qu) / N)} of tickets")
q2c = tk.filter(Q2_25).group_by("category").agg(pl.len().alias("n"))
q2c = {r["category"]: r["n"] / n_q2 for r in q2c.iter_rows(named=True)}
q2_floor, q2_app, q2_svc = q2c["low_range"] + q2c["other"], q2c["app_issue"], q2c["no_battery_available"] + q2c["long_queue"]
print(f"Apr-Jun-25 (n={n_q2:,}): battery floor {pct(q2_floor)} | central {pct(q2_floor + f_app * q2_app)} | upper {pct(q2_floor + q2_app)} vs "
      f"stock-outs {pct(q2c['no_battery_available'])} + queues {pct(q2c['long_queue'])} = {pct(q2_svc)}")
first, last = mon_w.index.min(), mon_w.index.max()
for m_ in [first, pd.Timestamp("2024-08-01"), pd.Timestamp("2024-09-01"), last]:
    r_ = mon_w.loc[m_]
    print(f"{m_:%b-%y}: per 1K 2W swaps low_range {r_['low_range']:.2f} | +other {r_['low_range'] + r_['other']:.2f} | +app {r_['low_range'] + r_['other'] + r_['app_issue']:.2f} "
          f"| Kyron share {m2w_pd.loc[m_, 'p_kyron'] * 100:.1f}% | packs <80% SoH {m2w_pd.loc[m_, 'share_lt80'] * 100:.1f}%")
batt_rate = mon_w["low_range"] + mon_w["other"] + mon_w["app_issue"]
oth_ratio = mon_w["other"] / mon_w["low_range"]
print(f"'other' / 'low_range' monthly ratio: min {oth_ratio.min():.2f}, median {oth_ratio.median():.2f}, max {oth_ratio.max():.2f} "
      f"(full window {n_other / n_lr:.2f}) -> the text changes the level, not the trend")
r_k = np.corrcoef(batt_rate, m2w_pd.loc[mon_w.index, "p_kyron"])[0, 1]
r_s = np.corrcoef(batt_rate, m2w_pd.loc[mon_w.index, "share_lt80"])[0, 1]
post_ix = [m for m in mon_w.index if m >= pd.Timestamp("2024-10-01")]
r_s_post = np.corrcoef(batt_rate.loc[post_ix], m2w_pd.loc[post_ix, "share_lt80"])[0, 1]
r_k_post = np.corrcoef(batt_rate.loc[post_ix], m2w_pd.loc[post_ix, "p_kyron"])[0, 1]
print(f"monthly battery-complaint rate vs Kyron share r={r_k:.2f} (Oct-24..Jun-25 r={r_k_post:.2f}); vs share of issued packs <80% SoH "
      f"r={r_s:.2f} (Oct-24..Jun-25 r={r_s_post:.2f})")
# Is the post-plateau 'tracks pack ageing' correlation specific? Placebo: stock-out+queue rate vs the same <80% series; detrended
# version: first differences. Both trending series correlate whatever drives them.
svc_rate = mon_w["no_battery_available"] + mon_w["long_queue"]
r_s_plc = np.corrcoef(svc_rate.loc[post_ix], m2w_pd.loc[post_ix, "share_lt80"])[0, 1]
r_s_fd = np.corrcoef(np.diff(batt_rate.loc[post_ix].values), np.diff(m2w_pd.loc[post_ix, "share_lt80"].values))[0, 1]
oct24, jan25 = pd.Timestamp("2024-10-01"), pd.Timestamp("2025-01-01")
print(f"ageing test, Oct-24..Jun-25: battery rate vs <80% SoH share r={r_s_post:.2f}, but stock-out+queue (placebo) r={r_s_plc:.2f} and first "
      f"differences r={r_s_fd:.2f} (n={len(post_ix) - 1}); battery rate rose {batt_rate.loc[oct24]:.2f} (Oct-24) -> {batt_rate.loc[jan25]:.2f} (Jan-25) "
      f"while packs <80% SoH were {m2w_pd.loc[oct24, 'share_lt80'] * 100:.1f}% -> {m2w_pd.loc[jan25, 'share_lt80'] * 100:.1f}% "
      f"-> not specific; monthly correlations cannot separate ageing from trend")
mon_w.assign(kyron_share=m2w_pd["p_kyron"], share_packs_lt80=m2w_pd["share_lt80"], mean_soh_2w=m2w_pd["soh_2w"]).to_csv(TAB / "q8_monthly_battery_complaints.csv")

# %%
# HERO 2: hidden battery complaints over time + Kyron share (aligned panel, shared time axis; no dual axis)
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 7.4), sharex=True, gridspec_kw={"height_ratios": [2.3, 1]})
x = pd.to_datetime(mon_w.index)
w_ = 20
b1 = ax.bar(x, mon_w["low_range"], width=w_, color=PALETTE["primary"], label="Filed as low_range (official battery count)")
b2 = ax.bar(x, mon_w["other"], width=w_, bottom=mon_w["low_range"], color=PALETTE["accent"],
            label="Filed as 'other': vehicle-phrased, battery fingerprint")
b3 = ax.bar(x, mon_w["app_issue"], width=w_, bottom=mon_w["low_range"] + mon_w["other"], color=PALETTE["warn"],
            label=f"Filed as 'app_issue': app wording, ~{f_app * 100:.0f}% battery-like by 3W mix")
ax.set_ylabel("Tickets per 1K 2W swaps")
ax.legend(loc="upper left", fontsize=10)
lr_last, tot_last = mon_w["low_range"].iloc[-1], batt_rate.iloc[-1]
lb_last = mon_w["low_range"].iloc[-1] + mon_w["other"].iloc[-1]
ax.annotate(f"Jun-25: official {lr_last:.1f}\n+ vehicle-phrased 'other' {lb_last:.1f} ({lb_last / lr_last:.1f}x)\n+ app_issue {tot_last:.1f} ({tot_last / lr_last:.1f}x)",
            xy=(x[-1], tot_last), xytext=(x[-5], tot_last * 1.02), fontsize=10, color=INK, ha="right", va="top",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
lr0, tot0 = mon_w["low_range"].iloc[0], batt_rate.iloc[0]
ax.text(x[0], tot0 + 0.25, f"Jan-24: {lr0:.2f} -> {tot0:.2f}", fontsize=9.5, color=INK2, ha="center")
ax.axvline(pd.Timestamp("2024-08-17"), color=INK2, lw=1, ls=":")
ax.text(pd.Timestamp("2024-08-20"), ax.get_ylim()[1] * 0.62, "Kyron packs\nenter the fleet", fontsize=9.5, color=INK2)
ax2.plot(x, m2w_pd.loc[mon_w.index, "p_kyron"] * 100, color=SUPPLIER_COLORS["Kyron"], lw=2, marker="o", ms=4, label="Kyron share of issued 2W packs")
ax2.plot(x, m2w_pd.loc[mon_w.index, "share_lt80"] * 100, color=PALETTE["muted"], lw=2, marker="s", ms=4, label="Issued 2W packs below 80% SoH")
ax2.set_ylabel("% of 2W swaps")
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax2.text(x[-1], m2w_pd["p_kyron"].iloc[-1] * 100 + 2.5, f"Kyron {m2w_pd['p_kyron'].iloc[-1] * 100:.0f}%", color=INK, fontsize=10, ha="right")
ax2.text(x[-1], m2w_pd["share_lt80"].iloc[-1] * 100 + 2.5, f"<80% SoH {m2w_pd['share_lt80'].iloc[-1] * 100:.0f}%", color=INK, fontsize=10, ha="right")
ax2.legend(loc="upper left", fontsize=9.5)
ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b-%y"))
top = headline(fig, f"Read the text and battery complaints are {tot_lb / tot_lr:.1f}-{tot_ub / tot_lr:.1f}x the official count: "
                    f"they jumped with Kyron and kept rising",
               f"Tickets from 2W riders per 1K completed 2W swaps, by agent category. The Sep-24 step coincides with Kyron (city dose test "
               f"{dz_b['ratio_of_ratios']:.2f}x; placebos {dose_tab.loc['billing (all riders, all swaps)', 'ratio_of_ratios']:.2f}-"
               f"{dz_s['ratio_of_ratios']:.2f}x).\nThe rise continued after Kyron's share plateaued, consistent with fleet ageing, but it began "
               f"while 0% of packs were below 80% SoH: monthly correlations cannot separate ageing from trend.")
fig.tight_layout(rect=[0, 0, 1, top])
save_fig(fig, "q8_03_hidden_battery_complaints_kyron")

# %% [markdown]
# ## 5b · What the text adds (2): rider-reported range in "scooter dying at X km"
# About 1,350 low_range tickets carry a number. If it is real signal, it should track the linked pack's SoH and
# supplier and move with the event log's km-per-swap.

# %%
km_ev = (events(["is_completed", "month", "vehicle_class", "km_since_last_swap"])
         .filter(pl.col("is_completed") & (pl.col("vehicle_class") == "2W"))
         .group_by("month").agg(pl.col("km_since_last_swap").median().alias("km_log_median")).collect(engine="streaming").sort("month"))
kmr = tk.filter(pl.col("km_reported").is_not_null())
n_km, n_km_bad = kmr.height, kmr.filter(~pl.col("km_reported").is_between(0, 150)).height
k2 = kmr.filter(pl.col("km_reported").is_between(0, 150) & (pl.col("rider_vc") == "2W"))
ks = k2.group_by("pack_supplier").agg(pl.len().alias("n"), pl.col("km_reported").median().alias("median_km"),
                                      pl.col("km_reported").mean().alias("mean_km"), pl.col("pack_soh").mean().alias("mean_pack_soh")).sort("pack_supplier")
ky_km = k2.filter(pl.col("pack_supplier") == "Kyron")["km_reported"].to_numpy()
nk_km = k2.filter(pl.col("pack_supplier").is_in(["Cellora", "Amptek"]))["km_reported"].to_numpy()
mw = stats.mannwhitneyu(ky_km, nk_km)
kk = k2.filter(pl.col("pack_soh").is_not_null())
r_km = stats.pearsonr(kk["km_reported"], kk["pack_soh"])
rho_km = stats.spearmanr(kk["km_reported"], kk["pack_soh"])
ols = np.polyfit(kk["pack_soh"].to_numpy(), kk["km_reported"].to_numpy(), 1)
km_m = k2.group_by("month").agg(pl.len().alias("n"), pl.col("km_reported").median().alias("km_text_median")).join(km_ev, on="month").sort("month")
r_km_month = np.corrcoef(km_m["km_text_median"], km_m["km_log_median"])[0, 1]
# the monthly r is two trending series; the within-month slope (month fixed effects) shows the km-SoH link is not just shared trend
ols_fe = smf.ols("km ~ soh + C(m)", data=pd.DataFrame({"km": kk["km_reported"].to_numpy(), "soh": kk["pack_soh"].to_numpy(),
                                                        "m": kk["month"].cast(pl.Utf8).to_numpy()})).fit()
km_fe_slope, km_fe_p = float(ols_fe.params["soh"]), float(ols_fe.pvalues["soh"])
n_km_gt60 = int((k2["km_reported"] > 60).sum())
print(f"'dying at X km' tickets {n_km:,} ({n_km_bad} implausible <0 or >150 km dropped); 2W valid {k2.height:,}; values above 60 km: {n_km_gt60} "
      f"({pct(n_km_gt60 / k2.height)}) -> truncated near 60 km (only complainers report), so medians describe complainers, not the fleet")
print(f"km vs linked-pack SoH within month (month fixed effects): {km_fe_slope:.2f} km per SoH point (p={km_fe_p:.1e}, n={kk.height:,})")
show("reported km by linked-pack supplier (2W)", ks)
print(f"Kyron vs Cellora+Amptek reported km: median {np.median(ky_km):.0f} vs {np.median(nk_km):.0f} km, Mann-Whitney p={mw.pvalue:.1e}; "
      f"km vs linked-pack SoH Pearson r={r_km.statistic:.2f} (p={r_km.pvalue:.1e}), Spearman rho={rho_km.statistic:.2f}, n={kk.height:,}; "
      f"slope {ols[0]:.2f} km per SoH point")
km_oct24 = float(km_m.filter(pl.col("month") == pl.date(2024, 10, 1))["km_text_median"].item())
print(f"monthly median reported km {km_m['km_text_median'][0]:.0f} (Jan-24) -> {km_oct24:.0f} (Oct-24, Kyron plateau) -> "
      f"{km_m['km_text_median'][-1]:.0f} (Jun-25); event-log 2W median km/swap "
      f"{km_m['km_log_median'][0]:.1f} -> {km_m['km_log_median'][-1]:.1f}; monthly r={r_km_month:.2f}")
km_m.write_csv(TAB / "q8_reported_km_monthly.csv")

fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})
for s in ["Cellora", "Amptek", "Kyron"]:
    d = kk.filter(pl.col("pack_supplier") == s)
    axa.scatter(d["pack_soh"], d["km_reported"] + RNG.uniform(-0.35, 0.35, d.height), s=12, alpha=0.45, color=SUPPLIER_COLORS[s],
                label=f"{s} (n={d.height}, median {float(d['km_reported'].median()):.0f} km)")
xs = np.linspace(float(kk["pack_soh"].min()), float(kk["pack_soh"].max()), 50)
axa.plot(xs, np.polyval(ols, xs), color=INK, lw=2)
axa.text(0.98, 0.04, f"Pearson r = {r_km.statistic:.2f} (n = {kk.height:,})\n{ols[0]:.2f} km per SoH point\nwithin month: {km_fe_slope:.2f} km/pt "
         f"(p = {km_fe_p:.0e})", fontsize=10.5, color=INK,
         ha="right", va="bottom", transform=axa.transAxes, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC"))
axa.set_xlabel("Linked pack's SoH in the ticket month (%)")
axa.set_ylabel("km at which rider says the scooter died")
axa.legend(loc="upper left", fontsize=9.5)
axa.set_title("Reported km vs pack health", fontsize=11.5)
xm = pd.to_datetime(km_m["month"].to_pandas())
axb.plot(xm, km_m["km_text_median"], color=PALETTE["accent"], lw=2, marker="o", ms=4, label="Median km in 'dying at X km' comments")
axb.plot(xm, km_m["km_log_median"], color=PALETTE["primary"], lw=2, marker="s", ms=4, label="Median km per 2W swap (event log)")
axb.set_ylabel("km")
axb.set_ylim(0, max(km_m["km_log_median"].max(), km_m["km_text_median"].max()) * 1.2)
axb.legend(loc="lower left", fontsize=9.5)
axb.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b-%y"))
axb.set_title(f"Text and telemetry fall together (monthly r = {r_km_month:.2f})", fontsize=11.5)
top = headline(fig, f"Riders' own 'dying at X km' numbers track pack health: Kyron packs die at {np.median(ky_km):.0f} km vs {np.median(nk_km):.0f} km",
               f"{k2.height:,} 2W low_range tickets that state a km figure ({n_km_bad} implausible value(s) <0 or >150 km dropped). "
               f"The category field cannot show this: only the text carries a range number.\nOnly complainers report, so values are "
               f"truncated near 60 km ({n_km_gt60} above 60): medians describe complainers, not the whole fleet.")
fig.tight_layout(rect=[0, 0, 1, top])
save_fig(fig, "q8_04_reported_range_km")

# %% [markdown]
# ## 6 · What the text adds (3): price complaints vs the base-price change and the peak/off-peak pilot
# Billing tickets per 1K completed swaps by station city. Pilot = Bengaluru + Pune from 2024-10-01 (peak = list + ₹15).
# DiD: Poisson GLM, city-month counts, offset log(swaps), city + month fixed effects, Pearson-scaled SEs.
#
# **Control-group trap.** Three of the four control cities (Delhi NCR, Hyderabad, Jaipur) received ~39% Kyron packs from
# Sep-24. Shorter range raised swaps per rider there, which deflates any per-swap ticket rate (and battery complaints
# may crowd out others). Mumbai is the only control city with a Kyron share like the pilot cities (~8.5%), so the pilot
# is also compared with Mumbai alone, and the Kyron cities are tested against Mumbai inside the control group.

# %%
swc = sw_cmv.group_by("city", "month").agg(pl.col("swaps").sum())
act = (events(["is_completed", "city", "month", "rider_id"]).filter(pl.col("is_completed"))
       .group_by("city", "month").agg(pl.col("rider_id").n_unique().alias("riders")).collect(engine="streaming"))
bill = tk.filter((pl.col("category") == "billing_dispute") & pl.col("station_city").is_not_null())
PHASES = ["1 Jan-Jun 24", "2 Jul-Sep 24", "3 Oct 24-Jun 25"]


def bill_panel(tickets):
    p = (swc.join(tickets.group_by("station_city", "month").agg(pl.len().alias("n")), left_on=["city", "month"], right_on=["station_city", "month"],
                  how="left").join(act, on=["city", "month"], how="left")
         .with_columns(pl.col("n").fill_null(0), pl.col("city").is_in(PILOT_CITIES).alias("pilot"), (pl.col("month") >= PILOT_START).alias("post_pilot"))
         .to_pandas())
    p["treat"] = (p["pilot"] & p["post_pilot"]).astype(int)
    p["month_s"] = p["month"].astype(str)
    p["t"] = (p["month"].dt.year - 2024) * 12 + p["month"].dt.month
    p["phase"] = np.select([p["month"] < pd.Timestamp(BASE_PRICE_CHANGE), p["month"] < pd.Timestamp(PILOT_START)], PHASES[:2], PHASES[2])
    return p


def pois(data, formula="n ~ treat + C(city) + C(month_s)", off="swaps"):
    return smf.glm(formula, data=data, family=sm.families.Poisson(), offset=np.log(data[off])).fit(scale="X2")


pan = bill_panel(bill)
did = pois(pan)
irr, irr_lo, irr_hi = np.exp(did.params["treat"]), *np.exp(did.conf_int().loc["treat"])
grp_rate = pan.assign(grp=np.where(pan["pilot"], "pilot", "control")).groupby(["grp", "phase"]).agg(n=("n", "sum"), swaps=("swaps", "sum"))
grp_rate["per_1k"] = grp_rate["n"] / grp_rate["swaps"] * 1000
print(grp_rate)
print(f"pilot DiD on billing tickets per swap, all four control cities: IRR {irr:.2f} (95% CI {irr_lo:.2f}-{irr_hi:.2f}), p={did.pvalues['treat']:.3f}")
# Robustness: placebo 'pilot' at the base-price change (pre-pilot months only); city-specific linear trends; active riders as
# offset. Control-group checks: (a) Mumbai as the only control; (b) inside the controls, Kyron cities after Sep-24 vs Mumbai;
# (c) billing tickets of independent riders only (they pay the PEAK price).
pre_p = pan[pan["month"] < pd.Timestamp(PILOT_START)].copy()
pre_p["treat_pl"] = (pre_p["pilot"] & (pre_p["month"] >= pd.Timestamp(BASE_PRICE_CHANGE))).astype(int)
pan_mum = pan[pan["pilot"] | (pan["city"] == "Mumbai")]
pan_ctl = pan[~pan["pilot"]].copy()
pan_ctl["kyc"] = (pan_ctl["city"].isin(KY_CITIES) & (pan_ctl["month"] >= pd.Timestamp("2024-09-01"))).astype(int)
pan_ind = bill_panel(bill.filter(pl.col("partner_id").is_null()))
fits = {
    "main: all 4 control cities, swaps offset, city+month FE": (did, "treat"),
    "placebo: 'pilot' at Jul-24, Jan-Sep 24 only": (pois(pre_p, "n ~ treat_pl + C(city) + C(month_s)"), "treat_pl"),
    "+ city-specific linear trends": (pois(pan, "n ~ treat + C(city) + C(month_s) + C(city):t"), "treat"),
    "active-riders offset, city+month FE": (pois(pan, off="riders"), "treat"),
    "(a) Mumbai-only control (pilot cities + Mumbai), swaps offset": (pois(pan_mum), "treat"),
    "(a) Mumbai-only control, active-riders offset": (pois(pan_mum, off="riders"), "treat"),
    "(b) control cities only: Kyron city x post-Sep-24 vs Mumbai": (pois(pan_ctl, "n ~ kyc + C(city) + C(month_s)"), "kyc"),
    "(c) independent riders' billing tickets, all 4 controls": (pois(pan_ind), "treat"),
}
rob = pd.DataFrame([dict(spec=k, IRR=np.exp(f.params[v]), lo=np.exp(f.conf_int().loc[v, 0]), hi=np.exp(f.conf_int().loc[v, 1]), p=f.pvalues[v])
                    for k, (f, v) in fits.items()]).set_index("spec")
with pd.option_context("display.width", 200):
    print(rob.to_string(float_format=lambda v: f"{v:.3f}"))
rob.to_csv(TAB / "q8_pricing_did_robustness.csv")
R_MUM, R_KYC, R_IND = rob.iloc[4], rob.iloc[6], rob.iloc[7]
grow = pan.groupby("pilot").apply(lambda d: d.loc[d["month"] == d["month"].max(), "swaps"].sum() / d.loc[d["month"] == d["month"].min(), "swaps"].sum())
print(f"swap growth Jan-24 -> Jun-25: pilot cities {grow[True]:.2f}x, control cities {grow[False]:.2f}x")
# (d) billing tickets per 1K swaps by city and phase
cty = pan.groupby(["city", "phase"]).agg(n=("n", "sum"), swaps=("swaps", "sum"))
cty_w = (cty["n"] / cty["swaps"] * 1000).unstack("phase")[PHASES].reindex(CITIES)
cty_w["post_vs_pre"] = cty_w[PHASES[2]] / cty_w[PHASES[0]]
cty_w["group"] = np.where(cty_w.index.isin(PILOT_CITIES), "pilot", np.where(cty_w.index.isin(KY_CITIES), "control: Kyron city", "control: Mumbai"))
cty_w.to_csv(TAB / "q8_billing_rate_by_city.csv")
print("billing tickets per 1K swaps by city, Jan-Jun-24 -> Oct-24..Jun-25: " + "; ".join(
    f"{c} {r[PHASES[0]]:.2f} -> {r[PHASES[2]]:.2f} ({r['post_vs_pre']:.2f}x)" for c, r in cty_w.iterrows()))
# why the Kyron cities' per-swap rates fall faster: shorter range -> more swaps per 2W rider (deflates every per-swap rate)
spr = (events(["is_completed", "city", "month", "rider_id", "vehicle_class"])
       .filter(pl.col("is_completed") & (pl.col("vehicle_class") == "2W") & (JA24 | OD24))
       .group_by(pl.col("city").is_in(KY_CITIES).alias("ky"), "month").agg(pl.len().alias("sw"), pl.col("rider_id").n_unique().alias("riders"))
       .collect(engine="streaming").with_columns((pl.col("sw") / pl.col("riders")).alias("spr"), JA24.alias("ja")))
spr_t = {(r["ky"], r["ja"]): r["spr"] for r in spr.group_by("ky", "ja").agg(pl.col("spr").mean()).iter_rows(named=True)}
print(f"completed swaps per active 2W rider per month, Jul-Aug-24 -> Oct-Dec-24: Kyron cities {spr_t[(True, True)]:.1f} -> {spr_t[(True, False)]:.1f}; "
      f"other cities {spr_t[(False, True)]:.1f} -> {spr_t[(False, False)]:.1f}")
bt = bill.with_columns(pl.col("station_city").is_in(PILOT_CITIES).alias("pilot"),
                       pl.when(pl.col("month") < BASE_PRICE_CHANGE).then(pl.lit("1 Jan-Jun 24")).when(pl.col("month") < PILOT_START)
                       .then(pl.lit("2 Jul-Sep 24")).otherwise(pl.lit("3 Oct 24-Jun 25")).alias("phase"),
                       (pl.col("rider_comment") == "surcharge laga diya bina bataye").alias("surcharge_txt"))
sur = bt.group_by("pilot", "phase").agg(pl.len().alias("n"), pl.col("surcharge_txt").mean().alias("surcharge_share")).sort("pilot", "phase")
show("share of billing tickets saying 'surcharge laga diya bina bataye' (surcharge applied without notice)", sur)
ct = pd.crosstab(bt.filter(pl.col("pilot"))["phase"].to_pandas(), bt.filter(pl.col("pilot"))["surcharge_txt"].to_pandas())
chi_s = stats.chi2_contingency(ct)
print(f"pilot cities: surcharge-template share by phase chi2 p={chi_s[1]:.2f}")
sur_tab = sur.to_pandas()
s_pil_pre = float(sur_tab.query("pilot and phase == '2 Jul-Sep 24'")["surcharge_share"].iloc[0])
s_pil_post = float(sur_tab.query("pilot and phase == '3 Oct 24-Jun 25'")["surcharge_share"].iloc[0])
s_con_post = float(sur_tab.query("not pilot and phase == '3 Oct 24-Jun 25'")["surcharge_share"].iloc[0])
s_con_pre0 = float(sur_tab.query("not pilot and phase == '1 Jan-Jun 24'")["surcharge_share"].iloc[0])
# base price change, all cities (no same-season baseline exists: caveat)
b_pre = grp_rate.xs("1 Jan-Jun 24", level="phase")[["n", "swaps"]].sum()
b_mid = grp_rate.xs("2 Jul-Sep 24", level="phase")[["n", "swaps"]].sum()
print(f"all cities billing per 1K swaps: Jan-Jun-24 {b_pre['n'] / b_pre['swaps'] * 1000:.2f} -> Jul-Sep-24 {b_mid['n'] / b_mid['swaps'] * 1000:.2f}")
billvc = (tk.filter(pl.col("category") == "billing_dispute").group_by("rider_vc", "month").agg(pl.len().alias("n"))
          .join(sw_mvs.group_by("vehicle_class", "month").agg(pl.col("swaps").sum()), left_on=["rider_vc", "month"], right_on=["vehicle_class", "month"])
          .with_columns(pl.when(pl.col("month") < BASE_PRICE_CHANGE).then(pl.lit("1 Jan-Jun 24")).when(pl.col("month") < PILOT_START)
                        .then(pl.lit("2 Jul-Sep 24")).otherwise(pl.lit("3 Oct 24-Jun 25")).alias("phase"))
          .group_by("rider_vc", "phase").agg((pl.col("n").sum() / pl.col("swaps").sum() * 1000).alias("billing_per_1k")).sort("rider_vc", "phase"))
show("billing tickets per 1K swaps by vehicle class and pricing phase", billvc)

G3 = {"pilot": "Pilot cities (Bengaluru, Pune)", "mum": "Mumbai (control, low Kyron)", "ky": f"Kyron cities ({', '.join(KY_CITIES)})"}
pan["grp3"] = np.where(pan["pilot"], G3["pilot"], np.where(pan["city"] == "Mumbai", G3["mum"], G3["ky"]))
mp = pan.groupby(["grp3", "month"]).agg(n=("n", "sum"), swaps=("swaps", "sum"))
mp["per_1k"] = mp["n"] / mp["swaps"] * 1000
fig, (axa, axb) = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
for g, col, mk in [(G3["pilot"], PALETTE["accent"], "o"), (G3["mum"], PALETTE["good"], "s"), (G3["ky"], PALETTE["primary"], "^")]:
    d = mp.xs(g, level="grp3")
    axa.plot(pd.to_datetime(d.index), d["per_1k"], color=col, lw=2, marker=mk, ms=4, label=g)
axa.set_ylim(0, 3.0)
for dte, lab, ha_, dx in [(BASE_PRICE_CHANGE, "Base price\n+₹5 / +₹10", "right", -4), ("2024-09-01", "Kyron\npacks", "right", -4),
                          (PILOT_START, "Peak/off-peak\npilot starts", "left", 4)]:
    axa.axvline(pd.Timestamp(dte), color=INK2, ls=":", lw=1)
    axa.text(pd.Timestamp(dte) + pd.Timedelta(days=dx), 2.95, lab, fontsize=9.5, color=INK2, va="top", ha=ha_)
axa.set_ylabel("Billing tickets per 1K completed swaps")
axa.legend(loc="lower left", fontsize=9.5)
axa.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b-%y"))
axa.set_title(f"Pilot vs Mumbai IRR {R_MUM['IRR']:.2f} ({R_MUM['lo']:.2f}-{R_MUM['hi']:.2f}). The {irr:.2f} vs all four controls comes from\n"
              f"Kyron cities falling faster from Sep-24, before the pilot (vs Mumbai IRR {R_KYC['IRR']:.2f}, {R_KYC['lo']:.2f}-{R_KYC['hi']:.2f})", fontsize=11)
phases = ["1 Jan-Jun 24", "2 Jul-Sep 24", "3 Oct 24-Jun 25"]
xx = np.arange(3)
for k_, (flag, col, lab) in enumerate([(True, PALETTE["accent"], "Pilot cities"), (False, PALETTE["primary"], "Control cities")]):
    vals = [float(sur_tab.query(f"pilot == {flag} and phase == '{p}'")["surcharge_share"].iloc[0]) * 100 for p in phases]
    axb.bar(xx + (k_ - 0.5) * 0.36, vals, width=0.34, color=col, label=lab)
    for xi, v_ in zip(xx + (k_ - 0.5) * 0.36, vals):
        axb.text(xi, v_ + 0.8, f"{v_:.0f}%", ha="center", fontsize=9.5, color=INK)
axb.set_xticks(xx, ["Jan-Jun 24", "Jul-Sep 24\n(new base price)", "Oct 24-Jun 25\n(pilot live)"])
axb.set_ylabel("% of billing tickets citing a 'surcharge'")
axb.set_ylim(0, 50)
axb.legend(loc="upper left", fontsize=9.5)
axb.set_title("'Surcharge' wording does not track the surcharge", fontsize=11.5)
top = headline(fig, "Billing complaints fell alike in pilot cities and Mumbai: the pilot shows no billing effect, and the text cannot explain "
                    "billing complaints either",
               f"Billing-dispute tickets by station city. Independent riders, who pay the peak price, show no effect either (IRR {R_IND['IRR']:.2f}, "
               f"p={R_IND['p']:.2f}). 'Surcharge laga diya bina bataye' is ~1/3 of billing tickets\nin every city and phase (pilot cities "
               f"{s_pil_pre * 100:.0f}% -> {s_pil_post * 100:.0f}%, chi2 p={chi_s[1]:.2f}), including control cities that never had a surcharge "
               f"({s_con_post * 100:.0f}%).")
fig.tight_layout(rect=[0, 0, 1, top])
save_fig(fig, "q8_05_price_complaints_pilot")

# %% [markdown]
# ## 7 · What the text adds (4): urgency, repeat-failure claims, competitor mentions and churn
# * Does agent `priority` reflect the text or change handling?
# * Is "empty station again, this is the third time this week" borne out by the rider's logged stock-outs?
# * Lapse = rider's last completed swap falls within 30 days after the ticket (tickets before 2025-06-01 so a full
#   30-day window is observed). Logit with month, city, vehicle class, fleet affiliation and tenure controls;
#   rider-clustered SEs. Association, not causation: riders who complain may already be disengaging.

# %%
ct_p = pd.crosstab(tk["template_id"].to_pandas(), tk["priority"].to_pandas())
chi_p = stats.chi2_contingency(ct_p)
cramer = np.sqrt(chi_p[0] / (ct_p.values.sum() * (min(ct_p.shape) - 1)))
pr = tk.group_by("priority").agg(pl.len().alias("n"), pl.col("resolution_hours").median().alias("median_res_h"),
                                 (pl.col("resolution_status") == "unresolved").mean().alias("unresolved"), pl.col("has_csat").mean().alias("csat_response")).sort("priority")
kw = stats.kruskal(*[tk.filter((pl.col("priority") == p) & pl.col("resolution_hours").is_not_null())["resolution_hours"].to_numpy() for p in ["low", "medium", "high"]])
hi_urg = tk.group_by("urgency_business_impact").agg((pl.col("priority") == "high").mean().alias("high_share"), pl.len().alias("n"))
show("priority vs handling", pr)
print(f"template x priority chi2 p={chi_p[1]:.2f}, Cramer's V={cramer:.3f}; resolution hours by priority Kruskal-Wallis p={kw.pvalue:.2f}")
show("high-priority share by business-impact wording ('late ... delivery')", hi_urg)

so_ev = events(["rider_id", "event_ts", "is_stockout"]).filter(pl.col("is_stockout")).select("rider_id", "event_ts").collect(engine="streaming")


def stockouts_prior_7d(df):
    j = (df.select("ticket_id", "rider_id", "created_ts").join(so_ev, on="rider_id", how="inner")
         .filter((pl.col("event_ts") <= pl.col("created_ts")) & (pl.col("event_ts") > pl.col("created_ts") - pl.duration(days=7)))
         .group_by("ticket_id").agg(pl.len().alias("so7")))
    return df.join(j, on="ticket_id", how="left").with_columns(pl.col("so7").fill_null(0))


st = stockouts_prior_7d(tk.filter(pl.col("category") == "no_battery_available"))
st_sum = st.group_by("urgency_repeat").agg(pl.len().alias("n"), pl.col("so7").mean().alias("mean_stockouts_7d"),
                                           (pl.col("so7") >= 1).mean().alias("ge1"), (pl.col("so7") >= 3).mean().alias("ge3"))
pl_b = stockouts_prior_7d(tk.filter(pl.col("category") == "billing_dispute"))
show("stock-out tickets: logged stock-outs in prior 7 days, 'third time this week' vs other stock-out wording", st_sum)
print(f"placebo - billing tickets with >=1 logged stock-out in prior 7 days: {pct(float((pl_b['so7'] >= 1).mean()))}")

ISSUE5 = {"battery_range_health": "Battery (range or vehicle wording)", "vehicle_phrased": "Battery (range or vehicle wording)",
          "app_tech": "App / QR (battery fingerprint)", "station_hw_system": "App / QR (battery fingerprint)",
          "no_battery_stockout": "Stock-out", "long_queue": "Long queue", "billing_wrong_amount": "Billing / pricing", "pricing_surcharge": "Billing / pricing"}
I5_ORDER = ["Battery (range or vehicle wording)", "App / QR (battery fingerprint)", "Stock-out", "Long queue", "Billing / pricing"]
tk = tk.with_columns(pl.col("text_issue").replace_strict(ISSUE5, default="Other").alias("issue5"))
lp = (tk.filter((pl.col("created_ts") < pl.datetime(2025, 6, 1)) & pl.col("last_swap_ts").is_not_null() & pl.col("rider_vc").is_not_null())
      .with_columns((pl.col("last_swap_ts") < pl.col("created_ts") + pl.duration(days=30)).alias("lapse30"),
                    ((pl.col("created_ts") - pl.col("first_swap_ts")).dt.total_hours() / 24).clip(0, None).alias("tenure_days"),
                    pl.col("station_city").fill_null("unknown").alias("city_f"), pl.col("partner_id").is_not_null().alias("fleet"),
                    pl.col("month").cast(pl.Utf8).alias("month_s"))
      .to_pandas())
lp["lapse30"] = lp["lapse30"].astype(int)
lp_rate = lp.groupby("issue5").agg(n=("lapse30", "size"), lapse30=("lapse30", "mean")).reindex(I5_ORDER)
lg = smf.logit("lapse30 ~ C(issue5, Treatment('Billing / pricing')) + C(month_s) + C(city_f) + C(rider_vc) + fleet + np.log1p(tenure_days)",
               data=lp).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(lp["rider_id"])[0]})
orr = pd.DataFrame({"OR": np.exp(lg.params), "lo": np.exp(lg.conf_int()[0]), "hi": np.exp(lg.conf_int()[1]), "p": lg.pvalues})
orr = orr[orr.index.str.startswith("C(issue5")]
orr.index = orr.index.str.extract(r"\[T\.(.*)\]")[0].values
lp_rate = lp_rate.join(orr)
print(f"\nlapse-30 sample {len(lp):,} tickets; overall lapse {lp['lapse30'].mean():.3f}")
print(lp_rate.to_string(float_format=lambda v: f"{v:.3f}"))
# Right-censoring: 'last swap < ticket + 30 d' needs time after the window to tell a lapse from a pause. Data end 30-Jun-25, so
# Apr/May-25 tickets have <= 60 days of follow-up and their lapse rates are inflated. Month FE absorb this in the ORs; the level
# is reported on tickets before Apr-25, and the logit is refit on that sample as a check.
lp_m = lp.groupby(lp["month"].dt.strftime("%b-%y"), sort=False)["lapse30"].mean()
lp_ok = lp[lp["created_ts"] < pd.Timestamp("2025-04-01")]
lg_ok = smf.logit("lapse30 ~ C(issue5, Treatment('Billing / pricing')) + C(month_s) + C(city_f) + C(rider_vc) + fleet + np.log1p(tenure_days)",
                  data=lp_ok).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(lp_ok["rider_id"])[0]})
or_ok = np.exp(lg_ok.params[lg_ok.params.index.str.startswith("C(issue5")])
p_ok = lg_ok.pvalues[lg_ok.pvalues.index.str.startswith("C(issue5")]
lp_rate["lapse30_pre_apr25"] = lp_ok.groupby("issue5")["lapse30"].mean().reindex(I5_ORDER)
lp_rate["OR_pre_apr25"] = pd.Series(or_ok.values, index=or_ok.index.str.extract(r"\[T\.(.*)\]")[0].values).reindex(I5_ORDER)
print(f"lapse by ticket month (right-censored at the end): {', '.join(f'{k} {v:.3f}' for k, v in lp_m.tail(5).items())}")
print(f"tickets before Apr-25: n={len(lp_ok):,}, lapse30 {lp_ok['lapse30'].mean():.3f}; issue-group ORs vs billing {or_ok.min():.2f}-{or_ok.max():.2f} "
      f"(min p={p_ok.min():.2f}) vs full sample {lp_rate['OR'].min():.2f}-{lp_rate['OR'].max():.2f} (min p={lp_rate['p'].min():.2f})")
lp_rate.to_csv(TAB / "q8_lapse_after_ticket_by_issue.csv")


def two_prop(mask_a, mask_b, label):
    a_, b_ = lp[mask_a]["lapse30"], lp[mask_b]["lapse30"]
    z, p = proportions_ztest([a_.sum(), b_.sum()], [len(a_), len(b_)])
    print(f"{label}: {a_.mean():.3f} (n={len(a_):,}) vs {b_.mean():.3f} (n={len(b_):,}), z={z:.2f}, p={p:.2f}")
    return a_.mean(), b_.mean(), p


lap_third = two_prop((lp["category"] == "no_battery_available") & lp["urgency_repeat"], (lp["category"] == "no_battery_available") & ~lp["urgency_repeat"],
                     "lapse: 'third time this week' vs other stock-out wording")
lap_late = two_prop((lp["category"] == "long_queue") & lp["urgency_business_impact"], (lp["category"] == "long_queue") & ~lp["urgency_business_impact"],
                    "lapse: 'late ho gaya delivery mein' vs other queue wording")
n_comp = int((tk["text_issue"] == "competitor_switching").sum())
print(f"competitor / switching-intent mentions: {n_comp} tickets, {int((tax['text_issue'] == 'competitor_switching').sum())} of 89 templates")

# %% [markdown]
# ## 8 · CSAT: who answers, and does reweighting change the score?
# CSAT exists only for resolved tickets (0% for unresolved and duplicate), so those are outside any reweighting
# (no overlap) and are bounded with scenarios instead. For resolved tickets we model P(has CSAT) with a logistic
# regression on resolution-time decile, category, channel, priority, month and city, then weight respondents by
# P(respond)/p_hat (stabilised IPW). IPW fixes bias from *observed* drivers of response only; if unhappy riders skip
# the survey for reasons not in the model (true MNAR), the delta-adjustment rows show how far the mean could move.

# %%
res = tk.filter(pl.col("resolution_status") == "resolved").with_columns(pl.col("station_city").fill_null("unknown").alias("city_f"),
                                                                       pl.col("month").cast(pl.Utf8).alias("month_s")).to_pandas()
res["res_dec"] = pd.qcut(res["resolution_hours"], 10, labels=False, duplicates="drop")
res["res_q"] = pd.qcut(res["resolution_hours"], 4, labels=["Q1 fastest", "Q2", "Q3", "Q4 slowest"])
FEATS = ["res_dec", "category", "channel", "priority", "month_s", "city_f"]
Xd = pd.get_dummies(res[FEATS].astype(str), drop_first=True).astype(float).values
yv = res["has_csat"].astype(int).values
sv = res["csat_score"].values


def ipw_fit(idx):
    m = LogisticRegression(C=1e4, max_iter=3000)
    m.fit(Xd[idx], yv[idx])
    p = m.predict_proba(Xd[idx])[:, 1]
    return p


p_hat = ipw_fit(np.arange(len(res)))
auc = roc_auc_score(yv, p_hat)
r_rate = yv.mean()
wts = np.where(yv == 1, r_rate / p_hat, 0.0)
resp = yv == 1
naive = float(np.mean(sv[resp]))
ipw = float(np.average(sv[resp], weights=wts[resp]))
ess = wts[resp].sum() ** 2 / (wts[resp] ** 2).sum()
boot = []
for _ in range(40):
    idx = RNG.integers(0, len(res), len(res))
    p_b = ipw_fit(idx)
    y_b, s_b = yv[idx], sv[idx]
    w_b = np.where(y_b == 1, y_b.mean() / p_b, 0.0)
    boot.append((np.mean(s_b[y_b == 1]), np.average(s_b[y_b == 1], weights=w_b[y_b == 1])))
boot = np.array(boot)
ci_naive, ci_ipw, ci_diff = np.percentile(boot[:, 0], [2.5, 97.5]), np.percentile(boot[:, 1], [2.5, 97.5]), np.percentile(boot[:, 1] - boot[:, 0], [2.5, 97.5])
print(f"resolved {len(res):,}; response {r_rate:.3f}; response-model AUC {auc:.3f}; weights min {wts[resp].min():.2f} max {wts[resp].max():.2f}; ESS {ess:,.0f} of {resp.sum():,}")
print(f"CSAT naive {naive:.3f} (95% CI {ci_naive[0]:.3f}-{ci_naive[1]:.3f}) | IPW {ipw:.3f} ({ci_ipw[0]:.3f}-{ci_ipw[1]:.3f}) | "
      f"IPW - naive {ipw - naive:+.3f} ({ci_diff[0]:+.3f} to {ci_diff[1]:+.3f}); 40 bootstrap refits")
rq = res.groupby("res_q", observed=True).agg(n=("has_csat", "size"), response=("has_csat", "mean"), csat_mean=("csat_score", "mean"),
                                             hours_min=("resolution_hours", "min"), hours_max=("resolution_hours", "max"))
print(rq.to_string(float_format=lambda v: f"{v:.3f}"))
rqc = res.pivot_table(index="category", columns="res_q", values="has_csat", aggfunc="mean", observed=True).reindex(CATS)
rqc.to_csv(TAB / "q8_csat_response_by_speed_category.csv")
print("response rate by category x speed quartile\n", rqc.to_string(float_format=lambda v: f"{v:.3f}"))
dec = res.groupby("res_dec").agg(hours_med=("resolution_hours", "median"), response=("has_csat", "mean"), csat=("csat_score", "mean"),
                                 csat_sd=("csat_score", "std"), n_resp=("has_csat", "sum"))
rho_speed = stats.spearmanr(res.loc[resp, "resolution_hours"], sv[resp])
print(f"among respondents, CSAT vs resolution hours Spearman rho={rho_speed.statistic:.3f} (p={rho_speed.pvalue:.2f})")
# by issue group
res["w"] = wts
res["issue5"] = res["text_issue"].map(ISSUE5)
cs_issue = []
for g in I5_ORDER:
    d = res[(res["issue5"] == g) & (res["has_csat"])]
    cs_issue.append(dict(issue=g, respondents=len(d), naive=d["csat_score"].mean(), ipw=np.average(d["csat_score"], weights=d["w"])))
cs_issue = pd.DataFrame(cs_issue).set_index("issue")
print(cs_issue.to_string(float_format=lambda v: f"{v:.3f}"))
# what IPW cannot fix
n_res_, n_unres = len(res), int((tk["resolution_status"] == "unresolved").sum())
n_dup = int((tk["resolution_status"] == "duplicate").sum())
scen = {"Naive (respondents only)": naive, "IPW (resolved tickets)": ipw}
for dlt in [0.5, 1.0]:
    scen[f"Non-respondents {dlt:.1f} pt lower (MNAR)"] = ipw - (1 - r_rate) * dlt
for s_ in [2.0, 1.0]:
    scen[f"+ unresolved tickets scored {s_:.0f}"] = (n_res_ * ipw + n_unres * s_) / (n_res_ + n_unres)
scen = pd.Series(scen)
print(f"never surveyed: unresolved {n_unres:,} + duplicate {n_dup:,} = {pct((n_unres + n_dup) / N)} of tickets")
print(scen.to_string(float_format=lambda v: f"{v:.3f}"))
scen.to_csv(TAB / "q8_csat_naive_ipw_scenarios.csv")

# %%
# HERO 3: CSAT naive vs adjusted
fig = plt.figure(figsize=(15.5, 5.6))
axs = [fig.add_axes([0.055, 0.15, 0.225, 0.58]), fig.add_axes([0.345, 0.15, 0.225, 0.58]), fig.add_axes([0.765, 0.15, 0.215, 0.58])]
xd = np.arange(1, len(dec) + 1)
xt = [f"D{i}\n{h:.0f}h" for i, h in zip(xd, dec["hours_med"].values)]
axs[0].plot(xd, dec["response"] * 100, color=PALETTE["primary"], lw=2, marker="o", ms=5)
axs[0].set_xticks(xd, xt, fontsize=9)
axs[0].set_xlabel("Resolution-time decile (median hours)")
axs[0].set_ylabel("% of resolved tickets with a CSAT score")
axs[0].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
axs[0].set_ylim(0, 55)
axs[0].set_title("1. Slow tickets answer less", fontsize=11.5)
axs[0].annotate(f"{dec['response'].iloc[-1] * 100:.0f}%", (xd[-1], dec["response"].iloc[-1] * 100), xytext=(-12, -18), textcoords="offset points", fontsize=10, color=INK)
axs[0].annotate(f"{dec['response'].iloc[0] * 100:.0f}%", (xd[0], dec["response"].iloc[0] * 100), xytext=(-5, 8), textcoords="offset points", fontsize=10, color=INK)
se = dec["csat_sd"] / np.sqrt(dec["n_resp"])
axs[1].errorbar(xd, dec["csat"], yerr=1.96 * se, color=PALETTE["accent"], lw=2, marker="o", ms=5, capsize=3)
axs[1].set_xticks(xd, xt, fontsize=9)
axs[1].set_xlabel("Resolution-time decile (median hours)")
axs[1].set_ylabel("Mean CSAT of respondents (1-5, 95% CI)")
axs[1].set_ylim(1, 5)
axs[1].set_title(f"2. ...but score the same (rho = {rho_speed.statistic:.2f})", fontsize=11.5)
labels = list(scen.index)
yy = np.arange(len(labels))[::-1]
cols = [PALETTE["muted"], PALETTE["primary"], PALETTE["warn"], PALETTE["warn"], PALETTE["accent"], PALETTE["accent"]]
axs[2].hlines(yy, 1, scen.values, color="#DDDDDD", lw=2)
axs[2].scatter(scen.values, yy, c=cols, s=80, zorder=3)
axs[2].errorbar([naive, ipw], yy[:2], xerr=[[naive - ci_naive[0], ipw - ci_ipw[0]], [ci_naive[1] - naive, ci_ipw[1] - ipw]], fmt="none", ecolor=INK, capsize=3)
for y_, v_ in zip(yy, scen.values):
    axs[2].text(v_ + 0.04, y_ + 0.18, f"{v_:.2f}", fontsize=10.5, color=INK)
axs[2].set_yticks(yy, labels, fontsize=10)
axs[2].set_xlim(1, 4.2)
axs[2].set_xlabel("Mean CSAT (1-5)")
axs[2].set_title("3. Naive vs adjusted vs what IPW cannot see", fontsize=11.5)
axs[2].grid(axis="y", visible=False)
top = headline(fig, f"CSAT survives the non-response trap (IPW moves it {ipw - naive:+.3f}); the blind spot is the "
                    f"{pct((n_unres + n_dup) / N, 0)} of tickets never surveyed",
               f"Resolved tickets: {pct(r_rate, 0)} answer, {rq['response'].iloc[0] * 100:.0f}% in the fastest quartile vs {rq['response'].iloc[-1] * 100:.0f}% in the slowest, "
               f"but scores do not depend on speed. Response model AUC {auc:.2f}.\nUnresolved and duplicate tickets are never surveyed, so no "
               f"weighting can recover them; the four bottom rows are scenarios, not estimates.")
save_fig(fig, "q8_06_csat_naive_vs_ipw")

# %% [markdown]
# ## 9 · Resolution by true issue, and ticket rates per 1K swaps by city, station and supplier

# %%
rb = (tk.group_by("issue5").agg(pl.len().alias("tickets"), (pl.col("resolution_status") == "unresolved").mean().alias("unresolved"),
                                (pl.col("resolution_status") == "duplicate").mean().alias("duplicate"),
                                pl.col("resolution_hours").median().alias("median_res_h"), pl.col("resolution_hours").quantile(0.9).alias("p90_res_h"),
                                pl.col("has_csat").mean().alias("csat_response"))
      .to_pandas().set_index("issue5").reindex(I5_ORDER))
rb = rb.join(cs_issue[["naive", "ipw"]].rename(columns={"naive": "csat_naive", "ipw": "csat_ipw"})).join(lp_rate[["lapse30"]])
print(rb.to_string(float_format=lambda v: f"{v:.3f}"))
rb.to_csv(TAB / "q8_resolution_by_true_issue.csv")
chi_u = stats.chi2_contingency(pd.crosstab(tk["issue5"].to_pandas(), (tk["resolution_status"] == "unresolved").to_pandas()))
print(f"unresolved share differs by issue? chi2 p={chi_u[1]:.2f}")

# city-month rates
svc = (tk.filter(pl.col("category").is_in(["no_battery_available", "long_queue"]) & pl.col("station_city").is_not_null())
       .group_by("station_city", "month").agg(pl.len().alias("n"))
       .join(swc, left_on=["station_city", "month"], right_on=["city", "month"]).with_columns((pl.col("n") / pl.col("swaps") * 1000).alias("rate")))
sw2c = sw_cmv.filter(pl.col("vehicle_class") == "2W").select("city", "month", pl.col("swaps").alias("swaps_2w"))
btc = (tk.filter(pl.col("category").is_in(["low_range", "other", "app_issue"]) & pl.col("station_city").is_not_null() & (pl.col("rider_vc") == "2W"))
       .group_by("station_city", "month").agg(pl.len().alias("n"))
       .join(sw2c, left_on=["station_city", "month"], right_on=["city", "month"]).with_columns((pl.col("n") / pl.col("swaps_2w") * 1000).alias("rate")))
# pooled H1-2025 city rates (large counts) + what differs between cities: Kyron share, pack SoH, heat
heat = (city_daily().filter(pl.col("date").is_between(pl.date(2025, 1, 1), pl.date(2025, 6, 30)))
        .group_by("city").agg(pl.col("max_temp_c").mean().alias("mean_max_temp_h1_25")))
kyc = (events(["is_completed", "city", "month", "vehicle_class", "supplier_out", "soh_out_pct"])
       .filter(pl.col("is_completed") & (pl.col("vehicle_class") == "2W") & (pl.col("month") >= pl.date(2025, 1, 1)))
       .group_by("city").agg((pl.col("supplier_out") == "Kyron").mean().alias("kyron_share_2w"), pl.col("soh_out_pct").mean().alias("mean_issued_soh"))
       .collect(engine="streaming"))
city_h1 = (svc.filter(H1_25).group_by("station_city").agg((pl.col("n").sum() / pl.col("swaps").sum() * 1000).alias("svc_per_1k"))
           .join(btc.filter(H1_25).group_by("station_city").agg((pl.col("n").sum() / pl.col("swaps_2w").sum() * 1000).alias("batt_per_1k_2w")), on="station_city")
           .join(kyc, left_on="station_city", right_on="city").join(heat, left_on="station_city", right_on="city").sort("batt_per_1k_2w", descending=True))
show("H1-2025 by city: service and battery ticket rates vs Kyron share, issued SoH and heat", city_h1)
city_h1.write_csv(TAB / "q8_city_ticket_rates_h1_2025.csv")
spread_svc = float(city_h1["svc_per_1k"].max() / city_h1["svc_per_1k"].min())
spread_bat = float(city_h1["batt_per_1k_2w"].max() / city_h1["batt_per_1k_2w"].min())
r_heat = np.corrcoef(city_h1["batt_per_1k_2w"], city_h1["mean_max_temp_h1_25"])[0, 1]
r_kyc = np.corrcoef(city_h1["batt_per_1k_2w"], city_h1["kyron_share_2w"])[0, 1]
print(f"H1-25 max/min city ratio: stock-out+queue tickets {spread_svc:.1f}x vs battery tickets {spread_bat:.1f}x; across 6 cities battery rate vs "
      f"mean max temp r={r_heat:.2f}, vs Kyron share r={r_kyc:.2f} (n=6, descriptive only)")
# heat and Kyron coincide across cities; separate them in time: before Kyron (H1-24) the city gap is heat (and anything else) only.
# KY_CITIES, ky_hi, ky_lo and the non-summer dose test (Jul-Aug-24 -> Oct-Dec-24, with placebos) come from section 4.


def ky_city_ratio(period):
    g = (btc.filter(period).with_columns(pl.col("station_city").is_in(KY_CITIES).alias("ky"))
         .group_by("ky").agg((pl.col("n").sum() / pl.col("swaps_2w").sum() * 1000).alias("r")))
    return float(g.filter(pl.col("ky"))["r"].item()), float(g.filter(~pl.col("ky"))["r"].item())


(h24, m24), (h25, m25) = ky_city_ratio(H1_24), ky_city_ratio(H1_25)
print(f"high-Kyron cities {KY_CITIES} (Kyron {ky_hi * 100:.0f}% of issued 2W packs in H1-25) vs others ({ky_lo * 100:.1f}%): battery tickets per 1K 2W swaps "
      f"H1-24 {h24:.2f} vs {m24:.2f} (ratio {h24 / m24:.2f}, pre-Kyron, includes the hot-city summer spike) -> H1-25 {h25:.2f} vs {m25:.2f} "
      f"(ratio {h25 / m25:.2f}); increase {h25 - h24:+.2f} vs {m25 - m24:+.2f}")
print(f"non-summer dose test (section 4), battery group per 1K 2W swaps: high-Kyron cities {dz_b['kyron_cities_JulAug24']:.2f} -> "
      f"{dz_b['kyron_cities_OctDec24']:.2f} ({dz_b['kyron_change']:.2f}x); other cities {dz_b['other_cities_JulAug24']:.2f} -> "
      f"{dz_b['other_cities_OctDec24']:.2f} ({dz_b['other_change']:.2f}x); ratio of ratios {dz_b['ratio_of_ratios']:.2f}")
peak = svc.filter(pl.col("month") == pl.date(2025, 5, 1)).sort("rate", descending=True)
show("May-25 stock-out+queue tickets per 1K swaps by city", peak.select("station_city", "n", "swaps", "rate"))
show("Jun-25 battery-group tickets per 1K 2W swaps by city", btc.filter(pl.col("month") == pl.date(2025, 6, 1)).sort("rate", descending=True).select("station_city", "n", "swaps_2w", "rate"))
# Supplier attribution per ticket pack, within city, Sep-24..Jun-25: tickets whose battery_id is a Kyron (vs Cellora+Amptek) 2W
# pack, per 1K 2W attempts at which riders INSERTED a Kyron (vs Cellora+Amptek) pack - the ticket pack is the inserted pack.
# Placebo: stock-out+queue tickets. The issued-pack denominator used originally is kept below, labelled superseded.
att_city = (ins_cms.filter(pl.col("month") >= pl.date(2024, 9, 1)).group_by("city", (pl.col("supplier_in") == "Kyron").alias("ky"))
            .agg(pl.col("att").sum().alias("den")))
sup_city_sw = (sw_cmvs.filter((pl.col("vehicle_class") == "2W") & (pl.col("month") >= pl.date(2024, 9, 1)))
               .group_by("city", (pl.col("supplier_out") == "Kyron").alias("ky")).agg(pl.col("swaps").sum().alias("den")))


def within_city(cats, denom):
    t = (tk.filter((pl.col("pack_vc") == "2W") & POST_KY & pl.col("category").is_in(cats) & pl.col("station_city").is_not_null())
         .group_by(pl.col("station_city").alias("city"), pl.col("is_kyron").alias("ky")).agg(pl.len().alias("n")))
    w = denom.join(t, on=["city", "ky"], how="left").with_columns(pl.col("n").fill_null(0)).pivot(on="ky", index="city", values=["n", "den"])
    w = w.rename({c: c.replace("_true", "_ky").replace("_false", "_ca") for c in w.columns})
    return (w.with_columns(((pl.col("n_ky") / pl.col("den_ky")) / (pl.col("n_ca") / pl.col("den_ca"))).alias("ratio"),
                           (pl.col("n_ky") - pl.col("n_ca") / pl.col("den_ca") * pl.col("den_ky")).alias("excess"))
            .sort(pl.col("city").replace_strict({c: i for i, c in enumerate(CITIES)})))


wc_in = {k: within_city(c, att_city) for k, c in [("battery group", BATT), ("stock-out+queue", SVC), ("billing", ["billing_dispute"])]}
wc_iss = {k: within_city(c, sup_city_sw) for k, c in [("battery group", BATT), ("stock-out+queue", SVC)]}
n_bat_linked_post = int(wc_in["battery group"]["n_ky"].sum() + wc_in["battery group"]["n_ca"].sum())
for basis, dd in [("SUPERSEDED, per 1K swaps ISSUING the supplier's pack", wc_iss), ("per 1K attempts INSERTING the supplier's pack", wc_in)]:
    for k, w in dd.items():
        print(f"{basis} | {k}: within-city Kyron / Cellora+Amptek ticket-rate ratio {w['ratio'].min():.2f}-{w['ratio'].max():.2f} "
              f"(median {w['ratio'].median():.2f}); excess {w['excess'].sum():,.0f} of {int(w['n_ky'].sum() + w['n_ca'].sum()):,} linked tickets "
              f"({pct(w['excess'].sum() / (w['n_ky'].sum() + w['n_ca'].sum()))})")
wcb, wcs = wc_in["battery group"], wc_in["stock-out+queue"]
wc_tab = (wcb.select("city", pl.col("ratio").alias("battery_group_ratio"), pl.col("excess").alias("battery_group_excess"),
                     pl.col("n_ky").alias("battery_tickets_kyron"), pl.col("den_ky").alias("kyron_inserted_attempts"))
          .join(wcs.select("city", pl.col("ratio").alias("stockout_queue_ratio")), on="city")
          .join(wc_in["billing"].select("city", pl.col("ratio").alias("billing_ratio")), on="city")
          .join(wc_iss["battery group"].select("city", pl.col("ratio").alias("battery_group_ratio_issued_superseded")), on="city"))
wc_tab.write_csv(TAB / "q8_kyron_within_city_inserted.csv")
show("within-city Kyron / Cellora+Amptek ticket-rate ratios per inserted-pack attempt (Sep-24..Jun-25)", wc_tab)
print(f"Kyron 2W swaps Sep-24..Jun-25 (issued): {int(wc_iss['battery group']['den_ky'].sum()):,}; Kyron 2W inserted-pack attempts: "
      f"{int(wcb['den_ky'].sum()):,}; battery-group tickets with a linked 2W pack and a station city: {n_bat_linked_post:,}")

# station level: do text labels line up with station telemetry?
smo = station_month().group_by("station_id").agg(pl.col("attempts").sum(), pl.col("completed").sum(), pl.col("system_errors").sum(),
                                                 (pl.col("stockouts").sum() + pl.col("abandoned").sum() + pl.col("system_errors").sum()).alias("failures"),
                                                 pl.col("city").first())
stt = tk.filter(pl.col("station_id").is_not_null()).group_by("station_id").agg(
    pl.len().alias("tickets"),
    pl.col("category").is_in(["no_battery_available", "long_queue"]).sum().alias("svc"),
    pl.col("category").is_in(["low_range", "other", "app_issue"]).sum().alias("batt"),
    (pl.col("text_issue") == "station_hw_system").sum().alias("qr"),
    (pl.col("rider_comment") == "app crash ho gaya swap ke time").sum().alias("crash"),
    pl.col("category").is_in(["billing_dispute"]).sum().alias("bill"))
stt = smo.join(stt, on="station_id", how="left").fill_null(0).filter(pl.col("completed") >= 5000).with_columns(
    (pl.col("failures") / pl.col("attempts")).alias("fail_rate"), (pl.col("system_errors") / pl.col("attempts")).alias("syserr_rate"),
    *[(pl.col(c) / pl.col("attempts") * 1000).alias(f"{c}_per_1k") for c in ["tickets", "svc", "batt", "qr", "crash", "bill"]])
sp = lambda a_, b_: stats.spearmanr(stt[a_], stt[b_])
s1, s2, s3, s4, s5 = sp("svc_per_1k", "fail_rate"), sp("batt_per_1k", "fail_rate"), sp("qr_per_1k", "syserr_rate"), sp("crash_per_1k", "syserr_rate"), sp("bill_per_1k", "fail_rate")
print(f"stations with >=5K swaps: {stt.height}. Spearman: stock-out+queue tickets vs failure rate {s1.statistic:.2f} (p={s1.pvalue:.1e}); "
      f"battery-group tickets vs failure rate {s2.statistic:.2f} (p={s2.pvalue:.2f}); billing vs failure rate {s5.statistic:.2f} (p={s5.pvalue:.2f}); "
      f"QR-scan tickets vs system-error rate {s3.statistic:.2f} (p={s3.pvalue:.2f}); app-crash vs system-error rate {s4.statistic:.2f} (p={s4.pvalue:.2f})")
top10 = stt.sort("tickets_per_1k", descending=True).head(10).select("station_id", "city", "attempts", "tickets", "tickets_per_1k", "svc_per_1k", "batt_per_1k", "fail_rate")
show("top-10 stations by tickets per 1K attempts (>=5K swaps)", top10)
stt.sort("tickets_per_1k", descending=True).write_csv(TAB / "q8_station_ticket_rates.csv")
net = tk.group_by("month").agg(pl.len().alias("tickets")).join(msw, on="month").with_columns((pl.col("tickets") / pl.col("swaps_all") * 1000).alias("per_1k")).sort("month")
print(f"network tickets per 1K swaps: Jan-24 {net['per_1k'][0]:.2f} -> Jun-25 {net['per_1k'][-1]:.2f}; H1-24 "
      f"{net.filter(H1_24)['tickets'].sum() / net.filter(H1_24)['swaps_all'].sum() * 1000:.2f} vs H1-25 {net.filter(H1_25)['tickets'].sum() / net.filter(H1_25)['swaps_all'].sum() * 1000:.2f}")

fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.0), sharex=False)
for c in CITIES:
    d = svc.filter(pl.col("station_city") == c).sort("month")
    axs[0].plot(pd.to_datetime(d["month"].to_pandas()), d["rate"], color=CITY_COLORS[c], lw=1.8, label=c)
    d = btc.filter(pl.col("station_city") == c).sort("month")
    axs[1].plot(pd.to_datetime(d["month"].to_pandas()), d["rate"], color=CITY_COLORS[c], lw=1.8, label=c)
axs[0].set_title(f"1. Stock-out + queue tickets per 1K swaps\n(summer spikes; H1-25 city spread {spread_svc:.1f}x)", fontsize=11.5)
axs[1].set_title(f"2. Battery-group tickets per 1K 2W swaps\n(highest where Kyron is {ky_hi * 100:.0f}% of packs)", fontsize=11.5)
axs[1].legend(fontsize=9, ncol=2, loc="upper left")
xc = np.arange(len(CITIES))
for k_, (wdf, col, lab) in enumerate([(wcb, PALETTE["accent"], "Battery group"), (wcs, PALETTE["primary"], "Stock-out + queue (placebo)")]):
    vals = wdf["ratio"].to_numpy()
    axs[2].bar(xc + (k_ - 0.5) * 0.38, vals, width=0.36, color=col, label=f"{lab}: median {np.median(vals):.2f}x")
    for xi, v_ in zip(xc + (k_ - 0.5) * 0.38, vals):
        axs[2].text(xi, v_ + 0.02, f"{v_:.2f}", ha="center", fontsize=8.5, color=INK)
axs[2].axhline(1, color=PALETTE["muted"], ls="--", lw=1.2)
axs[2].set_xticks(xc, [c.replace("Delhi NCR", "Delhi\nNCR") for c in wcb["city"].to_list()], fontsize=9.5)
axs[2].set_ylabel("Kyron / Cellora+Amptek ticket rate")
axs[2].set_ylim(0, 2.05)
axs[2].legend(loc="upper left", fontsize=9)
axs[2].set_title("3. Ticket pack's supplier, within city: Kyron draws\nonly slightly more battery tickets than the placebo", fontsize=11.5)
axs[2].grid(axis="x", visible=False)
for a_ in axs[:2]:
    a_.set_ylim(0, None)
    a_.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b-%y"))
    a_.tick_params(axis="x", labelrotation=30)
axs[0].set_ylabel("Tickets per 1K swaps")
top = headline(fig, "Stock-out complaints spike each summer; battery complaints climb in every city, most where Kyron packs went",
               f"Panels 1-2 by station city. Battery group = low_range + other + app_issue from 2W riders. {', '.join(KY_CITIES)} got Kyron as "
               f"{ky_hi * 100:.0f}% of issued 2W packs (others {ky_lo * 100:.1f}%):\noutside summer their battery-ticket rate rose "
               f"{dz_b['kyron_change']:.1f}x (Jul-Aug -> Oct-Dec 24) vs {dz_b['other_change']:.1f}x elsewhere. Panel 3: tickets whose battery_id is a "
               f"2W pack, per 1K attempts inserting each supplier's pack, Sep-24 to Jun-25.")
fig.tight_layout(rect=[0, 0, 1, top])
save_fig(fig, "q8_07_ticket_rates_city_supplier")

# %%
print(f"\nfigures: {sorted(p.name for p in FIG.glob('q8_*.png'))}")
print(f"tables: {sorted(p.name for p in TAB.glob('q8_*.csv'))}")
print(f"runtime {time.time() - T0:.1f}s")
