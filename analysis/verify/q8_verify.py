# %% [markdown]
# # Q8 verification - independent re-derivation of the support-ticket text findings
# Fresh code (does not import or copy analysis/q8_ticket_text.py). Re-derives every headline number from data/clean via
# src/common.py loaders, uses its own keyword taxonomy, and adds placebo tests the original analysis did not run.

# %%
import sys
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import *  # noqa: E402,F401,F403

T0 = time.time()
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)


def hdr(s):
    print(f"\n{'=' * 8} {s} {'=' * 8}")


# %% [markdown]
# ## A. Scope, categories and an independent comment taxonomy

# %%
tk_all = tickets()
tk = tk_all.filter(pl.col("in_window") & ~pl.col("is_test"))
N = tk.height
hdr("A1 scope")
print(f"raw {tk_all.height:,}; out-of-window {int((~tk_all['in_window']).sum())}; test {int(tk_all['is_test'].sum())}; "
      f"both {int((tk_all['is_test'] & ~tk_all['in_window']).sum())}; scoped {N:,}; distinct comments (all) {tk_all['rider_comment'].n_unique()}; "
      f"null comments (scoped) {int(tk['rider_comment'].is_null().sum())}")
cats_per_tpl = tk_all.group_by("rider_comment").agg(pl.col("category").n_unique().alias("k"))
print(f"templates with >1 agent category (all 44,000 tickets): {int((cats_per_tpl['k'] > 1).sum())} of {cats_per_tpl.height}")
cat_n = tk.group_by("category").agg(pl.len().alias("n")).with_columns((pl.col("n") / N).alias("share")).sort("n", descending=True)
print(cat_n)


def norm(s):
    s = s.lower()
    return s.replace("staion", "station").replace("battry", "battery").replace("rnage", "range")


def my_label(c):
    s = norm(c)
    if re.search(r"competitor|switch|another (app|company|service)|other (app|company|service)|\bleav|chhod|uninstall|cheaper|"
                 r"yulu|bounce|sun mobility|battery smart|esmito|voltup|\bola\b|rapido|dusr|doosr", s):
        return "competitor"
    if re.search(r"thank|great|\bgood\b|accha|achha|badhiya|shukriya|\bnice\b|happy|helpful", s):
        return "praise"
    if re.search(r"no battery|charged battery|empty station|batteries discharge", s):
        return "stockout"
    if re.search(r"queue|\bline\b", s):
        return "queue"
    if re.search(r"amount|surcharge|charged extra|refund|price", s):
        return "billing"
    if re.search(r"\bqr\b|scan", s):
        return "qr"
    if re.search(r"\bapp\b", s):
        return "app"
    if re.search(r"battery|range|discharge|dying at", s):
        return "battery_word"
    if re.search(r"vehicle|scooter|bike|gaadi", s):
        return "vehicle_only"
    return "unmatched"


HINDI = {"hai", "nahi", "ho", "gaya", "raha", "lag", "lagti", "mein", "ki", "bahut", "bohot", "kal", "se", "sab", "laga", "diya", "bina",
         "bataye", "kata", "galat", "koi", "mila", "thi", "tha", "ke", "liye", "aaj", "aajkal", "pe", "itna", "lamba", "lambi", "dikkat",
         "ka", "kam", "baar"}  # note: English "is" deliberately excluded ("this is the third time" is English)
tpl = tk.group_by("rider_comment").agg(pl.len().alias("n"), pl.col("category").first().alias("cat"))
tpl = tpl.with_columns(
    pl.col("rider_comment").map_elements(my_label, return_dtype=pl.Utf8).alias("lab"),
    pl.col("rider_comment").map_elements(lambda c: bool(HINDI & set(re.findall(r"[a-z]+", c.lower()))), return_dtype=pl.Boolean).alias("hinglish"),
    pl.col("rider_comment").str.contains(r"(?i)staion|battry|rnage").alias("typo"),
    pl.col("rider_comment").map_elements(lambda c: bool(re.search(r"vehicle|scooter|bike|gaadi", c.lower())), return_dtype=pl.Boolean).alias("veh"),
    pl.col("rider_comment").str.extract(r"dying at (-?\d+)", 1).cast(pl.Int64).alias("km"))
tk = tk.join(tpl.select("rider_comment", "lab", "hinglish", "typo", "veh", "km"), on="rider_comment", how="left")
hdr("A2 taxonomy (my rules)")
print(tpl.group_by("cat", "lab").agg(pl.len().alias("templates"), pl.col("n").sum().alias("tickets")).sort("cat", "lab"))
print(f"unmatched templates: {int((tpl['lab'] == 'unmatched').sum())}; competitor tickets {int((tk['lab'] == 'competitor').sum())}; "
      f"praise tickets {int((tk['lab'] == 'praise').sum())}")
print(f"Hinglish share {tk['hinglish'].mean():.4f}; typo share {tk['typo'].mean():.4f}")
NAT = {"low_range": {"battery_word"}, "other": {"praise", "unmatched"}, "app_issue": {"app"}, "no_battery_available": {"stockout"},
       "long_queue": {"queue"}, "billing_dispute": {"billing"}}
tk = tk.with_columns(pl.struct("category", "lab").map_elements(lambda s: s["lab"] in NAT[s["category"]], return_dtype=pl.Boolean).alias("agree"))
n_dis = int((~tk["agree"]).sum())
n_qr = int((tk["lab"] == "qr").sum())
n_other = int((tk["category"] == "other").sum())
print(f"disagree {n_dis:,} = {n_dis / N:.4f}; other {n_other:,} = {n_other / N:.4f} (vehicle_only share of 'other' "
      f"{tk.filter(pl.col('category') == 'other')['lab'].eq('vehicle_only').mean():.3f}); QR {n_qr:,} = {n_qr / N:.4f}")
bt = tk.filter(pl.col("lab").is_in(["battery_word", "vehicle_only"]))
print(f"battery-word + vehicle-only tickets {bt.height:,}; mention vehicle {int(bt['veh'].sum()):,} = {bt['veh'].mean():.4f}")
n_lr = int((tk["category"] == "low_range").sum())
n_app = int((tk["category"] == "app_issue").sum())
n_so = int((tk["category"] == "no_battery_available").sum())
n_q = int((tk["category"] == "long_queue").sum())
lb, ub = n_lr + n_other, n_lr + n_other + n_app
print(f"HEADLINE: low_range {n_lr:,}; +other {lb:,} ({lb / n_lr:.3f}x, {lb / N:.4f}); +app {ub:,} ({ub / n_lr:.3f}x, {ub / N:.4f}); "
      f"stock-out {n_so / N:.4f}; queue {n_q / N:.4f}; stock-out+queue {(n_so + n_q) / N:.4f}; official low_range share {n_lr / N:.4f}")
q2 = tk.filter(pl.col("month") >= pl.date(2025, 4, 1))
print(f"Apr-Jun-25 n={q2.height:,}: low_range+other {(q2['category'].is_in(['low_range', 'other'])).mean():.4f}; +app "
      f"{(q2['category'].is_in(['low_range', 'other', 'app_issue'])).mean():.4f}; stock-out {(q2['category'] == 'no_battery_available').mean():.4f}; "
      f"stock-out+queue {(q2['category'].is_in(['no_battery_available', 'long_queue'])).mean():.4f}")

# %% [markdown]
# ## B. Battery fingerprint: rider vehicle class, same-season step-up, supplier of the ticket's battery_id

# %%
rd = riders().select("rider_id", pl.col("vehicle_class").alias("rvc"), "home_city_clean", "partner_id", "first_swap_ts", "last_swap_ts")
bat = batteries().select("battery_id", pl.col("supplier").alias("sup"), pl.col("vehicle_class").alias("pvc"))
tk = tk.join(rd, on="rider_id", how="left").join(bat, on="battery_id", how="left")
print(f"tickets with unknown rider class {int(tk['rvc'].is_null().sum())}; with battery_id {tk['battery_id'].is_not_null().mean():.4f}; "
      f"battery_id not in batteries table {int((tk['battery_id'].is_not_null() & tk['sup'].is_null()).sum())}")

swaps_cm = (events(["is_completed", "month", "city", "vehicle_class", "supplier_out", "soh_out_pct"]).filter(pl.col("is_completed"))
            .group_by("month", "city", "vehicle_class", "supplier_out")
            .agg(pl.len().alias("sw"), pl.col("soh_out_pct").mean().alias("soh"), (pl.col("soh_out_pct") < 80).sum().alias("lt80"),
                 pl.col("soh_out_pct").count().alias("soh_n"))
            .collect(engine="streaming"))
sw3 = float(swaps_cm.filter(pl.col("vehicle_class") == "3W")["sw"].sum() / swaps_cm["sw"].sum())
hdr("B1 3W share by category; app_chat share by rider class")
print(f"3W share of completed swaps {sw3:.4f}")
print(tk.group_by("category").agg((pl.col("rvc") == "3W").mean().alias("share_3w"), pl.len().alias("n")).sort("share_3w"))
print(tk.group_by("rvc").agg((pl.col("channel") == "app_chat").mean().alias("app_chat")).sort("rvc"))
s3 = dict(tk.group_by("category").agg((pl.col("rvc") == "3W").mean()).iter_rows())
s_bat = (s3["low_range"] * n_lr + s3["other"] * n_other) / (n_lr + n_other)
s_gen_w = (s3["no_battery_available"] * n_so + s3["long_queue"] * n_q + s3["billing_dispute"] * int((tk["category"] == "billing_dispute").sum())) / (
    n_so + n_q + int((tk["category"] == "billing_dispute").sum()))
f_app = (s_gen_w - s3["app_issue"]) / (s_gen_w - s_bat)
print(f"mixing model: 3W share battery-like {s_bat:.4f}, service categories {s_gen_w:.4f}, app_issue {s3['app_issue']:.4f} -> "
      f"implied battery-like fraction of app_issue {f_app:.3f}")

m2w = (swaps_cm.filter(pl.col("vehicle_class") == "2W").group_by("month")
       .agg(pl.col("sw").sum().alias("sw2w"), (pl.col("sw").filter(pl.col("supplier_out") == "Kyron").sum() / pl.col("sw").sum()).alias("p_ky"),
            (pl.col("lt80").sum() / pl.col("soh_n").sum()).alias("lt80")).sort("month"))
H1_24 = pl.col("month").is_between(pl.date(2024, 1, 1), pl.date(2024, 6, 1))
H1_25 = pl.col("month").is_between(pl.date(2025, 1, 1), pl.date(2025, 6, 1))
s24, s25 = float(m2w.filter(H1_24)["sw2w"].sum()), float(m2w.filter(H1_25)["sw2w"].sum())
hdr("B2 same-season step-up, 2W-rider tickets per 1K 2W swaps")
step = (tk.filter(pl.col("rvc") == "2W").group_by("category")
        .agg(H1_24.sum().alias("n24"), H1_25.sum().alias("n25"))
        .with_columns((pl.col("n24") / s24 * 1000).alias("r24"), (pl.col("n25") / s25 * 1000).alias("r25"))
        .with_columns((pl.col("r25") / pl.col("r24")).alias("step")).sort("step", descending=True))
print(step)

# Kyron O/E vs city-month expectation (station city; fallback rider home city)
pky_cm = (swaps_cm.filter(pl.col("vehicle_class") == "2W").group_by("city", "month")
          .agg((pl.col("sw").filter(pl.col("supplier_out") == "Kyron").sum() / pl.col("sw").sum()).alias("pky_c")))
tk = (tk.with_columns(pl.coalesce("station_city", "home_city_clean").alias("city_any"))
      .join(pky_cm, left_on=["city_any", "month"], right_on=["city", "month"], how="left")
      .join(m2w.select("month", "p_ky"), on="month", how="left"))
POST = pl.col("month") >= pl.date(2024, 9, 1)
lk = tk.filter(POST & (pl.col("pvc") == "2W"))
hdr("B3 Kyron O/E on 2W-pack-linked tickets, Sep-24..Jun-25 (city-month expectation)")
rows = []
for c, d in lk.group_by("category"):
    O = int((d["sup"] == "Kyron").sum())
    E = float(d["pky_c"].sum())
    En = float(d["p_ky"].sum())
    rows.append(dict(category=c[0], linked=d.height, kyron_share=O / d.height, OE=O / E, OE_lo=stats.chi2.ppf(0.025, 2 * O) / 2 / E,
                     OE_hi=stats.chi2.ppf(0.975, 2 * O + 2) / 2 / E, OE_network=O / En))
oe = pd.DataFrame(rows).sort_values("OE", ascending=False)
print(oe.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# %% [markdown]
# ## B4. What does a ticket's battery_id actually point to?

# %%
pairs = tk.filter(pl.col("battery_id").is_not_null()).select("ticket_id", "rider_id", "battery_id", "created_ts", "category")
ev_link = events(["is_completed", "rider_id", "event_ts", "battery_out_id", "battery_in_id"]).filter(pl.col("is_completed"))
m_out = (ev_link.join(pairs.lazy().select("rider_id", "battery_id").unique(), left_on=["rider_id", "battery_out_id"],
                      right_on=["rider_id", "battery_id"], how="semi")
         .select("rider_id", pl.col("battery_out_id").alias("battery_id"), "event_ts").collect(engine="streaming"))
m_in = (ev_link.join(pairs.lazy().select("rider_id", "battery_id").unique(), left_on=["rider_id", "battery_in_id"],
                     right_on=["rider_id", "battery_id"], how="semi")
        .select("rider_id", pl.col("battery_in_id").alias("battery_id"), "event_ts").collect(engine="streaming"))
any_out = pairs.join(m_out, on=["rider_id", "battery_id"], how="semi").height
any_in = pairs.join(m_in, on=["rider_id", "battery_id"], how="semi").height
near = (pairs.join(m_out, on=["rider_id", "battery_id"], how="inner")
        .filter((pl.col("event_ts") - pl.col("created_ts")).abs() <= pl.duration(days=7)).select("ticket_id").unique().height)
hdr("B4 battery_id linkage")
print(f"linked tickets {pairs.height:,}; battery_id ever issued TO this rider {any_out:,} ({any_out / pairs.height:.4f}); ever returned BY this rider "
      f"{any_in:,} ({any_in / pairs.height:.4f}); issued to rider within +-7 days {near:,} ({near / pairs.height:.4f})")
pvr = tk.filter(pl.col("pvc").is_not_null() & pl.col("rvc").is_not_null())
print(f"pack class = rider class on {(pvr['pvc'] == pvr['rvc']).mean():.4f} of linked tickets; 3W riders' linked packs that are 3W: "
      f"{pvr.filter(pl.col('rvc') == '3W')['pvc'].eq('3W').mean():.4f}")
# is the ticket pack at least issued in the ticket's station city around then?
pk_city = (events(["is_completed", "battery_out_id", "city", "month"]).filter(pl.col("is_completed"))
           .group_by("battery_out_id", "city", "month").agg(pl.len().alias("k")).collect(engine="streaming"))
pc = tk.filter(pl.col("battery_id").is_not_null() & pl.col("station_city").is_not_null()).join(
    pk_city, left_on=["battery_id", "station_city", "month"], right_on=["battery_out_id", "city", "month"], how="left")
print(f"ticket pack issued in the ticket's station city in the ticket month: {pc['k'].is_not_null().mean():.4f}")
pk_tot = pk_city.group_by("battery_out_id", "month").agg(pl.col("k").sum().alias("ktot"))
pc2 = tk.filter(pl.col("battery_id").is_not_null()).join(pk_tot, left_on=["battery_id", "month"], right_on=["battery_out_id", "month"], how="left")
print(f"ticket pack issued anywhere in the ticket month: {pc2['ktot'].is_not_null().mean():.4f}")
del m_out, m_in, pk_city

# %% [markdown]
# ## C. True battery-complaint volume over time, dose test and supplier attribution (with placebos)

# %%
hdr("C1 monthly 2W-rider tickets per 1K 2W swaps")
mon = (tk.filter(pl.col("rvc") == "2W").group_by("month").agg(
    *[(pl.col("category") == c).sum().alias(c) for c in ["low_range", "other", "app_issue", "no_battery_available", "long_queue", "billing_dispute"]])
       .join(m2w, on="month").sort("month"))
mon = mon.with_columns(*[(pl.col(c) / pl.col("sw2w") * 1000).alias(f"r_{c}") for c in ["low_range", "other", "app_issue"]]).with_columns(
    (pl.col("r_low_range") + pl.col("r_other")).alias("r_lb"), (pl.col("r_low_range") + pl.col("r_other") + pl.col("r_app_issue")).alias("r_ub"),
    (pl.col("other") / pl.col("low_range")).alias("oth_ratio"))
print(mon.select("month", "r_low_range", "r_lb", "r_ub", "oth_ratio", "p_ky", "lt80"))
print(f"other/low_range monthly ratio min {mon['oth_ratio'].min():.2f} median {mon['oth_ratio'].median():.2f} max {mon['oth_ratio'].max():.2f}")
r_k = np.corrcoef(mon["r_ub"], mon["p_ky"])[0, 1]
post = mon.filter(pl.col("month") >= pl.date(2024, 10, 1))
print(f"r(battery rate, Kyron share) all months {r_k:.3f}; Oct-24..Jun-25: r with <80% SoH share {np.corrcoef(post['r_ub'], post['lt80'])[0, 1]:.3f}, "
      f"with Kyron share {np.corrcoef(post['r_ub'], post['p_ky'])[0, 1]:.3f} (n={post.height})")
# placebo: do the service categories also 'track' the <80% SoH share over Oct-24..Jun-25?
post = post.with_columns(((pl.col("no_battery_available") + pl.col("long_queue")) / pl.col("sw2w") * 1000).alias("r_svc"),
                         (pl.col("billing_dispute") / pl.col("sw2w") * 1000).alias("r_bill"))
print(f"placebo Oct-24..Jun-25: stock-out+queue rate vs <80% SoH share r={np.corrcoef(post['r_svc'], post['lt80'])[0, 1]:.3f}; "
      f"billing r={np.corrcoef(post['r_bill'], post['lt80'])[0, 1]:.3f}")
# detrended (first differences) version of the pack-ageing link
d_ub, d_lt = np.diff(post["r_ub"].to_numpy()), np.diff(post["lt80"].to_numpy())
print(f"first-difference r(battery rate, <80% SoH share) Oct-24..Jun-25 = {np.corrcoef(d_ub, d_lt)[0, 1]:.3f} (n={len(d_ub)})")

hdr("C2 non-summer dose test (Jul-Aug-24 -> Oct-Dec-24) + placebo on service tickets")
sw_city = (swaps_cm.group_by("city", "month", "vehicle_class").agg(pl.col("sw").sum()))
ky_city_h1 = (swaps_cm.filter((pl.col("vehicle_class") == "2W") & H1_25).group_by("city")
              .agg((pl.col("sw").filter(pl.col("supplier_out") == "Kyron").sum() / pl.col("sw").sum()).alias("pky")).sort("pky", descending=True))
print(ky_city_h1)
KYC = ky_city_h1["city"].head(3).to_list()
JA = pl.col("month").is_between(pl.date(2024, 7, 1), pl.date(2024, 8, 1))
OD = pl.col("month").is_between(pl.date(2024, 10, 1), pl.date(2024, 12, 1))


def city_grp_rate(cats, period, rider_cls, sw_cls):
    t = tk.filter(pl.col("category").is_in(cats) & pl.col("station_city").is_not_null() & period)
    if rider_cls:
        t = t.filter(pl.col("rvc") == rider_cls)
    t = t.with_columns(pl.col("station_city").is_in(KYC).alias("ky")).group_by("ky").agg(pl.len().alias("n"))
    s = sw_city.filter(period)
    if sw_cls:
        s = s.filter(pl.col("vehicle_class") == sw_cls)
    s = s.with_columns(pl.col("city").is_in(KYC).alias("ky")).group_by("ky").agg(pl.col("sw").sum())
    j = t.join(s, on="ky").with_columns((pl.col("n") / pl.col("sw") * 1000).alias("r"))
    return float(j.filter(pl.col("ky"))["r"].item()), float(j.filter(~pl.col("ky"))["r"].item())


for lab_, cats, rc, sc in [("battery group (2W riders / 2W swaps)", ["low_range", "other", "app_issue"], "2W", "2W"),
                           ("PLACEBO stock-out+queue (2W riders / 2W swaps)", ["no_battery_available", "long_queue"], "2W", "2W"),
                           ("PLACEBO stock-out+queue (all / all swaps)", ["no_battery_available", "long_queue"], None, None),
                           ("PLACEBO billing (all / all swaps)", ["billing_dispute"], None, None)]:
    (hja, mja), (hod, mod) = city_grp_rate(cats, JA, rc, sc), city_grp_rate(cats, OD, rc, sc)
    print(f"{lab_}: Kyron cities {hja:.2f}->{hod:.2f} ({hod / hja:.2f}x); others {mja:.2f}->{mod:.2f} ({mod / mja:.2f}x); "
          f"ratio of ratios {(hod / hja) / (mod / mja):.2f}")

hdr("C3 within-city supplier rate ratio (tickets linked to a 2W pack / swaps issuing that supplier), Sep-24..Jun-25 + placebos")
sw_city_ky = (events(["is_completed", "city", "month", "vehicle_class", "supplier_out"])
              .filter(pl.col("is_completed") & (pl.col("vehicle_class") == "2W") & (pl.col("month") >= pl.date(2024, 9, 1)))
              .group_by("city", (pl.col("supplier_out") == "Kyron").alias("ky")).agg(pl.len().alias("sw")).collect(engine="streaming"))


def within_city(cats):
    t = (tk.filter(POST & (pl.col("pvc") == "2W") & pl.col("category").is_in(cats) & pl.col("station_city").is_not_null())
         .group_by(pl.col("station_city").alias("city"), (pl.col("sup") == "Kyron").alias("ky")).agg(pl.len().alias("n")))
    j = sw_city_ky.join(t, on=["city", "ky"], how="left").with_columns(pl.col("n").fill_null(0), (pl.col("n").fill_null(0) / pl.col("sw") * 1000).alias("r"))
    w = j.pivot(on="ky", index="city", values=["r", "sw", "n"])
    w = w.rename({c: c.replace("_true", "_ky").replace("_false", "_ca") for c in w.columns})
    w = w.with_columns((pl.col("r_ky") / pl.col("r_ca")).alias("ratio"), ((pl.col("r_ky") - pl.col("r_ca")) / 1000 * pl.col("sw_ky")).alias("excess")).sort("city")
    tot_n = int(j["n"].sum())
    return w, tot_n


res_wc = {}
for lab_, cats in [("battery group", ["low_range", "other", "app_issue"]), ("stock-out+queue", ["no_battery_available", "long_queue"]),
                   ("billing", ["billing_dispute"]), ("low_range only", ["low_range"])]:
    w, tot_n = within_city(cats)
    res_wc[lab_] = w
    print(f"{lab_}: ratio range {w['ratio'].min():.2f}-{w['ratio'].max():.2f}; excess {w['excess'].sum():,.0f} of {tot_n:,} linked tickets "
          f"({w['excess'].sum() / tot_n:.3f}); Kyron swaps {int(w['sw_ky'].sum()):,}")
    print(w.select("city", "r_ky", "r_ca", "ratio", "excess"))
# battery-specific excess = battery-group excess over and above what the stock-out+queue placebo ratio predicts
wb, wp = res_wc["battery group"], res_wc["stock-out+queue"]
jj = wb.join(wp.select("city", pl.col("ratio").alias("ratio_placebo")), on="city").with_columns(
    ((pl.col("r_ky") - pl.col("r_ca") * pl.col("ratio_placebo")) / 1000 * pl.col("sw_ky")).alias("excess_net"),
    (pl.col("ratio") / pl.col("ratio_placebo")).alias("ratio_of_ratios"))
print(jj.select("city", "ratio", "ratio_placebo", "ratio_of_ratios", "excess", "excess_net"))
print(f"battery-specific excess (net of the stock-out+queue placebo ratio): {jj['excess_net'].sum():,.0f}")

# %% [markdown]
# ## D. Rider-reported km ("scooter dying at X km")

# %%
hdr("D reported km")
pack_month = (events(["is_completed", "battery_out_id", "month", "soh_out_pct"]).filter(pl.col("is_completed"))
              .group_by("battery_out_id", "month").agg(pl.col("soh_out_pct").mean().alias("pack_soh")).collect(engine="streaming"))
km = tk.filter(pl.col("km").is_not_null()).join(pack_month, left_on=["battery_id", "month"], right_on=["battery_out_id", "month"], how="left")
print(f"km tickets {km.height:,}; outside 0-150: {km.filter(~pl.col('km').is_between(0, 150)).height}; values outside: "
      f"{km.filter(~pl.col('km').is_between(0, 150))['km'].to_list()}; >100: {km.filter(pl.col('km') > 100)['km'].to_list()}")
k2 = km.filter(pl.col("km").is_between(0, 150) & (pl.col("rvc") == "2W"))
print(f"2W valid {k2.height:,}; by supplier:")
print(k2.group_by("sup").agg(pl.len(), pl.col("km").median().alias("med"), pl.col("km").mean().alias("mean")).sort("sup"))
ky = k2.filter(pl.col("sup") == "Kyron")["km"].to_numpy()
ca = k2.filter(pl.col("sup").is_in(["Cellora", "Amptek"]))["km"].to_numpy()
print(f"Kyron median {np.median(ky):.1f} vs Cellora+Amptek {np.median(ca):.1f}; MW p={stats.mannwhitneyu(ky, ca).pvalue:.2e}")
kk = k2.filter(pl.col("pack_soh").is_not_null())
pr_ = stats.pearsonr(kk["km"], kk["pack_soh"])
print(f"km vs linked pack SoH (ticket month) r={pr_.statistic:.3f} n={kk.height}; slope {np.polyfit(kk['pack_soh'], kk['km'], 1)[0]:.3f}")
# does the km figure carry info beyond month (time trend)? partial correlation via month FE
kkp = kk.select("km", "pack_soh", pl.col("month").cast(pl.Utf8).alias("m"), "sup").to_pandas()
ols_fe = smf.ols("km ~ pack_soh + C(m)", data=kkp).fit()
print(f"km ~ pack_soh + month FE: slope {ols_fe.params['pack_soh']:.3f} (p={ols_fe.pvalues['pack_soh']:.1e})")
km_ev = (events(["is_completed", "month", "vehicle_class", "km_since_last_swap"]).filter(pl.col("is_completed") & (pl.col("vehicle_class") == "2W"))
         .group_by("month").agg(pl.col("km_since_last_swap").median().alias("km_log")).collect(engine="streaming"))
kmm = k2.group_by("month").agg(pl.len().alias("n"), pl.col("km").median().alias("km_txt")).join(km_ev, on="month").sort("month")
print(kmm)
print(f"monthly r(text median, log median) = {np.corrcoef(kmm['km_txt'], kmm['km_log'])[0, 1]:.3f}")

# %% [markdown]
# ## E. Pricing pilot DiD on billing tickets (+ vehicle-class split and composition checks)

# %%
hdr("E pricing DiD")
swc = sw_city.group_by("city", "month").agg(pl.col("sw").sum())
act = (events(["is_completed", "city", "month", "rider_id"]).filter(pl.col("is_completed")).group_by("city", "month")
       .agg(pl.col("rider_id").n_unique().alias("riders")).collect(engine="streaming"))


def did_panel(filter_expr=None, sw_cls=None):
    t = tk.filter((pl.col("category") == "billing_dispute") & pl.col("station_city").is_not_null())
    if filter_expr is not None:
        t = t.filter(filter_expr)
    t = t.group_by(pl.col("station_city").alias("city"), "month").agg(pl.len().alias("n"))
    s = sw_city.filter(pl.col("vehicle_class") == sw_cls) if sw_cls else sw_city
    s = s.group_by("city", "month").agg(pl.col("sw").sum())
    p = s.join(t, on=["city", "month"], how="left").join(act, on=["city", "month"], how="left").with_columns(pl.col("n").fill_null(0)).to_pandas()
    p["pilot"] = p["city"].isin(PILOT_CITIES)
    p["post"] = p["month"] >= pd.Timestamp(PILOT_START)
    p["treat"] = (p["pilot"] & p["post"]).astype(int)
    p["ms"] = p["month"].astype(str)
    p["t"] = (p["month"].dt.year - 2024) * 12 + p["month"].dt.month
    return p


def fit(p, f="n ~ treat + C(city) + C(ms)", off="sw", var="treat"):
    m = smf.glm(f, data=p, family=sm.families.Poisson(), offset=np.log(p[off])).fit(scale="X2")
    lo, hi = np.exp(m.conf_int().loc[var])
    return np.exp(m.params[var]), lo, hi, m.pvalues[var]


p_all = did_panel()
ph = p_all.assign(g=np.where(p_all["pilot"], "pilot", "control"),
                  ph=np.select([p_all["month"] < pd.Timestamp(BASE_PRICE_CHANGE), p_all["month"] < pd.Timestamp(PILOT_START)], ["1", "2"], "3"))
gr = ph.groupby(["g", "ph"]).agg(n=("n", "sum"), sw=("sw", "sum"))
gr["per1k"] = gr["n"] / gr["sw"] * 1000
print(gr)
print("main IRR %.3f (%.3f-%.3f) p=%.4f" % fit(p_all))
print("city trends IRR %.3f (%.3f-%.3f) p=%.4f" % fit(p_all, "n ~ treat + C(city) + C(ms) + C(city):t"))
print("per active rider IRR %.3f (%.3f-%.3f) p=%.4f" % fit(p_all, off="riders"))
pre = p_all[p_all["month"] < pd.Timestamp(PILOT_START)].copy()
pre["tp"] = (pre["pilot"] & (pre["month"] >= pd.Timestamp(BASE_PRICE_CHANGE))).astype(int)
print("placebo Jul-24 IRR %.3f (%.3f-%.3f) p=%.4f" % fit(pre, "n ~ tp + C(city) + C(ms)", var="tp"))
p_ex = p_all[(p_all["month"] < pd.Timestamp(BASE_PRICE_CHANGE)) | (p_all["month"] >= pd.Timestamp(PILOT_START))]
print("drop Jul-Sep-24 IRR %.3f (%.3f-%.3f) p=%.4f" % fit(p_ex))
p_short = p_all[(p_all["month"] >= pd.Timestamp(BASE_PRICE_CHANGE))]
print("pre = Jul-Sep-24 only (post base-price) IRR %.3f (%.3f-%.3f) p=%.4f" % fit(p_short))
p2 = did_panel(pl.col("rvc") == "2W", "2W")
p3 = did_panel(pl.col("rvc") == "3W", "3W")
print("2W riders / 2W swaps IRR %.3f (%.3f-%.3f) p=%.4f" % fit(p2))
print("3W riders / 3W swaps IRR %.3f (%.3f-%.3f) p=%.4f" % fit(p3))
pind = did_panel(pl.col("partner_id").is_null())
pflt = did_panel(pl.col("partner_id").is_not_null())
print("independent riders' billing tickets / all swaps IRR %.3f (%.3f-%.3f) p=%.4f" % fit(pind))
print("fleet riders' billing tickets / all swaps IRR %.3f (%.3f-%.3f) p=%.4f" % fit(pflt))
# placebo outcome: the same DiD on non-billing tickets
for c in ["no_battery_available", "long_queue", "low_range"]:
    t = tk.filter((pl.col("category") == c) & pl.col("station_city").is_not_null()).group_by(pl.col("station_city").alias("city"), "month").agg(pl.len().alias("n"))
    pp = swc.join(t, on=["city", "month"], how="left").with_columns(pl.col("n").fill_null(0)).to_pandas()
    pp["treat"] = (pp["city"].isin(PILOT_CITIES) & (pp["month"] >= pd.Timestamp(PILOT_START))).astype(int)
    pp["ms"] = pp["month"].astype(str)
    print(f"PLACEBO outcome {c}: IRR %.3f (%.3f-%.3f) p=%.4f" % fit(pp))
gw = p_all.groupby("pilot").apply(lambda d: d.loc[d["month"] == d["month"].max(), "sw"].sum() / d.loc[d["month"] == d["month"].min(), "sw"].sum())
print(f"swap growth Jan-24->Jun-25 pilot {gw[True]:.2f}x control {gw[False]:.2f}x")
ctl = p_all[~p_all["pilot"] & p_all["post"]]
print(f"control-city post-pilot billing tickets/month {ctl['n'].sum() / ctl['month'].nunique():.1f}")
sur = (tk.filter((pl.col("category") == "billing_dispute") & pl.col("station_city").is_not_null())
       .with_columns(pl.col("station_city").is_in(PILOT_CITIES).alias("pilot"),
                     pl.when(pl.col("month") < BASE_PRICE_CHANGE).then(pl.lit("1")).when(pl.col("month") < PILOT_START).then(pl.lit("2")).otherwise(pl.lit("3")).alias("ph"),
                     (pl.col("rider_comment") == "surcharge laga diya bina bataye").alias("s"))
       .group_by("pilot", "ph").agg(pl.len(), pl.col("s").mean()).sort("pilot", "ph"))
print(sur)
# billing by vehicle class over phases
bvc = (tk.filter(pl.col("category") == "billing_dispute").with_columns(
    pl.when(pl.col("month") < BASE_PRICE_CHANGE).then(pl.lit("1")).when(pl.col("month") < PILOT_START).then(pl.lit("2")).otherwise(pl.lit("3")).alias("ph"))
       .group_by("rvc", "ph").agg(pl.len().alias("n")))
svc_ph = sw_city.with_columns(pl.when(pl.col("month") < BASE_PRICE_CHANGE).then(pl.lit("1")).when(pl.col("month") < PILOT_START).then(pl.lit("2"))
                              .otherwise(pl.lit("3")).alias("ph")).group_by(pl.col("vehicle_class").alias("rvc"), "ph").agg(pl.col("sw").sum())
print(bvc.join(svc_ph, on=["rvc", "ph"]).with_columns((pl.col("n") / pl.col("sw") * 1000).alias("per1k")).sort("rvc", "ph"))
# Do billing-ticket riders in pilot cities post-pilot pay more PEAK swaps than other ticket riders there? (exposure check)
peak_r = (events(["is_completed", "rider_id", "city", "month", "tariff_code"])
          .filter(pl.col("is_completed") & pl.col("city").is_in(PILOT_CITIES) & (pl.col("month") >= PILOT_START))
          .group_by("rider_id").agg((pl.col("tariff_code") == "PEAK").mean().alias("peak_share"), pl.len().alias("sw")).collect(engine="streaming"))
pt = (tk.filter(pl.col("station_city").is_in(PILOT_CITIES) & (pl.col("month") >= PILOT_START)).join(peak_r, on="rider_id", how="inner")
      .group_by(pl.col("category") == "billing_dispute").agg(pl.len(), pl.col("peak_share").mean(), (pl.col("peak_share") > 0).mean().alias("any_peak")))
print("pilot-city post-pilot tickets: PEAK exposure of billing vs other ticket filers\n", pt)

# %% [markdown]
# ## F. Lapse after ticket - right-censoring check of the analyst's definition

# %%
hdr("F lapse censoring")
lp = tk.filter((pl.col("created_ts") < pl.datetime(2025, 6, 1)) & pl.col("last_swap_ts").is_not_null() & pl.col("rvc").is_not_null()).with_columns(
    (pl.col("last_swap_ts") < pl.col("created_ts") + pl.duration(days=30)).alias("lapse30"))
print(f"n={lp.height:,}; lapse30 {lp['lapse30'].mean():.4f}")
print(lp.group_by("month").agg(pl.len(), pl.col("lapse30").mean()).sort("month").tail(6))
lp_ok = lp.filter(pl.col("created_ts") < pl.datetime(2025, 4, 1))
print(f"tickets before Apr-25 (>=60 days of observation after the 30-day mark): n={lp_ok.height:,}, lapse30 {lp_ok['lapse30'].mean():.4f}")
# true 30-day post-ticket inactivity (no completed swap in (ticket, ticket+30d]) for comparison
evr = (events(["is_completed", "rider_id", "event_ts"]).filter(pl.col("is_completed"))
       .join(lp.lazy().select("rider_id").unique(), on="rider_id", how="semi").select("rider_id", "event_ts").collect(engine="streaming").sort("event_ts"))
nx = lp.select("ticket_id", "rider_id", "created_ts", "month").sort("created_ts").join_asof(
    evr.rename({"event_ts": "next_ts"}), left_on="created_ts", right_on="next_ts", by="rider_id", strategy="forward")
nx = nx.with_columns(((pl.col("next_ts").is_null()) | (pl.col("next_ts") > pl.col("created_ts") + pl.duration(days=30))).alias("inactive30"))
print(f"alternative: no completed swap within 30 days after the ticket = {nx['inactive30'].mean():.4f}")
del evr

# %% [markdown]
# ## G. CSAT non-response

# %%
hdr("G CSAT")
res = tk.filter(pl.col("resolution_status") == "resolved").to_pandas()
print(f"resolved {len(res):,}; response {res['has_csat'].mean():.4f}; CSAT on non-resolved: {int(tk.filter(pl.col('resolution_status') != 'resolved')['has_csat'].sum())}")
res["q"] = pd.qcut(res["resolution_hours"], 4, labels=False)
res["dec"] = pd.qcut(res["resolution_hours"], 10, labels=False)
print(res.groupby("q").agg(resp=("has_csat", "mean"), csat=("csat_score", "mean"), n=("has_csat", "size")))
naive = res.loc[res["has_csat"], "csat_score"].mean()
# post-stratification (cell-based IPW) on resolution decile x category x channel
cells = res.groupby(["dec", "category", "channel"]).agg(N=("has_csat", "size"), R=("has_csat", "sum"), m=("csat_score", "mean")).reset_index()
cells = cells[cells["R"] > 0]
ps = np.average(cells["m"], weights=cells["N"])
print(f"naive {naive:.4f}; post-stratified (decile x category x channel) {ps:.4f}; diff {ps - naive:+.4f}; covered resolved share {cells['N'].sum() / len(res):.4f}")
rr = res[res["has_csat"]]
print(f"Spearman CSAT vs hours rho={stats.spearmanr(rr['resolution_hours'], rr['csat_score']).statistic:.4f} p={stats.spearmanr(rr['resolution_hours'], rr['csat_score']).pvalue:.3f}")
lr = smf.ols("csat_score ~ np.log1p(resolution_hours) + C(category) + C(channel) + C(priority)", data=rr).fit()
print(f"CSAT OLS: log-hours coef {lr.params['np.log1p(resolution_hours)']:.4f} (p={lr.pvalues['np.log1p(resolution_hours)']:.3f}); R2 {lr.rsquared:.4f}")
n_unres, n_dup = int((tk["resolution_status"] == "unresolved").sum()), int((tk["resolution_status"] == "duplicate").sum())
print(f"never surveyed {n_unres:,}+{n_dup:,} = {(n_unres + n_dup) / N:.4f}")
r_ = res["has_csat"].mean()
for d in [0.5, 1.0]:
    print(f"non-respondents {d} lower -> {ps - (1 - r_) * d:.3f}")
for s_ in [2, 1]:
    print(f"unresolved scored {s_} -> {(len(res) * ps + n_unres * s_) / (len(res) + n_unres):.3f}")
# what does drive CSAT? (rider repeat failures, issue, city)
print(res.groupby("category")["csat_score"].mean().round(3).to_dict())

# %% [markdown]
# ## H. Handling: priority, resolution by issue, repeat-claim, station validation, network rates

# %%
hdr("H handling")
ct = pd.crosstab(tk["rider_comment"].to_pandas(), tk["priority"].to_pandas())
c2 = stats.chi2_contingency(ct)
print(f"template x priority chi2 p={c2[1]:.3f}, V={np.sqrt(c2[0] / (ct.values.sum() * (min(ct.shape) - 1))):.4f}")
print(tk.group_by("priority").agg(pl.col("resolution_hours").median(), (pl.col("resolution_status") == "unresolved").mean().alias("unres")).sort("priority"))
print(f"KW p={stats.kruskal(*[tk.filter((pl.col('priority') == p) & pl.col('resolution_hours').is_not_null())['resolution_hours'].to_numpy() for p in ['low', 'medium', 'high']]).pvalue:.3f}")
G5 = {"low_range": "battery", "other": "battery", "app_issue": "app/qr", "no_battery_available": "stockout", "long_queue": "queue", "billing_dispute": "billing"}
tk = tk.with_columns(pl.col("category").replace_strict(G5).alias("g5"))
print(tk.group_by("g5").agg(pl.len(), (pl.col("resolution_status") == "unresolved").mean().alias("unres"), pl.col("resolution_hours").median().alias("med"),
                            pl.col("resolution_hours").quantile(0.9).alias("p90")).sort("g5"))
print(f"unresolved by g5 chi2 p={stats.chi2_contingency(pd.crosstab(tk['g5'].to_pandas(), (tk['resolution_status'] == 'unresolved').to_pandas()))[1]:.3f}")
so = events(["rider_id", "event_ts", "is_stockout"]).filter(pl.col("is_stockout")).select("rider_id", "event_ts").collect(engine="streaming")
for c in ["no_battery_available", "billing_dispute"]:
    t = tk.filter(pl.col("category") == c).select("ticket_id", "rider_id", "created_ts", "rider_comment")
    j = (t.join(so, on="rider_id").filter((pl.col("event_ts") <= pl.col("created_ts")) & (pl.col("event_ts") > pl.col("created_ts") - pl.duration(days=7)))
         .group_by("ticket_id").agg(pl.len().alias("k")))
    t = t.join(j, on="ticket_id", how="left").with_columns(pl.col("k").fill_null(0), pl.col("rider_comment").str.contains("third time").alias("third"))
    print(c, t.group_by("third").agg(pl.len(), (pl.col("k") >= 1).mean().alias("ge1"), (pl.col("k") >= 3).mean().alias("ge3")).sort("third"))
del so
smo = station_month().group_by("station_id").agg(pl.col("attempts").sum(), pl.col("completed").sum(), pl.col("system_errors").sum(),
                                                 (pl.col("stockouts") + pl.col("abandoned") + pl.col("system_errors")).sum().alias("fail"))
stt = (tk.filter(pl.col("station_id").is_not_null()).group_by("station_id")
       .agg(pl.col("category").is_in(["no_battery_available", "long_queue"]).sum().alias("svc"), (pl.col("lab") == "qr").sum().alias("qr")))
stt = smo.join(stt, on="station_id", how="left").fill_null(0).filter(pl.col("completed") >= 5000)
print(f"stations {stt.height}; rho(svc/attempts, fail/attempts) {stats.spearmanr(stt['svc'] / stt['attempts'], stt['fail'] / stt['attempts']).statistic:.3f}; "
      f"rho(qr/attempts, syserr/attempts) {stats.spearmanr(stt['qr'] / stt['attempts'], stt['system_errors'] / stt['attempts']).statistic:.3f} "
      f"p={stats.spearmanr(stt['qr'] / stt['attempts'], stt['system_errors'] / stt['attempts']).pvalue:.2f}")
net = tk.group_by("month").agg(pl.len().alias("t")).join(swaps_cm.group_by("month").agg(pl.col("sw").sum()), on="month").sort("month")
print(f"network tickets/1K swaps Jan-24 {net['t'][0] / net['sw'][0] * 1000:.2f}, Jun-25 {net['t'][-1] / net['sw'][-1] * 1000:.2f}; "
      f"H1-24 {net.filter(H1_24)['t'].sum() / net.filter(H1_24)['sw'].sum() * 1000:.2f}, H1-25 {net.filter(H1_25)['t'].sum() / net.filter(H1_25)['sw'].sum() * 1000:.2f}")
print(f"runtime {time.time() - T0:.1f}s")

# %% [markdown]
# ## I. Follow-ups: (1) pricing DiD control-group confounding, (2) what the ticket battery_id is, (3) a placebo-validated Kyron excess

# %%
hdr("I1 pricing DiD: is the control group (3 of 4 = Kyron-heavy cities) driving it?")
print("Mumbai-only control IRR %.3f (%.3f-%.3f) p=%.4f" % fit(p_all[p_all["pilot"] | (p_all["city"] == "Mumbai")]))
print("Mumbai-only control, per active rider IRR %.3f (%.3f-%.3f) p=%.4f" % fit(p_all[p_all["pilot"] | (p_all["city"] == "Mumbai")], off="riders"))
pc_ = p_all[~p_all["pilot"]].copy()
pc_["kyc"] = (pc_["city"].isin(KYC) & (pc_["month"] >= pd.Timestamp("2024-09-01"))).astype(int)
print("placebo inside control cities: Kyron cities x post-Sep-24 vs Mumbai, billing IRR %.3f (%.3f-%.3f) p=%.4f"
      % fit(pc_, "n ~ kyc + C(city) + C(ms)", var="kyc"))
for c in CITIES:
    d = p_all[p_all["city"] == c]
    pre_r = d[d["month"] < pd.Timestamp(BASE_PRICE_CHANGE)]
    post_r = d[d["month"] >= pd.Timestamp(PILOT_START)]
    print(f"  {c:10s} billing/1K swaps Jan-Jun-24 {pre_r['n'].sum() / pre_r['sw'].sum() * 1000:.2f} -> Oct-24..Jun-25 "
          f"{post_r['n'].sum() / post_r['sw'].sum() * 1000:.2f} ({post_r['n'].sum() / post_r['sw'].sum() / (pre_r['n'].sum() / pre_r['sw'].sum()):.2f}x)")
pa = partners()
print(pa.select("partner_id", "partner_name", "partner_segment", "vehicle_class", "peak_surcharge_billable", "cities_active", "amendment_date"))
# billing tickets by fleet partner, pilot vs control, pre vs post
bp = (tk.filter((pl.col("category") == "billing_dispute") & pl.col("station_city").is_not_null())
      .with_columns(pl.col("station_city").is_in(PILOT_CITIES).alias("pilot"), (pl.col("month") >= PILOT_START).alias("post"),
                    pl.col("partner_id").fill_null("independent").alias("pid"))
      .group_by("pilot", "post", "pid").agg(pl.len().alias("n")).sort("pilot", "post", "pid"))
swp = (events(["is_completed", "city", "month", "partner_id"]).filter(pl.col("is_completed"))
       .with_columns(pl.col("city").is_in(PILOT_CITIES).alias("pilot"), (pl.col("month") >= PILOT_START).alias("post"),
                     pl.col("partner_id").fill_null("independent").alias("pid"))
       .group_by("pilot", "post", "pid").agg(pl.len().alias("sw")).collect(engine="streaming"))
bpp = (bp.join(swp, on=["pilot", "post", "pid"], how="full", coalesce=True).with_columns(pl.col("n").fill_null(0))
       .with_columns((pl.col("n") / pl.col("sw") * 1000).alias("r")).pivot(on="post", index=["pilot", "pid"], values=["r", "n"]))
bpp = bpp.rename({c: c.replace("_true", "_post").replace("_false", "_pre") for c in bpp.columns}).with_columns((pl.col("r_post") / pl.col("r_pre")).alias("ratio"))
print(bpp.sort("pid", "pilot"))
# PEAK exposure among INDEPENDENT riders only (fleet riders are billed on the PARTNER tariff code)
pt2 = (tk.filter(pl.col("station_city").is_in(PILOT_CITIES) & (pl.col("month") >= PILOT_START) & pl.col("partner_id").is_null())
       .join(peak_r, on="rider_id", how="inner")
       .group_by(pl.col("category") == "billing_dispute").agg(pl.len(), pl.col("peak_share").mean(), (pl.col("peak_share") > 0).mean().alias("any_peak")))
print("independent riders, pilot cities post-pilot: PEAK exposure billing vs other ticket filers\n", pt2)

# %%
hdr("I2 what does a ticket's battery_id point to? (returned pack = battery_in_id)")
ev_in = (events(["rider_id", "event_ts", "battery_in_id", "soh_in_pct", "event_type", "month", "city"])
         .filter(pl.col("battery_in_id").is_not_null())
         .join(pairs.lazy().select("rider_id", "battery_id").unique(), left_on=["rider_id", "battery_in_id"], right_on=["rider_id", "battery_id"], how="semi")
         .collect(engine="streaming"))
mm = (pairs.join(ev_in.rename({"battery_in_id": "battery_id"}), on=["rider_id", "battery_id"], how="inner")
      .with_columns(((pl.col("event_ts") - pl.col("created_ts")).dt.total_minutes() / 60).alias("dt_h"))
      .with_columns(pl.col("dt_h").abs().alias("adt")).sort("adt").group_by("ticket_id").first())
print(f"linked tickets {pairs.height:,}; battery_id seen as a pack this rider inserted (any attempt type) {mm.height:,} ({mm.height / pairs.height:.4f})")
print("closest such insertion relative to the ticket (hours):", mm.select(pl.col("dt_h").quantile(q).alias(f"q{int(q * 100)}") for q in [0.05, 0.25, 0.5, 0.75, 0.95]))
print(f"share within -48h..0h (pack returned before the ticket) {mm['dt_h'].is_between(-48, 0).mean():.3f}; within 0..+48h {mm['dt_h'].is_between(0, 48).mean():.3f}; "
      f"|dt|>30 days {(mm['adt'] > 720).mean():.3f}")
print(mm.group_by("event_type").agg(pl.len()).sort("len", descending=True))
# per-ticket SoH of the returned pack vs same-supplier, same-city-month returned packs
soh_ref = (events(["battery_in_id", "soh_in_pct", "month", "city", "vehicle_class"]).filter(pl.col("soh_in_pct").is_not_null() & (pl.col("vehicle_class") == "2W"))
           .join(bat.lazy(), left_on="battery_in_id", right_on="battery_id", how="left")
           .group_by("sup", "city", "month").agg(pl.col("soh_in_pct").mean().alias("ref_soh")).collect(engine="streaming"))
near_ = (mm.filter(pl.col("adt") <= 72).join(tk.select("ticket_id", "sup", "pvc"), on="ticket_id")
         .filter(pl.col("pvc") == "2W").join(soh_ref, on=["sup", "city", "month"], how="left")
         .with_columns((pl.col("soh_in_pct") - pl.col("ref_soh")).alias("gap")))
print(f"2W tickets whose pack was inserted by the rider within 72h of the ticket: {near_.height:,}")
print(near_.group_by("category").agg(pl.len(), pl.col("gap").mean().alias("soh_gap_mean"), (pl.col("gap").std() / pl.len().sqrt()).alias("se"),
                                     pl.col("soh_in_pct").mean().alias("soh_in")).sort("soh_gap_mean"))

# %%
hdr("I3 Kyron excess from the placebo-validated city dose test (Oct-Dec-24)")
ky_sw_od = float(sw_city.filter(OD & (pl.col("vehicle_class") == "2W") & pl.col("city").is_in(KYC))["sw"].sum())
(hja, mja), (hod, mod) = city_grp_rate(["low_range", "other", "app_issue"], JA, "2W", "2W"), city_grp_rate(["low_range", "other", "app_issue"], OD, "2W", "2W")
cf = hja * (mod / mja)
exc_od = (hod - cf) / 1000 * ky_sw_od
obs_od = hod / 1000 * ky_sw_od
print(f"Kyron cities Oct-Dec-24: observed {hod:.2f}, counterfactual {cf:.2f} per 1K 2W swaps -> excess {hod - cf:.2f}/1K x {ky_sw_od:,.0f} 2W swaps = "
      f"{exc_od:,.0f} tickets ({exc_od / 3:,.0f}/month; {exc_od / obs_od:.3f} of the {obs_od:,.0f} battery-group tickets there)")
(sja, smja), (sod, smod) = city_grp_rate(["no_battery_available", "long_queue"], JA, "2W", "2W"), city_grp_rate(["no_battery_available", "long_queue"], OD, "2W", "2W")
print(f"same calculation on stock-out+queue (placebo): excess {(sod - sja * smod / smja):.2f}/1K = {(sod - sja * smod / smja) / 1000 * ky_sw_od:,.0f} tickets")
print(f"H1-25 city rates (station city): see analyst table; battery group per 1K 2W swaps by city:")
print(tk.filter(pl.col("category").is_in(["low_range", "other", "app_issue"]) & (pl.col("rvc") == "2W") & H1_25 & pl.col("station_city").is_not_null())
      .group_by(pl.col("station_city").alias("city")).agg(pl.len().alias("n"))
      .join(sw_city.filter(H1_25 & (pl.col("vehicle_class") == "2W")).group_by("city").agg(pl.col("sw").sum()), on="city")
      .with_columns((pl.col("n") / pl.col("sw") * 1000).alias("per1k_2w")).sort("per1k_2w"))
print(f"runtime {time.time() - T0:.1f}s")

# %%
hdr("I4 Kyron O/E and within-city rates using the RIGHT population: packs riders INSERTED at attempts (supplier_in)")
ins = (events(["month", "city", "vehicle_class", "supplier_in", "battery_in_id", "is_completed", "supplier_out"])
       .filter((pl.col("vehicle_class") == "2W") & pl.col("battery_in_id").is_not_null())
       .group_by("city", "month").agg((pl.col("supplier_in") == "Kyron").mean().alias("pky_in"), pl.len().alias("ins"),
                                      ((pl.col("supplier_out") == "Kyron") & pl.col("is_completed")).sum().alias("ky_out"),
                                      pl.col("is_completed").sum().alias("comp"))
       .collect(engine="streaming"))
post_ins = ins.filter(pl.col("month") >= pl.date(2024, 9, 1))
print(f"Sep-24..Jun-25 network Kyron share: inserted 2W packs {float((post_ins['pky_in'] * post_ins['ins']).sum() / post_ins['ins'].sum()):.3f} vs issued "
      f"{float(post_ins['ky_out'].sum() / post_ins['comp'].sum()):.3f}")
lk2 = lk.join(ins.select("city", "month", "pky_in"), left_on=["city_any", "month"], right_on=["city", "month"], how="left")
rows = []
for c, d in lk2.group_by("category"):
    O = int((d["sup"] == "Kyron").sum())
    E = float(d["pky_in"].sum())
    rows.append(dict(category=c[0], linked=d.height, OE_inserted=O / E, lo=stats.chi2.ppf(0.025, 2 * O) / 2 / E, hi=stats.chi2.ppf(0.975, 2 * O + 2) / 2 / E))
print(pd.DataFrame(rows).sort_values("OE_inserted", ascending=False).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
att_city = (events(["month", "city", "vehicle_class", "supplier_in", "battery_in_id"])
            .filter((pl.col("vehicle_class") == "2W") & pl.col("battery_in_id").is_not_null() & (pl.col("month") >= pl.date(2024, 9, 1)))
            .group_by("city", (pl.col("supplier_in") == "Kyron").alias("ky")).agg(pl.len().alias("att")).collect(engine="streaming"))
for lab_, cats in [("battery group", ["low_range", "other", "app_issue"]), ("stock-out+queue", ["no_battery_available", "long_queue"]), ("billing", ["billing_dispute"])]:
    t = (tk.filter(POST & (pl.col("pvc") == "2W") & pl.col("category").is_in(cats) & pl.col("station_city").is_not_null())
         .group_by(pl.col("station_city").alias("city"), (pl.col("sup") == "Kyron").alias("ky")).agg(pl.len().alias("n")))
    j = att_city.join(t, on=["city", "ky"], how="left").with_columns((pl.col("n").fill_null(0) / pl.col("att") * 1000).alias("r"))
    w = j.pivot(on="ky", index="city", values="r").rename({"true": "ky", "false": "ca"}).with_columns((pl.col("ky") / pl.col("ca")).alias("ratio"))
    print(f"{lab_}: within-city Kyron/other ticket rate per 1K inserted-pack attempts: {w['ratio'].min():.2f}-{w['ratio'].max():.2f} "
          f"(median {w['ratio'].median():.2f})")

# %%
hdr("I5 lapse wording tests (raw two-proportion, as reported)")
lpp = lp.with_columns(pl.col("rider_comment").str.contains("third time").alias("third"), pl.col("rider_comment").str.contains("late ho gaya").alias("late"))
for cat_, flag in [("no_battery_available", "third"), ("long_queue", "late")]:
    d = lpp.filter(pl.col("category") == cat_)
    a, b = d.filter(pl.col(flag))["lapse30"], d.filter(~pl.col(flag))["lapse30"]
    from statsmodels.stats.proportion import proportions_ztest
    z, p = proportions_ztest([int(a.sum()), int(b.sum())], [a.len(), b.len()])
    print(f"{cat_} '{flag}': {a.mean():.4f} (n={a.len()}) vs {b.mean():.4f} (n={b.len()}), p={p:.3f}")
print(f"runtime {time.time() - T0:.1f}s")

# %%
hdr("I6 the attempt each ticket links to (rider's last insertion of the ticket pack before the ticket)")
lnk = (mm.filter(pl.col("dt_h").is_between(-48, 0)).join(tk.select("ticket_id", "rvc"), on="ticket_id")
       .join(ev_in.select("rider_id", "event_ts", "battery_in_id", "soh_in_pct").rename({"battery_in_id": "battery_id", "soh_in_pct": "s2"}),
             on=["rider_id", "battery_id", "event_ts"], how="left"))
print(lnk.group_by("category").agg(pl.len().alias("n"), (pl.col("event_type") == "swap_completed").mean().alias("completed"),
                                   (pl.col("event_type") == "failed_no_charged_battery").mean().alias("stockout"),
                                   (pl.col("event_type") == "abandoned_queue").mean().alias("abandoned"),
                                   pl.col("dt_h").median().alias("median_h_before")).sort("category"))
soc = (events(["rider_id", "event_ts", "battery_in_id", "soc_in_pct"]).filter(pl.col("battery_in_id").is_not_null())
       .join(lnk.lazy().select("rider_id", "event_ts").unique(), on=["rider_id", "event_ts"], how="semi").collect(engine="streaming"))
lnk2 = lnk.join(soc.rename({"battery_in_id": "battery_id"}), on=["rider_id", "battery_id", "event_ts"], how="left")
print(lnk2.group_by("category").agg(pl.col("soc_in_pct").mean().alias("mean_soc_in"), (pl.col("soc_in_pct") < 10).mean().alias("soc_lt10")).sort("mean_soc_in"))

hdr("I7 dose test per battery category; per-swap denominators vs swaps per rider")
for c in ["low_range", "other", "app_issue"]:
    (a1, b1), (a2, b2) = city_grp_rate([c], JA, "2W", "2W"), city_grp_rate([c], OD, "2W", "2W")
    print(f"{c}: Kyron cities {a1:.2f}->{a2:.2f} ({a2 / a1:.2f}x) vs others {b1:.2f}->{b2:.2f} ({b2 / b1:.2f}x); ratio of ratios {(a2 / a1) / (b2 / b1):.2f}")
spr = (events(["is_completed", "city", "month", "rider_id", "vehicle_class"]).filter(pl.col("is_completed") & (pl.col("vehicle_class") == "2W"))
       .with_columns(pl.col("city").is_in(KYC).alias("ky"))
       .group_by("ky", "month").agg(pl.len().alias("sw"), pl.col("rider_id").n_unique().alias("riders")).collect(engine="streaming")
       .with_columns((pl.col("sw") / pl.col("riders")).alias("sw_per_rider")).sort("ky", "month"))
w_ = spr.pivot(on="ky", index="month", values="sw_per_rider").rename({"true": "kyron_cities", "false": "other_cities"}).sort("month")
import datetime as _dt
print(w_.filter(pl.col("month").is_in([_dt.date(2024, 7, 1), _dt.date(2024, 8, 1), _dt.date(2024, 10, 1), _dt.date(2024, 12, 1), _dt.date(2025, 6, 1)])))

hdr("I8 corrected Kyron per-pack excess using inserted-pack attempts as the denominator")
t = (tk.filter(POST & (pl.col("pvc") == "2W") & pl.col("category").is_in(["low_range", "other", "app_issue"]) & pl.col("station_city").is_not_null())
     .group_by(pl.col("station_city").alias("city"), (pl.col("sup") == "Kyron").alias("ky")).agg(pl.len().alias("n")))
j = att_city.join(t, on=["city", "ky"], how="left").with_columns(pl.col("n").fill_null(0))
w8 = j.pivot(on="ky", index="city", values=["n", "att"])
w8 = w8.rename({c: c.replace("_true", "_ky").replace("_false", "_ca") for c in w8.columns}).with_columns(
    (pl.col("n_ky") - pl.col("n_ca") / pl.col("att_ca") * pl.col("att_ky")).alias("excess"))
print(f"excess battery-group tickets on Kyron packs vs same-city Cellora+Amptek per inserted-pack attempt: {w8['excess'].sum():,.0f} of "
      f"{int(w8['n_ky'].sum() + w8['n_ca'].sum()):,} ({w8['excess'].sum() / (w8['n_ky'].sum() + w8['n_ca'].sum()):.3f})")
svc_h1 = (tk.filter(pl.col("category").is_in(["no_battery_available", "long_queue"]) & H1_25 & pl.col("station_city").is_not_null())
          .group_by(pl.col("station_city").alias("city")).agg(pl.len().alias("n"))
          .join(sw_city.filter(H1_25).group_by("city").agg(pl.col("sw").sum()), on="city").with_columns((pl.col("n") / pl.col("sw") * 1000).alias("r")).sort("r"))
print(f"H1-25 stock-out+queue per 1K swaps by city: {dict(zip(svc_h1['city'], svc_h1['r'].round(2)))}")
print(f"runtime {time.time() - T0:.1f}s")

# %%
hdr("I9 rider-level check of app_issue/QR text: a logged system error or cancellation by the rider in the 7 days before the ticket?")
se = (events(["rider_id", "event_ts", "event_type"]).filter(pl.col("event_type").is_in(["failed_system_error", "cancelled_by_rider"]))
      .select("rider_id", "event_ts", "event_type").collect(engine="streaming"))
tt = tk.select("ticket_id", "rider_id", "created_ts", "category", "lab")
jj9 = (tt.join(se, on="rider_id").filter((pl.col("event_ts") <= pl.col("created_ts")) & (pl.col("event_ts") > pl.col("created_ts") - pl.duration(days=7)))
       .group_by("ticket_id").agg((pl.col("event_type") == "failed_system_error").sum().alias("se7"), (pl.col("event_type") == "cancelled_by_rider").sum().alias("cx7")))
tt = tt.join(jj9, on="ticket_id", how="left").with_columns(pl.col("se7").fill_null(0), pl.col("cx7").fill_null(0))
print(tt.group_by("lab").agg(pl.len(), (pl.col("se7") >= 1).mean().alias("sys_err_7d"), (pl.col("cx7") >= 1).mean().alias("cancel_7d")).sort("sys_err_7d"))
