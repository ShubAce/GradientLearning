# 1. Problem understanding

## 1.1 The business
VoltRelay Energy runs unmanned battery-swap cabinets for electric two- and three-wheelers in six Indian cities
(Bengaluru, Delhi NCR, Hyderabad, Pune, Mumbai, Jaipur). Its customers are gig and logistics riders (food delivery, quick
commerce, e-commerce logistics, cargo, bike-taxi) for whom time off the road is lost income: a rider swaps a depleted
pack for a charged one in about two minutes, instead of charging for hours. Independent riders pay per swap or with a
prepaid pack; riders affiliated with one of 12 fleet partners are billed to their employer at a contracted discount.

## 1.2 The paradox leadership faces
Between January 2024 and June 2025 completed swaps roughly tripled and monthly revenue more than tripled, yet leadership
sees three things moving the wrong way: **service failures**, **new-rider retention** and **per-swap profitability**.
At the same time the company changed almost everything at once: two station-expansion waves, a base-price increase,
a peak/off-peak pricing pilot in two cities, a new battery supplier and a renegotiated contract with its largest fleet
partner. External shocks were layered on top: two Indian summers, a competitor opening near ten stations and a firmware
clock bug.

:::fig q0_02_timeline.png | **Figure 1.1** Nine overlapping changes in 18 months. Any before/after comparison has to separate these, especially the summer heat that recurs in April–June of both years.

The decision at stake is the **next operating budget**. Internal teams have proposed four uses for it: more stations, more
batteries, a network-wide pricing rollout, and a long-term exclusive with the largest fleet partner. Leadership needs to
know what is actually driving the three outcomes before choosing.

## 1.3 How we framed the question
We decomposed leadership's question into the three outcomes that deteriorated and turned each into testable hypotheses,
each tied to a data source and a method (Figure 1.2). The six core questions of the brief map onto this tree: network
performance over time (Q1) measures the outcomes; service failures (Q2), stations and geography (Q3), batteries (Q4),
pricing and partners (Q5) and retention (Q6) test the branches.

:::fig q0_01_issue_tree.png | **Figure 1.2** Hypothesis tree. Each hypothesis was tested against the data; Section 5 reports which ones survived and whether each is a primary or a secondary driver.

## 1.4 What a good answer looks like
| Outcome | KPI we track | Definition |
|---|---|---|
| Growth | Completed swaps, revenue | `swap_completed` events; `amount_charged_inr` (test stations excluded) |
| Service quality | Service-failure rate; queue wait | (stockouts + abandoned-in-queue + system errors) / attempts; p90 wait of completed swaps |
| Retention | New-rider return rate | share of riders who sign up in the window and swap again in days 31–60 after their first attempt |
| Profitability | Contribution margin per swap | **cm1** = revenue − energy − battery wear; **cm2** = cm1 − station rent & maintenance |

Two principles guide the analysis. First, **same-season comparisons**: failure rates swing from about 4% to 12% between
winter and summer, so comparing January 2024 with June 2025 manufactures a trend. Second, **association is not
causation**: where the data allows we use designs that get closer to cause (difference-in-differences for the pricing
pilot, dose-response and within-rider comparisons for batteries, pre-outcome features for churn) and we label the strength
of each claim.
