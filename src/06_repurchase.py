"""Step 6: Post-purchase evaluation and repurchase within 180 days (RQ5).

Unit: customer (customer_unique_id). Index order = first non-cancelled order.
Eligibility: index order delivered and reviewed; index purchase on or before 2018-03-04 so that the full
180-day window ends by 2018-08-31 (the last complete month in the data).
Outcome: any further non-cancelled order placed more than 1 day and at most 180 days after the index purchase
(orders placed within 1 day are treated as split baskets, not repurchases).
Usage: python 06_repurchase.py <out_dir>
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
OUT = Path(sys.argv[1])
od = pd.read_parquet(OUT / "orders_analytic.parquet")
ts = pd.read_parquet(OUT / "text_scored.parquet")[["order_id", "p_neg_text", "text_negative", "leia_compound"]]
od = od.merge(ts, on="order_id", how="left")

valid = od[~od.order_status.isin(["canceled", "unavailable"])].sort_values("order_purchase_timestamp")
first = valid.groupby("customer_unique_id").head(1).copy()
END = pd.Timestamp("2018-08-31 23:59:59")
H = 180
elig = first[(first.order_status == "delivered") & first.review_score.notna() &
             first.order_delivered_customer_date.notna() &
             (first.order_purchase_timestamp >= "2017-01-01") &
             (first.order_purchase_timestamp <= END - pd.Timedelta(days=H))].copy()

nxt = valid[["customer_unique_id", "order_purchase_timestamp"]].rename(columns={"order_purchase_timestamp": "t2"})
mm = elig[["customer_unique_id", "order_purchase_timestamp"]].merge(nxt, on="customer_unique_id")
mm["gap"] = (mm.t2 - mm.order_purchase_timestamp).dt.total_seconds() / 86400
rep = mm[(mm.gap > 1) & (mm.gap <= H)].customer_unique_id.unique()
elig["repurchase"] = elig.customer_unique_id.isin(rep).astype(int)

elig["low_rating"] = (elig.review_score <= 2).astype(int)
elig["late"] = elig.late.astype(int)
elig["log_value"] = np.log(elig.item_value)
elig["freight_ratio10"] = elig.freight_ratio * 10
elig["pay"] = elig.payment_type_main.fillna("credit_card").replace({"not_defined": "credit_card"})
top_cat = elig.primary_category.value_counts().head(15).index
elig["cat"] = np.where(elig.primary_category.isin(top_cat), elig.primary_category, "other")
elig["score_cat"] = elig.review_score.astype(int).astype(str)
elig = elig.dropna(subset=["log_value", "freight_ratio10", "customer_region"])

res = {"n_eligible": int(len(elig)), "repurchases": int(elig.repurchase.sum()),
       "rate": float(elig.repurchase.mean()),
       "rate_by_score": elig.groupby("review_score").repurchase.mean().round(4).to_dict(),
       "n_by_score": elig.review_score.value_counts().sort_index().to_dict(),
       "rate_low_vs_high": {"low(1-2)": float(elig[elig.low_rating == 1].repurchase.mean()),
                            "high(4-5)": float(elig[elig.review_score >= 4].repurchase.mean())},
       "rate_late": elig.groupby("late").repurchase.mean().round(4).to_dict()}

ctrl = ("log_value + freight_ratio10 + C(pay, Treatment('credit_card')) + C(customer_region, Treatment('Southeast')) "
        "+ C(cat, Treatment('bed_bath_table')) + C(ym)")


def fit(formula, data):
    m = smf.logit(formula, data=data).fit(disp=0, maxiter=200, cov_type="cluster",
                                          cov_kwds={"groups": pd.factorize(data.primary_seller.fillna("na"))[0]})
    return m


def tidy(m, terms):
    out = {}
    for t in terms:
        if t in m.params:
            out[t] = [float(np.exp(m.params[t])), float(np.exp(m.conf_int().loc[t, 0])),
                      float(np.exp(m.conf_int().loc[t, 1])), float(m.pvalues[t])]
    return out


m1 = fit(f"repurchase ~ low_rating + late + {ctrl}", elig)
m1b = fit(f"repurchase ~ C(score_cat, Treatment('5')) + late + {ctrl}", elig)
res["M1_all"] = {"n": int(m1.nobs), "events": int(elig.repurchase.sum()),
                 "OR": tidy(m1, ["low_rating", "late"])}
res["M1b_score_levels"] = {"OR": tidy(m1b, [f"C(score_cat, Treatment('5'))[T.{s}]" for s in "1234"] + ["late"])}
txt = elig[elig.p_neg_text.notna()].copy()
m2 = fit(f"repurchase ~ text_negative + late + {ctrl}", txt)
m3 = fit(f"repurchase ~ text_negative + low_rating + late + {ctrl}", txt)
res["M2_text"] = {"n": int(m2.nobs), "events": int(txt.repurchase.sum()),
                  "rate_text_neg": float(txt[txt.text_negative == 1].repurchase.mean()),
                  "rate_text_nonneg": float(txt[txt.text_negative == 0].repurchase.mean()),
                  "OR": tidy(m2, ["text_negative", "late"])}
res["M3_text_plus_stars"] = {"n": int(m3.nobs), "OR": tidy(m3, ["text_negative", "low_rating", "late"])}
# Sensitivity: 90-day and 365-day windows
for HH in [90, 365]:
    e2 = first[(first.order_status == "delivered") & first.review_score.notna() &
               (first.order_purchase_timestamp >= "2017-01-01") &
               (first.order_purchase_timestamp <= END - pd.Timedelta(days=HH))].copy()
    mm2 = e2[["customer_unique_id", "order_purchase_timestamp"]].merge(nxt, on="customer_unique_id")
    mm2["gap"] = (mm2.t2 - mm2.order_purchase_timestamp).dt.total_seconds() / 86400
    rr = mm2[(mm2.gap > 1) & (mm2.gap <= HH)].customer_unique_id.unique()
    e2 = elig.drop(columns=["repurchase"]).merge(
        e2[["customer_unique_id"]].assign(repurchase=lambda x: x.customer_unique_id.isin(rr).astype(int)),
        on="customer_unique_id", how="inner") if HH > H else None
    if HH < H:
        e2 = elig.copy()
        e2["repurchase"] = e2.customer_unique_id.isin(rr).astype(int)
    m = fit(f"repurchase ~ low_rating + late + {ctrl}", e2)
    res[f"S_window_{HH}"] = {"n": int(m.nobs), "events": int(e2.repurchase.sum()), "OR": tidy(m, ["low_rating", "late"])}

with open(OUT / "repurchase_results.json", "w") as f:
    json.dump(res, f, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else int(o))
print(json.dumps(res, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else int(o)))
