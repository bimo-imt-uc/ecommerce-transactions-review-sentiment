"""Step 7: Sentiment-validation figure, complaint themes by delivery status, and the share of
negative evaluations coming from late and multi-item orders.
Usage: python 07_themes_and_validation.py <out_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_curve

OUT = Path(sys.argv[1])
FIG = OUT / "figures"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

# ---------------- Fig. 3 sentiment validation
tp = pd.read_parquet(OUT / "sentiment_test_predictions.parquet")
ts = pd.read_parquet(OUT / "text_scored.parquet")
od = pd.read_parquet(OUT / "orders_analytic.parquet")
sr = json.load(open(OUT / "sentiment_results.json"))
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45))
ax = axes[0]
for col, lab, c in [("p_lr", "TF-IDF + LR", "#E45756"), ("p_nb", "TF-IDF + NB", "#F58518"),
                    ("leia", "LeIA lexicon", "#54A24B")]:
    s = -tp.leia_compound if col == "leia" else tp[col]
    fpr, tpr, _ = roc_curve(tp.y_neg, s)
    key = {"p_lr": "TFIDF_LR", "p_nb": "TFIDF_NB", "leia": "LeIA"}[col]
    ax.plot(fpr, tpr, color=c, lw=1.2, label=f"{lab} (AUC {sr[key]['AUC']:.3f})")
ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.7)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("(a) Temporal hold-out ROC", loc="left", fontsize=8)
ax.legend(frameon=False, fontsize=6.5, loc="lower right")

ax = axes[1]
stars = [1, 2, 3, 4, 5]
lr = [sr["share_text_negative_by_star_LR"][f"{s}.0"] * 100 for s in stars]
le = [sr["share_text_negative_by_star_LeIA"][f"{s}.0"] * 100 for s in stars]
x = np.arange(5)
ax.bar(x - 0.18, lr, width=0.36, color="#E45756", label="TF-IDF + LR (out-of-fold)")
ax.bar(x + 0.18, le, width=0.36, color="#54A24B", label="LeIA lexicon")
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in stars])
ax.set_xlabel("Star rating")
ax.set_ylabel("Texts classified negative (%)")
ax.set_ylim(0, 105)
ax.legend(frameon=False, fontsize=6.5, loc="upper right")
ax.set_title("(b) Negative text by star rating", loc="left", fontsize=8)

ax = axes[2]
wt = [sr["share_with_text_by_star"][f"{s}.0"] * 100 for s in stars]
ax.bar(x, wt, width=0.6, color="#4C78A8")
for i, v in enumerate(wt):
    ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=6.5)
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in stars])
ax.set_xlabel("Star rating")
ax.set_ylabel("Reviews with written text (%)")
ax.set_ylim(0, 90)
ax.set_title("(c) Who writes text", loc="left", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "sentiment_validation.png", dpi=600)
plt.close(fig)

# ---------------- topics: theme table
tr = json.load(open(OUT / "topics_results.json"))
nt = pd.read_parquet(OUT / "neg_topics.parquet")
theme_map = {0: "Non-receipt and waiting", 2: "Non-receipt and waiting", 3: "Non-receipt and waiting",
             4: "Non-receipt and waiting", 8: "Non-receipt and waiting", 5: "Delivery delay",
             13: "Incomplete order", 14: "Incomplete order", 1: "Wrong or defective item",
             6: "Wrong or defective item", 7: "Product quality", 12: "Product quality",
             9: "After-sales and administrative", 10: "After-sales and administrative",
             11: "After-sales and administrative"}
nt["theme"] = nt.topic.map(theme_map).fillna("Unassigned")
ct = pd.crosstab(nt.theme, nt.status_group)
chi2, p, dof, _ = chi2_contingency(ct[ct.index != "Unassigned"])
n = ct[ct.index != "Unassigned"].values.sum()
V = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
themes = {"overall_share": (nt.theme.value_counts(normalize=True)).round(4).to_dict(),
          "col_pct_by_status": (ct / ct.sum(axis=0)).round(4).to_dict(),
          "counts_by_status": ct.to_dict(), "chi2": float(chi2), "dof": int(dof), "p": float(p), "cramers_v": V,
          "fulfilment_share_overall": float(nt.theme.isin(["Non-receipt and waiting", "Delivery delay",
                                                          "Incomplete order"]).mean())}
for g in ["delivered_on_time_or_early", "delivered_late", "not_delivered"]:
    sub = nt[nt.status_group == g]
    themes[f"fulfilment_share_{g}"] = float(sub.theme.isin(["Non-receipt and waiting", "Delivery delay",
                                                           "Incomplete order"]).mean())
    themes[f"product_share_{g}"] = float(sub.theme.isin(["Wrong or defective item", "Product quality"]).mean())

# ---------------- where negative evaluations come from (delivered, reviewed, window)
dd = od[(od.order_status == "delivered") & od.order_delivered_customer_date.notna() & od.review_score.notna() &
        (od.order_purchase_timestamp >= "2017-01-01") & (od.order_purchase_timestamp < "2018-09-01")].copy()
dd = dd.merge(ts[["order_id", "text_negative"]], on="order_id", how="left")
dd["low"] = (dd.review_score <= 2).astype(int)
orig = {"n": int(len(dd)), "late_share_orders": float(dd.late.mean()),
        "share_of_low_ratings_from_late_orders": float(dd[dd.low == 1].late.mean()),
        "share_of_neg_text_from_late_orders": float(dd[dd.text_negative == 1].late.mean()),
        "share_of_low_ratings_from_multi_item": float((dd[dd.low == 1].n_items > 1).mean()),
        "low_rating_rate_late": float(dd[dd.late == 1].low.mean()),
        "low_rating_rate_not_late": float(dd[dd.late == 0].low.mean()),
        "low_rating_rate_multi_item": float(dd[dd.n_items > 1].low.mean()),
        "low_rating_rate_single_item": float(dd[dd.n_items == 1].low.mean())}
# multi-item orders and incomplete-order complaints
mi = nt.merge(od[["order_id", "n_items"]], on="order_id", how="left")
mi["multi"] = mi.n_items > 1
inc = mi.theme == "Incomplete order"
orig["multi_item_share_of_negative_texts"] = float(mi.multi.mean())
orig["multi_item_share_of_incomplete_order_texts"] = float(mi[inc].multi.mean())
json.dump({"themes": themes, "origin_of_negatives": orig}, open(OUT / "supplementary_numbers.json", "w"), indent=1)
print(json.dumps({"themes": themes, "origin_of_negatives": orig}, indent=1))
