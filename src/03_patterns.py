"""Step 3: Descriptive consumption-pattern analytics (RQ1).

Outputs JSON with KPIs and two figures:
  consumption_patterns.png  (a) monthly orders and gross merchandise value; (b) weekday x hour purchase heat map;
                    (c) category concentration (Lorenz curve); (d) customer purchase frequency
Usage: python 03_patterns.py <out_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(sys.argv[1])
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False})

od = pd.read_parquet(OUT / "orders_analytic.parquet")
items = pd.read_parquet(OUT / "items_enriched.parquet")
pay = pd.read_parquet(OUT / "payments.parquet")

# analysis window with complete months
win = od[(od.order_purchase_timestamp >= "2017-01-01") & (od.order_purchase_timestamp < "2018-09-01")].copy()
valid = win[~win.order_status.isin(["canceled", "unavailable"])].copy()
valid["gmv"] = valid.item_value.fillna(0) + valid.freight.fillna(0)

k = {}
k["window"] = "2017-01-01 to 2018-08-31"
k["orders_window_all"] = int(len(win))
k["orders_window_valid"] = int(len(valid))
k["customers_valid"] = int(valid.customer_unique_id.nunique())
k["gmv_total_BRL"] = float(valid.gmv.sum())
k["item_value_total_BRL"] = float(valid.item_value.sum())
k["aov_median_BRL"] = float(valid.gmv.median())
k["aov_mean_BRL"] = float(valid.gmv.mean())
k["aov_IQR"] = [float(valid.gmv.quantile(.25)), float(valid.gmv.quantile(.75))]
k["items_per_order_mean"] = float(valid.n_items.mean())
k["share_multi_item"] = float((valid.n_items > 1).mean())
k["freight_ratio_median"] = float(valid.freight_ratio.median())
pm = valid.payment_type_main.value_counts(normalize=True)
k["payment_main_share"] = pm.round(4).to_dict()
cc = valid[valid.payment_type_main == "credit_card"]
k["credit_card_installments_median"] = float(cc.installments_max.median())
k["credit_card_share_installments_gt1"] = float((cc.installments_max > 1).mean())
k["region_share_orders"] = valid.customer_region.value_counts(normalize=True).round(4).to_dict()
k["state_SP_share"] = float((valid.customer_state == "SP").mean())
k["cross_state_share"] = float(valid.cross_state.mean())

# monthly series
m = valid.groupby("ym").agg(orders=("order_id", "size"), gmv=("gmv", "sum")).reset_index()
k["monthly"] = m.round(2).to_dict(orient="list")
# growth: Jan-Aug 2017 vs Jan-Aug 2018
o17 = m[m.ym.between("2017-01", "2017-08")].orders.sum()
o18 = m[m.ym.between("2018-01", "2018-08")].orders.sum()
k["orders_JanAug2017"] = int(o17)
k["orders_JanAug2018"] = int(o18)
k["yoy_growth_JanAug"] = float(o18 / o17 - 1)
daily = valid.groupby(valid.order_purchase_timestamp.dt.date).size()
k["peak_day"] = [str(daily.idxmax()), int(daily.max())]
k["daily_median"] = float(daily.median())
nov = m.set_index("ym").orders
k["nov2017_vs_mean_neighbours"] = float(nov["2017-11"] / ((nov["2017-10"] + nov["2017-12"]) / 2) - 1)

# weekday-hour
valid["wd"] = valid.order_purchase_timestamp.dt.dayofweek
valid["hr"] = valid.order_purchase_timestamp.dt.hour
heat = valid.groupby(["wd", "hr"]).size().unstack(fill_value=0)
k["share_weekday_mon_fri"] = float((valid.wd <= 4).mean())
k["share_hours_10_22"] = float(valid.hr.between(10, 22).mean())
k["share_hours_0_6"] = float(valid.hr.between(0, 6).mean())
wdshare = valid.wd.value_counts(normalize=True).sort_index()
k["weekday_share"] = wdshare.round(4).to_dict()

# category concentration (item level, valid orders)
it = items[items.order_id.isin(valid.order_id)].copy()
it["rev"] = it.price
catg = it.groupby("category").rev.sum().sort_values(ascending=False)
k["n_categories"] = int(catg.shape[0])
k["top10_categories_rev_share"] = float(catg.head(10).sum() / catg.sum())
k["top10_categories"] = (catg.head(10) / catg.sum()).round(4).to_dict()
s = np.sort(catg.values)
cum = np.cumsum(s) / s.sum()
n = len(s)
gini_cat = 1 - 2 * np.sum(cum) / n + 1 / n
k["gini_category_revenue"] = float(gini_cat)
# seller concentration
sel = it.groupby("seller_id").rev.sum().sort_values(ascending=False)
k["n_sellers_active"] = int(len(sel))
k["top10pct_sellers_rev_share"] = float(sel.head(int(np.ceil(0.1 * len(sel)))).sum() / sel.sum())

# customer behaviour: frequency on distinct purchase days
cust_days = valid.groupby("customer_unique_id").order_purchase_timestamp.apply(lambda x: x.dt.normalize().nunique())
k["customers_repeat_share_distinct_days"] = float((cust_days >= 2).mean())
k["freq_dist"] = cust_days.value_counts().sort_index().head(6).to_dict()
spend = valid.groupby("customer_unique_id").gmv.sum().sort_values(ascending=False)
k["top10pct_customers_spend_share"] = float(spend.head(int(np.ceil(0.1 * len(spend)))).sum() / spend.sum())
s2 = np.sort(spend.values)
cum2 = np.cumsum(s2) / s2.sum()
k["gini_customer_spend"] = float(1 - 2 * np.sum(cum2) / len(s2) + 1 / len(s2))
# inter-purchase interval for repeaters (first to second distinct purchase day)
rp = valid.assign(d=valid.order_purchase_timestamp.dt.normalize()).groupby("customer_unique_id").d.apply(
    lambda x: sorted(set(x)))
gaps = rp[rp.map(len) >= 2].map(lambda x: (x[1] - x[0]).days)
k["interpurchase_days_median"] = float(gaps.median())
k["interpurchase_days_IQR"] = [float(gaps.quantile(.25)), float(gaps.quantile(.75))]

# delivery performance (delivered)
dl = od[(od.order_status == "delivered") & od.order_delivered_customer_date.notna()]
k["delivered_n"] = int(len(dl))
k["delivery_days_median"] = float(dl.delivery_days.median())
k["promised_days_median"] = float(dl.promised_days.median())
k["delay_days_median"] = float(dl.delay_days.median())
k["late_share"] = float(dl.late.mean())
k["late_share_by_region"] = dl.groupby("customer_region").late.mean().round(4).to_dict()
k["delivery_days_median_by_region"] = dl.groupby("customer_region").delivery_days.median().round(2).to_dict()
k["late_share_by_month"] = dl.groupby("ym").late.mean().round(4).to_dict()

# ---------------- figure
fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2))
ax = axes[0, 0]
x = np.arange(len(m))
ax.bar(x, m.orders, color="#4C78A8", width=0.75, label="Orders")
ax.set_ylabel("Orders per month")
ax.set_xticks(x[::2])
ax.set_xticklabels([v[2:] for v in m.ym.values[::2]], rotation=60)
ax2 = ax.twinx()
ax2.plot(x, m.gmv / 1e6, color="#E45756", marker="o", ms=2.5, lw=1.2, label="GMV")
ax2.set_ylabel("GMV (million BRL)")
ax2.spines["top"].set_visible(False)
ax.set_title("(a) Monthly orders and gross merchandise value", loc="left", fontsize=8)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")

ax = axes[0, 1]
im = ax.imshow(heat.values / heat.values.sum() * 100, aspect="auto", cmap="Blues")
ax.set_yticks(range(7))
ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
ax.set_xticks(range(0, 24, 3))
ax.set_xlabel("Hour of purchase")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cb.set_label("% of orders")
ax.set_title("(b) Purchase timing by weekday and hour", loc="left", fontsize=8)

ax = axes[1, 0]
xs = np.concatenate([[0], np.arange(1, n + 1) / n])
ax.plot(xs, np.concatenate([[0], cum]), color="#4C78A8", lw=1.4, label=f"Categories (Gini = {gini_cat:.2f})")
xs2 = np.concatenate([[0], np.arange(1, len(s2) + 1) / len(s2)])
ax.plot(xs2, np.concatenate([[0], cum2]), color="#F58518", lw=1.4,
        label=f"Customers (Gini = {k['gini_customer_spend']:.2f})")
ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
ax.set_xlabel("Cumulative share of units (ascending)")
ax.set_ylabel("Cumulative share of value")
ax.legend(frameon=False, loc="upper left")
ax.set_title("(c) Concentration of revenue (Lorenz curves)", loc="left", fontsize=8)

ax = axes[1, 1]
fd = cust_days.clip(upper=4).value_counts().sort_index()
labels = ["1", "2", "3", "4+"]
ax.bar(range(len(fd)), fd.values / fd.sum() * 100, color="#54A24B")
ax.set_yscale("log")
for i, v in enumerate(fd.values / fd.sum() * 100):
    ax.text(i, v * 1.15, f"{v:.2f}%", ha="center", fontsize=7)
ax.set_xticks(range(len(fd)))
ax.set_xticklabels(labels)
ax.set_xlabel("Distinct purchase days per customer")
ax.set_ylabel("% of customers (log scale)")
ax.set_title("(d) Purchase frequency per customer", loc="left", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "consumption_patterns.png", dpi=600)

with open(OUT / "patterns_results.json", "w") as f:
    json.dump(k, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, np.integer) else str(o))
print(json.dumps({kk: v for kk, v in k.items() if kk not in ("monthly", "late_share_by_month")}, indent=1, default=str))
