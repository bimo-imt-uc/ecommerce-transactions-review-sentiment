"""Step 4: Transaction and fulfilment correlates of negative customer evaluation (RQ3).

Outcomes (delivered orders with a linked review):
  Y1 low_rating      : review score <= 2 (all reviewed orders)
  Y2 neg_text_lr     : negative text, TF-IDF + logistic regression, out-of-fold (text subsample)
  Y3 neg_text_leia   : negative text, LeIA compound <= -0.05 (text subsample)
  Y4 low_rating_text : review score <= 2 within the text subsample (separates selection from measurement)

Model: logistic regression, standard errors clustered by primary seller.
Also: average marginal effects, covariate-adjusted predicted probabilities by delay bin,
a gradient-boosting early-warning model evaluated on a temporal hold-out, and sensitivity analyses.
Usage: python 04_drivers.py <out_dir>
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")
OUT = Path(sys.argv[1])
FIG = OUT / "figures"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

od = pd.read_parquet(OUT / "orders_analytic.parquet")
ts = pd.read_parquet(OUT / "text_scored.parquet")[["order_id", "leia_compound", "p_neg_text", "text_negative"]]

d = od[(od.order_status == "delivered") & od.order_delivered_customer_date.notna() & od.review_score.notna()].copy()
d = d.merge(ts, on="order_id", how="left")
d = d[(d.order_purchase_timestamp >= "2017-01-01") & (d.order_purchase_timestamp < "2018-09-01")]

# ---- variables
bins = [-1e9, -15, -8, -1, 0, 3, 7, 14, 1e9]
labs = ["early15", "early8_14", "early1_7", "ontime", "late1_3", "late4_7", "late8_14", "late15"]
d["delay_bin"] = pd.cut(d.delay_days, bins=bins, labels=labs).astype(str)
d["low_rating"] = (d.review_score <= 2).astype(int)
d["neg_text_lr"] = d.text_negative
d["neg_text_leia"] = (d.leia_compound <= -0.05).astype(float)
d.loc[d.leia_compound.isna(), "neg_text_leia"] = np.nan
d["log_value"] = np.log(d.item_value)
d["freight_ratio10"] = d.freight_ratio * 10  # per 10 percentage points
d["promised_wk"] = d.promised_days / 7
d["dist500"] = d.distance_km / 500
d["multi_item"] = (d.n_items > 1).astype(int)
d["multi_seller"] = (d.n_sellers > 1).astype(int)
d["inst_cat"] = pd.cut(d.installments_max.fillna(1), [-1, 1, 5, 100], labels=["i1", "i2_5", "i6p"]).astype(str)
d["pay"] = d.payment_type_main.fillna("credit_card").replace({"not_defined": "credit_card"})
top_cat = d.primary_category.value_counts().head(15).index
d["cat"] = np.where(d.primary_category.isin(top_cat), d.primary_category, "other")
d["photos2"] = (d.photos_mean.fillna(1) >= 2).astype(int)

covs = ("C(delay_bin, Treatment('early1_7')) + promised_wk + log_value + freight_ratio10 + dist500 + "
        "multi_item + multi_seller + C(inst_cat, Treatment('i1')) + C(pay, Treatment('credit_card')) + photos2 + "
        "C(customer_region, Treatment('Southeast')) + C(cat, Treatment('bed_bath_table')) + C(ym)")
need = ["delay_bin", "promised_wk", "log_value", "freight_ratio10", "dist500", "multi_item", "multi_seller",
        "inst_cat", "pay", "photos2", "customer_region", "cat", "ym", "primary_seller"]
d = d.dropna(subset=["log_value", "freight_ratio10", "dist500", "customer_region", "primary_seller"])

res = {"n_all_reviewed_delivered": int(len(d)), "n_text": int(d.p_neg_text.notna().sum()),
       "delay_bin_counts": d.delay_bin.value_counts().to_dict(),
       "raw_rates_by_bin": {}}
for y in ["low_rating", "neg_text_lr", "neg_text_leia"]:
    res["raw_rates_by_bin"][y] = d.groupby("delay_bin")[y].mean().round(4).to_dict()
res["outcome_prevalence"] = {"low_rating_all": float(d.low_rating.mean()),
                             "neg_text_lr": float(d.neg_text_lr.mean()),
                             "neg_text_leia": float(d.neg_text_leia.mean()),
                             "low_rating_text": float(d[d.p_neg_text.notna()].low_rating.mean())}


def fit(y, data):
    dd = data.dropna(subset=[y]).copy()
    f = covs
    for v in ["multi_item", "multi_seller", "photos2"]:
        if dd[v].nunique() < 2:
            f = f.replace(f"{v} + ", "")
    if dd.customer_region.nunique() < 2:
        f = f.replace("C(customer_region, Treatment('Southeast')) + ", "")
    m = smf.logit(f"{y} ~ {f}", data=dd).fit(disp=0, maxiter=200, cov_type="cluster",
                                                 cov_kwds={"groups": pd.factorize(dd.primary_seller)[0]})
    return m, dd


models = {}
samples = {}
spec = {"Y1_low_rating_all": ("low_rating", d),
        "Y2_neg_text_lr": ("neg_text_lr", d[d.p_neg_text.notna()]),
        "Y3_neg_text_leia": ("neg_text_leia", d[d.p_neg_text.notna()]),
        "Y4_low_rating_text": ("low_rating", d[d.p_neg_text.notna()])}
for name, (y, data) in spec.items():
    m, dd = fit(y, data)
    models[name] = m
    samples[name] = dd
    tab = pd.DataFrame({"OR": np.exp(m.params), "lo": np.exp(m.conf_int()[0]), "hi": np.exp(m.conf_int()[1]),
                        "p": m.pvalues})
    tab = tab[~tab.index.str.startswith("C(ym)")]
    res[name] = {"n": int(m.nobs), "events": int(dd[y].sum()), "pseudo_R2": float(m.prsquared),
                 "AUC_in_sample": float(roc_auc_score(dd[y], m.predict(dd))),
                 "clusters": int(dd.primary_seller.nunique()),
                 "table": tab.round(4).reset_index().rename(columns={"index": "term"}).to_dict(orient="records")}
    print(name, res[name]["n"], res[name]["events"], round(res[name]["pseudo_R2"], 3))

# ---- covariate-adjusted predicted probabilities by delay bin (g-computation), with cluster bootstrap CI
order_bins = labs


def adj_probs(m, dd):
    out = {}
    for b in order_bins:
        tmp = dd.copy()
        tmp["delay_bin"] = b
        out[b] = float(m.predict(tmp).mean())
    return out


adj = {name: adj_probs(models[name], samples[name]) for name in models}
res["adjusted_probs"] = adj

# delta-method 95% CIs for the adjusted probabilities, using the seller-clustered covariance matrix
# (equivalent to the "margins" approach: g(theta) = mean_i F(x_i(b)'theta); grad = mean_i f(.) x_i(b))
def adj_probs_ci(m, dd):
    V = m.cov_params().values
    beta = m.params.values
    X0 = pd.DataFrame(m.model.exog, columns=m.model.exog_names)
    dcols = [c for c in X0.columns if c.startswith("C(delay_bin")]
    out = {}
    for b in order_bins:
        Xb = X0.copy()
        Xb[dcols] = 0.0
        col = f"C(delay_bin, Treatment('early1_7'))[T.{b}]"
        if col in Xb.columns:
            Xb[col] = 1.0
        Xb = Xb.values
        p = 1 / (1 + np.exp(-(Xb @ beta)))
        grad = ((p * (1 - p))[:, None] * Xb).mean(axis=0)
        se = float(np.sqrt(grad @ V @ grad))
        est = float(p.mean())
        out[b] = [est - 1.96 * se, est + 1.96 * se]
    return out


res["adjusted_probs_CI"] = {name: adj_probs_ci(models[name], samples[name])
                            for name in ["Y1_low_rating_all", "Y2_neg_text_lr", "Y3_neg_text_leia", "Y4_low_rating_text"]}
res["adjusted_probs_CI_method"] = "delta method with seller-clustered covariance"

# ---- average marginal effects of key continuous / binary covariates (Y1, Y2)
for name in ["Y1_low_rating_all", "Y2_neg_text_lr"]:
    me = models[name].get_margeff(at="overall", method="dydx")
    sf = me.summary_frame()
    keep = sf[sf.index.str.contains("promised_wk|log_value|freight_ratio10|dist500|multi_item|multi_seller|photos2")]
    res[name + "_AME"] = keep.round(5).reset_index().rename(columns={"index": "term"}).to_dict(orient="records")

# ---- sensitivity analyses on Y1 and Y2 (late indicator coefficient for >14 and 4-7)
sens = {}


def late_or(m):
    p = m.params
    ci = m.conf_int()
    out = {}
    for b in ["late1_3", "late4_7", "late8_14", "late15", "late4p"]:
        t = f"C(delay_bin, Treatment('early1_7'))[T.{b}]"
        if t in p.index:
            out[b] = [float(np.exp(p[t])), float(np.exp(ci.loc[t, 0])), float(np.exp(ci.loc[t, 1]))]
    return out


base_y1 = d.copy()
# S1: exclude reviews created before delivery
s1 = base_y1[base_y1.review_creation_date >= base_y1.order_delivered_customer_date.dt.normalize()].copy()
# almost no post-delivery reviews exist for orders >7 days late, so late bins are pooled (>=4 days late)
s1["delay_bin"] = s1.delay_bin.replace({"late4_7": "late4p", "late8_14": "late4p", "late15": "late4p"})
res["review_before_delivery_by_bin"] = (base_y1.review_creation_date < base_y1.order_delivered_customer_date.dt.normalize()).groupby(base_y1.delay_bin).mean().round(4).to_dict()
res["review_minus_estimated_days_late_orders"] = (base_y1[base_y1.delay_days > 0].review_creation_date - base_y1[base_y1.delay_days > 0].order_estimated_delivery_date).dt.days.describe().round(2).to_dict()
# S2: single-item single-seller orders only
s2 = base_y1[(base_y1.n_items == 1)]
# S3: Southeast customers only (largest, best-served region)
s3 = base_y1[base_y1.customer_region == "Southeast"]
for nm, data in {"S1_review_after_delivery": s1, "S2_single_item": s2, "S3_southeast": s3}.items():
    for y, lab in [("low_rating", "Y1"), ("neg_text_lr", "Y2")]:
        dat = data if y == "low_rating" else data[data.p_neg_text.notna()]
        m, _ = fit(y, dat)
        sens[f"{nm}_{lab}"] = {"n": int(m.nobs), "late_ORs": late_or(m)}
# S4: alternative threshold for LR negative text (p >= 0.7)
dd4 = d[d.p_neg_text.notna()].copy()
dd4["neg_text_lr70"] = (dd4.p_neg_text >= 0.7).astype(int)
m4, _ = fit("neg_text_lr70", dd4)
sens["S4_threshold_0.7_Y2"] = {"n": int(m4.nobs), "late_ORs": late_or(m4)}
# S5: continuous delay (late days, capped at 30) instead of bins
dd5 = d.copy()
dd5["late_days"] = dd5.delay_days.clip(lower=0, upper=30)
dd5["early_days"] = (-dd5.delay_days).clip(lower=0, upper=30)
covs5 = covs.replace("C(delay_bin, Treatment('early1_7'))", "late_days + early_days")
m5 = smf.logit(f"low_rating ~ {covs5}", data=dd5).fit(disp=0, maxiter=200, cov_type="cluster",
                                                      cov_kwds={"groups": pd.factorize(dd5.primary_seller)[0]})
sens["S5_continuous_Y1"] = {"OR_per_late_day": [float(np.exp(m5.params["late_days"])),
                                                 float(np.exp(m5.conf_int().loc["late_days", 0])),
                                                 float(np.exp(m5.conf_int().loc["late_days", 1]))],
                            "OR_per_early_day": [float(np.exp(m5.params["early_days"])),
                                                  float(np.exp(m5.conf_int().loc["early_days", 0])),
                                                  float(np.exp(m5.conf_int().loc["early_days", 1]))]}
res["sensitivity"] = sens

# ---- early-warning model: predict low rating / negative text from transaction data only (temporal hold-out)
feat_num = ["delay_days", "delivery_days", "promised_days", "log_value", "freight_ratio", "distance_km", "n_items",
            "n_sellers", "installments_max", "photos_mean", "desc_len_mean", "weight_total"]
feat_cat = ["pay", "customer_region", "cat"]
X = d[feat_num + feat_cat].copy()
for c in feat_cat:
    X[c] = X[c].astype("category")
cut = d.order_purchase_timestamp.quantile(0.75)
trm = d.order_purchase_timestamp <= cut
ew = {"cut": str(cut)}
for y in ["low_rating", "neg_text_lr"]:
    mask = d[y].notna()
    Xtr, Xte = X[mask & trm], X[mask & ~trm]
    ytr, yte = d.loc[mask & trm, y].astype(int), d.loc[mask & ~trm, y].astype(int)
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                                        l2_regularization=1.0, categorical_features="from_dtype", random_state=7)
    gb.fit(Xtr, ytr)
    p = gb.predict_proba(Xte)[:, 1]
    # logistic benchmark with the same raw features (one-hot)
    Xl = pd.get_dummies(X[feat_num + feat_cat].fillna(X[feat_num].median()), columns=feat_cat, drop_first=True).astype(float)
    lr = sm.Logit(ytr, sm.add_constant(Xl[mask & trm])).fit(disp=0, maxiter=200)
    pl = lr.predict(sm.add_constant(Xl[mask & ~trm]))
    # delay-only benchmark
    pd_only = smf.logit(f"{y} ~ C(delay_bin)", data=d[mask & trm]).fit(disp=0).predict(d[mask & ~trm])
    pi = permutation_importance(gb, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=7)
    imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False)
    ew[y] = {"n_train": int(len(ytr)), "n_test": int(len(yte)), "prev_test": float(yte.mean()),
             "AUC_GBM": float(roc_auc_score(yte, p)), "AP_GBM": float(average_precision_score(yte, p)),
             "AUC_logit": float(roc_auc_score(yte, pl)), "AUC_delay_only": float(roc_auc_score(yte, pd_only)),
             "perm_importance_top": imp.head(8).round(4).to_dict()}
res["early_warning"] = ew

# ---- figure: adjusted probabilities by delay bin
nice = ["≥15 early", "8–14 early", "1–7 early", "On date", "1–3 late", "4–7 late", "8–14 late", ">14 late"]
fig, ax = plt.subplots(figsize=(6.3, 3.0))
cols = {"Y1_low_rating_all": "#4C78A8", "Y2_neg_text_lr": "#E45756", "Y3_neg_text_leia": "#54A24B"}
labels = {"Y1_low_rating_all": "Low star rating (1–2), all reviews",
          "Y2_neg_text_lr": "Negative text, TF-IDF + LR",
          "Y3_neg_text_leia": "Negative text, LeIA lexicon"}
xs = np.arange(len(order_bins))
for j, name in enumerate(cols):
    pr = np.array([adj[name][b] for b in order_bins]) * 100
    ci = res["adjusted_probs_CI"][name]
    lo = np.array([ci[b][0] for b in order_bins]) * 100
    hi = np.array([ci[b][1] for b in order_bins]) * 100
    off = (j - 1) * 0.12
    ax.errorbar(xs + off, pr, yerr=[pr - lo, hi - pr], fmt="o-", ms=3, lw=1.1, capsize=2, color=cols[name],
                label=labels[name])
ax.axvline(3.5, color="grey", ls=":", lw=0.8)
ax.text(3.55, 2, "estimated delivery date", fontsize=7, color="grey")
ax.set_xticks(xs)
ax.set_xticklabels([n + "" for n in nice])
ax.set_xlabel("Delivery timing relative to the estimated date (days)")
ax.set_ylabel("Adjusted probability (%)")
ax.set_ylim(0, 100)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "adjusted_probability_by_delay.png", dpi=600)


def conv(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


with open(OUT / "drivers_results.json", "w") as f:
    json.dump(res, f, indent=1, default=conv)
print(json.dumps({k: v for k, v in res.items() if not k.startswith("Y")}, indent=1, default=conv)[:8000])
