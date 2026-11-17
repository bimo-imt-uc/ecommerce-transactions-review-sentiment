"""Step 5: Themes of negative review text (RQ4) with non-negative matrix factorisation (NMF).

Corpus: reviews whose text is classified negative by the out-of-fold TF-IDF + LR model.
Number of topics k chosen from 4..12 by mean NPMI coherence of the top-10 terms (document co-occurrence),
with seed stability (mean Jaccard overlap of top-10 terms across three seeds) reported for the chosen k.
Usage: python 05_topics.py <out_dir>
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

OUT = Path(sys.argv[1])
ts = pd.read_parquet(OUT / "text_scored.parquet")
od = pd.read_parquet(OUT / "orders_analytic.parquet")[["order_id", "order_status", "late", "delay_days",
                                                         "order_purchase_timestamp"]]
neg = ts[ts.text_negative == 1].merge(od, on="order_id", how="left")
neg = neg[neg.clean.str.split().map(len) >= 2].reset_index(drop=True)
print("negative docs", len(neg))

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_df=0.4, sublinear_tf=True)
X = vec.fit_transform(neg.clean)
vocab = np.array(vec.get_feature_names_out())
Bm = (X > 0).astype(np.int8).tocsc()
N = X.shape[0]
df_counts = np.asarray(Bm.sum(axis=0)).ravel()


def npmi_topic(idx):
    vals = []
    for i, j in combinations(idx, 2):
        pij = Bm[:, i].multiply(Bm[:, j]).sum() / N
        pi, pj = df_counts[i] / N, df_counts[j] / N
        if pij == 0:
            vals.append(-1.0)
        else:
            vals.append(np.log(pij / (pi * pj)) / (-np.log(pij)))
    return float(np.mean(vals))


def top_idx(H, n=10):
    return [np.argsort(-H[t])[:n] for t in range(H.shape[0])]


sel = {}
for k in range(4, 17):
    nmf = NMF(n_components=k, init="nndsvda", random_state=1, max_iter=600, l1_ratio=0.0)
    W = nmf.fit_transform(X)
    tops = top_idx(nmf.components_)
    coh = [npmi_topic(t) for t in tops]
    sel[k] = {"mean_npmi": float(np.mean(coh)), "min_npmi": float(np.min(coh)),
              "recon_err": float(nmf.reconstruction_err_)}
    print(k, round(np.mean(coh), 4), round(np.min(coh), 4))
best_k = max(sel, key=lambda kk: (round(sel[kk]["mean_npmi"], 3), -kk))

# stability across seeds for best_k (random init)
tops_by_seed = []
for seed in [11, 22, 33]:
    m = NMF(n_components=best_k, init="random", random_state=seed, max_iter=1000)
    m.fit(X)
    tops_by_seed.append([set(vocab[t]) for t in top_idx(m.components_)])
nmf = NMF(n_components=best_k, init="nndsvda", random_state=1, max_iter=600)
W = nmf.fit_transform(X)
ref = [set(vocab[t]) for t in top_idx(nmf.components_)]
jac = []
for tb in tops_by_seed:
    for r in ref:
        jac.append(max(len(r & t) / len(r | t) for t in tb))
stability = float(np.mean(jac))

neg["topic"] = W.argmax(axis=1)
neg["topic_weight"] = W.max(axis=1)
neg.loc[neg.topic_weight == 0, "topic"] = -1
neg["status_group"] = np.where(neg.order_status != "delivered", "not_delivered",
                               np.where(neg.late == 1, "delivered_late", "delivered_on_time_or_early"))
ct = pd.crosstab(neg.topic, neg.status_group)
chi2, pval, dof, _ = chi2_contingency(ct)
cramer_v = float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))

topics = []
for t in range(best_k):
    idx = np.argsort(-nmf.components_[t])[:12]
    sub = neg[neg.topic == t]
    topics.append({
        "topic": t,
        "top_terms": vocab[idx].tolist(),
        "npmi": npmi_topic(idx[:10]),
        "n_docs": int(len(sub)),
        "share": float(len(sub) / len(neg)),
        "mean_stars": float(sub.review_score.mean()),
        "share_by_status": (sub.status_group.value_counts() / neg.status_group.value_counts()).round(4).to_dict(),
        "examples": sub.sort_values("topic_weight", ascending=False).text.head(6).tolist(),
    })
res = {"n_docs": int(len(neg)), "vocab": int(len(vocab)), "selection": sel, "best_k": int(best_k),
       "stability_jaccard": stability, "chi2": float(chi2), "dof": int(dof), "p": float(pval),
       "cramers_v": cramer_v, "status_counts": neg.status_group.value_counts().to_dict(),
       "crosstab_col_pct": (ct / ct.sum(axis=0)).round(4).to_dict(), "topics": topics}
with open(OUT / "topics_results.json", "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
neg[["order_id", "topic", "topic_weight", "status_group"]].to_parquet(OUT / "neg_topics.parquet", index=False)
for t in topics:
    print(t["topic"], t["share"], round(t["npmi"], 3), t["top_terms"])
    for e in t["examples"][:3]:
        print("    -", e[:140].replace("\n", " "))
print("best_k", best_k, "stability", stability, "chi2", chi2, "p", pval, "V", cramer_v)
print(res["crosstab_col_pct"])
