"""Step 2: NLP sentiment measurement of review text and its validation against star ratings.

Instruments
  A. LeIA lexicon (VADER adapted to Brazilian Portuguese; external, no training on these data)
  B. Supervised TF-IDF + logistic regression, weakly supervised by star ratings
     (1-2 = negative, 4-5 = positive; 3-star excluded from training)
  C. Supervised TF-IDF + multinomial naive Bayes (baseline)

Evaluation: temporal hold-out (latest 30% of reviews by creation date), bootstrap CIs.
Scoring of the full corpus: 5-fold out-of-fold predictions grouped by customer, so no
review is scored by a model that saw it during training.

Usage: python 02_sentiment.py <out_dir> <leia_dir> <stopwords_zip>
"""
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr
from unidecode import unidecode

OUT = Path(sys.argv[1])
sys.path.insert(0, sys.argv[2])
from leia import SentimentIntensityAnalyzer  # noqa: E402

RNG = np.random.default_rng(20261112)
SEED = 20261112

od = pd.read_parquet(OUT / "orders_analytic.parquet")
txt = od[(od.has_text == 1) & od.review_score.notna()].copy()
txt = txt[txt.text.str.strip().str.len() > 0]
print("text reviews:", len(txt))

# ---------------- preprocessing for supervised models
with zipfile.ZipFile(sys.argv[3]) as z:
    sw = z.read("stopwords/portuguese").decode("utf-8").split()
NEG_KEEP = {"nao", "nem", "nunca", "sem", "mas", "muito", "pouco", "mais", "menos", "ainda", "ja", "so"}
STOP = {unidecode(w.lower()) for w in sw} - NEG_KEEP


def clean(s: str) -> str:
    s = unidecode(str(s).lower())
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)
    toks = [t for t in s.split() if len(t) > 1 and t not in STOP]
    return " ".join(toks)


txt["clean"] = txt.text.map(clean)
txt["n_tokens_raw"] = txt.text.str.split().map(len)

# ---------------- Instrument A: LeIA
sia = SentimentIntensityAnalyzer()
txt["leia_compound"] = txt.text.map(lambda s: sia.polarity_scores(str(s))["compound"])
txt["leia_class"] = np.select([txt.leia_compound >= 0.05, txt.leia_compound <= -0.05], ["pos", "neg"], "neu")

# ---------------- labels
txt["star_class"] = np.select([txt.review_score <= 2, txt.review_score >= 4], ["neg", "pos"], "neu")
txt["y_neg"] = (txt.review_score <= 2).astype(int)
bin_mask = txt.star_class != "neu"

# temporal split on review creation date
cut = txt.review_creation_date.quantile(0.70)
txt["split"] = np.where(txt.review_creation_date <= cut, "train", "test")
tr = txt[bin_mask & (txt.split == "train")]
te = txt[bin_mask & (txt.split == "test")]
print("train", len(tr), "test", len(te), "cut", cut)


def boot_ci(y, s, fn, B=1000):
    y = np.asarray(y)
    s = np.asarray(s)
    n = len(y)
    vals = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        vals.append(fn(y[idx], s[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def metrics(y, score, pred):
    auc = roc_auc_score(y, score)
    f1m = f1_score(y, pred, average="macro")
    return {
        "AUC": auc,
        "AUC_CI": boot_ci(y, score, roc_auc_score),
        "macroF1": f1m,
        "macroF1_CI": boot_ci(y, pred, lambda a, b: f1_score(a, b, average="macro")),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "F1_negative": f1_score(y, pred, pos_label=1),
        "confusion": confusion_matrix(y, pred).tolist(),
        "n": int(len(y)),
        "prevalence_neg": float(np.mean(y)),
    }


res = {"split_cut": str(cut), "n_text": int(len(txt)), "n_train_binary": int(len(tr)), "n_test_binary": int(len(te)),
       "n_neutral_3star_text": int((~bin_mask).sum())}

# A. LeIA on test
res["LeIA"] = metrics(te.y_neg, -te.leia_compound, (te.leia_compound <= -0.05).astype(int))
# LeIA alternative decision rule: negative if compound < 0.05 (i.e., not positive)
res["LeIA_rule_notpositive"] = metrics(te.y_neg, -te.leia_compound, (te.leia_compound < 0.05).astype(int))

# B. TF-IDF + LR with grouped CV tuning on train
groups_tr = tr.customer_unique_id.values
lr_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
])
gs = GridSearchCV(lr_pipe, {"clf__C": [0.25, 1.0, 4.0], "tfidf__ngram_range": [(1, 1), (1, 2)]},
                  scoring="roc_auc", cv=GroupKFold(n_splits=5), n_jobs=2)
gs.fit(tr.clean, tr.y_neg, groups=groups_tr)
res["LR_best_params"] = {k: str(v) for k, v in gs.best_params_.items()}
res["LR_cv_auc"] = float(gs.best_score_)
p_te = gs.predict_proba(te.clean)[:, 1]
res["TFIDF_LR"] = metrics(te.y_neg, p_te, (p_te >= 0.5).astype(int))

# C. TF-IDF + NB
nb_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True)),
    ("clf", MultinomialNB()),
])
gsn = GridSearchCV(nb_pipe, {"clf__alpha": [0.1, 0.5, 1.0]}, scoring="roc_auc", cv=GroupKFold(n_splits=5), n_jobs=2)
gsn.fit(tr.clean, tr.y_neg, groups=groups_tr)
p_nb = gsn.predict_proba(te.clean)[:, 1]
# NB is not class-balanced; use prevalence-matched threshold from train
thr_nb = float(np.quantile(gsn.predict_proba(tr.clean)[:, 1], 1 - tr.y_neg.mean()))
res["TFIDF_NB"] = metrics(te.y_neg, p_nb, (p_nb >= thr_nb).astype(int))
res["TFIDF_NB"]["threshold"] = thr_nb
res["NB_best_params"] = {k: str(v) for k, v in gsn.best_params_.items()}

# subgroup robustness of LR on test: short vs long texts
te = te.assign(p_lr=p_te, p_nb=p_nb)
te[["review_id", "review_score", "y_neg", "leia_compound", "p_lr", "p_nb"]].to_parquet(OUT / "sentiment_test_predictions.parquet", index=False)
short = te.n_tokens_raw <= te.n_tokens_raw.median()
res["TFIDF_LR_short"] = {"AUC": float(roc_auc_score(te[short].y_neg, te[short].p_lr)), "n": int(short.sum())}
res["TFIDF_LR_long"] = {"AUC": float(roc_auc_score(te[~short].y_neg, te[~short].p_lr)), "n": int((~short).sum())}
res["LeIA_short"] = {"AUC": float(roc_auc_score(te[short].y_neg, -te[short].leia_compound))}
res["LeIA_long"] = {"AUC": float(roc_auc_score(te[~short].y_neg, -te[~short].leia_compound))}

# ---------------- out-of-fold scoring of the full text corpus (incl. 3-star) with the selected LR spec
best_C = gs.best_params_["clf__C"]
best_ng = gs.best_params_["tfidf__ngram_range"]
txt["p_neg_text"] = np.nan
gkf = GroupKFold(n_splits=5)
all_idx = np.arange(len(txt))
grp = txt.customer_unique_id.values
for k, (tri, tei) in enumerate(gkf.split(all_idx, groups=grp)):
    fit_rows = txt.iloc[tri]
    fit_rows = fit_rows[fit_rows.star_class != "neu"]
    m = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=best_ng, min_df=3, max_df=0.9, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", C=best_C)),
    ])
    m.fit(fit_rows.clean, fit_rows.y_neg)
    txt.iloc[tei, txt.columns.get_loc("p_neg_text")] = m.predict_proba(txt.iloc[tei].clean)[:, 1]
txt["text_negative"] = (txt.p_neg_text >= 0.5).astype(int)

oof_bin = txt[bin_mask]
res["OOF_LR_all_binary"] = {"AUC": float(roc_auc_score(oof_bin.y_neg, oof_bin.p_neg_text)),
                            "macroF1": float(f1_score(oof_bin.y_neg, oof_bin.text_negative, average="macro"))}

# ---------------- agreement with star ratings
res["spearman_leia_stars"] = list(map(float, spearmanr(txt.leia_compound, txt.review_score)))
res["spearman_plr_stars"] = list(map(float, spearmanr(-txt.p_neg_text, txt.review_score)))
order = ["neg", "neu", "pos"]
res["kappa_leia3_vs_stars3_linear"] = float(cohen_kappa_score(txt.star_class, txt.leia_class, labels=order, weights="linear"))
res["crosstab_leia_vs_stars"] = pd.crosstab(txt.review_score, txt.leia_class).to_dict()
# share of text classified negative by star level
res["share_text_negative_by_star_LR"] = txt.groupby("review_score").text_negative.mean().round(4).to_dict()
res["share_text_negative_by_star_LeIA"] = txt.groupby("review_score").apply(lambda g: (g.leia_compound <= -0.05).mean()).round(4).to_dict()
res["mean_p_neg_by_star"] = txt.groupby("review_score").p_neg_text.mean().round(4).to_dict()

# ---------------- comment-writing selection: share of reviews with text by star
od_r = od[od.review_score.notna()]
res["share_with_text_by_star"] = od_r.groupby("review_score").has_text.mean().round(4).to_dict()
res["share_with_text_overall"] = float(od_r.has_text.mean())

# ---------------- discordance: high-star (4-5) reviews whose text is classified negative
disc = txt[(txt.review_score >= 4)]
res["discordant_highstar_text_negative_n"] = int(disc.text_negative.sum())
res["discordant_highstar_text_negative_share"] = float(disc.text_negative.mean())

# top weighted terms of final model fitted on all binary data (for interpretability table)
mfull = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=best_ng, min_df=3, max_df=0.9, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", C=best_C)),
])
mfull.fit(txt[bin_mask].clean, txt[bin_mask].y_neg)
vocab = np.array(mfull.named_steps["tfidf"].get_feature_names_out())
coef = mfull.named_steps["clf"].coef_[0]
res["vocab_size"] = int(len(vocab))
# report coefficients (log-odds toward the negative class) only for terms in >= 30 documents, to avoid idiosyncratic n-grams
dfreq = np.asarray((mfull.named_steps["tfidf"].transform(txt[bin_mask].clean) > 0).sum(axis=0)).ravel()
ok = dfreq >= 30
cv, vv, dv = coef[ok], vocab[ok], dfreq[ok]
o_neg = np.argsort(-cv)[:20]
o_pos = np.argsort(cv)[:20]
res["top_negative_terms"] = [(vv[i], round(float(cv[i]), 3), int(dv[i])) for i in o_neg]
res["top_positive_terms"] = [(vv[i], round(float(cv[i]), 3), int(dv[i])) for i in o_pos]

txt[["order_id", "review_id", "customer_unique_id", "review_score", "text", "clean", "n_tokens_raw",
     "leia_compound", "leia_class", "p_neg_text", "text_negative", "split"]].to_parquet(OUT / "text_scored.parquet", index=False)


def conv(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


with open(OUT / "sentiment_results.json", "w") as f:
    json.dump(res, f, indent=2, default=conv)
print(json.dumps(res, indent=1, default=conv))
