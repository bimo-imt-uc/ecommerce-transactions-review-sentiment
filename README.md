# Linking e-commerce transactions and review sentiment

Experiment code for the article *Big Data Analytics Linking E-Commerce Transactions and Review Sentiment Using Natural Language Processing* (Setyorini, Gumelar, Kholid, Junaedi, Winardi, and Abadi). The code rebuilds every reported result from public data: the data audit, consumption patterns, validation of the sentiment measures, the models of negative evaluation, the complaint topics, and the repurchase cohort.

## Data

`run_all.sh` downloads all inputs. Nothing has to be added by hand.

| Input | Source | Licence |
|---|---|---|
| Brazilian E-Commerce Public Dataset by Olist, 9 CSV files, 1,550,922 rows | https://github.com/olist/work-at-olist-data (same tables as https://doi.org/10.34740/KAGGLE/DSV/195341) | MIT (GitHub repository); CC BY-NC-SA 4.0 (Kaggle release) |
| LeIA sentiment lexicon for Brazilian Portuguese | https://github.com/rafjaa/LeIA | MIT |
| NLTK Portuguese stopword list | https://github.com/nltk/nltk_data | Apache 2.0 |

`olist_sha256.txt` contains SHA-256 checksums of the nine CSV files, and `run_all.sh` stops if a file does not match.

## Running

Locally, with Python 3.11 or later:

```bash
pip install -r requirements.txt
bash run_all.sh              # outputs go to ./work/outputs
```

In Google Colab, open `notebooks/reproduce_colab.ipynb`, set `REPO_URL` in the first cell, and run all cells.

A full run takes about two minutes on a two-core machine. Seeds are fixed, and two independent runs produced identical result files.

## Pipeline

| Script | Analysis | Main outputs |
|---|---|---|
| `src/01_load_audit.py` | Loads the nine tables with DuckDB, audits keys, timestamps and review linkage, builds the order-level table | `audit.json`, `orders_analytic.parquet` |
| `src/02_sentiment.py` | Sentiment measurement: LeIA lexicon, TF-IDF with logistic regression and naive Bayes, temporal hold-out, out-of-fold scoring | `sentiment_results.json`, `text_scored.parquet` |
| `src/03_patterns.py` | Consumption patterns: time, category, region, payment, concentration, purchase frequency | `patterns_results.json`, `figures/consumption_patterns.png` |
| `src/04_drivers.py` | Seller-clustered logistic regressions of negative evaluation, adjusted probabilities, sensitivity analyses, gradient-boosting early warning | `drivers_results.json`, `figures/adjusted_probability_by_delay.png` |
| `src/05_topics.py` | Non-negative matrix factorization of negative texts, topic number chosen by NPMI coherence | `topics_results.json` |
| `src/06_repurchase.py` | First-order cohort and repurchase within 180 days (90 and 365 days as checks) | `repurchase_results.json` |
| `src/07_themes_and_validation.py` | Complaint themes by delivery status, sources of negative evaluations, validation figure | `supplementary_numbers.json`, `figures/sentiment_validation.png` |

`results/` holds the JSON files and figures from the run reported in the article, so the numbers can be checked without rerunning.

## Notes

The supervised sentiment classifiers use star ratings as weak labels (one or two stars negative, four or five positive, three excluded); no manual annotation was done. Text-based outcomes in step 04 use out-of-fold predictions from five folds grouped by customer. Seeds: `20261112` for resampling, and `1`, `7`, `11`, `22` and `33` for model initialization.

## Citation

See `CITATION.cff`. The article reference will be added after publication.
