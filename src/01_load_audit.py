"""Step 1: Load the nine relational tables with DuckDB, run a data-quality audit,
and build the order-level analytical table.

Usage: python 01_load_audit.py <data_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DATA = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
tables = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "cat_translation": "product_category_name_translation.csv",
}
for name, fn in tables.items():
    # all_varchar for zip prefixes so leading zeros are kept; types cast later
    con.execute(
        f"CREATE TABLE {name} AS SELECT * FROM read_csv('{DATA / fn}', header=true, "
        f"all_varchar=true, quote='\"', escape='\"', strict_mode=false)"
    )

audit = {}
inv = []
for name in tables:
    n, ncol = con.execute(f"SELECT COUNT(*), (SELECT COUNT(*) FROM information_schema.columns WHERE table_name='{name}') FROM {name}").fetchone()
    inv.append({"table": name, "rows": int(n), "columns": int(ncol)})
audit["inventory"] = inv
audit["total_rows"] = int(sum(r["rows"] for r in inv))

# ---- convert to pandas with types
def df(q):
    return con.execute(q).df()

orders = df("SELECT * FROM orders")
for c in ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
          "order_delivered_customer_date", "order_estimated_delivery_date"]:
    orders[c] = pd.to_datetime(orders[c], errors="coerce")
items = df("SELECT * FROM order_items")
items["price"] = items["price"].astype(float)
items["freight_value"] = items["freight_value"].astype(float)
items["order_item_id"] = items["order_item_id"].astype(int)
pay = df("SELECT * FROM payments")
pay["payment_value"] = pay["payment_value"].astype(float)
pay["payment_installments"] = pay["payment_installments"].astype(int)
pay["payment_sequential"] = pay["payment_sequential"].astype(int)
rev = df("SELECT * FROM reviews")
rev["review_score"] = pd.to_numeric(rev["review_score"], errors="coerce")
rev["review_creation_date"] = pd.to_datetime(rev["review_creation_date"], errors="coerce")
rev["review_answer_timestamp"] = pd.to_datetime(rev["review_answer_timestamp"], errors="coerce")
cust = df("SELECT * FROM customers")
sell = df("SELECT * FROM sellers")
# zip prefixes lost leading zeros in some files; normalise to 5 digits
cust["customer_zip_code_prefix"] = cust.customer_zip_code_prefix.str.strip().str.zfill(5)
sell["seller_zip_code_prefix"] = sell.seller_zip_code_prefix.str.strip().str.zfill(5)
prod = df("SELECT * FROM products")
for c in ["product_name_lenght", "product_description_lenght", "product_photos_qty", "product_weight_g",
          "product_length_cm", "product_height_cm", "product_width_cm"]:
    prod[c] = pd.to_numeric(prod[c], errors="coerce")
cat = df("SELECT * FROM cat_translation")
cat.columns = [c.replace("﻿", "") for c in cat.columns]
geo = df("SELECT * FROM geolocation")
geo["geolocation_lat"] = geo["geolocation_lat"].astype(float)
geo["geolocation_lng"] = geo["geolocation_lng"].astype(float)
geo["geolocation_zip_code_prefix"] = geo.geolocation_zip_code_prefix.str.strip().str.zfill(5)

# ---- audit checks
a = {}
a["orders_unique_ids"] = int(orders.order_id.nunique())
a["order_status_counts"] = orders.order_status.value_counts().to_dict()
a["purchase_min"] = str(orders.order_purchase_timestamp.min())
a["purchase_max"] = str(orders.order_purchase_timestamp.max())
a["orders_by_month"] = orders.order_purchase_timestamp.dt.to_period("M").astype(str).value_counts().sort_index().to_dict()
dl = orders[orders.order_status == "delivered"]
a["delivered_missing_customer_date"] = int(dl.order_delivered_customer_date.isna().sum())
a["delivered_before_purchase"] = int((dl.order_delivered_customer_date < dl.order_purchase_timestamp).sum())
a["carrier_before_approval"] = int((orders.order_delivered_carrier_date < orders.order_approved_at).sum())
a["customers_rows"] = int(len(cust))
a["customer_id_unique"] = int(cust.customer_id.nunique())
a["customer_unique_id_unique"] = int(cust.customer_unique_id.nunique())
a["items_orders"] = int(items.order_id.nunique())
a["orders_without_items"] = int(len(set(orders.order_id) - set(items.order_id)))
a["payments_orders"] = int(pay.order_id.nunique())
a["orders_without_payment"] = int(len(set(orders.order_id) - set(pay.order_id)))
a["reviews_rows"] = int(len(rev))
a["reviews_unique_review_id"] = int(rev.review_id.nunique())
a["reviews_unique_order_id"] = int(rev.order_id.nunique())
rc = rev.groupby("order_id").size()
a["orders_with_multiple_reviews"] = int((rc > 1).sum())
rid = rev.groupby("review_id").order_id.nunique()
a["review_ids_linked_to_multiple_orders"] = int((rid > 1).sum())
rev["msg"] = rev.review_comment_message.fillna("").str.strip()
rev["ttl"] = rev.review_comment_title.fillna("").str.strip()
a["reviews_with_message"] = int((rev.msg != "").sum())
a["reviews_with_title"] = int((rev.ttl != "").sum())
a["review_score_dist"] = rev.review_score.value_counts().sort_index().to_dict()
a["products_missing_category"] = int(prod.product_category_name.isna().sum())
a["categories_n"] = int(prod.product_category_name.nunique())
a["categories_without_translation"] = sorted(set(prod.product_category_name.dropna()) - set(cat.product_category_name))
a["sellers"] = int(len(sell))
a["geolocation_rows"] = int(len(geo))
a["geolocation_zip_prefixes"] = int(geo.geolocation_zip_code_prefix.nunique())
audit["checks"] = a

# ---- geolocation: median coordinates per zip prefix, restricted to Brazil bounding box
gbox = geo[(geo.geolocation_lat.between(-34.0, 5.5)) & (geo.geolocation_lng.between(-74.5, -34.5))]
audit["checks"]["geolocation_outside_brazil_box"] = int(len(geo) - len(gbox))
zipc = gbox.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]].median()

# ---- order-level table
items = items.merge(prod[["product_id", "product_category_name", "product_photos_qty", "product_description_lenght",
                          "product_weight_g"]], on="product_id", how="left")
items = items.merge(cat, on="product_category_name", how="left")
items["category"] = items.product_category_name_english.fillna(items.product_category_name).fillna("unknown")
items = items.merge(sell[["seller_id", "seller_zip_code_prefix", "seller_state"]], on="seller_id", how="left")

agg = items.groupby("order_id").agg(
    n_items=("order_item_id", "size"),
    n_products=("product_id", "nunique"),
    n_sellers=("seller_id", "nunique"),
    item_value=("price", "sum"),
    freight=("freight_value", "sum"),
    photos_mean=("product_photos_qty", "mean"),
    desc_len_mean=("product_description_lenght", "mean"),
    weight_total=("product_weight_g", "sum"),
).reset_index()
# primary item = highest price item (ties: lowest order_item_id)
prim = items.sort_values(["order_id", "price", "order_item_id"], ascending=[True, False, True]).drop_duplicates("order_id")
prim = prim[["order_id", "category", "seller_id", "seller_zip_code_prefix", "seller_state"]].rename(
    columns={"category": "primary_category", "seller_id": "primary_seller"})
agg = agg.merge(prim, on="order_id", how="left")

payo = pay.sort_values(["order_id", "payment_value"], ascending=[True, False])
pagg = pay.groupby("order_id").agg(payment_total=("payment_value", "sum"),
                                   installments_max=("payment_installments", "max"),
                                   n_payment_methods=("payment_type", "nunique")).reset_index()
ptype = payo.drop_duplicates("order_id")[["order_id", "payment_type"]].rename(columns={"payment_type": "payment_type_main"})
pagg = pagg.merge(ptype, on="order_id", how="left")

od = orders.merge(cust, on="customer_id", how="left").merge(agg, on="order_id", how="left").merge(pagg, on="order_id", how="left")

# distance customer-seller (haversine, km) using zip-prefix centroids
od = od.merge(zipc.rename(columns={"geolocation_lat": "c_lat", "geolocation_lng": "c_lng"}),
              left_on="customer_zip_code_prefix", right_index=True, how="left")
od = od.merge(zipc.rename(columns={"geolocation_lat": "s_lat", "geolocation_lng": "s_lng"}),
              left_on="seller_zip_code_prefix", right_index=True, how="left")
R = 6371.0
lat1, lon1, lat2, lon2 = map(np.radians, [od.c_lat, od.c_lng, od.s_lat, od.s_lng])
h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
od["distance_km"] = 2 * R * np.arcsin(np.sqrt(h))

od["purchase_date"] = od.order_purchase_timestamp.dt.normalize()
od["delivery_days"] = (od.order_delivered_customer_date - od.order_purchase_timestamp).dt.total_seconds() / 86400
od["promised_days"] = (od.order_estimated_delivery_date - od.purchase_date).dt.days
# lateness at calendar-day resolution because the estimated date carries no time-of-day
od["delay_days"] = (od.order_delivered_customer_date.dt.normalize() - od.order_estimated_delivery_date.dt.normalize()).dt.days
od["late"] = (od.delay_days > 0).astype(float)
od.loc[od.delay_days.isna(), "late"] = np.nan
od["freight_ratio"] = od.freight / (od.item_value + od.freight)
audit["checks"]["orders_distance_missing"] = int(od.distance_km.isna().sum())
od["cross_state"] = (od.customer_state != od.seller_state).astype(float)
region = {
    "AC": "North", "AP": "North", "AM": "North", "PA": "North", "RO": "North", "RR": "North", "TO": "North",
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast", "MA": "Northeast", "PB": "Northeast", "PE": "Northeast",
    "PI": "Northeast", "RN": "Northeast", "SE": "Northeast",
    "DF": "Center-West", "GO": "Center-West", "MT": "Center-West", "MS": "Center-West",
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast", "SP": "Southeast",
    "PR": "South", "RS": "South", "SC": "South",
}
od["customer_region"] = od.customer_state.map(region)
od["ym"] = od.order_purchase_timestamp.dt.to_period("M").astype(str)

# ---- review linkage: keep orders with exactly one review, and reviews linked to exactly one order
single_order_reviews = rid[rid == 1].index
r1 = rev[rev.review_id.isin(single_order_reviews)]
rc1 = r1.groupby("order_id").size()
r1 = r1[r1.order_id.isin(rc1[rc1 == 1].index)].copy()
r1["text"] = (r1.ttl + ". " + r1.msg).str.strip(". ").where(r1.ttl != "", r1.msg)
r1["has_text"] = (r1.msg != "").astype(int)
audit["checks"]["reviews_after_linkage_rules"] = int(len(r1))
od = od.merge(r1[["order_id", "review_id", "review_score", "text", "has_text", "review_creation_date",
                  "review_answer_timestamp"]], on="order_id", how="left")

audit["checks"]["orders_distance_missing"] = int(od.distance_km.isna().sum())
od.to_parquet(OUT / "orders_analytic.parquet", index=False)
items.to_parquet(OUT / "items_enriched.parquet", index=False)
pay.to_parquet(OUT / "payments.parquet", index=False)

with open(OUT / "audit.json", "w") as f:
    json.dump(audit, f, indent=2, default=str)
print(json.dumps(audit, indent=1, default=str)[:6000])
print("order table:", od.shape)
