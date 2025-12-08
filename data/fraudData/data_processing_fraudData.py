#!/usr/bin/env python
# coding: utf-8

# ## Indexing

# ### Import

# In[65]:


import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

import matplotlib.pyplot as plt

from sqlalchemy import create_engine, text


# ### Data Pre-Processing

# In[66]:


train_path = "fraudTrain.csv"
test_path  = "fraudTest.csv"

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)


# In[67]:


train["split"] = "train"
test["split"]  = "test"

df = pd.concat([train, test], ignore_index=True)


# In[68]:


print(df)


# #### Data Cleaning

# In[69]:


if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])


# #### Data Normalization & Data Parsing

# In[70]:


# Parse timestamps and dates
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["dob"] = pd.to_datetime(df["dob"])

# Treat cc_num and zip as identifiers: convert to string
df["cc_num"] = df["cc_num"].astype(str)
df["zip"] = df["zip"].astype(str)


# #### Feature Creation: Time-Based Features

# In[71]:


# Transaction date and related features
df["trans_date"] = df["trans_date_trans_time"].dt.date
df["year"]       = df["trans_date_trans_time"].dt.year
df["month"]      = df["trans_date_trans_time"].dt.month
df["year_month"] = df["trans_date_trans_time"].dt.to_period("M").astype(str)
df["day_of_week"] = df["trans_date_trans_time"].dt.day_name()
df["hour"]        = df["trans_date_trans_time"].dt.hour
df["is_weekend"]  = df["day_of_week"].isin(["Saturday", "Sunday"])


# #### Feature Creation: Age

# In[72]:


age_days = (df["trans_date_trans_time"] - df["dob"]).dt.days
df["age_at_txn_years"] = age_days / 365.25


# #### Feature Creation: Haversine Distance

# In[73]:


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


# In[74]:


df["cust_merch_distance_km"] = haversine_km(df["lat"], df["long"],
                                            df["merch_lat"], df["merch_long"])


# #### Data Checking

# In[75]:


# Check for any missing values
df.isna().sum()


# ### Database: PostgreSQL

# In[76]:


import os
import subprocess

if subprocess.run(["docker", "start", "postgresql"]).returncode != 0:
    subprocess.run([
        "docker", "run",
        "--name", "postgresql",
        "-e", "POSTGRES_USER=user",
        "-e", "POSTGRES_PASSWORD=password",
        "-e", "POSTGRES_DB=database",
        "-p", "5432:5432",
        "-d", "postgres:16"
    ])


# In[77]:


DB_USER = "user"
DB_PASS = "password"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "database"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# In[78]:


engine = create_engine(DATABASE_URL, echo=False, future=True)


# In[79]:


with engine.connect() as conn:
    result = conn.execute(text("SELECT version();"))
    print(result.scalar())


# #### Star Schema Creation

# ##### Star Schema Creation: Customer Dimension

# In[80]:


# factorize returns consistent integer codes for each unique cc_num
df["customer_id"], customer_unique = pd.factorize(df["cc_num"])
df["customer_id"] = df["customer_id"] + 1  # make it 1-based instead of 0-based


# In[81]:


customer_cols = [
    "customer_id",
    "cc_num", "first", "last", "gender",
    "street", "city", "state", "zip",
    "lat", "long", "city_pop", "job", "dob"
]


# In[82]:


dim_customer = (
    df[customer_cols]
    .drop_duplicates("customer_id")
    .sort_values("customer_id")
    .reset_index(drop=True)
)


# ##### Star Schema Creation: Merchant Dimension

# In[83]:


df["merchant_id"], merchant_unique = pd.factorize(df["merchant"])
df["merchant_id"] = df["merchant_id"] + 1


# In[84]:


dim_merchant = (
    df[["merchant_id", "merchant", "merch_lat", "merch_long"]]
    .drop_duplicates("merchant_id")
    .sort_values("merchant_id")
    .reset_index(drop=True)
)


# ##### Star Schema Creation: Category Dimension

# In[85]:


df["category_id"], category_unique = pd.factorize(df["category"])
df["category_id"] = df["category_id"] + 1


# In[86]:


dim_category = (
    df[["category_id", "category"]]
    .drop_duplicates("category_id")
    .sort_values("category_id")
    .reset_index(drop=True)
)


# ##### Star Schema Creation: Date Dimension

# In[87]:


# This uses the actual transaction date (not datetime) as the key
df["date_id"], date_unique = pd.factorize(df["trans_date"])
df["date_id"] = df["date_id"] + 1


# In[88]:


dim_date = (
    df[[
        "date_id",
        "trans_date",
        "year",
        "month",
        "day_of_week",
        "is_weekend",
        "year_month"
    ]]
    .copy()
)

# Extract day from trans_date
dim_date["day"] = pd.to_datetime(dim_date["trans_date"]).dt.day

dim_date = (
    dim_date
    .drop_duplicates("date_id")
    .sort_values("date_id")
    .reset_index(drop=True)
)


# ##### Fact Table Creation

# In[89]:


fact_transactions = df[[
    "trans_num",
    "customer_id",
    "merchant_id",
    "category_id",
    "date_id",
    "trans_date_trans_time",
    "unix_time",
    "amt",
    "is_fraud",
    "year",
    "month",
    "hour",
    "is_weekend",
    "cust_merch_distance_km",
    "split"
]].copy()

# Create surrogate transaction_id
fact_transactions.insert(0, "transaction_id", range(1, len(fact_transactions) + 1))


# #### Star Schema Loading

# In[ ]:


schema_ddl = """
DROP TABLE IF EXISTS fact_transactions CASCADE;
DROP TABLE IF EXISTS dim_date CASCADE;
DROP TABLE IF EXISTS dim_category CASCADE;
DROP TABLE IF EXISTS dim_merchant CASCADE;
DROP TABLE IF EXISTS dim_customer CASCADE;

CREATE TABLE dim_customer (
    customer_id      BIGINT PRIMARY KEY,
    cc_num           TEXT UNIQUE,
    first            TEXT,
    last             TEXT,
    gender           VARCHAR(1),
    street           TEXT,
    city             TEXT,
    state            TEXT,
    zip              TEXT,
    lat              DOUBLE PRECISION,
    long             DOUBLE PRECISION,
    city_pop         BIGINT,
    job              TEXT,
    dob              DATE
);

CREATE TABLE dim_merchant (
    merchant_id      BIGINT PRIMARY KEY,
    merchant_name    TEXT,
    merch_lat        DOUBLE PRECISION,
    merch_long       DOUBLE PRECISION
);

CREATE TABLE dim_category (
    category_id      BIGINT PRIMARY KEY,
    category_name    TEXT UNIQUE
);

CREATE TABLE dim_date (
    date_id       BIGINT PRIMARY KEY,
    trans_date    DATE UNIQUE,
    year          INT,
    month         INT,
    day           INT,
    day_of_week   TEXT,
    is_weekend    BOOLEAN,
    year_month    TEXT
);

CREATE TABLE fact_transactions (
    transaction_id          BIGINT PRIMARY KEY,
    trans_num               TEXT UNIQUE,
    customer_id             BIGINT REFERENCES dim_customer(customer_id),
    merchant_id             BIGINT REFERENCES dim_merchant(merchant_id),
    category_id             BIGINT REFERENCES dim_category(category_id),
    date_id                 BIGINT REFERENCES dim_date(date_id),
    trans_ts                TIMESTAMP,
    unix_time               BIGINT,
    amt                     DOUBLE PRECISION,
    is_fraud                SMALLINT,
    year                    INT,
    month                   INT,
    hour                    INT,
    is_weekend              BOOLEAN,
    cust_merch_distance_km  DOUBLE PRECISION,
    split                   TEXT
);
"""

with engine.begin() as conn:
    conn.execute(text(schema_ddl))


# ##### Star Schema Loading: Customer Dimension, Merchant Dimension, Category Dimension, Date Dimension

# In[91]:


dim_customer_for_db = dim_customer.rename(columns={
    "cc_num": "cc_num",
    "first": "first",
    "last": "last",
    "gender": "gender",
    "street": "street",
    "city": "city",
    "state": "state",
    "zip": "zip",
    "lat": "lat",
    "long": "long",
    "city_pop": "city_pop",
    "job": "job",
    "dob": "dob"
})

dim_merchant_for_db = dim_merchant.rename(columns={
    "merchant": "merchant_name"
})

dim_category_for_db = dim_category.rename(columns={
    "category": "category_name"
})

dim_date_for_db = dim_date.rename(columns={
    "trans_date": "trans_date"
})


# In[ ]:


# Use chunksize to avoid memory issues, though these dims are small.
dim_customer_for_db.to_sql("dim_customer", con=engine, if_exists="append", index=False, method="multi")
dim_merchant_for_db.to_sql("dim_merchant", con=engine, if_exists="append", index=False, method="multi")
dim_category_for_db.to_sql("dim_category", con=engine, if_exists="append", index=False, method="multi")
dim_date_for_db.to_sql("dim_date", con=engine, if_exists="append", index=False, method="multi")


# ##### Fact Table Loading

# In[93]:


fact_for_db = fact_transactions.rename(columns={
    "trans_date_trans_time": "trans_ts"
})

fact_for_db.to_sql(
    "fact_transactions",
    con=engine,
    if_exists="append",
    index=False,
    method="multi",
    chunksize=10000
)


# #### Indexing and Peformance Tuning

# In[ ]:


index_sql = """
CREATE INDEX IF NOT EXISTS idx_fact_date_id ON fact_transactions(date_id);
CREATE INDEX IF NOT EXISTS idx_fact_merchant_id ON fact_transactions(merchant_id);
CREATE INDEX IF NOT EXISTS idx_fact_category_id ON fact_transactions(category_id);
CREATE INDEX IF NOT EXISTS idx_fact_is_fraud ON fact_transactions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_fact_date_fraud ON fact_transactions(date_id, is_fraud);
CREATE INDEX IF NOT EXISTS idx_fact_merchant_fraud ON fact_transactions(merchant_id, is_fraud);
CREATE INDEX IF NOT EXISTS idx_fact_category_fraud ON fact_transactions(category_id, is_fraud);
"""

with engine.begin() as conn:
    conn.execute(text(index_sql))


# #### Metric Definitions and Aggregate Tables

# ##### Materialized View: Aggregate Daily Fraud

# In[95]:


agg_daily_sql = """
DROP MATERIALIZED VIEW IF EXISTS agg_daily_fraud;

CREATE MATERIALIZED VIEW agg_daily_fraud AS
SELECT
    d.trans_date,
    d.year,
    d.month,
    d.day,
    d.day_of_week,
    d.is_weekend,
    d.year_month,
    COUNT(*) AS total_tx,
    SUM(CASE WHEN f.is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_tx,
    (SUM(CASE WHEN f.is_fraud = 1 THEN 1 ELSE 0 END)::float
        / NULLIF(COUNT(*), 0)) AS fraud_rate,
    SUM(f.amt) AS total_amount,
    SUM(CASE WHEN f.is_fraud = 1 THEN f.amt ELSE 0 END) AS fraud_amount,
    (SUM(CASE WHEN f.is_fraud = 1 THEN f.amt ELSE 0 END)
        / NULLIF(SUM(f.amt), 0)) AS fraud_share_by_value
FROM fact_transactions f
JOIN dim_date d ON f.date_id = d.date_id
GROUP BY d.trans_date, d.year, d.month, d.day, d.day_of_week, d.is_weekend, d.year_month
ORDER BY d.trans_date;
"""

with engine.begin() as conn:
    conn.execute(text(agg_daily_sql))


# In[96]:


idx_daily_sql = """
CREATE INDEX IF NOT EXISTS idx_agg_daily_date ON agg_daily_fraud(trans_date);
CREATE INDEX IF NOT EXISTS idx_agg_daily_year_month ON agg_daily_fraud(year_month);
"""

with engine.begin() as conn:
    conn.execute(text(idx_daily_sql))


# ##### Materialized View: Aggregate Monthly Fraud

# In[97]:


agg_monthly_sql = """
DROP MATERIALIZED VIEW IF EXISTS agg_monthly_fraud;

CREATE MATERIALIZED VIEW agg_monthly_fraud AS
SELECT
    d.year,
    d.month,
    d.year_month,
    COUNT(*) AS total_tx,
    SUM(CASE WHEN f.is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_tx,
    (SUM(CASE WHEN f.is_fraud = 1 THEN 1 ELSE 0 END)::float
        / NULLIF(COUNT(*), 0)) AS fraud_rate,
    SUM(f.amt) AS total_amount,
    SUM(CASE WHEN f.is_fraud = 1 THEN f.amt ELSE 0 END) AS fraud_amount,
    (SUM(CASE WHEN f.is_fraud = 1 THEN f.amt ELSE 0 END)
        / NULLIF(SUM(f.amt), 0)) AS fraud_share_by_value
FROM fact_transactions f
JOIN dim_date d ON f.date_id = d.date_id
GROUP BY d.year, d.month, d.year_month
ORDER BY d.year, d.month;
"""

with engine.begin() as conn:
    conn.execute(text(agg_monthly_sql))


# In[98]:


idx_monthly_sql = """
CREATE INDEX IF NOT EXISTS idx_agg_monthly_year_month ON agg_monthly_fraud(year_month);
"""

with engine.begin() as conn:
    conn.execute(text(idx_monthly_sql))


# ##### Materialized View: Aggregate Merchant Fraud

# In[99]:


agg_merchant_sql = """
DROP MATERIALIZED VIEW IF EXISTS agg_merchant_fraud;

CREATE MATERIALIZED VIEW agg_merchant_fraud AS
SELECT
    m.merchant_id,
    m.merchant_name,
    COUNT(*) AS total_tx,
    SUM(CASE WHEN f.is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_tx,
    (SUM(CASE WHEN f.is_fraud = 1 THEN 1 ELSE 0 END)::float
        / NULLIF(COUNT(*), 0)) AS fraud_rate,
    SUM(f.amt) AS total_amount,
    SUM(CASE WHEN f.is_fraud = 1 THEN f.amt ELSE 0 END) AS fraud_amount,
    (SUM(CASE WHEN f.is_fraud = 1 THEN f.amt ELSE 0 END)
        / NULLIF(SUM(f.amt), 0)) AS fraud_share_by_value
FROM fact_transactions f
JOIN dim_merchant m ON f.merchant_id = m.merchant_id
GROUP BY m.merchant_id, m.merchant_name
ORDER BY fraud_rate DESC;
"""

with engine.begin() as conn:
    conn.execute(text(agg_merchant_sql))


# In[100]:


idx_merchant_sql = """
CREATE INDEX IF NOT EXISTS idx_agg_merchant_fraud_rate
    ON agg_merchant_fraud(fraud_rate DESC);

CREATE INDEX IF NOT EXISTS idx_agg_merchant_name
    ON agg_merchant_fraud(merchant_name);
"""

with engine.begin() as conn:
    conn.execute(text(idx_merchant_sql))


# ##### Materialized View: Aggregate Category Fraud

# In[101]:


agg_category_sql = """
DROP MATERIALIZED VIEW IF EXISTS agg_category_fraud;

CREATE MATERIALIZED VIEW agg_category_fraud AS
SELECT
    c.category_id,
    c.category_name,
    COUNT(*) AS total_tx,
    SUM(CASE WHEN f.is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_tx,
    (SUM(CASE WHEN f.is_fraud = 1 THEN 1 ELSE 0 END)::float
        / NULLIF(COUNT(*), 0)) AS fraud_rate,
    SUM(f.amt) AS total_amount,
    SUM(CASE WHEN f.is_fraud = 1 THEN f.amt ELSE 0 END) AS fraud_amount,
    (SUM(CASE WHEN f.is_fraud = 1 THEN f.amt ELSE 0 END)
        / NULLIF(SUM(f.amt), 0)) AS fraud_share_by_value
FROM fact_transactions f
JOIN dim_category c ON f.category_id = c.category_id
GROUP BY c.category_id, c.category_name
ORDER BY fraud_rate DESC;
"""

with engine.begin() as conn:
    conn.execute(text(agg_category_sql))


# In[102]:


idx_category_sql = """
CREATE INDEX IF NOT EXISTS idx_agg_category_fraud_rate
    ON agg_category_fraud(fraud_rate DESC);

CREATE INDEX IF NOT EXISTS idx_agg_category_name
    ON agg_category_fraud(category_name);
"""

with engine.begin() as conn:
    conn.execute(text(idx_category_sql))


# #### Exporting: Snapshot

# In[ ]:


import os
import subprocess

subprocess.run([
    "docker", "exec",
    "-e", "PGPASSWORD=password",
    "-t",
    "postgresql",
    "pg_dump",
    "-U", "user",
    "-d", "database",
    "-Fc",
    "-f", "/tmp/fraudData_snapshot.dump"
], check=True)

subprocess.run([
    "docker", "cp",
    "postgresql:/tmp/fraudData_snapshot.dump",
    "./fraudData_snapshot.dump"
], check=True)


# ### Validation

# In[104]:


with engine.connect() as conn:
    print("Daily Fraud Head:")
    res = conn.execute(text("SELECT * FROM agg_daily_fraud ORDER BY trans_date LIMIT 5;"))
    for row in res:
        print(row)

    print("\nTop 5 Merchants by fraud_rate:")
    res = conn.execute(text("""
        SELECT merchant_name, total_tx, fraud_tx, fraud_rate
        FROM agg_merchant_fraud
        ORDER BY fraud_rate DESC
        LIMIT 5;
    """))
    for row in res:
        print(row)

    print("\nTop 5 Categories by fraud_rate:")
    res = conn.execute(text("""
        SELECT category_name, total_tx, fraud_tx, fraud_rate
        FROM agg_category_fraud
        ORDER BY fraud_rate DESC
        LIMIT 5;
    """))
    for row in res:
        print(row)


# ## Inference

# ### Import

# In[1]:


import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

import matplotlib.pyplot as plt

from sqlalchemy import create_engine, text


# ### Database: PostgreSQL

# In[14]:


import os
import time
import subprocess

if subprocess.run(["docker", "start", "postgresql"]).returncode != 0:
    subprocess.run([
        "docker", "run",
        "--name", "postgresql",
        "-e", "POSTGRES_USER=user",
        "-e", "POSTGRES_PASSWORD=password",
        "-e", "POSTGRES_DB=database",
        "-p", "5432:5432",
        "-d", "postgres:16"
    ], check=True)

time.sleep(10)

subprocess.run([
    "docker", "cp",
    "fraudData_snapshot.dump",              
    "postgresql:/tmp/fraudData_snapshot.dump"
], check=True)

time.sleep(10)

subprocess.run([
    "docker", "exec",
    "-e", "PGPASSWORD=password",
    "postgresql",
    "pg_restore",
    "-U", "user",
    "-d", "database",
    "/tmp/fraudData_snapshot.dump"
], check=True)


# In[15]:


DB_USER = "user"
DB_PASS = "password"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "database"

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# In[16]:


engine = create_engine(DATABASE_URL, echo=False, future=True)


# In[17]:


with engine.connect() as conn:
    result = conn.execute(text("SELECT version();"))
    print(result.scalar())


# ### Testing

# In[18]:


with engine.connect() as conn:
    print("Daily Fraud Head:")
    res = conn.execute(text("SELECT * FROM agg_daily_fraud ORDER BY trans_date LIMIT 5;"))
    for row in res:
        print(row)

    print("\nTop 5 Merchants by fraud_rate:")
    res = conn.execute(text("""
        SELECT merchant_name, total_tx, fraud_tx, fraud_rate
        FROM agg_merchant_fraud
        ORDER BY fraud_rate DESC
        LIMIT 5;
    """))
    for row in res:
        print(row)

    print("\nTop 5 Categories by fraud_rate:")
    res = conn.execute(text("""
        SELECT category_name, total_tx, fraud_tx, fraud_rate
        FROM agg_category_fraud
        ORDER BY fraud_rate DESC
        LIMIT 5;
    """))
    for row in res:
        print(row)

