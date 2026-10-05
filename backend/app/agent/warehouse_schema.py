SCHEMA_DESCRIPTION = """
You can query the following PostgreSQL tables and materialized views.

Dimension tables:

1) dim_customer(
    customer_id BIGINT,
    cc_num TEXT,
    first TEXT,
    last TEXT,
    gender VARCHAR(1),
    street TEXT,
    city TEXT,
    state TEXT,
    zip TEXT,
    lat DOUBLE PRECISION,
    long DOUBLE PRECISION,
    city_pop BIGINT,
    job TEXT,
    dob DATE
)

2) dim_merchant(
    merchant_id BIGINT,
    merchant_name TEXT,
    merch_lat DOUBLE PRECISION,
    merch_long DOUBLE PRECISION
)

3) dim_category(
    category_id BIGINT,
    category_name TEXT
)

4) dim_date(
    date_id BIGINT,
    trans_date DATE,
    year INT,
    month INT,
    day INT,
    day_of_week TEXT,
    is_weekend BOOLEAN,
    year_month TEXT
)

Fact table:

5) fact_transactions(
    transaction_id BIGINT,
    trans_num TEXT,
    customer_id BIGINT,
    merchant_id BIGINT,
    category_id BIGINT,
    date_id BIGINT,
    trans_ts TIMESTAMP,
    unix_time BIGINT,
    amt DOUBLE PRECISION,
    is_fraud SMALLINT,
    year INT,
    month INT,
    hour INT,
    is_weekend BOOLEAN,
    cust_merch_distance_km DOUBLE PRECISION,
    split TEXT
)

Pre-aggregated materialized views (preferred when appropriate):

6) agg_daily_fraud(
    trans_date DATE,
    year INT,
    month INT,
    day INT,
    day_of_week TEXT,
    is_weekend BOOLEAN,
    year_month TEXT,
    total_tx BIGINT,
    fraud_tx BIGINT,
    fraud_rate DOUBLE PRECISION,
    total_amount DOUBLE PRECISION,
    fraud_amount DOUBLE PRECISION,
    fraud_share_by_value DOUBLE PRECISION
)

7) agg_monthly_fraud(
    year INT,
    month INT,
    year_month TEXT,
    total_tx BIGINT,
    fraud_tx BIGINT,
    fraud_rate DOUBLE PRECISION,
    total_amount DOUBLE PRECISION,
    fraud_amount DOUBLE PRECISION,
    fraud_share_by_value DOUBLE PRECISION
)

8) agg_merchant_fraud(
    merchant_id BIGINT,
    merchant_name TEXT,
    total_tx BIGINT,
    fraud_tx BIGINT,
    fraud_rate DOUBLE PRECISION,
    total_amount DOUBLE PRECISION,
    fraud_amount DOUBLE PRECISION,
    fraud_share_by_value DOUBLE PRECISION
)

9) agg_category_fraud(
    category_id BIGINT,
    category_name TEXT,
    total_tx BIGINT,
    fraud_tx BIGINT,
    fraud_rate DOUBLE PRECISION,
    total_amount DOUBLE PRECISION,
    fraud_amount DOUBLE PRECISION,
    fraud_share_by_value DOUBLE PRECISION
)

Guidance:

- Whenever possible, prefer the agg_* materialized views for questions about overall
  daily/monthly fraud rates, top merchants, top categories, and similar aggregated metrics.
- If the question requires aggregate time-of-day patterns or metrics not present in the views,
  then use fact_transactions and join it to the dimension tables as needed:
  - fact_transactions.customer_id = dim_customer.customer_id
  - fact_transactions.merchant_id = dim_merchant.merchant_id
  - fact_transactions.category_id = dim_category.category_id
  - fact_transactions.date_id = dim_date.date_id

Examples:

Q: "How does the monthly fraud rate evolve over the entire period?"
SQL:
  SELECT year_month, fraud_rate
  FROM agg_monthly_fraud
  ORDER BY year, month;

Q: "Which merchants have the highest fraud rate?"
SQL:
  SELECT merchant_name, total_tx, fraud_tx, fraud_rate
  FROM agg_merchant_fraud
  ORDER BY fraud_rate DESC
  LIMIT 10;

Q: "Which merchant categories exhibit the highest incidence of fraudulent transactions?"
SQL:
  SELECT category_name, total_tx, fraud_tx, fraud_rate
  FROM agg_category_fraud
  ORDER BY fraud_tx DESC, total_tx DESC
  LIMIT 10;

Q: "How many fraudulent transactions occurred each day, and what was their total amount?"
SQL:
  SELECT
      d.trans_date,
      COUNT(*) AS fraud_tx,
      SUM(f.amt) AS fraud_amount
  FROM fact_transactions f
  JOIN dim_date d ON f.date_id = d.date_id
  WHERE f.is_fraud = 1
  GROUP BY d.trans_date
  ORDER BY d.trans_date;

Q: "What is the average transaction amount by hour of day for fraudulent transactions?"
SQL:
  SELECT
      f.hour,
      AVG(f.amt) AS avg_fraud_amount,
      COUNT(*) AS fraud_tx
  FROM fact_transactions f
  WHERE f.is_fraud = 1
  GROUP BY f.hour
  ORDER BY f.hour;
"""
