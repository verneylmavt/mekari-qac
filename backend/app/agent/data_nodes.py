# backend/app/agent/data_nodes.py

from typing import List
from decimal import Decimal
from statistics import mean
import numbers

from .state import AgentState
from ..llm.client import call_gpt5_mini
from ..repositories.metrics_repo import run_sql_query


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
- If the question requires raw transaction-level details (e.g., specific customers, card numbers,
  time-of-day patterns, individual transactions) or metrics that are not present in the views,
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
  ORDER BY fraud_rate DESC
  LIMIT 10;

Q: "List the last 10 fraudulent transactions with customer name and merchant."
SQL:
  SELECT
      f.trans_ts,
      c.first AS customer_first,
      c.last AS customer_last,
      m.merchant_name,
      f.amt,
      f.is_fraud
  FROM fact_transactions f
  JOIN dim_customer c ON f.customer_id = c.customer_id
  JOIN dim_merchant m ON f.merchant_id = m.merchant_id
  WHERE f.is_fraud = 1
  ORDER BY f.trans_ts DESC
  LIMIT 10;

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

SQL_GEN_SYSTEM = (
    "You generate a single PostgreSQL SELECT query against the tables and views described below.\n"
    "Rules:\n"
    "- You may use any of the listed tables and views.\n"
    "- Prefer the agg_* materialized views when they already contain the metrics needed.\n"
    "- If the views are not sufficient to answer the question (for example, if you need "
    "transaction-level detail, specific customers, card numbers, or time-of-day patterns), "
    "then use fact_transactions and join it with the dimension tables as appropriate.\n"
    "- Only reference the tables and views that are listed in the schema description.\n"
    "- Do not add comments.\n"
    "- Do not wrap the query in backticks.\n\n"
    f"{SCHEMA_DESCRIPTION}"
)


def generate_sql_node(state: AgentState) -> AgentState:
    question = state["question"]
    user_prompt = (
        "Write a single PostgreSQL SELECT statement that answers the question.\n"
        f"Question: {question}\n\n"
        "Output only the SQL query."
    )
    sql = call_gpt5_mini(
        system_prompt=SQL_GEN_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.0,
        max_tokens=512,
    )
    print("Text-to-SQL (by GPT-5 Mini) is Called")
    sql = sql.strip().strip(";")
    state["generated_sql"] = sql  # type: ignore
    return state


def _is_numeric(value) -> bool:
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, Decimal):
        return True
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return True
    return False


def run_sql_node(state: AgentState) -> AgentState:
    sql = state.get("generated_sql")
    if not sql:
        state["sql_result_rows"] = []  # type: ignore
        state["sql_result_preview"] = "No SQL generated."
        return state

    try:
        print("Querying Data from Relational Database...")
        rows = run_sql_query(sql, default_limit=200)
        state["sql_result_rows"] = rows  # type: ignore

        preview_lines: List[str] = []
        preview_lines.append(
            f"Total rows returned (capped at backend limit): {len(rows)}"
        )

        # Show up to 5 example rows
        max_preview_rows = 5
        for idx, row in enumerate(rows[:max_preview_rows]):
            preview_lines.append(f"Row {idx+1}: {row}")

        # Generic numeric summary statistics across all returned rows
        if rows:
            first_row = rows[0]
            numeric_fields = [
                k for k, v in first_row.items() if _is_numeric(v)
            ]

            stats_lines: List[str] = []
            for field in numeric_fields:
                values: List[float] = []
                for r in rows:
                    v = r.get(field)
                    if _is_numeric(v):
                        if isinstance(v, Decimal):
                            v = float(v)
                        values.append(float(v))
                if values:
                    stats_lines.append(
                        f"Summary for '{field}': "
                        f"min={min(values):.6g}, "
                        f"mean={mean(values):.6g}, "
                        f"max={max(values):.6g}"
                    )

            if stats_lines:
                preview_lines.append("Summary statistics over all returned rows:")
                preview_lines.extend(stats_lines)

        state["sql_result_preview"] = (
            "\n".join(preview_lines) if preview_lines else "No rows returned."
        )
    except Exception as e:
        state["sql_result_rows"] = []  # type: ignore
        state["sql_result_preview"] = f"Error executing SQL: {e}"

    return state


DATA_ANSWER_SYSTEM = (
    "You are a data analyst. You are given:\n"
    "- A user question.\n"
    "- The SQL query used to answer it.\n"
    "- A preview of the result rows, which may include summary statistics.\n\n"
    "Your job:\n"
    "1) First, directly answer the user's question using the information from the SQL results "
    "(including any summary statistics). Use 1–3 short paragraphs.\n"
    "2) Then, briefly mention any important limitations or caveats (for example, if the result "
    "set is small, heavily filtered, or if the query may not perfectly match the question).\n"
    "3) Do not spend more than one short paragraph describing the SQL itself.\n"
    "If the result is empty or if there was an error, clearly say that and explain what might be wrong."
)


def data_answer_node(state: AgentState) -> AgentState:
    question = state["question"]
    sql = state.get("generated_sql") or ""
    preview = state.get("sql_result_preview") or ""

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"SQL used:\n{sql}\n\n"
        f"Result preview:\n{preview}\n\n"
        "Now provide a concise but informative answer to the user, following the instructions above."
    )

    answer = call_gpt5_mini(
        system_prompt=DATA_ANSWER_SYSTEM,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=512,
    )
    print("Reasoning for Queried Data (by GPT-5 Mini) is Called")
    
    state["answer"] = answer  # type: ignore
    state["answer_type"] = "data"  # type: ignore
    return state