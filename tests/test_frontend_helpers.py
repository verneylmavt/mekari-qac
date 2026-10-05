"""Turn state and evidence tests use only deterministic data."""

from frontend import app


def test_history_excludes_failed_pairs_and_bounds_completed_turns():
    messages = [
        {"role": "user", "content": "Failed", "turn_id": "bad", "state": "failed"},
        {"role": "assistant", "content": "Error", "turn_id": "bad", "state": "failed"},
    ]
    for index in range(8):
        messages += [
            {"role": "user", "content": f"Q{index}", "state": "completed"},
            {"role": "assistant", "content": f"A{index}", "state": "completed"},
        ]
    assert app.build_history_for_backend(messages) == [
        {"role": role, "content": f"{prefix}{index}"}
        for index in range(2, 8)
        for role, prefix in [("user", "Q"), ("assistant", "A")]
    ]


def test_retry_reuses_snapshot_without_duplicate_user_message():
    messages = [{"role": "user", "content": "Before"}, {"role": "assistant", "content": "Previous"}]
    turn = app.begin_turn(messages, "Question", "eba")
    turn["state"] = "failed"
    messages[-1]["state"] = "failed"
    retry = app.retry_turn(messages, turn)
    assert len(messages) == 3
    assert retry["id"] == turn["id"]
    assert retry["history"] == [
        {"role": "user", "content": "Before"},
        {"role": "assistant", "content": "Previous"},
    ]
    assert retry["document_id"] == "eba"
    assert retry["state"] == "pending"


def test_chart_uses_only_sql_rows_and_converts_fractional_rate():
    sources = [
        {
            "type": "sql_result",
            "rows_preview": [
                {"year_month": "2020-01", "fraud_rate": 0.012},
                {"year_month": "2020-02", "fraud_rate": 0.02},
            ],
        }
    ]
    chart = app.chart_spec(sources)
    assert chart["rows"] == [
        {"Month": "2020-01", "Fraud rate (%)": 1.2},
        {"Month": "2020-02", "Fraud rate (%)": 2.0},
    ]
    assert chart["kind"] == "line"
    assert (
        app.chart_spec([{"type": "document_chunks", "chunks": [{"text": "fraud rate 5%"}]}]) is None
    )
    assert (
        app.chart_spec([{"type": "sql_result", "rows_preview": [{"random": "A", "value": 12}]}])
        is None
    )


def test_charts_support_actual_warehouse_category_and_merchant_names():
    category = app.chart_spec(
        [
            {
                "type": "sql_result",
                "rows_preview": [{"category_name": "grocery_pos", "fraud_rate": 0.015}],
            }
        ]
    )
    assert category["rows"] == [{"Category": "grocery_pos", "Fraud rate (%)": 1.5}]
    merchant = app.chart_spec(
        [{"type": "sql_result", "rows_preview": [{"merchant_name": "Demo store", "fraud_tx": 42}]}]
    )
    assert merchant["rows"] == [{"Merchant": "Demo store", "Fraud transactions (count)": 42}]


def test_amount_charts_do_not_invent_currency_without_source_metadata():
    fraud = app.chart_spec(
        [
            {
                "type": "sql_result",
                "rows_preview": [{"year_month": "2020-01", "fraud_amount": 1200.5}],
            }
        ]
    )
    assert fraud["rows"] == [{"Month": "2020-01", "Fraud amount (dataset units)": 1200.5}]
    transactions = app.chart_spec(
        [
            {
                "type": "sql_result",
                "rows_preview": [{"category_name": "grocery_pos", "total_amount": 4200}],
            }
        ]
    )
    assert transactions["rows"] == [
        {"Category": "grocery_pos", "Transaction amount (dataset units)": 4200}
    ]


def test_export_includes_answer_sql_and_page_citations_without_failed_answers():
    messages = [
        {"role": "user", "content": "Question", "state": "completed"},
        {
            "role": "assistant",
            "content": "Answer [EBA-1]",
            "state": "completed",
            "sql": "SELECT 1",
            "quality_score": None,
            "quality_available": False,
            "sources": [
                {
                    "type": "document_chunks",
                    "chunks": [
                        {
                            "citation_id": "EBA-1",
                            "title": "EBA 2024",
                            "page": 7,
                            "snippet": "Evidence",
                        }
                    ],
                }
            ],
        },
        {"role": "assistant", "content": "private error", "state": "failed"},
    ]
    exported = app.export_markdown(messages)
    assert "Answer [EBA-1]" in exported
    assert "EBA 2024 · page 7" in exported
    assert "SELECT 1" in exported
    assert "Quality: N/A" in exported
    assert "private error" not in exported
