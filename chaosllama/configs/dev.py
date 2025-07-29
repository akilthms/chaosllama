# config.py

# 🤝 Unity Catalog
CATALOG = "retail_consumer_goods"
SCHEMA = "store_ops"
CHAOS_ANALYTICS_TABLE_TEMPLATE = "{CATALOG}.{SCHEMA}.chaos_analytics"
EVAL_TABLE_NAME = ""  # You can set this dynamically if needed

# 🏃 Runtime Configuration
MAX_TOKENS = 200
LIMIT = 1
CONSISTENCY_FACTOR = 1
EPOCHS = 1
BATCH_SIZE = 5
N_JOBS = 8
VALIDATION_SET_LIMIT = 20
REFRESH_DASHBOARD = False
RUN_BASELINE = False
INTROSPECTION_LOOKBACK = 3
DEBUG = False
IS_TRIGGERED_FROM_CHECKPOINT = False

# Base line / Null Hypothesis Config
BASELINE_CATALOG = ""
BASELINE_SCHEMA = ""

# 🤖 AI Configuration
INTROSPECT_AGENT_LLM_ENDPOINT = "databricks-claude-3-7-sonnet"
    # Options:
    # - databricks-meta-llama-3-3-70b-instruct
    # - databricks-claude-3-7-sonnet
    # - databricks-llama-4-maverick
SMALL_LLM_ENDPOINTS = "databricks-meta-llama-3-1-405b-instruct"
    # Options:
    # - databricks-meta-llama-3-1-8b-instruct
    # - databricks-mixtral-8x7b-instruct
    # - databricks-meta-llama-3-1-405b-instruct
    # - databricks-llama-4-maverick

# 🧞‍♂️ Genie Space Ids
NULL_HYPOTHESIS_GENIE_SPACE_ID = ""
TEST_GENIE_SPACE_ID = ""
TEST_GENIE_SPACE_ID_2 = ""
TEST_GENIE_SPACE_ID_3 = ""

metrics = {
    "QUALITY_THRESHOLD": 0.90,
    "METRICS_DEFINITION": (
        "SQL Accuracy: The response query does not match the expected query in terms of metrics, countries, and years.\n"
        "Dimension Ambiguity: The response uses different dimension tables than those specified in the request.\n"
        "Correctness: The response does not match the context of the question or the expected response.\n"
        "Guideline Adherence: The response does not address issues with multiple columns and omits necessary filters."
    )
}

BEST_MLFLOW_RUN = "",
MLFLOW_EXPERIMENTS =  {
    "🦙ChaosLlama": ""
}


global_guidelines = {
    "v1": {
        "sql_accuracy": [
            "Ensure that response and the ground truth are semantically equivalent in ansi spark sql"
        ],
        "count_joins": [
            "Count the number of joins are the same in both queries."
        ],
        "dimension_ambiguity": [
            "Identify if the query chooses the wrong dimension table"
        ],
        "has_select": [
            "Ensure that the expected response has a sql SELECT statement , ignoring case, in the ansi sql query"
        ],
        "financial_planning": [
            "For every generated query the following dimensions MUST BE PRESENT. Dimensions: [scenario, report_type, year]"
        ]
    },
    "v2": {
        "sql_accuracy": [
            "Ensure that response and the ground truth are semantically equivalent in ansi spark sql"]
    }
}
