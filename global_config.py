dbutils.widgets.text("CATALOG", "")
dbutils.widgets.text("SCHEMA", "")
dbutils.widgets.text("MAX_TOKENS", "200")
dbutils.widgets.text("TRIGGER_RUN", "False")
dbutils.widgets.text("CLEAN_UP", "False")
dbutils.widgets.text("TRIGGER_SCALE_TEST", "False")

dbutils.widgets.text("LIMIT", "1")
dbutils.widgets.text("CONSISTENCY_FACTOR", "1")
dbutils.widgets.text("EPOCHS", "1")
dbutils.widgets.text("BATCH_SIZE", "5")
dbutils.widgets.text("N_JOBS", "8")
dbutils.widgets.text("VALIDATION_SET_LIMIT", "20")

dbutils.widgets.dropdown("Introspection LLM Endpoint",
                         defaultValue='databricks-meta-llama-3-3-70b-instruct',
                         choices=["databricks-meta-llama-3-3-70b-instruct", "databricks-claude-3-7-sonnet",
                                  'databricks-llama-4-maverick'])

dbutils.widgets.dropdown("JUDGE_LLM_ENDPOINT",
                         defaultValue='databricks-llama-4-maverick',
                         choices=['databricks-llama-4-maverick', "databricks-meta-llama-3-3-70b-instruct",
                                  "databricks-claude-3-7-sonnet"])

dbutils.widgets.dropdown("SMALL_LLM_ENDPOINTS",
                         defaultValue='databricks-meta-llama-3-1-405b-instruct',
                         choices=['databricks-meta-llama-3-1-8b-instruct', "databricks-mixtral-8x7b-instruct",
                                  "databricks-meta-llama-3-1-405b-instruct", "databricks-llama-4-maverick"])

dbutils.widgets.text("EXPERIMENT_ID", "713974745739565")
dbutils.widgets.text("RUN_BASELINE", "False")
dbutils.widgets.text("REFRESH_DASHBOARD", "False")
dbutils.widgets.text("INTROSPECTION_LOOKBACK", "3")
dbutils.widgets.text("DEBUG", "False")
dbutils.widgets.text("IS_TRIGGERED_FROM_CHECKPOINT", "False")

# 🦙 Chaos LLama Parameters
IS_TRIGGERED_FROM_CHECKPOINT = dbutils.widgets.get("IS_TRIGGERED_FROM_CHECKPOINT").lower() == "true"
DEBUG = dbutils.widgets.get("DEBUG").lower() == "true"
REFRESH_DASHBOARD = (dbutils.widgets.get("REFRESH_DASHBOARD").lower() == "true")
SMALL_LLM_ENDPOINTS = dbutils.widgets.get("SMALL_LLM_ENDPOINTS")
LIMIT = int(dbutils.widgets.get("LIMIT"))
VALIDATION_SET_LIMIT = int(dbutils.widgets.get("VALIDATION_SET_LIMIT"))
CONSISTENCY_FACTOR = int(dbutils.widgets.get("CONSISTENCY_FACTOR"))
EPOCHS = int(dbutils.widgets.get("EPOCHS"))
BATCH_SIZE = int(dbutils.widgets.get("BATCH_SIZE"))
N_JOBS = int(dbutils.widgets.get("N_JOBS"))
MAX_TOKENS = int(dbutils.widgets.get("MAX_TOKENS"))
RUN_BASELINE = dbutils.widgets.get("RUN_BASELINE").lower() == "true"
EVAL_TABLE_NAME = dbutils.widgets.get("EVAL_TABLE_NAME")
# Data Module
CATALOG = dbutils.widgets.get("CATALOG")
SCHEMA = dbutils.widgets.get("SCHEMA")
TRIGGER_RUN = dbutils.widgets.get("TRIGGER_RUN").lower() == "true"
TRIGGER_SCALE_TEST = dbutils.widgets.get("TRIGGER_SCALE_TEST").lower() == "true"
CLEAN_UP = dbutils.widgets.get("CLEAN_UP").lower() == "true"
EVAL_TABLE = f"{CATALOG}.{SCHEMA}.{EVAL_TABLE_NAME}"
JUDGES_TABLE = f"{CATALOG}.{SCHEMA}.judges"
CHAOS_ANALYTICS_TABLE = f"{CATALOG}.{SCHEMA}.chaos_analytics"

BASELINE_SCHEMA = "sem_glbl_fincl_cpgpt"
# AI/ML Module
EXPERIMENT_NAME = ""
EXPERIMENT_ID = dbutils.widgets.get("EXPERIMENT_ID")
LLM = ""
LLM_PARAMETERS = {"temperature": 0.0}
HOST = "https://adb-6209649103177418.18.azuredatabricks.net"
INTROSPECT_AGENT_LLM_ENDPOINT = dbutils.widgets.get("Introspection LLM Endpoint")
AI_QUERY_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
JUDGE_LLM_ENDPOINT = dbutils.widgets.get("JUDGE_LLM_ENDPOINT")
TEST_TABLE = f"{CATALOG}.{SCHEMA}.market_dim"

GENIE_SPACE_ID = "01f004e394f21cdbafb2a1df68f53c3b"
CPGPT_GENIE_SPACE_ID_WITH_DATA_SAMPLING = "01effdc9c84518888b036a015a4924b7"
CPGPT_GENIE_SPACE_ID_WITHOUT_DATA_SAMPLING = "01efed4c07061097bbd9330461a22b9f"
DBRX_CPGPT_GENIE_SPACE_ID_WITH_DATA_SAMPLING = "01f00e773a2e19f9ab163cc576bde01f"
MANUAL_DBRX_CPGPT_GENIE_SPACE_ID_WITH_DATA_SAMPLING = "01f015947d28105cb6da1f044b7f7c47"
INTROSPECTION_LOOKBACK = int(dbutils.widgets.get("INTROSPECTION_LOOKBACK"))

QUALITY_THRESHOLD = .90  # 90%

NULL_HYPOTHESIS_GENIE_SPACE_ID = "01f0086525d917ddb7527333c5af68ba"

METRICS_DEFINITION = """ 
SQL Accuracy: The response query does not match the expected query in terms of metrics, countries, and years.
Dimension Ambiguity: The response uses different dimension tables than those specified in the request.
Correctness: The response does not match the context of the question or the expected response.
Guideline Adherence: The response does not address issues with multiple columns and omits necessary filters.
"""

BEST_MLFLOW_RUN = "stately-hound-205"
BEST_MLFLOW_RUN_PROVING_IMPROVEMENT = "persistent-snipe-320"

BEST_MLFLOW_RUN_AT_SCALE = "6322d6636e0d4c478fffba1fabc456e5"

BEST_MLFLOW_RUNS_MAP = {
    "skittish-croc-724": "3ed7d94979bc4764beea08b3ea07b547"  # "20 EPOCHs, 70pct accurarcy, 20 questions"
}

MLFLOW_EXPERIMENTS = {
    "Benchmark Lab": "283295167155400",
    "🦙ChaosLlama": "713974745739565"
}

DDL_HISTORY_TABLE = "ddl_history"

GLOBAL_GUIDELINES = dict(
    global_guidelines={
        "sql_accuracy": ["Ensure that response and the ground truth are semantically equivalent in ansi spark sql"],
        "count_joins": ["Count the number of joins are the same in both queries."],
        "dimension_ambiguity": ["Identify if the query chooses the wrong dimension table"],
        "has_select": [
            "Ensure that the expected response has a sql SELECT statement , ignoring case, in the ansi sql query"],
        "financial_planning": [
            "For every generated query the following dimensions MUST BE PRESENT. Dimensions: [scenario, report_type, year]"]
    }
)

GLOBAL_GUIDELINES_v2 = dict(
    global_guidelines={
        "sql_accuracy": ["Ensure that response and the ground truth are semantically equivalent in ansi spark sql"],
        "count_joins": ["Count the number of joins are the same in both queries."],
        "has_select": [
            "Ensure that the expected response has a sql SELECT statement , ignoring case, in the ansi sql query"],
        "financial_planning": [
            "For every generated query the following dimensions MUST BE PRESENT. Dimensions: [scenario, report_type, year]"]
    }
)
w = WorkspaceClient()