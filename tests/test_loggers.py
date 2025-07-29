from chaosllama.loggers import DeltaLogger
# 🧪 Test Loggers
def test_logging(name="genie_manager_logger"):
    genie_logger = DeltaLogger("genie_manager_logger")
    genie_logger.info("🧪 Test message")
    df = spark.table(f"{CATALOG}.{SCHEMA}.{name}")
    df.display()


DELETE_LOGGING_TABLES = True
CREATE_LOGGING_TABLES = False
if DELETE_LOGGING_TABLES: delete_logging_tables(loggers)
if CREATE_LOGGING_TABLES: create_logging_tables(loggers)