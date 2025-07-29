from chaosllama import logging
from databricks.sdk.service import catalog
from pyspark.sql.types import StructType, StructField, StringType, TimestampType

logger_names = ["genie_manager", "mlflow_eval_manager", "uc_manager"]
loggers = {f"{name}_logger": logging.getLogger(f"{name}_logger") for name in logger_names}
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

log_schema = StructType([
    StructField("timestamp", TimestampType(), False),
    StructField("logger_name", StringType(), False),
    StructField("level", StringType(), False),
    StructField("message", StringType(), False),
])


# Delete Logging Tables
def delete_logging_tables(loggers: logging.Logger):
    for logger_name, _ in loggers.items():
        UCManager().drop_table(logger_name)


# Create Logging Tables
def create_logging_tables(loggers: logging.Logger):
    for logger_name, _ in loggers.items():
        UCManager().create_table(logger_name)


class DeltaLogger:
    def __init__(self, name: str, level=logging.INFO, user=None):
        self.name = name
        self.logger = loggers[name]
        self.logger.setLevel(level)
        self.table_name = f"{CATALOG}.{SCHEMA}.{self.name}"

    def _write_log(self, level_name, msg):
        log_row = Row(
            timestamp=datetime.now(),
            level=level_name,
            message=msg,
            logger_name=self.logger.name,
        )

        (spark.createDataFrame([log_row], schema=log_schema)
         .write
         .option('mergeSchema', True)
         .format("delta")
         .mode("append")
         .saveAsTable(self.table_name))

    def truncate_log(self):
        spark.sql(f"TRUNCATE TABLE {self.table_name}")

    def info(self, msg): self._write_log("INFO", msg)

    def warning(self, msg): self._write_log("WARNING", msg)

    def error(self, msg): self._write_log("ERROR", msg)

    def debug(self, msg): self._write_log("DEBUG", msg)


genie_logger = DeltaLogger("genie_manager_logger")
mlflow_eval_logger = DeltaLogger("mlflow_eval_manager_logger")
uc_logger = DeltaLogger("uc_manager_logger")


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
# test_logging()
# if is_true:=test_logging(): genie_logger.truncate_log()
