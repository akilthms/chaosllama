


def class ChaosLLamaAnalytics():
""" The purpose of this class is to maintain and manage the metrics from the ChaosLLama run for vizualization purposes """


def __init__(self, parent_run_id: str, table_name: str = CHAOS_ANALYTICS_TABLE, experiment_id: str = EXPERIMENT_ID,
             catalog=CATALOG, schema=SCHEMA):
    self.parent_run_id = parent_run_id
    self.experiment_id = experiment_id
    self.table_name = table_name
    self.runs = mlflow.search_runs(experiment_ids=[self.experiment_id],
                                   filter_string=f"tags.mlflow.rootRunId = '{self.parent_run_id}'",
                                   output_format="pandas")


def save_run_metadata(self, mode="append", is_test=False, ):
    (pdf := self.runs).display()
    pdf.columns = pdf.columns.str.replace(" ", "_")
    if not is_test:
        (
            spark.createDataFrame(pdf)
            .write
            .option("mergeSchema", "true")
            .option("delta.enableChangeDataFeed", "true")
            .format("delta")
            .saveAsTable(self.table_name, mode=mode)
        )


def update_data(self, df, mode="overwrite"):
    (
        df.write
        .option("mergeSchema", "true")
        .option("delta.enableChangeDataFeed", "true")
        .format("delta")
        .saveAsTable(self.table_name, mode=mode)
    )

    df.display()


def add_column(self, column_name, transform: Callable = None) -> None:
    if isinstance(transform, Callable) or isinstance(transform, pyspark.sql.column.Column):
        df = spark.table(self.table_name).withColumn(column_name, transform)

    self.update_data(df, mode="overwrite")


def update_chaos_analytics_tbl():
    chaos_analytics = ChaosLLamaAnalytics(parent_run_id=None)  # mlflow_parent_run.run_id)
    chaos_analytics.add_column("parentRunName", get_run_name_udf("`tags.mlflow.parentRunId`"))


def test_chaos_analytics(parent_run_id):
    chaos_analytics = ChaosLLamaAnalytics(parent_run_id=parent_run_id)  # mlflow_parent_run.run_id)
    chaos_analytics.save_run_metadata("chaos_analytics", mode="append")
    spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.chaos_analytics")


