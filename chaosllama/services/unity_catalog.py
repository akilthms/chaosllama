
from databricks.connect import DatabricksSession
from dotenv import dotenv_values
from dataclasses import dataclass, asdict
import pyspark
from pyspark.sql import functions as F
from databricks.sdk import WorkspaceClient
from deltalake import DeltaTable

env = dotenv_values(".env")
PROFILE = env["DATABRICKS_PROFILE"]
spark = DatabricksSession.builder.profile(PROFILE).serverless(True).getOrCreate()

@dataclass
class UCService:
    """ The purpose of this class is to manage the various interactions with the metadata in unity catalog"""
    _workspace_client = None
    def __init__(self, catalog, schema):
        self.catalog = catalog
        self.schema = schema
        self.eval_set = None
        if UCService._workspace_client is None:
            UCService._workspace_client = WorkspaceClient()

    @property
    def w(self):
        """Provides access to the shared WorkspaceClient instance"""
        return UCService._workspace_client

    def get_tables(self):
        tables = [ table.name for table in list(self.w.tables.list(self.catalog, self.schema))]
        self.tables = tables
        return tables

    def change_col_desc(self, table_name :str, column_name :str ,new_comment :str) -> None:
        spark.sql \
            (f"ALTER TABLE {self.catalog}.{self.schema}.{table_name} CHANGE COLUMN {column_name} COMMENT \'{new_comment}\'")

    def get_ddl(self, table_name :str):
        return spark.sql(f"DESCRIBE TABLE {self.catalog}.{self.schema}.{table_name}")
        # return spark.sql(f"SHOW CREATE TABLE {table_name}").first().createtab_stmt

    def get_create_stmnt(self, table_name :str):
        return spark.sql(f"SHOW CREATE TABLE {self.catalog}.{self.schema}.{table_name}").first().createtab_stmt

    def get_all_ddls(self):
        ddls = [ self.get_create_stmnt(table) for table in self.get_tables()]
        return ddls

    def get_table_history(self, table_name :str):
        dtbl = DeltaTable.forName(spark, f"{self.catalog}.{self.schema}.{table_name}")
        history = dtbl.history()
        history.display()
        return history

    def get_evals(self, table_name:str):
        self.eval_set = spark.table(table_name)
        return self

    def get_eval_questions(self, table_name, limit=None):
        if self.eval_set:
            return self.eval_set.limit(limit).toPandas()["question"].values
        else:
            self.get_evals(table_name)
            return self.eval_set.limit(limit).toPandas()["question"].values

    def update_ground_truth(self, replace_terms :dict) -> pyspark.sql.DataFrame:
        for source, target in replace_terms.items():
            regex_replace = F.regexp_replace("ground_truth_query", source, target)
            self.eval_set = self.eval_set.withColumn("updated_ground_truth" ,regex_replace)
        return self

    def create_table(self, table_name :str):
        spark.sql(f"""
                  CREATE TABLE IF NOT EXISTS {self.catalog}.{self.schema}.{table_name}
                  """)

    def drop_table(self, table_name: str):
        spark.sql(f"""DROP TABLE IF EXISTS {self.catalog}.{self.schema}.{table_name}""")

    @classmethod
    def run_ai_query(cls, df :pyspark.sql.DataFrame, column :str ,query :str, ai_query_endpoint=None):
        ai_query =  f"""
            SELECT ai_query(endpoint => '{ai_query_endpoint}', 
                  request => CONCAT( '{query}', {column}),
                  returnType => "STRING")


                    """

        ai_query = f"""
          ai_query('databricks-meta-llama-3-3-70b-instruct', CONCAT('{query}', {column})) as guideline_name
        """
        return df.selectExpr("*", ai_query)

