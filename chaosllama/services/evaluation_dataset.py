from typing import Optional, Literal, Self
from dotenv import dotenv_values
from chaosllama.entities.models import EvalSetTable
from databricks.connect import DatabricksSession
from chaosllama.synthetic_data.synthesize import SyntheticDataGenerator
import pyspark
from pyspark.sql import functions as F
import pandas as pd
from rich.console import Console
from rich.panel import Panel

env = dotenv_values(".env")
PROFILE = env["DATABRICKS_PROFILE"]
spark = DatabricksSession.builder.profile(PROFILE).serverless(True).getOrCreate()
console = Console()

class EvalSetManager:
    def __init__(self, table_name:str=None,limit=None, consistency_factor= None, eval_set: Optional[EvalSetTable | str] = None):
        self.eval_set = eval_set
        self.table_name = table_name
        self.limit = limit
        self.consistency_factor = consistency_factor

    def write_evalset(self) -> Self:
        """ Write the evaluation set to the specified table in Unity Catalog. """

        if self.eval_set.data is not None:
            if isinstance(self.eval_set.data, pd.DataFrame):
                self.eval_set.data = spark.createDataFrame(self.eval_set.data)

            try:
                (self.eval_set.data
                              .write
                              .mode("overwrite")
                              .option("mergeSchema", "true")
                              .saveAsTable(self.table_name))
                print(f"Evaluation set written to {self.table_name}")
            except Exception as e:
                print(f"Error writing evaluation set to {self.table_name}: {e}")

        else:
            raise ValueError("eval_set must be an instance of EvalSetTable")

        return self


    def simulate_system_instruction_update(self, ai_suggested_instruction: str) -> Self:
        """ Simulate the update of the system instruction in the evaluation set. """
        if not self.eval_set:
            raise ValueError("EvalSet does not exist, please create it first.")

        if isinstance(self.eval_set.data, pd.DataFrame):
            self.eval_set.data["question"] = ai_suggested_instruction + "\n" + self.eval_set.data["question"]
        else:
            self.eval_set.data = (
                self.eval_set.data.withColumn("question", F.concat(F.lit(ai_suggested_instruction),
                                                                   F.lit("\n"),
                                                                   F.col("question")))
            )

        return self


    def get_evalset(self, mode: Literal["synthetic", "existing"] = "existing", **kwargs) -> Self:
        """ Create the evaluation set for the optimization run, based on the provided configuration. """


        if isinstance(self.eval_set, str) and self.eval_set.count("." ) == 2:
            self.eval_set = spark.table(self.eval_set)

        elif isinstance(self.eval_set, str) and self.eval_set.count("." ) != 2:
            raise ValueError("eval_set must be a valid unity catalog 3 namespace scheme")

        else:
            try:
                if mode == "existing":
                    data = spark.table(self.table_name)
                    self.eval_set = EvalSetTable(data=data)

                if mode == "local":
                    self.eval_set = EvalSetTable(data=pd.read_csv("assets/debug_evalset3.csv"))

                if mode == "synthetic":
                    print("Creating Synthetic Evaluation Set")
                    syn_gen = SyntheticDataGenerator(
                        target_table="synthetic_evals",
                        num_evals=5,
                        agent_description="""
                            The agent needs to generate accurate sql queries given the question only. The Agent has access to the metadata
                            of the tables in the schema (data types, column descriptions, primary and foreign key relationships
                            and can use that to generate the sql queries.
                            """,
                        question_guidelines="""        
                            # User personas
                            - A analyst who is trying to understand the data

                            # Example questions
                            - Can you give me information on sku STB-KCP-001?
                            - What was the top selling item sku last year?

                            # Additional Guidelines
                            - The 
                            - Questions should be business analyst questions related to the provided content 
                            - The questions generated should be analytical and relatively simple for someone that is college educated.
                            - The question must relate to one or more tables.
                            - The question should not reference any table names. 
                            i
                            """
                    )

                    data = syn_gen.run()
                    data["issues"] = None
                    self.eval_set = EvalSetTable(data=data)



            except Exception as e:
                print(f"Error loading table {self.table_name}: {e}")
                # raise ValueError(f"Table {self.table_name} does not exist or is not accessible.")




        return self

    def prepare_evals(self, mode: Literal["synthetic", "existing"] = "existing", **kwargs) -> Self:
        console.print(Panel("📐Prepping Evaluation Dataset", expand=False, style="bold cyan"))

        if not self.eval_set:
            print("EvalSet does not exist, creating a new one")
            self.get_evalset(mode=mode)

        evaluation_dataset = (
            self.eval_set
                .limit(self.limit)
                .replicate_rows(self.consistency_factor)
        )

        print("Created Evaluation Set")
        print(evaluation_dataset.data)
        return self