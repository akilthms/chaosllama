import mlflow
from databricks.agents.evals import generate_evals_df
import pandas as pd
from typing import List, Dict, Optional, Literal
from chaosllama.entities.models import SourceDocument, AgentConfig, EvalSetTable
from chaosllama.services.unity_catalog import UCService
from chaosllama.profiles.config import config
from pprint import pprint
from functools import reduce
from dataclasses import asdict
from chaosllama.utils.utilities import ask_ai
from langchain.prompts import PromptTemplate
from pathlib import Path

class SyntheticDataGenerator:
    """A class for generating synthetic data for evaluation purposes.

    Args:
        target_table (str): The name of the target table.
        source_docs (List[SourceDocument]): List of source documents to use for generation.
        num_evals (int): Number of evaluations to generate.
        agent_description (str): Description of the agent used for generation.
    """

    def __init__(self,
                 target_table: str,
                 num_evals: int,
                 agent_description: str,
                 source_docs: List[SourceDocument] = None ,
                 question_guidelines: Optional[str] = None,
                 example_questions: Optional[List[str]] = None,
                 uc_service: UCService = UCService(config.CATALOG, config.SCHEMA)):
        self.source_docs = source_docs
        self.target_table = target_table
        self.num_evals = num_evals
        self.agent_description = agent_description
        self.question_guidelines = question_guidelines
        self.example_questions = example_questions
        self.uc_service = uc_service

    def create_source_docs(self, catalog:str, schema:str) -> List[SourceDocument]:
        """Create source documents from a given schema.

        Args:
            catalog (str): The catalog to use for document creation.
            schema (str): The schema to use for document creation.

        Returns:
            List[SourceDocument]: List of created source documents.
        """
        pass

    def group_source_ddls(self,
                          source_ddls: List[str],
                          num_groups: int=1,
                          strategy:Literal["random", "ai"] = "random") -> List[str]:
        """Group source DDLs into a specified number of groups based upon a strategy.

        If the strategy is "random", it will randomly shuffle the DDLs and group them.
        If the strategy is "ai", it will use an AI-based approach to group the DDLs (not implemented here).

        Args:
            source_ddls (List[str]): List of source DDLs.
            num_groups (int): Number of groups to create.
            strategy (string): The strategy to use for grouping. Options are "random" or "ai".

        Returns:
            List[str]: List of grouped DDLs as strings.

        """
        match strategy:
            case "ai":
                pass
            case "random":

                if num_groups <= 0:
                    raise ValueError("Number of groups must be greater than zero.")

                elif num_groups == 1:
                    return [reduce(lambda x, y: f"{x}\n{y}", source_ddls)]

                elif num_groups > 1:
                    raise NotImplementedError("Random grouping strategy greater than 1 is not implemented yet!")

                elif num_groups > len(source_ddls):
                    raise ValueError("Number of groups cannot exceed the number of source DDLs.")
            case _:
                raise ValueError(f"Unknown strategy: {strategy}. Supported strategies are 'random' and 'ai'.")

    def convert_to_chaos_llama_evalset(self, eval_df:pd.DataFrame) -> EvalSetTable:
        pass

    def get_source_docs(self) -> pd.DataFrame:
        """Retrieve source documents.

        Returns:
            List[SourceDocument]: List of source documents.
        """
        tables = self.uc_service.get_tables()
        ddls = [ self.uc_service.get_create_stmnt(t) for t in tables]
        ddl_groups = self.group_source_ddls(ddls, num_groups=1, strategy="random")

        doc_uri = f"{self.uc_service.catalog}.{self.uc_service.schema}.*"
        source_docs = [
                       asdict(SourceDocument(content=grp, doc_uri=doc_uri))
                        for grp in ddl_groups
                      ]
        return pd.DataFrame(columns=list(source_docs[0].keys()), data=source_docs)


    def run(self, mode: Literal["databricks","custom"]="custom") -> Optional[pd.DataFrame]:
        """Execute the synthetic data generation process.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing generated evaluations,
                                  or None if generation fails.
        """
        if self.source_docs is None:
            print("Source documents not provided, generating from UCService...")
            self.source_docs = self.get_source_docs()

        try:
            match mode:

                case "databricks":
                    evals_df = generate_evals_df(
                        docs=self.source_docs,
                        num_evals=self.num_evals,
                        agent_description=self.agent_description
                    )
                    return evals_df
                case "custom":
                    agent_config = AgentConfig(
                        endpoint="databricks-claude-3-7-sonnet",
                        llm_parameters={"temperature": 0.0},
                        system_prompt=PromptTemplate(
                            input_variables=["schema_ddl", "num_q"],
                            template="""
                                                    Look at the following DDL for an entire schema
                                                    {schema_ddl}

                                                    Generate {num_q} analytical question that can be answered using the schema.
                                                    The question should be specific and relevant to the schema provided.
                                                    The question should be in the form of a question, not a statement.
                                                    Example: 'What was the top selling item sku last year?'

                                                    # Formatting
                                                    Structure the questions into a single jsonl, formatted with the following keys:
                                                    - question: The question to be asked
                                                    - ground_truth_query: The SQL query that can be used to answer the question

                                                    """)

                    )

                    # TODO: Actually implement the AI-based generation of evaluations
                    # schema_ddl = open(Path("assets/ddls.txt")).read()
                    # response = ask_ai(dict(schema_ddl=schema_ddl, num_q=5), agent_config)

                    return pd.read_csv(Path("assets/debug_evalset.csv"))  # Placeholder for actual AI generation logic


        except Exception as e:
            print(f"Error generating synthetic data: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage

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


    pprint(syn_gen.run())

    # 🤖 Generate synthetic evaluations
    #
    # syn_gen.create_source_docs(schema=config.SCHEMA)
    #
    # evals_df = generator.run()
    # if evals_df is not None:
    #     print("Synthetic evaluations generated successfully.")
    #     print(evals_df.head())
    # else:
    #     print("Failed to generate synthetic evaluations.")