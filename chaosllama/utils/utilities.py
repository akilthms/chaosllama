from chaosllama.entities.models import AgentConfig
from langchain.chains.llm import LLMChain
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
)
from langchain.prompts import PromptTemplate
from pathlib import Path

def ask_ai(inputs:str | dict, agent_config:AgentConfig, context:str=None):
    # Update this to latest lanchain version
    qa_chain = LLMChain(
        llm=ChatDatabricks(endpoint=agent_config.endpoint, **agent_config.llm_parameters),
        prompt=agent_config.system_prompt
    )

    return qa_chain.run(inputs)


if __name__ == "__main__":
    # Example usage
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



    schema_ddl = open(Path("assets/ddls.txt")).read()
    response = ask_ai(dict(schema_ddl=schema_ddl, num_q=5), agent_config)
    print(response)  # Should print the AI's response to the question


