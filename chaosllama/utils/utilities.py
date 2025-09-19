#from chaosllama.entities.models import AgentConfig
from langchain.chains.llm import LLMChain
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
)
from langchain.prompts import PromptTemplate
from pathlib import Path
import mlflow
from databricks.connect import DatabricksSession
from dotenv import dotenv_values


def ask_ai(inputs:str | dict, agent_config, context:str=None):
    # Update this to latest lanchain version
    qa_chain = LLMChain(
        llm=ChatDatabricks(endpoint=agent_config.endpoint, **agent_config.llm_parameters),
        prompt=agent_config.system_prompt
    )

    return qa_chain.run(inputs)



def ask_genie(question:str, space_id):
    from chaosllama.services.genie import GenieService
    """Ask Genie a question and get the response."""

    genie_manager = GenieService(space_id=space_id)

    message = genie_manager.start_conversation_and_wait_v2(question)
    message = genie_manager.poll_status(
        genie_manager.get_message_v2,
        message_id=message.message_id,
        conversation_id=message.conversation_id,
    )

    response = message.attachments[0].text.content

    return response

def get_spark_session():
    try:
        from dotenv import dotenv_values
        env = dotenv_values(".env")
        PROFILE = env["DATABRICKS_PROFILE"]
        spark = DatabricksSession.builder.profile(PROFILE).serverless(True).getOrCreate()
        print(spark)
    except Exception as e:
        print(f"Error with serverless: {e}")
        spark = DatabricksSession.builder.getOrCreate()
    return spark





if __name__ == "__main__":
    # Example usage
    message = ask_genie("Give me a list of 5 questions I can ask this genie room", space_id="01f05dd06c421ad6b522bf7a517cf6d2")
    print(message)

