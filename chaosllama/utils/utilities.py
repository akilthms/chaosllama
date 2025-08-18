from chaosllama.entities.models import AgentConfig
from langchain.chains.llm import LLMChain
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
)
from langchain.prompts import PromptTemplate
from pathlib import Path
import mlflow
from chaosllama.services.genie import GenieService
from chaosllama.profiles.config import config, RuntimeConfig

def ask_ai(inputs:str | dict, agent_config:AgentConfig, context:str=None):
    # Update this to latest lanchain version
    qa_chain = LLMChain(
        llm=ChatDatabricks(endpoint=agent_config.endpoint, **agent_config.llm_parameters),
        prompt=agent_config.system_prompt
    )

    return qa_chain.run(inputs)



def ask_genie(question:str, space_id):
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


def display_chaosllama_params(self, config: RuntimeConfig=config.runtime):
        # TODO clean this up
        banner = f"\n{'='*100}\n"
        print(
        f"""
        {banner} 
        🦙 Trigger Chaos LLama with parameters: 
            {config.EPOCHS=} 
            {config.LIMIT=}
            {config.CONSISTENCY_FACTOR=}
            {config.N_JOBS=}
            {config.BATCH_SIZE=}
            Genie Space Id: {self.genie_manager.space_id}
        {banner}
        """
        )




if __name__ == "__main__":
    # Example usage
    message = ask_genie("Give me a list of 5 questions I can ask this genie room", space_id="01f05dd06c421ad6b522bf7a517cf6d2")
    print(message)

