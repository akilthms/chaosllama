from chaosllama.services import genie, mosaic, unity_catalog
from chaosllama.entities.models import AgentConfig, IntrospectionManager
from dataclasses import dataclass



@dataclass
class ChaosLlamaServicesConfig:
    mlflow_manager: mosaic.MosaicEvalService
    genie_manager: genie.GenieService
    uc_manager: unity_catalog.UCService
    agent_config: AgentConfig


class ChaosLlama():
    """
    Chaos Llama is a novel framework for testing and improving the accuracy of Text-to-SQL systems. It
        uses large language models (LLMs) not just for generation, but for judging correctness, analyzing
        errors, and evolving system prompts through introspective learning.
    """

    def __init__(self, config: ChaosLlamaServicesConfig = None):
        self.mlflow_manager = config.mlflow_manager
        self.genie_manager = config.genie_manager
        self.uc_manager = config.uc_manager
        self.agent_config = config.agent_config


    def run(self, epochs=1, is_test=True, limit=None, is_cached=True, run_baseline=False) -> IntrospectionManager:
        pass