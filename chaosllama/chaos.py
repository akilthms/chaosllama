from chaosllama.services import genie, mosaic, unity_catalog
from typing import Tuple
# from chaosllama.entities.models import AgentConfig, IntrospectionManager
from chaosllama.services.introspection import IntrospectionAIAgent
from dataclasses import dataclass
import mlflow
from mlflow.entities import SpanType
from abc import ABC
from chaosllama.profiles.config import config
from pyspark.sql import functions as F
from chaosllama.entities.models import AgentConfig, IntrospectionManager, AgentInput


@dataclass
class ChaosLlamaServicesConfig:
    mlflow_manager: mosaic.MosaicEvalService
    genie_manager: genie.GenieService
    uc_manager: unity_catalog.UCService
    agent_config: AgentConfig


class ChaosLlamaABC(ABC):
    def __init__(self, config: ChaosLlamaServicesConfig = None):
        self.config = config
        self._configure()

    def _configure(self,):
        self.mlflow_manager = self.config.mlflow_manager
        self.genie_manager = self.config.genie_manager
        self.uc_manager = self.config.uc_manager
        self.agent_config = self.config.agent_config

    def baseline_evaluation(self, enable_baseline_test: bool = False):
        """
        This method is used to run the baseline evaluation test.
        :param enable_baseline_test: If True, it will run the baseline evaluation test.
        :return: None
        """
        raise NotImplemented()

    def introspect_from_checkpoint(self, enable_checkpoint: bool = False,checkpoint_run_id: str=None, ):
        raise NotImplemented()

    def optimize(self, feedbacks) -> IntrospectionManager:
        """
        This method is used to optimize the system by running the introspection agent.
        :return: IntrospectionManager
        """
        raise NotImplemented()

    def update_dashboard(self, introspection_manager: IntrospectionManager):
        """
        This method is used to update the dashboard with the introspection manager data.
        :param introspection_manager: IntrospectionManager
        :return: None
        """
        raise NotImplemented()

    def run(self, runtime_config: dict) -> (IntrospectionManager, str):
        """
        :param runtime_config:
        :return:
        """
        self.baseline_evaluation(enable_baseline_test=runtime_config["enable_baseline_test"])

        feedbacks = self.introspect_from_checkpoint(enable_checkpoint=runtime_config["enable_baseline_test"])
        introspection_director, mlflow_parent_run = self.optimize(feedbacks)

        self.update_dashboard(introspection_director)

        return introspection_director, mlflow_parent_run




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

    @mlflow.trace(name="🦙ChaosLlama App",span_type=SpanType.CHAIN)
    def run(self,
            epochs=1,
            is_test=True,
            limit=None,
            is_cached=True,
            run_baseline=False,
            run_null_hypothesis=False) -> Tuple[IntrospectionManager, mlflow.entities.Run]:
        """ The main entry point for running the ChaosLlama framework."""
        mlfmg = self.mlflow_manager
        # primer = PrimerManager(mlflow_parent_run_id=BEST_MLFLOW_RUNS_MAP.get(PARENT_RUN_NAME))
        # 🕵 Initialize Agent
        intropsective_agent = IntrospectionAIAgent(self.agent_config)

        # 💰 Cache
        if is_cached: mlfmg.eval_set.data.cache().count()

        with mlflow.start_run(experiment_id=mlfmg.experiment_id) as parent_run:
            mlfmg.eval_manager.eval_set.data = mlfmg.eval_manager.eval_set.data.withColumn("original_question", F.col("question"))

            if run_baseline: mlfmg.create_experiment_run(parent_run_id=parent_run.run_id, mode="baseline")
            if run_null_hypothesis: mlfmg.create_experiment_run(parent_run_id=parent_run.run_id, mode="null_hypothesis")

            # [TODO]: create a function in Primer class called clone_runs
            if config.runtime.IS_TRIGGERED_FROM_CHECKPOINT:
                raise NotImplementedError("ChaosLlama run from checkpoint is not implemented yet.")
                # primer.run_introspection() # Trigger Chaos LLama run from prexisting runs

            introspective_data = []
            introspection_director = IntrospectionManager() # Manages All IntrospectiveManagers
            for i in range(epochs):
                if introspection_director.metadata_suggestions:
                    ai_system_instruction = introspection_director.metadata_suggestions[-1].ai_system_instruction
                    mlfmg.eval_manager.simulate_system_instruction_update(ai_system_instruction)

                print(f"{'=' * 10} 🖥️ Displaying 🔄 Cycle {i + 1} Eval Data {"=" * 10}")
                introspection_director, exp_run = mlfmg.create_experiment_run(introspection_director,
                                                                        parent_run_id=parent_run.info.run_id,
                                                                        experiment_id=mlfmg.experiment_id,
                                                                        mode="optimization",
                                                                        optimization_id=i+1)

                lookback = config.runtime.INTROSPECTION_LOOKBACK
                reflection_data = AgentInput(
                    quality_threshold=config.scorers.QUALITY_THRESHOLD,
                    data_intelligence=introspection_director.feedback[-lookback:],
                    overall_quality_score=introspection_director.overall_quality_score[-lookback:],
                    system_instructions_history=introspection_director.metadata_suggestions[-lookback:],
                    optimization_id=introspection_director.optimization_id
                )

                # 🤖🎤 AI Suggestion as a result of introspection
                ai_suggestion = intropsective_agent.introspect(reflection_data)
                introspection_director.add_ai_suggestion(ai_suggestion)

                # 📝 MLFlow Logging of Optimization Loop tags
                # with mlflow.start_run(experiment_id=parent_run.info.experiment_id, run_id=exp_run.info.run_id):
                with mlflow.start_run(nested=True,
                                      run_id=exp_run.info.run_id,
                                      experiment_id=exp_run.info.experiment_id) as optimization_run:

                    mlflow.log_params(
                        {
                        "ai_system_instruction": ai_suggestion.ai_system_instruction,
                        "introspection ai prompt": self.agent_config.system_prompt,
                        "epochs": epochs,
                        "introspection ai agent": config.runtime.INTROSPECT_AGENT_LLM_ENDPOINT,
                        "max tokens": config.runtime.MAX_TOKENS
                        }
                    )

                #

            # RUN VALIDATION SET ✅
            # mlfmg.run_validation_set()

        return introspection_director, parent_run