from chaosllama.services import genie, mosaic, unity_catalog
# from chaosllama.entities.models import AgentConfig, IntrospectionManager
from chaosllama.services.introspection import IntrospectionAIAgent
from dataclasses import dataclass
import mlflow
from mlflow.entities import SpanType
from abc import ABC

from chaosllama.entities.models import AgentConfig, IntrospectionManager

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

    @mlflow.trace(name="🦙ChaosLlama App", span_type=SpanType.CHAIN)
    def run(self, epochs=1, is_test=True, limit=None, is_cached=True, run_baseline=False) -> IntrospectionManager:

        ucmg = self.uc_manager
        mlfmg = self.mlflow_manager
        limit = limit if is_test else None

        # 💰 Cache
        if is_cached: mlfmg.eval_set.data.cache().count()
        questions = mlfmg.eval_set.get_questions()

        with mlflow.start_run() as parent_run:
            mlfmg.eval_set.data = mlfmg.eval_set.data.withColumn("original_question", F.col("question"))
            if run_baseline:
                # mlflow_eval_logger.info(f"🏃 Running CockpitGPT Baseline Test w/ Data Sampling")
                mlfmg.create_baseline_experiment_run(run_name="1 - 🚀 CockpitGPT w/ Data Sampling Baseline Test",
                                                     genie_space_id=CPGPT_GENIE_SPACE_ID_WITH_DATA_SAMPLING,
                                                     parent_run_id=test_suite.info.run_id)

            instrmg = IntrospectionManager()
            # 🕵 Initialize Agent
            agent = IntrospectionAIAgent(self.agent_config)
            introspective_data = []
            # Manages All IntrospectiveManagers
            introspection_director = IntrospectionManager()

            # Trigger Chaos LLama run from prexisting runs
            # [TODO]: create a function in Primer class called clone_runs
            if IS_TRIGGERED_FROM_CHECKPOINT:
                primer = PrimerManager(mlflow_parent_run_id=BEST_MLFLOW_RUNS_MAP.get(PARENT_RUN_NAME))
                primer.run_introspection()

            for i in range(epochs):
                with mlflow.start_run(
                        run_name=f"🔄 Optimization Cycle - {i + 1}",
                        nested=True,
                        parent_run_id=parent_run.info.run_id) as optimization_run:

                    # print(introspection_director.metadata_suggestions)

                    if introspection_director.metadata_suggestions:
                        # TODO: convert this logic from spark to pandas
                        simulate_system_instruction_update = (
                            F.concat(
                                F.lit(introspection_director.metadata_suggestions[-1].ai_system_instruction),
                                F.lit("\n"),
                                F.col("original_question")
                            )
                        )

                        mlfmg.eval_set.data = (
                            mlfmg.eval_set
                            .data
                            .withColumn("question", simulate_system_instruction_update)
                        )

                    print(f"{'=' * 10} 🖥️ Displaying 🔄 Cycle {i + 1} Eval Data {"=" * 10}")

                    # Get Data Intelligence from evaluation run
                    instrmg: IntrospectionManager = mlfmg.run_evaluations_v2(self.genie_manager.space_id)

                    # Add data intelligence from all managers to director
                    introspection_director.add_data_intelligence(instrmg.feedback)
                    introspection_director.add_overall_quality_score(instrmg.overall_quality_score)
                    introspection_director.optimization_id = i

                    # TODO: Implement mlflow.validate_evaluation_results() for baseline testing

                    reflection_data = AgentInput_v3(
                        data_intelligence=introspection_director.data_intelligence[-INTROSPECTION_LOOKBACK:],
                        overall_quality_score=introspection_director.overall_quality_score[-INTROSPECTION_LOOKBACK:],
                        system_instructions_history=introspection_director.metadata_suggestions[
                                                    -INTROSPECTION_LOOKBACK:],
                        optimization_id=introspection_director.optimization_id
                    )

                    ai_suggestion = agent.introspect(reflection_data)

                    instrmg.add_ai_suggestion(ai_suggestion)
                    introspection_director.add_ai_suggestion(ai_suggestion)

                    # 📝 MLFlow Logging
                    mlflow.log_param("ai_system_instruction", ai_suggestion.ai_system_instruction)
                    mlflow.log_param("introspection ai prompt", chaos_config.agent_config.system_prompt)
                    mlflow.log_param("epochs", epochs)
                    mlflow.log_param("introspection ai agent", INTROSPECT_AGENT_LLM_ENDPOINT)
                    mlflow.log_param("max tokens", MAX_TOKENS)
                    mlflow.log_metric("step", i)

                    introspective_data.append(instrmg)

                    # RUN VALIDATION SET ✅
            # mlfmg.run_validation_set()

        return introspective_data, parent_run