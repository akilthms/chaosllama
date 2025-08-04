from typing import Literal, Callable, Tuple

import pandas as pd
from chaosllama.services.genie import GenieService, GenieAgent
from chaosllama.services.judges import JudgeService
from chaosllama.services.evaluation_dataset import EvalSetManager
from chaosllama.entities.models import EvalSetTable, IntrospectionManager
import mlflow
from mlflow.entities import SpanType, Feedback
from mlflow.genai.scorers import Correctness, RelevanceToQuery, ExpectationsGuidelines, Guidelines
from databricks.sdk import WorkspaceClient
from chaosllama.profiles.config import config
from functools import reduce

GLOBAL_GUIDELINES = config.scorers.global_guidelines["v2"]


class MosaicEvalService():
    """ The purpose of this class to manage the evaluation data """
    def __init__(self, eval_manager: EvalSetManager, judge_manager: JudgeService=None, genie_manager:GenieService=None,validation_set: pd.DataFrame=None, experiment_id:str=None, experiment_name:str=None):
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.eval_manager = eval_manager
        self.judge_manager = judge_manager
        self.validation_set = validation_set
        self.genie_manager = genie_manager
        self._w = WorkspaceClient()

    @staticmethod
    def _prepare_inputs(eval_set: EvalSetTable):
        records = eval_set.data.toPandas().to_dict(orient='records')

        eval_data = []
        for rec in records:
            inputs = dict(inputs=dict(inputs={"question": rec["question"]}),
                          expectations=dict(guidelines=[rec["issues"]],
                                            expected_response=rec["ground_truth_query"])
                          )
            eval_data.append(inputs)

        return eval_data

    def get_feedback(self, assessment: mlflow.models.evaluation.base.EvaluationResult) -> list[Feedback]:
        """ The purpose of this function is to ingest the evaluation dataset and produce a set of telemetry data that can be used to for the IntrospectionAI """
        traces = mlflow.search_traces(experiment_ids=[self.experiment_id], run_id=assessment.run_id)
        feedback = reduce(lambda x, y: x + y, traces["assessments"].to_list())
        return feedback

    @mlflow.trace(name="🧪 Mosaic Evaluation WorkFlow", span_type=SpanType.CHAIN)
    def run_evaluations(self, genie_space_id, timeout=1, validation_set=None) -> IntrospectionManager:
        """ The purpose of this function is to ingest the evaluation dataset and produce a set of telemetry data that can be used to for the IntrospectionAI"""
        genie_manager = GenieService(space_id=genie_space_id, should_reply=True)
        intrsmg = IntrospectionManager()

        genie_agent = GenieAgent(space_id=genie_space_id)
        global_guidelines = GLOBAL_GUIDELINES
        guidelines = [Guidelines(name=name, guidelines=g[0]) for name, g in global_guidelines.items()]

        scorers: list[Callable] = self.judge_manager.scorers
        print("🥅",scorers)
        eval_dataset = self._prepare_inputs(self.eval_manager.eval_set)

        completed_assessment = mlflow.genai.evaluate(
            data=eval_dataset,
            predict_fn=genie_agent.invoke,
            scorers=[
                *scorers,
                *guidelines,
                Correctness(),
                RelevanceToQuery(),
                ExpectationsGuidelines(),
            ]
        )

        intrsmg.add_feedback(self.get_feedback(completed_assessment)) \
            .add_overall_quality_score(completed_assessment.metrics)

        self.log_metadata(completed_assessment)

        return intrsmg

    def create_experiment_run(
            self,
            introspection_director: IntrospectionManager=None,
            parent_run_id=None,
            experiment_id:str=None,
            mode: Literal["null_hypothesis", "baseline", "validation", "optimization"] = None,
            optimization_id:int=None
    ) -> Tuple[IntrospectionManager, mlflow.entities.Run]:
        """
        The purpose of this function is to create an experiment run in MLFlow for the Mosaic Evaluation Service.
        :param introspection_director:
        :param parent_run_id:
        :param mode:
        :return:
        """

        match mode:
            case "null_hypothesis":
                genie_space_name = self._w.genie.get_space(config.genie.BASELINE_GENIE_SPACE_ID).title
                genie_space_id = config.genie.BASELINE_GENIE_SPACE_ID
                run_name = f"0 - 🫙 Null Hypothesis Test - {genie_space_name}"
                experiment_type = "null_hypothesis_test"


            case "baseline":
                genie_space_name = self._w.genie.get_space(config.genie.BASELINE_GENIE_SPACE_ID).title
                genie_space_id = config.genie.BASELINE_GENIE_SPACE_ID
                run_name = f"0 - ⛺️ Baseline Test - {genie_space_name}"
                experiment_type = "baseline_test"


            case "validation":
                genie_space_name = self._w.genie.get_space(config.genie.VALIDATION_GENIE_SPACE_ID).title
                genie_space_id = config.genie.VALIDATION_GENIE_SPACE_ID
                run_name = f"0 - ✅ Validation Test - {genie_space_name}"
                experiment_type = "validation_test"

            case "optimization":
                genie_space_id = config.genie.RUNTIME_GENIE_SPACE_ID
                run_name = f" 🔄 Optimization Cycle {optimization_id}"
                experiment_type = "optimization"

            case _:
                raise ValueError("Invalid mode. Please choose 'null_hypothesis' or 'baseline'.")



        with mlflow.start_run(run_name=run_name,
                              nested=True,
                              parent_run_id=parent_run_id,
                              experiment_id=experiment_id) as experiment_run:
            mlflow.set_tag(experiment_type, True)
            instrmg = self.run_evaluations(genie_space_id)
            mlflow.log_param("step", optimization_id)

        return instrmg, experiment_run

    def log_metadata(self, assessment:mlflow.models.EvaluationResult) -> None:
      """ Log all the metadata for the evaluation run """
      metadata = assessment.metrics
      for k,v in metadata.items():
        # rename metric name
        metric_name = "avg_" + "_".join(k.split("/")[:-1]).replace("/", "_")
        mlflow.log_metric(metric_name,v)