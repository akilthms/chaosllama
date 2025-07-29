import pandas as pd
from chaosllama.services.genie import GenieService
from chaosllama.services.judges import JudgeService
from chaosllama.entities.models import EvalSetTable, IntrospectionManager
import mlflow
from mlflow.entities import SpanType
from mlflow.genai.scorers import Correctness, RelevanceToQuery, ExpectationsGuidelines


class MosaicEvalService():
    """ The purpose of this class to manage the evaluation data """
    def __init__(self, eval_set:EvalSetTable, judge_manager: JudgeService=None, genie_manager:GenieService=None,validation_set: pd.DataFrame=None, experiment_id:str=None, experiment_name:str=None):
        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.eval_set = eval_set
        self.judge_manager = judge_manager
        self.validation_set = validation_set
        self.genie_manager = genie_manager

    @staticmethod
    def _prepare_inputs(self, eval_set: EvalSetTable):
        records = eval_set.data.toPandas().to_dict(orient='records')

        eval_data = []
        for rec in records:
            inputs = dict(inputs=dict(inputs={"question": rec["question"]}),
                          expectations=dict(guidelines=[rec["issues"]],
                                            expected_response=rec["ground_truth_query"])
                          )
            eval_data.append(inputs)

        return eval_data

    @mlflow.trace(name="🧪 Mosaic Evaluation WorkFlow", span_type=SpanType.CHAIN)
    def run_evaluations(self, genie_space_id, timeout=1, validation_set=None) -> IntrospectionManager:
        """ The purpose of this function is to ingest the evaluation dataset and produce a set of telemetry data that can be used to for the IntrospectionAI"""
        genie_manager = GenieService(space_id=genie_space_id, should_reply=True)
        intrsmg = IntrospectionManager()

        genie_agent = GenieAgent(space_id=DBRX_CPGPT_GENIE_SPACE_ID_WITH_DATA_SAMPLING)
        global_guidelines = GLOBAL_GUIDELINES_v2["global_guidelines"]
        guidelines = [Guidelines(name=name, guidelines=g[0]) for name, g in global_guidelines.items()]

        scorers: list[Callable] = self.judge_manager.scorers
        eval_dataset = self._prepare_inputs(self.eval_set)

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

        self.log_metadata(completed_assessment.metrics)

        return intrsmg