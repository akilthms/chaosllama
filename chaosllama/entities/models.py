from dataclasses import dataclass, field, fields, asdict
from typing import Optional, Literal, Self
from dotenv import dotenv_values
from databricks.connect import DatabricksSession
import pyspark
from chaosllama.profiles.config import config
from pyspark.sql import functions as F
from pathlib import Path

env = dotenv_values(".env")
PROFILE = env["DATABRICKS_PROFILE"]
spark = DatabricksSession.builder.profile(PROFILE).serverless(True).getOrCreate()

CATALOG = config.CATALOG
SCHEMA = config.SCHEMA

# ================ ChaosLlama Unity Catalog Tables ====================
@dataclass
class ChaosLlamaTable():
    name: str = field(default_factory=str)
    data: pyspark.sql.DataFrame = None
    catalog: str = CATALOG
    schema: str = SCHEMA

    def to_spark_row(self):
        attributes = list(self.__dataclass_fields__.keys())
        attributes = [attribute for attribute in attributes if attribute not in ["data", "name", "catalog", "schema"]]
        data = [{attribute: getattr(self, attribute) for attribute in attributes}]
        return spark.createDataFrame(data, schema=attributes)

    def drop_table(self):
        spark.sql(f"DROP TABLE IF EXISTS {self.catalog}.{self.schema}.{self.name}")
        print(f"🫳 table: {self.catalog}.{self.schema}.{self.name}")

    def save(self, opts: dict = None, mode="append"):
        df = self.to_spark()
        (
            df.write
            .option("mergeSchema", True)
            .option("delta.enableChangeDataFeed", True)
            .saveAsTable(name=f"{self.catalog}.{self.schema}.{self.name}",
                         options=opts,
                         mode=mode)
        )
        return self

    def load(self):
        self.data = spark.table(f"{self.catalog}.{self.schema}.{self.name}")
        return self


class PromptsTable(ChaosLlamaTable):
    name: str = "prompts"
    prompt: str = field(default_factory=str)
    creation_ts = datetime.now()

    def load_prompts(self):
        pass




@dataclass
class SystemInstructionsHistoryTable(ChaosLlamaTable):
    name: str = "system_history"
    suggested_change_type: Literal["systems_instructions", "column_description"] = field(default_factory=str)
    updated_metadata: str = field(default_factory=str)
    data_intelligence: list = field(default_factory=list)  # TODO: Replace with JudgeMetrics
    creation_ts: datetime.timestamp = field(default_factory=datetime.timestamp)


# =====================================================================


# ===================== AI Data Model  ================================
@dataclass
class SuggestionUpdate:
    field_name: str
    metadata_type: Literal["system_instructions", "column_description"]
    updated_value: str
    updated_ts: datetime.timestamp


@dataclass
class AgentConfig:
    system_prompt: PromptTemplate
    endpoint: str
    llm_parameters: dict = field(default_factory=lambda: {"temperature": 0.00})


@dataclass
class SuggestionUpdate:
    field_name: str
    metadata_type: Literal["system_instructions", "column_description"]
    updated_value: str
    updated_ts: datetime.timestamp


@dataclass
class AgentSuggestion:
    content: str
    update_suggestion: SuggestionUpdate
    creation_tm: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentInput:
    data_intelligence: list
    prev_overall_quality_score: float
    drift_metrics: list = field(default_factory=dict)
    previous_data_intelligence: dict = field(default_factory=dict)
    metrics_definition: str = METRICS_DEFINITION
    quality_threshold: float = QUALITY_THRESHOLD
    previous_suggestion: str = field(default_factory=str)
    optional_column_modifications: str = None


@dataclass
class AgentInput_v2:
    previous_prompt: str
    previous_overall_quality_score: dict
    current_overall_quality_score: dict
    data_intelligence: list = field(default_factory=list)
    previous_data_intelligence: dict = field(default_factory=list)
    quality_threshold: float = QUALITY_THRESHOLD


@dataclass
class AgentInput_v3:
    data_intelligence: list = field(default_factory=list)
    overall_quality_score: list = field(default_factory=list)
    system_instructions_history: list = field(default_factory=list)
    quality_threshold: float = QUALITY_THRESHOLD
    optimization_id: int = field(default_factory=int)


@dataclass
class MosaicAssessment:
    name: str = field(default_factory=str)
    expectation: str = field(default_factory=str)
    feedback: str = field(default_factory=str)
    rationale: str = field(default_factory=str)
    metadata: dict = field(default_factory=dict)
    root_cause_assessment: str = field(default_factory=str)
    root_cause_rationale: str = field(default_factory=str)


@dataclass
class DataIntelligence:
    question: str
    genie_generated_query: str
    ground_truth_query: str
    genie_generated_sql_thought_process_description: str
    mosaic_evaluation: list[MosaicAssessment]


@dataclass
class GenieTelemetry:
    """ Container for MLFlow Telemtry """
    conversation_id: str
    statement_id: str
    genie_query: str
    original_question: str
    row_count: int
    genie_generated_sql_thought_process_description: str  # describes the thought process of Genie
    created_timestamp: datetime.timestamp
    query_result_metadata: dict
    genie_question: str = field(default_factory=str)
    space_id: str = GENIE_SPACE_ID


# =================== AI Data Model  ================================
@dataclass
class IntrospectionManager:
    genie_telemetry: list[GenieTelemetry] = field(default_factory=list)
    introspections: list[dict] = field(default_factory=list)
    metadata_suggestions: list[AgentSuggestion] = field(default_factory=list)
    data_intelligence: list[dict] = field(default_factory=list)
    feedback: list[Feedback] = field(default_factory=list)
    overall_quality_score: list[dict] = field(default_factory=list)
    optimization_id: int = field(default_factory=int)

    def add_genie_telemetry(self, telemetry: GenieTelemetry):
        self.genie_telemetry.append(telemetry)
        return self

    def add_feedback(self, feedback: Feedback):
        self.feedback.append(feedback)
        return self

    def add_introspection(self, introspection: dict):
        self.instrospections.append(introspection)
        return self

    def add_ai_suggestion(self, suggestion: AgentSuggestion):
        self.metadata_suggestions.append(suggestion)
        return self

    def add_data_intelligence(self, data_intelligence: list[DataIntelligence]):
        self.data_intelligence.extend(data_intelligence)
        return self

    def add_overall_quality_score(self, score: dict):
        self.overall_quality_score.append(score)

    def get_prev_quality_score(self):
        try:
            return self.overall_quality_score[-2]
        except IndexError:
            return None

    def get_curr_quality_score(self):
        return self.overall_quality_score[-1]

    def get_prev_ai_prompt(self):
        try:
            if len(self.metadata_suggestions) > 2:
                return self.metadata_suggestions[-2]
            else:
                return self.metadata_suggestions[-1]
        except IndexError:
            return None

    def get_curr_ai_prompt(self):
        sys_prompt = self.metadata_suggestions[-1] if self.metadata_suggestions else None
        return sys_prompt

    def get_prev_intelligence(self):
        try:
            return self.data_intelligence[-2]
        except IndexError:
            return None

    def get_curr_intelligence(self):
        return self.data_intelligence[-1] if self.data_intelligence else None

    def as_dict(self):
        return asdict(self)


# ====================================================================


# ================= 👨‍⚖️ Custom AI Judges ==============================


@dataclass
class JudgeTable(ChaosLlamaTable):
    judge_name: str = field(default_factory=str)
    definition: str = field(default_factory=str)
    grading_prompt: str = field(default_factory=str)
    creation_dt: datetime = field(default_factory=datetime.utcnow)


@dataclass
class JudgeMetric():
    def from_dict(self, d):
        for k, v in d.items():
            setattr(self, k, v)
        return self


@dataclass
class EvalModel:
    question: str
    genie_repsonse: str
    ground_truth: str


# DATA Model For changes to the column descriptions / table descriptions
@dataclass
class ColumnChange:
    column_name: str
    new_comment: str


@dataclass
class DDLHistoryTable(ChaosLlamaTable):
    name: str = "ddl_history"
    col_name: str = field(default_factory=str)
    data_type: str = field(default_factory=str)
    comment: str = field(default_factory=str)
    table_name: str = field(default_factory=str)

@dataclass
class ChaosLlamaConfig:
    mlflow_manager: MLFlowEvalManager
    genie_manager: GenieManager
    uc_manager: UCManager
    agent_config: AgentConfig


@dataclass
class IntrospectionManager:
    introspections: list[dict] = field(default_factory=list)
    metadata_suggestions: list[AgentSuggestion] = field(default_factory=list)
    feedback: list[Feedback] = field(default_factory=list)
    overall_quality_score: list[dict] = field(default_factory=list)
    optimization_id: int = field(default_factory=int)

    def add_feedback(self, feedback: Feedback):
        self.feedback.append(feedback)
        return self

    def add_introspection(self, introspection: dict):
        self.instrospections.append(introspection)
        return self

    def add_ai_suggestion(self, suggestion: AgentSuggestion):
        self.metadata_suggestions.append(suggestion)
        return self

    def add_overall_quality_score(self, score: dict):
        self.overall_quality_score.append(score)

    def get_prev_quality_score(self):
        try:
            return self.overall_quality_score[-2]
        except IndexError:
            return None

    def get_curr_quality_score(self):
        return self.overall_quality_score[-1]

    def get_prev_ai_prompt(self):
        try:
            if len(self.metadata_suggestions) > 2:
                return self.metadata_suggestions[-2]
            else:
                return self.metadata_suggestions[-1]
        except IndexError:
            return None

    def get_curr_ai_prompt(self):
        sys_prompt = self.metadata_suggestions[-1] if self.metadata_suggestions else None
        return sys_prompt

    def get_prev_feedback(self):
        try:
            return self.get_prev_feedback[-2]
        except IndexError:
            return None

    def get_curr_feedback(self):
        return self.feedback[-1] if self.feedback else None

    def as_dict(self):
        return asdict(self)


# Evaluation Data Set Models
@dataclass
class EvalSetTable(ChaosLlamaTable):
    name: str = "eval_set"
    data: pyspark.sql.DataFrame = None
    question: str = field(default_factory=str)
    ground_truth_query: str = field(default_factory=str)
    inputs: dict = field(default_factory=dict)
    issues: str = field(default_factory=str)

    def get_questions(self, limit=None):
        return self.data.toPandas()[["question", "original_question"]].values

    def update_ground_truth(self, replace_terms: dict) -> Self:
        for source, target in replace_terms.items():
            regex_replace = F.regexp_replace("ground_truth_query", source, target)
            self.data = self.data.withColumn("updated_ground_truth", regex_replace)
        return self

    def limit(self, limit=None) -> Self:
        if limit:
            self.data = self.data.limit(limit)
        return self

    def replicate_rows(self, consistency_factor: int = 1) -> Self:
        """
        Duplicate each row in the given DataFrame `n` times.

        Parameters:
            df (DataFrame): Input PySpark DataFrame
            n (int): Number of times to duplicate each row

        Returns:
            DataFrame: A new DataFrame with each row duplicated `n` times
        """
        dup_logic =  F.explode(F.array([F.lit(i) for i in range(consistency_factor)]))
        self.data = (
                        self.data
                            .withColumn("dup", dup_logic)
                            .drop("dup")
                     )
        return self


class EvalSetManager:
    def __init__(self,limit=None, consistency_factor= None, eval_set: Optional[EvalSetTable | str] = None):
        self.eval_set = eval_set
        self.limit = limit
        self.consistency_factor = consistency_factor

    def create_evalset(self):
        """ Create the evaluation set for the optimization run, based on the provided configuration. """
        if isinstance(self.eval_set, str) and self.eval_set.count("." ) == 2:
            self.eval_set = spark.table(self.eval_set)

        elif isinstance(self.eval_set, str) and self.eval_set.count("." ) != 2:
            raise ValueError("eval_set must be a valid unity catalog 3 namespace scheme")

        else:
            self.eval_set = EvalSetTable()


        return self

    def prepare_evals(self):
        print("📐 Evaluation Dataset....")
        if not self.eval_set:
            self.create_evalset()

        evaluation_dataset = (
            self.eval_set
                .limit(self.limit)
                .replicate_rows(self.consistency_factor)
        )

        evaluation_dataset.data.show()
        return evaluation_dataset



