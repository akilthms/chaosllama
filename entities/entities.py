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
class EvalSetTable(ChaosLlamaTable):
    name: str = "eval_set"
    data: pyspark.sql.DataFrame = None
    question: str = field(default_factory=str)
    ground_truth_query: str = field(default_factory=str)
    inputs: dict = field(default_factory=dict)
    issues: str = field(default_factory=str)

    def get_questions(self, limit=None):
        return self.data.toPandas()[["question", "original_question"]].values

    def update_ground_truth(self, replace_terms: dict) -> pyspark.sql.DataFrame:
        for source, target in replace_terms.items():
            regex_replace = F.regexp_replace("ground_truth_query", source, target)
            self.data = self.data.withColumn("updated_ground_truth", regex_replace)
        return self

    def limit(self, limit=None) -> None:
        if limit:
            self.data = self.data.limit(limit)
        return self

    def replicate_rows(self, consistency_factor: int = 1) -> pyspark.sql.DataFrame:
        """
        Duplicate each row in the given DataFrame `n` times.

        Parameters:
            df (DataFrame): Input PySpark DataFrame
            n (int): Number of times to duplicate each row

        Returns:
            DataFrame: A new DataFrame with each row duplicated `n` times
        """
        self.data = self.data.withColumn("dup", F.explode(F.array([F.lit(i) for i in range(consistency_factor)]))).drop(
            "dup")
        return self


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


# @dataclass
# class DataIntelligence:
#     telemtry: field(default_factory = list[GenieTelemtry])
#     instrospections: field(default_factory = list[dict])
#     metadata_suggestions: field(default_factory = list[AgentSuggestion])

#     def add_telemtry(self, telemtry: GenieTelemtry):
#         self.telemtry.append(telemtry)

#     def add_introspection(self, introspection: dict):
#         self.instrospections.append(introspection)

#     def add_ai_suggestion(self, suggestion: AgentSuggestion):
#         self.metadata_suggestions.append(suggestion)

#     def get_latest_ai_prompt(self):
#         sys_prompt = self.metadata_suggestions[-1] if self.metadata_suggestions else None
#         return sys_prompt

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
