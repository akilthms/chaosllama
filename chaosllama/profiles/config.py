from pydantic import BaseModel
import yaml
import os

class MLFLowConfig(BaseModel):
    MLFLOW_RUNTIME_EXPERIMENT: str
    BEST_MLFLOW_RUN: str
    MLFLOW_EXPERIMENTS: list
    MLFLOW_EXPERIMENT_PATH: str

class GenieConfig(BaseModel):
    RUNTIME_GENIE_SPACE_ID: str
    NULL_HYPOTHESIS_GENIE_SPACE_ID: str
    TEST_GENIE_SPACE_ID: str
    BASELINE_GENIE_SPACE_ID: str
    VALIDATION_GENIE_SPACE_ID: str

class RuntimeConfig(BaseModel):
    MAX_TOKENS: int
    LIMIT: int
    CONSISTENCY_FACTOR: int
    EPOCHS: int
    BATCH_SIZE: int
    N_JOBS: int
    VALIDATION_SET_LIMIT: int
    REFRESH_DASHBOARD: bool
    RUN_BASELINE: bool
    RUN_NULL_HYPOTHESIS: bool
    INTROSPECTION_LOOKBACK: int
    DEBUG: bool
    IS_TRIGGERED_FROM_CHECKPOINT: bool
    INTROSPECT_AGENT_LLM_ENDPOINT: str
    BASELINE_CATALOG: str
    BASELINE_SCHEMA: str
    IS_CACHED: bool


class ScorerConfig(BaseModel):
    QUALITY_THRESHOLD: float
    METRICS_DEFINITION: str
    global_guidelines: dict

class Config(BaseModel):
    CATALOG: str
    SCHEMA: str
    SMALL_LLM_ENDPOINTS: str
    EVAL_TABLE_NAME: str
    mlflow: MLFLowConfig
    genie: GenieConfig
    runtime: RuntimeConfig
    scorers: ScorerConfig



with open("chaos_llama_config.yaml") as f:
    # TODO: Add logic to determine which mode to run chaos llama in (dev, prod, etc.)
    data = yaml.safe_load(f)
    config = Config(**data)

