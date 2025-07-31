from pydantic import BaseModel
import yaml
import os

class Config(BaseModel):
    CATALOG: str
    SCHEMA: str
    SMALL_LLM_ENDPOINTS: str
    BATCH_SIZE: int
    QUALITY_THRESHOLD: float
    EVAL_TABLE_NAME: str
    LIMIT: int
    CONSISTENCY_FACTOR: int


with open("chaos_llama_config.yaml") as f:
    # TODO: Add logic to determine which mode to run chaos llama in (dev, prod, etc.)
    data = yaml.safe_load(f)
    config = Config(**data)

