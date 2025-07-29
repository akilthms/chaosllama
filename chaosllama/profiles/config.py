from pydantic import BaseModel
import yaml

class Config(BaseModel):
    CATALOG: str
    SCHEMA: str


with open("chaos_llama_config.yaml") as f:
    # TODO: Add logic to determine which mode to run chaos llama in (dev, prod, etc.)
    data = yaml.safe_load(f)
    config = Config(**data)

