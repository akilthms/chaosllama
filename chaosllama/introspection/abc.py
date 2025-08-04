from chaosllama.entities.models import AgentConfig
from abc import ABC
import pandas as pd
from typing import Optional, Union

class InstrospectiveAI(ABC):
    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config

    def instrospect(self, data_intellegence: pd.DataFrame):
        pass

    def optimize(self,
                 agent_input,
                 mode="system_instructions") -> Optional[Union[str, pd.DataFrame]]:
        pass