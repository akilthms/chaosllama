from pathlib import Path
import marko
from typing import List, Callable
from chaosllama.profiles.config import config
import importlib
from mlflow.genai.judges import custom_prompt_judge, make_judge
from mlflow.genai.scorers import scorer



def load_function(module_name: str, func_name: str):
    # Import the module by string
    module = importlib.import_module(module_name)
    # Retrieve the function from the module
    func = getattr(module, func_name)
    return func

def extract_text(node):
    if hasattr(node, 'children'):
        return ''.join(extract_text(child) for child in node.children)
    return str(node)

class JudgeService():
    """
    The purpose of this class is to manage the judges that will be used to evaluate the AI generated responses.
    """
    def __init__(self, 
                 guidelines_path="./customer_guidelines.md", 
                 global_guidelines: dict=None, 
                 scorers=None):
        self.guidelines = None
        self.guidelines_path = guidelines_path
        self.global_guidelines = global_guidelines
        self.judges = []
        self.ai_judges = None
        self.scorers = scorers

    def load_guidelines(self, path: Path = None) -> str:
        self.guidelines = open(self.guidelines_path if self.guidelines_path else path, "r").read()
        return self

    def load_scorers_from_config(self):
        """ Load custom scorers from the configuration file. """
        MODULE = "chaosllama.scorers.scorers"
        if not self.scorers:
            self.scorers = []
            for scorer in config.scorers.custom_scorers:
                func = load_function(MODULE, scorer)
                self.scorers.append(func)
        return self


    def extract_text(self, node):
        if hasattr(node, 'children'):
            return ''.join(extract_text(child) for child in node.children)
        return str(node)

    def parse_guidelines(self, markdown: str = None):
        if not self.guidelines:
            self.load_guidelines()

        else:
            parsed = marko.parse(self.guidelines)
            result = {}
            current_heading = None
            for node in parsed.children:
                if isinstance(node, Heading):
                    current_heading = extract_text(node).strip(':')  # remove trailing colon
                    result[current_heading] = []
                elif isinstance(node, List) and current_heading:
                    for item in node.children:
                        bullet = extract_text(item).strip()
                        result[current_heading].append(bullet)

            self.guidelines = result
        return self, result

    # def load_judges_from_config(self) -> List[Callable]:
    #     judges_config:dict = config.judges

    #     for name, criteria in judges_config.items():

    #         judge = custom_prompt_judge(name=name, prompt_template=criteria)

    #         @scorer(name=name)
    #         def _make_prompt_judge(inputs: dict, outputs: dict, expectations: Optional[dict[str, Any]]):
    #             return judge(
    #                 outputs=outputs,
    #                 expectation=expectations.get("expected_response")
    #         )

    #         self.judges.append(_make_prompt_judge)

    #     return self.judges

    @property
    def judge_template(self):

        JUDGE_TEMPLATE = ("""
        [Task Description]
        You are an expert Databricks SQL analyst. Your task is to judge whether the criteria is met or not between the output and the expected_response.
        expected_response is the ground source of truth

        [Guideline]
        Guideline:{guideline}

        [Context]
        Output: {{outputs}}
        Expected Response: {{expected_response}}

        [Choices]
        [[True]]: Passes the aforementioned guideline
        [[False]]: Does not pass the aforemented guideline
        

        """)
        
        return JUDGE_TEMPLATE

    def load_judges_from_config(self) -> List[Callable]:
        
        judges_config:dict = config.scorers.global_guidelines.get(config.runtime.JUDGE_VERSION)
        for name, guideline in judges_config.items():
            #judge = custom_prompt_judge(name=name, prompt_template=criteria)
            judge = make_judge(
                name=name,
                instructions=self.judge_template.format(guideline=guideline),
                model=config.runtime.JUDGE_ENDPOINT
            )

            self.judges.append(judge)

        return self.judges















        