import pytest
import pandas as pd
import yaml
from chaosllama.services.evaluation_dataset import EvalSetManager, EvalSetTable
from chaosllama.services.genie import GenieService
from chaosllama.profiles.config import config
import mlflow
import os

from chaosllama.services.tracking import MLFlowExperimentManager

pd.set_option('display.width', pd.get_option('display.width') * 2)
pd.set_option('display.max_columns', None)


mlf_mngr = MLFlowExperimentManager(experiment_path=config.mlflow.MLFLOW_EXPERIMENT_PATH).get_or_create_mlflow_experiment("TestChaosLlama")

def get_genie_sql(question: str) -> str:
    """
    Mock function to simulate query generation from a question.
    In real implementation, this would call the actual LLM or service.
    """
    genie_mngr = GenieService(space_id=config.genie.RUNTIME_GENIE_SPACE_ID)
    return genie_mngr.genie_workflow_v2(dict(question=question)).genie_query


def test_hardware_store_evalset(yaml_path, limit=1):
    """
    Test function that reads YAML content and converts to pandas DataFrame.
    Creates DataFrame with columns: [question, ground_truth_sql, issues]
    Only populates 'question' column from YAML content field.
    """
    pd.set_option('display.expand_frame_repr', False)

    with open(yaml_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    
    questions = []
    for example_key, example_data in data['examples'].items():
        if 'messages' in example_data:
            for message in example_data['messages']:
                if message.get('role') == 'user' and 'content' in message:
                    questions.append(message['content'])

    questions = questions[:limit]

    df = pd.DataFrame({
        'question': questions,
        'ground_truth_sql': [None] * len(questions),
        'issues': [None] * len(questions)
    })

    df["ground_truth_sql"] = df["question"].apply(get_genie_sql)

    
    print(df)
    print(os.getcwd())
    df.to_csv("assets/debug_evalset3.csv", index=False)
    
    # assert isinstance(df, pd.DataFrame)
    # assert df.size > 0
    # assert list(df.columns) == ['question', 'ground_truth_sql', 'issues']
    # assert df['question'].notna().all()
    # assert df['ground_truth_sql'].isna().all()
    # assert df['issues'].isna().all()
    
    return df


if __name__ == "__main__":
    yaml_path = "assets/examples.yaml"
    print(test_hardware_store_evalset(yaml_path, limit=10))