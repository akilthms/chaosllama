from dotenv import dotenv_values
from databricks import sql
from collections import Counter
import sqlparse
from typing import Optional, Any
from mlflow.genai.scorers import scorer
from concurrent.futures import ThreadPoolExecutor
from sqlparse.tokens import Keyword
from mlflow.entities import Feedback
import pandas as pd

env = dotenv_values(".env")
HOST = env["DATABRICKS_HOST"]
WAREHOUSE_ID = dotenv_values(".env")["CHAOS_LLAMA_WAREHOUSE_ID"]

def execute_query(query: str, warehouse_id: str = WAREHOUSE_ID, server_hostname: str = HOST):
    with sql.connect(
            server_hostname=server_hostname,
            http_path=f"/sql/1.0/warehouses/{warehouse_id}",
            access_token=env["DATABRICKS_TOKEN"],
            _tls_no_verify=True
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
    return result


def interleave_list(lst_a, lst_b):
    new_list = [item for pair in zip(lst_a, lst_b) for item in pair]
    return new_list


def count_sql_keywords(sql: str) -> dict:
    # Tokenize the SQL statement(s)
    tokens = sqlparse.parse(sql)[0].tokens

    # Flatten nested tokens and extract only keywords
    def extract_keywords(tokens):
        for token in tokens:
            if token.ttype in Keyword:
                yield token.value.upper()
            elif token.is_group:
                yield from extract_keywords(token.tokens)

    # Count frequencies
    keyword_counts = Counter(extract_keywords(tokens))
    return dict(keyword_counts)


def process_eval_request(inputs: dict | str) -> str:
    return inputs


def process_eval_output(output: str):
    # [TODO]: Implemet logic
    return output


def process_eval_expectations(expectations: Optional[dict[str, Any]]):
    return expectations["expected_response"]


@scorer
def eval_sql_clauses_distro(inputs: dict, outputs: str, expectations: Optional[dict[str, Any]]):
    # ground_truth_sql = process_eval_request(inputs)
    outputs = process_eval_output(outputs)
    ground_truth_sql = process_eval_expectations(expectations)

    ground_truth_sql_distro = count_sql_keywords(ground_truth_sql)
    genie_sql_distro = count_sql_keywords(outputs)
    all_keys = set(ground_truth_sql_distro) | set(genie_sql_distro)

    diff_distro = {k: ground_truth_sql_distro.get(k, 0) - genie_sql_distro.get(k, 0) for k in all_keys}

    is_sql_token_distro_equal = all(value == 0 for value in diff_distro.values())

    PASS_RATIONALE = "The SQL clauses produced by the predicted sql query and the ground truth sql query are equal given that the count of SQL clauses are equivalent"
    FAIL_RATIONALE = f"""The SQL key word tokens produced by the predicted sql query and the ground truth sql query are different: 
        {diff_distro}
    The mapping represents the difference in the number of tokens between the ground truth sql query and the predicted sql query. The positive values indicate that the ground truth sql query has more of that token than the predicted sql query, while the negative values indicate the opposite.
    """
    _metadata = None if is_sql_token_distro_equal else {"difference_in_distrubtion": diff_distro}

    return Feedback(
        name="sql_clauses_distribution_equivalence",
        value=True if is_sql_token_distro_equal else False,
        metadata=_metadata,
        rationale=PASS_RATIONALE if is_sql_token_distro_equal else FAIL_RATIONALE
    )


@scorer(name="sql_results_equivalence")
def eval_query_results(inputs: dict, outputs: dict, expectations: Optional[dict[str, Any]]):
    """
    [TODO]:
    ground_truth_sql = process_eval_request(inputs)
    outputs = process_eval_output(outputs)
    """
    # request = process_eval_request(inputs)
    outputs = process_eval_output(outputs)
    ground_truth_sql = process_eval_expectations(expectations)
    print(f"ground_truth_sql: {ground_truth_sql}")
    print(f"outputs: {outputs}")
    # queries_list = interleave_list([request], [outputs]) if isinstance(request,dict) or (isinstance(request,str)) else interleave_list(request, outputs)

    queries_list = interleave_list([ground_truth_sql], [outputs])

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(execute_query, q) for q in queries_list]
        scores = [future.result() for future in futures]
        scores = [tuple(scores[i:i + 2]) for i in range(0, len(scores), 2)]  # group the predictions and targets

    if len(scores) > 1: raise ValueError("Only one pair of queries is supported")

    results_are_equal = []

    for pred_results, ground_truth_results in scores:
        pred_df = pd.DataFrame(pred_results).sort_index(axis=0).sort_index(axis=1)
        ground_truth_df = pd.DataFrame(ground_truth_results).sort_index(axis=0).sort_index(axis=1)
        is_equal = pred_df.equals(ground_truth_df)
        results_are_equal.append("yes" if is_equal else "no")

        # Limit the count of rows in scores
        RESULT_SET_LIMIT = 5
        scores = [tuple([lst[:RESULT_SET_LIMIT] for lst in score_pair]) for score_pair in scores]

    CORRECT_SQL_RATIONALE = f"The results sets produced by the predicted sql query and the ground truth sql query are equal given that the results sets are equivalent"
    INCORRECT_SQL_RATIONALE = f"The results sets produced by the predicted sql query and the ground truth sql query are different: {scores}"

    rationale_logic = (
        CORRECT_SQL_RATIONALE if (results_are_equal[0] == "yes")
        else INCORRECT_SQL_RATIONALE
    )
    _metadata = {
        "row_count": {
            "pred_df": pred_df.size,
            "ground_truth_df": ground_truth_df.size
        },
        "result_set": scores
    }

    _value = True if (results_are_equal[0] == "yes") else False

    return Feedback(
        name="sql_results_equivalence",
        value=_value,
        metadata=_metadata,
        rationale=rationale_logic
    )