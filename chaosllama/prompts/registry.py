from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from mlflow.genai.scorers import ExpectationsGuidelines, Guidelines
from chaosllama.profiles.config import config

def format_guidelines():
  global_guidelines = config.scorers.global_guidelines["v4"] 
  return "\n".join([f"- {name}: {g[0]}" for name, g in global_guidelines.items()])

METRICS_DEFINITION = """

"""
OPTIONAL_COLUMN_MODIFICATIONS = "or metadata (e.g., column descriptions) to guide "
INPUT_VARS = ["data_intelligence",
              "metrics_definition",
              "drift_metrics",
              "optional_column_modifications",
              "quality_threshold"]
FORMAT_INSTRUCTIONS = """ For your output format, output a json object with the following schema:
                          suggested_change_type str: ["system_instructions" | "column_description"],
                          updated_metadata: str
"""

INTROSPECT_PROMPT = PromptTemplate(input_variables=INPUT_VARS,
                                   partial_variables=dict(format_instructions=FORMAT_INSTRUCTIONS),
                                   optional_variables=["optional_column_modifications"],
                                   output_parser=JsonOutputParser(),
                                   template=""" 

  You are a specialized AI agent within the 🦙Chaos Llama framework, a system designed to test and improve the accuracy of Text-to-SQL models through introspective feedback loops. Your role is to analyze evaluation metrics including but not limited to execution correctness, SQL similarity, token confidence, and semantic errors—produced by mlflow.evaluate. Based on these insights, you must reason about why the model succeeded or failed and suggest targeted updates to the system prompt {optional_column_modifications} in future generations. You will be leveraging the insights plus your previous suggestions from previous invocations of yourself.You are not modifying query outputs directly; instead, your job is to tune the model's system prompts that it performs better in subsequent runs. Think like a prompt engineer, a data detective, and a metadata strategist. Always provide concise, actionable suggestions grounded in the observed data intelligence.

  Review the following intelligence gathered from the MLflow evaluation module and the drift metrics.

  {data_intelligence}

  Here are the following descriptions of the metrics used in the MLFlow Evaluation Module

  {metrics_definition}

  Your goal is to achieve an accuracy rate of {quality_threshold} or higher. 

  If the accuracy is not high enough, then you will attempt to update the System prompt to increase the accuracy. Observe the drift metrics of the current data_intelligence and the previous iteration of the data intelligence. 

  {previous_data_intelligence}

  Here is the drift of the previous computed metrics and this iteration's metrics: {drift_metrics}

  The following is the previous quality metric value :{prev_overall_quality_score}



  With all of this information create a brand new system prompt for this text2sql solution to the aim of increasing the quality metrics. Make sure you only output the new system prompt. 

  🤞 Good Luck.

""")

# TODO: Add Metrics Definition
INPUT_VARS_V2 = ["data_intelligence",
                 "quality_threshold",
                 "previous_data_intelligence",
                 "previous_overall_quality_score",
                 "current_overall_quality_score",
                 "previos_prompt"]

FORMAT_INSTRUCTIONS_V2 = """ For your output format, output a json object with the following schema:
                          suggested_change_type str: ["system_instructions" | "column_description"],
                          updated_metadata: str
"""


class ReflectionModel(BaseModel):
    ai_system_instruction: str = Field(decription="The optimized system prompt after evaluating the data intelligence")
    # rationale: str = Field(description="The rationale behind the creation / modification of the system prompt in the aim of improving the quality metrics")


introspection_parser = PydanticOutputParser(pydantic_object=ReflectionModel)

INTROSPECT_PROMPT_V2 = PromptTemplate(input_variables=INPUT_VARS_V2,
                                      partial_variables={
                                          "format_instructions": introspection_parser.get_format_instructions()},
                                      output_parser=introspection_parser,
                                      template="""
                                        You are a specialized AI Agent within the system prompt optimization framework called 🦙Chaos Llama. You task is to analyze various evaluation metrics to produce a more optimized system prompt than your predecesor (yourself in a previous iteration). 

                                        The previous prompt: {previous_prompt}

                                        This current iteration's  data intelligence from the evaluation metrics: {data_intelligence}
                                        The previous iteration's data intelligence from the evaluation metrics: {previous_data_intelligence}

                                        Your single and only objective is to create a new system prompt that beats the previous quality score.

                                        The previous overall quality score: {previous_overall_quality_score}
                                        The current overall_quality_score: {current_overall_quality_score}
                                        This quality threshold you must try to beat: {quality_threshold}

                                        Lets define some of the following contraints for you crafting the new system prompt.
                                        - You may add to / modify the previous system prompt. Reason about what should be removed and what new content should be added to increase accuracy metrics
                                        - Ensure that your suggestions for the new system prompt should be generalizable such that the system instructions perform well on unseen scenarios. 
                                        - MAKE SURE TO ONLY OUTPUT YOUR RESPONSE AS JSON with two keys [ai_system_instruction, rationale] as per the format instructions !!

                                        🤞 Good Luck.
                                      """)

INPUT_VARS_V3 = [
    "data_intelligence",
    "system_instructions_history",
    "overall_quality_score"

]

INSTROSPECT_PROMPT_V3 = PromptTemplate(input_variables=INPUT_VARS_V3,
                                       partial_variables={
                                           "format_instructions": introspection_parser.get_format_instructions()},
                                       output_parser=introspection_parser,
                                       template="""
                                        # Objective Overview
                                        You are a specialized AI Agent within the system prompt optimization framework called 🦙Chaos Llama. You task is to analyze various evaluation metrics to produce a more optimized system prompt than your predecesor (yourself in a previous iteration). Ensure that when you make changes from the previous prompt that you make smaller incremental changes such that we do not regress the performance accuracy too much. Observe the previous prompt and the historical running history of data intelligence (telemetry) of the evaluation process to determine how to improve the system instructions such that we increase quality metrics. Weight the sql_resultset_equivalence metric sql_results_equivalence metric more heavily than the other metrics. Your single and only objective is to create a new system prompt that beats the previous quality score.

                                        # Metadata on the Evaluation
                                        The previous prompts: {system_instructions_history}

                                        The Data Intelligence History over optimization cycles. Optimization id is the iteration of the optimization cycle in ascending order: {data_intelligence}

                                        # Quality Metrics
                                        The overall quality score history over optimization cycles: {overall_quality_score}
                                        This quality threshold you must try to beat: {quality_threshold}

                                        # Contraints and Parameters for output suggested metadata

                                        Lets define some of the following contraints for you crafting the new system prompt.
                                        - You may add to / modify the previous system prompt. Reason about what should be removed and what new content should be added to increase accuracy metrics
                                        - Ensure that your suggestions for the new system prompt should be generalizable such that the system instructions perform well on unseen scenarios. 
                                        - MAKE SURE TO ONLY OUTPUT YOUR RESPONSE AS JSON with two keys [ai_system_instruction, rationale] as per the format instructions !!

                                        🤞 Good Luck.
                                      """)

INSTROSPECT_PROMPT_V4 = PromptTemplate(input_variables=INPUT_VARS_V3,
                                       partial_variables={
                                           "format_instructions": introspection_parser.get_format_instructions()},
                                       output_parser=introspection_parser,
                                       template="""
                                        # Objective Overview
                                        You are a specialized AI Agent within the system prompt optimization framework called 🦙Chaos Llama. You task is to analyze various evaluation metrics to produce a more optimized system prompt than your predecesor (yourself in a previous iteration). Ensure that when you make changes from the previous prompt that you make smaller incremental changes such that we do not regress the performance accuracy too much. Observe the previous prompt and the historical running history of data intelligence (telemetry) of the evaluation process to determine how to improve the system instructions such that we increase quality metrics. Weight the sql_resultset_equivalence metric sql_results_equivalence metric more heavily than the other metrics. Your single and only objective is to create a new system prompt that beats the previous quality score.

                                        # Metadata on the Evaluation
                                        The previous prompts: {system_instructions_history}

                                        The Data Intelligence History over optimization cycles. Optimization id is the iteration of the optimization cycle in ascending order: {data_intelligence}

                                        # Quality Metrics
                                        The overall quality score history over optimization cycles: {overall_quality_score}
                                        This quality threshold you must try to beat: {quality_threshold}



                                        # Analyze the difference between the previous optimization cycle and the current optimization cycle. Reason about how the differences in the follow analytics has an impact on the overall_quality_score diff.

                                      Difference between current overall_quality_score and the previous overall_quality_score: {overall_quality_score_diff}
                                      Difference between the word token frequency difference: {word_token_diff}





                                        # Contraints and Parameters for output suggested metadata

                                        Lets define some of the following contraints for you crafting the new system prompt.
                                        - You may add to / modify the previous system prompt. Reason about what should be removed and what new content should be added to increase accuracy metrics 
                                        - MAKE SURE TO ONLY OUTPUT YOUR RESPONSE AS JSON with two keys [ai_system_instruction, rationale] as per the format instructions !!

                                        🤞 Good Luck.
                                      """)

INPUT_VARS_V5 = [
    "data_intelligence",
    "system_instructions_history",
    "overall_quality_score"

]
INSTROSPECT_PROMPT_V5 = PromptTemplate(input_variables=INPUT_VARS_V5,
                                       partial_variables={
                                           "format_instructions": introspection_parser.get_format_instructions()},
                                       output_parser=introspection_parser,
                                       template="""
                                        # Objective Overview
                                        You are a specialized AI Agent within the system prompt optimization framework called 🦙Chaos Llama. You task is to analyze various evaluation metrics to produce a more optimized system prompt than your predecesor (yourself in a previous iteration). Ensure that when you make changes from the previous prompt that you make smaller incremental changes such that we do not regress the performance accuracy too much. Observe the previous prompt and the historical running history of data intelligence (telemetry) of the evaluation process to determine how to improve the system instructions such that we increase quality metrics. Weight the sql_resultset_equivalence metric sql_results_equivalence metric more heavily than the other metrics. Your single and only objective is to create a new system prompt that beats the previous quality score.

                                        # Metadata on the Evaluation
                                        The previous prompts: {system_instructions_history}

                                        The Data Intelligence History over optimization cycles. Optimization id is the iteration of the optimization cycle in ascending order: {data_intelligence}

                                        # Quality Metrics
                                        The overall quality score history over optimization cycles: {overall_quality_score}
                                        This quality threshold you must try to beat: {quality_threshold}



                                        # Analyze the difference between the previous optimization cycle and the current optimization cycle. Reason about how the differences in the follow analytics has an impact on the overall_quality_score diff.

                                      Difference between current overall_quality_score and the previous overall_quality_score: {overall_quality_score_diff}
                                      Difference between the word token frequency difference: {word_token_diff}





                                        # Contraints and Parameters for output suggested metadata

                                        Lets define some of the following contraints for you crafting the new system prompt.
                                        - You may add to / modify the previous system prompt. Reason about what should be removed and what new content should be added to increase accuracy metrics
                                        - Ensure that your suggestions for the new system prompt should be generalizable such that the system instructions perform well on unseen scenarios. 
                                        - Make Sure to not overfit to the dataintelligence history.
                                        - MAKE SURE TO ONLY OUTPUT YOUR RESPONSE AS JSON with two keys [ai_system_instruction, rationale] as per the format instructions !!

                                        🤞 Good Luck.
                                      """)


global_guidelines = config.scorers.global_guidelines["v4"]
guidelines = [Guidelines(name=name, guidelines=g[0]) for name, g in global_guidelines.items()]

INPUT_VARS_V6 = [
    "data_intelligence",
    "system_instructions_history",
    "overall_quality_score"
]

ONE_DATA_INTELLIGENCE_PAYLOAD = """
 # Data for Introspection
  {feedback_intelligence}
"""

EXAMPLE_SYSTEM_INSTRUCTIONS = """
# Example Optimized System Instructions
Below is an example of the structure of optimized system instructions. Besides each heading is a tag of either required or optional. Based upon the feedback


  **Guidelines and Policies you must follow:** [Required]
    - <FEEDBACK_BASED_RULE1>
    - <FEEDBACK_BASED_RULE2>
    - <FEEDBACK_BASED_RULEN>

  ** Example synonyms / alliases: ** [Optional]

  ** Query Structure [Optional]
"""

INSTROSPECT_PROMPT_V6 = PromptTemplate(input_variables=INPUT_VARS_V6,
                                       partial_variables={
                                           "format_instructions": introspection_parser.get_format_instructions()},
                                       output_parser=introspection_parser,
                                       template= f"""
# Context Overview 
You are apart of an agentic system called Chaos Llama; that iteratively optimizes the system instructions of a Databricks Genie Space ( A Text2SQL system ) by introspecting on feedback provided by an array of llm-as-a-judge on an evaluation dataset with schema (question, ground_truth_query).

Here is a Process Flow of Chaos:
1. The system (Chaos Llama) is initialized with an empty set of system instructions
2. The system takes as input an evaluation dataset of schema (question, ground_truth_query) and during runtime generates natural language feedback from an array of llm-as-a-judge. 
3. The introspection module, **YOU**, introspects (analyzes) the natural language feedback and produces an optimized system instruction that endeavors to improve the overall quality score averaged across all questions in the evaluation dataset. 

# Objectives
Your primary task is to analyze feedback from Databricks LLM-as-a-Judge (scorers) and devise rules from that feedback to improve your own system prompt to perform better on the evaluation dataset for the next interation. Weight the sql_resultset_equivalence metric and sql_semantic_equivalence metric more heavily than the other metrics.

# Feedback Description Map
The following is a mapping of the name of the llm-as-a-judge to the criteria for the judge.
{format_guidelines()}
- eval_query_results: Ensures the result set of the expected_response is equal to the query in the outputs
- eval_sql_clauses_distro: Ensures the distribution of the sql clauses in the expected_response is equal to the distribution of the sql clauses in the outputs

# Data for introspection

## Data Schema for Intropection
- feedback: A list of feedback objects return by the llm-as-a-judge for each optimization cycle
- overall_quality_score: A list of quality score objects; the quality score is the average metric score across the entire validation dataset
- optimization_id: The iteration id of the optimization cycle. optimization_id = 0 is the first iteration, and incremental positive values are subsequent optimization runs. 
- suggestion: a list of optimized system instructions produced by the chaos llama system. mapped to the optimization_id. 


### Feedback Schema
The feedback is a JSON object with the following keys:
- `name`: A string containing the name of the feedback.
- `value`: A string containing either pass/fail, yes/no, or a numeric value as a result of the criteria of the judge/scorer .
- `rationale`: A string containing the rationale for the value of the feedback by the judge/scorer.

### Introspection Data
Provides Introspection Data for the current and {config.runtime.INTROSPECTION_LOOKBACK} previous optimization cycles. 

{{data_intelligence}}
## System Instructions History
{{system_instructions_history}}
## Overall Quality Score History
{{overall_quality_score}}


# Constraints on producing the optimize system instructions
- You can reference specific details in the feedback if you find that helpful to increase the quality metrics
- MAKE SURE TO ONLY OUTPUT YOUR RESPONSE AS JSON with two keys [ai_system_instruction, rationale] as per the format instructions !!

                                                

                                                """)







INPUT_VARS_V7 = ["introspections"]
INSTROSPECT_PROMPT_V7 = PromptTemplate(input_variables=INPUT_VARS_V7,
                                       partial_variables={
                                           "format_instructions": introspection_parser.get_format_instructions()},
                                       output_parser=introspection_parser,
                                       template= f"""
# Context Overview 
You are apart of an agentic system called Chaos Llama; that iteratively optimizes the system instructions of a Databricks Genie Space ( A Text2SQL system ) by introspecting on feedback provided by an array of llm-as-a-judge on an evaluation dataset with schema (question, ground_truth_query).

Here is a Process Flow of Chaos:
1. The system (Chaos Llama) is initialized with an empty set of system instructions
2. The system takes as input an evaluation dataset of schema (question, ground_truth_query) and during runtime generates natural language feedback from an array of llm-as-a-judge. 
3. The introspection module, **YOU**, introspects (analyzes) the natural language feedback and produces an optimized system instruction that endeavors to improve the overall quality score averaged across all questions in the evaluation dataset. 

# Objectives
Your primary task is to analyze feedback from Databricks LLM-as-a-Judge (scorers) and devise rules from that feedback to improve your own system prompt to perform better on the evaluation dataset for the next interation. Weight the sql_resultset_equivalence metric and sql_semantic_equivalence metric more heavily than the other metrics.

# Feedback Description Map
The following is a mapping of the name of the llm-as-a-judge to the criteria for the judge.
{format_guidelines()}
- eval_query_results: Ensures the result set of the expected_response is equal to the query in the outputs
- eval_sql_clauses_distro: Ensures the distribution of the sql clauses in the expected_response is equal to the distribution of the sql clauses in the outputs

# Data for introspection
## Data Schema for Intropection
- feedback: A list of feedback objects return by the llm-as-a-judge for each optimization cycle
- overall_quality_score: A list of quality score objects; the quality score is the average metric score for the judge or scrorer across the entire evaluation dataset
- optimization_id: The iteration id of the optimization cycle. optimization_id = 0 is the first iteration, and incremental positive values are subsequent optimization runs. 
- suggestion: a list of optimized system instructions produced by the chaos llama system. mapped to the optimization_id. 

### Feedback Schema
The feedback is a JSON object with the following keys:
- `name`: A string containing the name of the feedback.
- `value`: A string containing either pass/fail, yes/no, or a numeric value as a result of the criteria of the judge/scorer .
- `rationale`: A string containing the rationale for the value of the feedback by the judge/scorer.

### Introspection Data
Provides Introspection Data for the current and {config.runtime.INTROSPECTION_LOOKBACK} previous optimization cycles. 

{{introspections}}


# Constraints on producing the optimize system instructions
- You can reference specific details in the feedback if you find that helpful to increase the quality metrics
- MAKE SURE TO ONLY OUTPUT YOUR RESPONSE AS JSON with two keys [ai_system_instruction, rationale] as per the format instructions !!

                                                

                                                """)