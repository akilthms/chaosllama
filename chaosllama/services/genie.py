from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import (
   GenieMessage,
   GenieGetMessageQueryResultResponse,
   GenieAttachment, GenieQueryAttachment,
   GenieConversation, OperationFailed, MessageStatus)
from typing import Callable
import time
from concurrent.futures import ThreadPoolExecutor

from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

from chaosllama.profiles.config import config
from chaosllama.entities.models import GenieTelemetry, CATALOG
import mlflow
from mlflow.entities import SpanType
from datetime import datetime
import requests
from random import random
from databricks.connect import DatabricksSession
from dotenv import dotenv_values
from chaosllama.profiles.config import config
from pyspark.sql.functions import col as F
from time import sleep


env = dotenv_values(".env")
PROFILE = env["DATABRICKS_PROFILE"]
HOST = env["DATABRICKS_HOST"]
SMALL_LLM_ENDPOINTS = config.SMALL_LLM_ENDPOINTS
spark = DatabricksSession.builder.profile(PROFILE).serverless(True).getOrCreate()
EVAL_TABLE = f"{config.CATALOG}.{config.SCHEMA}.{config.EVAL_TABLE_NAME}"



_w = WorkspaceClient()

class GenieService():
    """ The purpose of this class is to manage the various interactions with the Genie Conversational API"""


    def __init__(self, space_id: str, should_reply: bool = False):
        self.space_id = space_id
        self.conversation_id = None
        self.message_id = None
        self.client = _w.genie
        self.should_reply = should_reply
        self.token = _w.tokens.create().token_value

    @mlflow.trace(span_type=SpanType.TOOL)
    def poll_status(self, func_call: Callable,
                    timeout_seconds: int = 300,
                    poll_interval: int = 5,
                    **func_kwargs) -> dict:
        EXPECTED_STATUS = MessageStatus.COMPLETED
        start_time = time.time()
        elapsed = 0
        while elapsed < timeout_seconds:
            # print(f"Current elapsed time: {elapsed}")
            response = func_call(**func_kwargs)
            genie_message_status = response.status
            # print(f"Current status of conversation_id {response.conversation_id}: {genie_message_status}")

            if genie_message_status == MessageStatus.FAILED:
                print(f"❌ Genie message failed: {response=}")

            if genie_message_status == EXPECTED_STATUS:
                # print(f"✅ Reached desired status: {EXPECTED_STATUS}")
                return response

            elapsed = time.time() - start_time
            time.sleep(poll_interval)

        raise TimeoutError(
            f"Polling timed out after {timeout_seconds} seconds without reaching status '{EXPECTED_STATUS}'.")

    def retry_message(max_retries: int = 3, delay: int = 10):
        def decorator(func):
            def wrapper(*args, **kwargs):
                attempt = 0
                while attempt < max_retries:
                    try:
                        res = func(*args, **kwargs)
                        return res
                    except (OperationFailed, ValueError) as e:
                        print(
                            f"\t❌ Attempt {attempt} failed for function `{func.__name__}` Exception occured: {e} attempting retry {max_retries - attempt} more times")
                        attempt += 1
                        if attempt < max_retries:
                            print("\t", f"Retrying in {delay} seconds...")
                            time.sleep(delay * (attempt + 1))  # Increasing back off
                        else:
                            print(f"😵 All {max_retries} attempts failed.")
                            return None
                    finally:
                        pass

            return wrapper

        return decorator

    def _parse_message_id(self, url):
        return url.split("m=")[-1]

    @retry_message()
    def start_conversation_and_wait(self, content: str):
        return self.client.start_conversation_and_wait(self.space_id, content)

    @mlflow.trace(span_type=SpanType.TOOL)
    def start_conversation_and_wait_v2(self, content: str = ""):
        api = f"api/2.0/genie/spaces/{self.space_id}/start-conversation"
        payload = dict(content=content)
        headers = dict(Authorization=f"Bearer {self.token}")
        response = requests.post(f"{HOST}/{api}", json=payload, headers=headers).json()
        return GenieMessage.from_dict(response['message'])

    @mlflow.trace
    @retry_message(max_retries=2, delay=10)
    def create_message_and_wait_v2(self, content: str, conversation_id: str = None) -> GenieMessage:
        api = f"api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages"
        payload = dict(content=content)
        headers = dict(Authorization=f"Bearer {self.token}")
        time.sleep(random() * 4)
        resp = requests.post(f"{HOST}/{api}", json=payload, headers=headers).json()
        print(f"✉️ Sending Message {resp=}")
        try:
            if resp.get("error_code", None):
                print(resp["error_code"])
                raise ValueError(f"❌ Error creating a message in {conversation_id=}")
        except Exception as e:
            print(f"Error: {e}")
            print(f"{resp=}")
        return GenieMessage.from_dict(resp)

    @mlflow.trace(span_type=SpanType.TOOL)
    def get_message_v2(self, conversation_id="", message_id: str = ""):
        # HOST
        api = f"/api/2.0/genie/spaces/{self.space_id}/conversations/{conversation_id}/messages/{message_id}"

        headers = dict(Authorization=f"Bearer {self.token}")
        resp = requests.get(f"{HOST}/{api}", headers=headers).json()
        return GenieMessage.from_dict(resp)
        # return requests.get(f"{HOST}/{api}", headers=headers).json()

    @retry_message(max_retries=2, delay=10)
    def create_message_and_wait(self, content: str, conversation_id: str = None) -> GenieMessage:
        conversation_id = conversation_id if conversation_id else self.conversation_id
        print(f"{conversation_id=}, {self.space_id=}")
        return self.client.create_message(self.space_id, conversation_id, content)
        # return (
        #     self.client.create_message_and_wait(self.space_id, conversation_id, content)
        #     if response is None else response
        # )

    def execute_message_attachment_query(self, conversation_id, message_id, attachement_id):
        return self.client.execute_message_query(self.space_id, conversation_id, message_id, attachement_id)

    def get_message(self, conversation_id: str = "", message_id: str = ""):
        return self.client.get_message(self.space_id, conversation_id, message_id)

    def delete_conversation(self, conversation_id: str):
        return self.client.delete_conversation(self.space_id, conversation_id)

    @mlflow.trace(span_type=SpanType.TOOL)
    def check_message_attachments(self, message: GenieMessage) -> GenieAttachment:
        if len(message.attachments) > 1:
            raise Exception("Expected only one attachment, but found more")
        else:
            genie_attachment = message.attachments[0]
        return genie_attachment

    @mlflow.trace(span_type=SpanType.TOOL)
    def check_followup_question(self, attachment: GenieAttachment) -> bool:
        return attachment.query is None

    @mlflow.trace
    def stub_reply(self, message: GenieMessage
                   ):
        # Respond when Genie is asking for clarification due to ambiguity
        REBUTTAL_WHEN_AMBIGUITY = "BU"
        message = self.create_message_and_wait(REBUTTAL_WHEN_AMBIGUITY, conversation_id=message.conversation_id)
        attachment = self.check_message_attachments(message)
        return message, attachment

    @mlflow.trace
    @retry_message(max_retries=4, delay=3)
    def get_ground_truth_query(self, original_question, eval_table=EVAL_TABLE):
        print(f"🪲 Ground Truth Query for {original_question=}")
        return \
        spark.table(eval_table).filter(F.col("question") == original_question).select("ground_truth_query").first()[
            "ground_truth_query"]

    @mlflow.trace
    def synthesize_reply(self, context: dict, message: GenieMessage, response_llm: str = SMALL_LLM_ENDPOINTS,
                         params={"max_tokens": 20}):
        original_question = context["original_question"]
        MAX_RETRIES = 2
        attempt = 0

        ground_truth_query = self.get_ground_truth_query(original_question)
        followup_question = context["followup_question"]

        PROMPT = """
            You are apart of a system that automates the process of evaluating sql queries. A part of the system generates a sql query based upon the question. At times, the system may ask a followup question. Your main purpose is to respond to the followup question taking into consideration the context. Respond as concisely as possible.

            Context:
                - Original Question: {original_question}
                - Ground Source Truth Query: {ground_truth_query}
        """

        reply = _w.serving_endpoints.query(
            name=response_llm,
            **params,
            messages=[
                ChatMessage(
                    role=ChatMessageRole.SYSTEM, content=PROMPT.format(original_question=original_question,
                                                                       ground_truth_query=ground_truth_query)
                ),
                ChatMessage(
                    role=ChatMessageRole.USER, content=followup_question
                ),
            ],

        ).choices[0].message.content

        print(f"🐛BUG {reply=}")
        message.id = message.message_id
        # print(f"📞 Conversation: {message.conversation_id=}")
        message = self.create_message_and_wait_v2(reply, conversation_id=message.conversation_id)
        # print(f"🪲🪲🪲: {message}")
        message = self.poll_status(
            self.get_message_v2,
            message_id=message.message_id,
            conversation_id=message.conversation_id,
        )

        attachment = self.check_message_attachments(message)
        return message, attachment

    @mlflow.trace(span_type=SpanType.CHAIN)
    def genie_workflow_v2(self, inputs, timeout=1) -> GenieTelemetry:

        question = inputs["question"]  # [TODO]: Add the system instructions to the question
        original_question = inputs["question"]

        message = self.start_conversation_and_wait_v2(content=question)
        message = self.poll_status(
            self.get_message_v2,
            message_id=message.message_id,
            conversation_id=message.conversation_id,
        )

        if message:
            genie_attachment = self.check_message_attachments(message)
            is_followup = self.check_followup_question(genie_attachment)
            if is_followup:
                print(f"🗣️ Followup question By Genie: {genie_attachment}")
                reply_context = {
                    "original_question": question,
                    "followup_question": genie_attachment.text.content
                }

            # TODO: CHANGE VARIABLE NAME MESSAGE TO REPSONSE
            message, genie_attachment = self.synthesize_reply(reply_context, message) if (
                        is_followup and self.should_reply) else (message, genie_attachment)
            genie_query_attachment = genie_attachment.query
            # print(f"  🤔 Question {i+1}: {question} | Completed ☑️")
            genie_telem = GenieTelemetry(
                genie_question=question,
                original_question=original_question,
                genie_query=genie_query_attachment.query if genie_query_attachment else None,
                conversation_id=message.conversation_id,
                space_id=message.space_id,
                created_timestamp=message.created_timestamp,
                statement_id=message.query_result.statement_id if message.query_result else None,
                genie_generated_sql_thought_process_description=genie_attachment.query.description if genie_attachment.query else None,
                query_result_metadata=message.query_result.as_dict() if message.query_result else None,
                row_count=genie_query_attachment.query_result_metadata.row_count if genie_query_attachment else None
            )

            sleep(timeout)
        else:
            print("❎ No Message")
            genie_telem = GenieTelemetry(
                genie_question=question,
                original_question=original_question,
                genie_query=None,
                conversation_id=None,
                space_id=self.space_id,
                created_timestamp=datetime.now().timestamp(),
                statement_id=None,
                genie_generated_sql_thought_process_description="Genie Failed to Provide a Response due to Internal Error",
                query_result_metadata=None,
                row_count=None
            )
        return genie_telem


    def concurrent_execution(self, questions, batch_size=1, n_jobs=1):
        """ The purpose of this function is to accelarate testing by executing multiple questions in parallel. """
        print(f"∥ Concurrent request {n_jobs}")
        batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(self.genie_workflow, batch) for batch in batches]
            results = [future.result() for future in futures]

        return [item for sublist in results for item in sublist]  # 📦 unpack results



class GenieAgent:
    """ Refactored Version of the Genie Manager into an Agent to fit into MLFlow 3.0 paradigm"""

    def __init__(self, space_id:str, should_reply:bool = True):
        self.space_id = space_id
        self._w = _w
        self.client = self._w.genie
        self.should_reply = should_reply
        self.genie_mgr = GenieService(self.space_id, should_reply=True)
        self.token = self._w.tokens.create().token_value

    @mlflow.trace(name="🧞‍♂️ Genie Agent")
    def invoke(self, inputs):
        question = inputs['question']
        # TODO: Uncomment and implement update_current_trace
        #mlflow.update_current_trace(request_preview=f"{question}")
        return self.genie_mgr.genie_workflow_v2(inputs).genie_query