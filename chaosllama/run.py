from chaosllama.chaos import ChaosLlama, ChaosLlamaServicesConfig
from chaosllama.entities.models import AgentConfig
from chaosllama.services import genie, mosaic, unity_catalog, judges
from chaosllama.services.evaluation_dataset import EvalSetManager
from chaosllama.profiles.config import config
from chaosllama.services.tracking import MLFlowExperimentManager
from chaosllama.scorers.scorers import eval_sql_clauses_distro, eval_query_results, eval_query_results_single_thread
import chaosllama.prompts.registry as prompt_registry
import pyfiglet
import rich
from rich.console import Console


console = Console()
RUN_CREATE_EVALS = False

if __name__ == "__main__":
    # 🪵Logging Configuration

    # TODO: Add logging configuration

    banner = pyfiglet.figlet_format("Chaos Llama 🦙")
    print(banner)

    # 🧑‍🍳 Prepare Evaluation Data Set
    evmngr = EvalSetManager(table_name=f"{config.CATALOG}.{config.SCHEMA}.{config.EVAL_TABLE_NAME}",
                        limit=config.runtime.LIMIT,
                        consistency_factor=config.runtime.CONSISTENCY_FACTOR)
    evmngr.prepare_evals(mode="synthetic")
    evmngr.write_evalset()

    # 🧑‍🔬Set MLFLow Experiment
    exp_mngr = MLFlowExperimentManager(experiment_path=config.mlflow.MLFLOW_EXPERIMENT_PATH).get_or_create_mlflow_experiment(config.mlflow.MLFLOW_RUNTIME_EXPERIMENT)

    # 🧑‍🔧️Configure Services
    jmngr = judges.JudgeService(scorers=[eval_sql_clauses_distro,eval_query_results_single_thread]) #🧑‍⚖️Judges eval_query_results
    mlfmngr = mosaic.MosaicEvalService(eval_manager=evmngr,judge_manager=jmngr, experiment_id=exp_mngr.experiment_id) # 🧪Mosaic Evaluations
    gmngr = genie.GenieService(space_id=config.genie.RUNTIME_GENIE_SPACE_ID) # 🧞‍♂️Genie Service
    ucmngr = unity_catalog.UCService(catalog=config.CATALOG, schema=config.SCHEMA) # 🤝 Unity Catalog Manager
    agent_config = AgentConfig(
            system_prompt=prompt_registry.INSTROSPECT_PROMPT_V3,
            endpoint=config.runtime.INTROSPECT_AGENT_LLM_ENDPOINT,
            llm_parameters={"temperature": 0.0, "max_tokens": config.runtime.MAX_TOKENS},
    )

    # ⚙️🦙Configure Chaos Llama
    chaos_config = ChaosLlamaServicesConfig(
        mlflow_manager=mlfmngr,
        genie_manager=gmngr,
        uc_manager=ucmngr,
        agent_config=agent_config
    )

    chaos_llama = ChaosLlama(config=chaos_config)

    data_intelligence, mlflow_parent_run = chaos_llama.run(
        epochs=config.runtime.EPOCHS,
        is_test=config.runtime.DEBUG,
        is_cached=config.runtime.IS_CACHED,
        run_baseline=config.runtime.RUN_BASELINE,
        run_null_hypothesis=config.runtime.RUN_NULL_HYPOTHESIS,
    )

    print("👍Done!")

