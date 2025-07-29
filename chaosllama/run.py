# from chaosllama.chaosllama import ChaosLlama
from chaosllama.services import genie, mosaic
from chaosllama.entities.models import EvalSetTable, EvalSetManager
import chaosllama
from chaosllama.profiles.config import config



if __name__ == "__main__":
    # Logging Configuration
    # Services Configuration

    # 👓 Read in Runtime Configurations

    # 🧑‍🍳 Prepare Evaluation Data Set
    evmngr = EvalSetManager(limit=config.LIMIT, consistency_factor=config.CONSISTENCY_FACTOR)
    eval_set = evmngr.prepare_evals()
    print("hello world")
    #
    #
    #
    # genie_mngr = genie.GenieService()
    # mosaic_eval_mngr = mosaic.MosaicEvalService()

    # cll_config = ChaosLlamaServicesConfig(
    #     genie_manager=genie_mngr,
    #     mlflow_manager=mosaic_eval_mngr,
    # )
    #
    # cll = ChaosLlama(config=cll_config)
    #
    # cll.run(epochs=2,
    #         is_test=True,
    #         limit=None,
    #         is_cached=True,
    #         run_baseline=False)
    #
