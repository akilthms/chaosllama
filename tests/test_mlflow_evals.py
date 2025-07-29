from chaosllama.services.mosaic import MosaicEvalService

def test_mosaic_eval_service():
    # Assuming EvalSetTable and JudgesManager are properly defined and imported
    eval_set = EvalSetTable()  # Replace with actual initialization
    judge_manager = None  # Replace with actual JudgesManager instance if needed
    genie_manager = None  # Replace with actual GenieService instance if needed

    mosaic_eval_service = MosaicEvalService(
        eval_set=eval_set,
        judge_manager=judge_manager,
        genie_manager=genie_manager,
        experiment_id="test_experiment",
        experiment_name="Test Experiment"
    )

    assert mosaic_eval_service is not None, "MosaicEvalService should be initialized successfully"
    print("MosaicEvalService initialized successfully")