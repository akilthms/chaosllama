import mlflow
from rich.console import Console
from rich.panel import Panel

console = Console()

class MLFlowExperimentManager:
    """Manager for MLflow experiments."""

    def __init__(self, experiment_name: str = "", experiment_id: str = "", experiment_path:str =""):
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.experiment_path = experiment_path
        self.experiment = None
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks")

    # TODO: CREATE A UNIT TEST FOR THIS FUNCTION
    def get_or_create_mlflow_experiment(self, experiment_name: str):
        """Get or create an MLflow experiment."""

        console.print(Panel(" 🔬MLFlow Experiment Instrumentation", expand=False, style="bold cyan"))

        EXPERIMENT_NAME = self.experiment_path + experiment_name

        try:
            print(f"🧪Creating MLFlow Experiment {EXPERIMENT_NAME}")
            self.experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
            if self.experiment is not None:
                print(f"Experiment {EXPERIMENT_NAME} already exists with ID: {self.experiment.experiment_id}")
                self.experiment_id = self.experiment.experiment_id
            else:
                print(f"Experiment {experiment_name} does not exist, creating a new one.")
                self.experiment = mlflow.create_experiment(EXPERIMENT_NAME)
                self.experiment_id = self.experiment
                print(f"Created new experiment {EXPERIMENT_NAME} with ID: {self.experiment_id}")

        except Exception as e:
            print(f"Error getting experiment {EXPERIMENT_NAME} with Error: {e}")

        return self

    def set_experiment(self, experiment_name: str):
        """Set the current MLFlow experiment."""
        try:
            mlflow.set_experiment(experiment_name)
            self.experiment_name = experiment_name
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        except Exception as e:
            print(f"Error setting experiment: {e}")
            raise
        return self