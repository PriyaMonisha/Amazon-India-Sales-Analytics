"""Shared MLflow helpers — use this instead of duplicating _setup_mlflow per model file."""
import mlflow

from config import MLFLOW_TRACKING_URI


def setup_mlflow(experiment_name: str) -> None:
    """Configure MLflow tracking URI and set/create the named experiment."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
