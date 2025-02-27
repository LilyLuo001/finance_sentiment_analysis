import mlflow
from mlflow.tracking import MlflowClient

class ExperimentTracker:
    """Comprehensive experiment tracking with MLflow"""
    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        
    def log_run(self, params: dict, metrics: dict, artifacts: list = None):
        """Log complete experiment details"""
        with mlflow.start_run():
            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            
            # Log artifacts
            if artifacts:
                for artifact in artifacts:
                    mlflow.log_artifact(artifact)
                    
            # Log model
            mlflow.pytorch.log_model(mlflow.pytorch.get_default_conda_env(), "model")
            
    def get_best_run(self, metric: str = 'sharpe_ratio') -> dict:
        """Retrieve best performing run based on metric"""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = self.client.search_runs(experiment.experiment_id)
        return max(runs, key=lambda r: r.data.metrics.get(metric, 0))
