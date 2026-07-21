from mlflow.tracking import MlflowClient
import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
client = MlflowClient()
print(client.get_model_version_by_alias("my_model", "development"))