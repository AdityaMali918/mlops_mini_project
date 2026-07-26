
import json
import mlflow
import logging
import os
from mlflow.tracking import MlflowClient

from dotenv import load_dotenv

load_dotenv()

# mlflow.set_tracking_uri('http://127.0.0.1:5000/')

dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "AdityaMali918"
repo_name = "mlops_mini_project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

client = MlflowClient()
# model_info_path = 'reports/experiment_info.json'

# with open(model_info_path, 'r') as file:
#             model_info = json.load(file)


# model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
#         # Register the model
# model_version = mlflow.register_model(model_uri, "my_model")

# Set an alias instead of a stage
# client = mlflow.tracking.MlflowClient()
# client.set_registered_model_alias(
#     name="my_model",
#     alias="development",
#     version=model_version.version
# )
# print(model_version)            

run_id = "3509a89dfd9d42b5a8886e0f730a7cb5"

for artifact in client.list_artifacts(run_id):
    print(artifact.path)