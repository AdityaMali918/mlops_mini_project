import os
import mlflow
from dotenv import load_dotenv

load_dotenv()
def promote_model():
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")

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

    client = mlflow.MlflowClient()
    # client = mlflow.MlflowClient()

    model_name = "my_model"

    # Get current development model
    dev_model = client.get_model_version_by_alias(
        model_name,
        "development"
    )

    # Check if a production model already exists
    try:
        prod_model = client.get_model_version_by_alias(
            model_name,
            "production"
        )

        # Move old production to archived
        client.set_registered_model_alias(
            name=model_name,
            alias="archived",
            version=prod_model.version
        )

    except Exception:
        # No production model yet
        pass

    # Promote development to production
    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=dev_model.version
    )

    print(f"Version {dev_model.version} promoted to production.")