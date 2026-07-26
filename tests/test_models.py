import os
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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

client = mlflow.MlflowClient()

def test_development_alias_points_to_latest_version():
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # client = mlflow.MlflowClient()
    model_name = "my_model"

    # Get all versions
    versions = client.search_model_versions(f"name='{model_name}'")
    latest_version = max(int(v.version) for v in versions)

    # Get version assigned to development alias
    alias_version = client.get_model_version_by_alias(
        model_name,
        "development"
    )

    assert int(alias_version.version) == latest_version


def test_registered_model_performance():
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")

    model = mlflow.pyfunc.load_model("models:/my_model@development")

    test_data = pd.read_csv("data/processed/test_bow.csv")

    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    y_pred = model.predict(X_test)

    assert accuracy_score(y_test, y_pred) >= 0.60
    assert precision_score(y_test, y_pred) >= 0.60
    assert recall_score(y_test, y_pred) >= 0.60
    assert f1_score(y_test, y_pred) >= 0.60
