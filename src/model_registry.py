import json
import mlflow
import logging
import os
import sys
import warnings
from typing import Dict, Any
import yaml
from mlflow.tracking import MlflowClient

# ------------------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("model_registry.log")
    ]
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------
def load_params(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Params file not found: {file_path}")

    with open(file_path, "r") as f:
        params = yaml.safe_load(f) or {}

    logger.info(f"Loaded parameters from {file_path}")
    return params


def validate_params(params: Dict[str, Any]) -> None:
    if "model_registry" not in params:
        raise KeyError("Missing 'model_registry' section in params.yaml")

    required = ["registry_path", "model_name"]
    missing = [k for k in required if k not in params["model_registry"]]

    if missing:
        raise KeyError(f"Missing keys in model_registry: {missing}")


def load_model_info(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model info file not found: {file_path}")

    with open(file_path, "r") as f:
        model_info = json.load(f)

    for key in ["run_id", "model_path"]:
        if key not in model_info or not model_info[key]:
            raise ValueError(f"Invalid model_info: '{key}' is missing or empty")

    logger.info("Loaded model info successfully")
    return model_info


def setup_mlflow(tracking_uri: str) -> bool:
    """
    Configure MLflow tracking and verify server availability.
    Returns True if server is reachable, else False.
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        client.search_experiments(max_results=1)
        logger.info(f"Connected to MLflow tracking server: {tracking_uri}")
        return True
    except Exception as e:
        logger.warning(f"MLflow server not reachable at {tracking_uri}")
        logger.warning(str(e))
        return False


def register_model(model_name: str, model_info: Dict[str, Any]) -> str:
    model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
    logger.info(f"Registering model from URI: {model_uri}")

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    logger.info(
        f"Model '{model_name}' registered as version {model_version.version}"
    )

    return model_version.version


# ------------------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------------------
def main():
    try:
        logger.info("Starting model registry stage")

        # Load & validate params
        params = load_params("params.yaml")
        validate_params(params)

        registry_cfg = params["model_registry"]
        model_name = registry_cfg["model_name"]
        registry_path = registry_cfg["registry_path"]

        # Resolve MLflow tracking URI (ENV > params > fallback)
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI",
            params.get("mlflow", {}).get("tracking_uri", "file:./mlruns")
        )

        # Setup MLflow
        mlflow_available = setup_mlflow(tracking_uri)

        # 🚨 Graceful exit if MLflow server is unavailable
        if not mlflow_available:
            logger.warning(
                "Skipping model registration because MLflow server is unavailable"
            )
            sys.exit(0)

        # Load model info
        model_info = load_model_info(registry_path)

        # Register model
        version = register_model(model_name, model_info)
        logger.info(f"Model registry completed successfully (version={version})")

    except Exception as e:
        logger.error(f"Model registry failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
