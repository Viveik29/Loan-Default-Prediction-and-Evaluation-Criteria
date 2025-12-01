import json
import mlflow
import logging
import os
import sys
import warnings
from typing import Dict, Any
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('model_registry.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


def params_load(file_path: str) -> Dict[str, Any]:
    """
    Load parameters from YAML file with error handling.
    
    Args:
        file_path (str): Path to the YAML parameters file
        
    Returns:
        Dict[str, Any]: Loaded parameters
        
    Raises:
        FileNotFoundError: If the parameters file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parameters file not found: {file_path}")
        
        with open(file_path, "r") as f:
            params = yaml.safe_load(f)
            
        if params is None:
            logger.warning(f"Parameters file '{file_path}' is empty")
            return {}
            
        logger.info(f"Successfully loaded parameters from {file_path}")
        return params
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading parameters from {file_path}: {e}")
        raise


def load_model_info(file_path: str) -> Dict[str, Any]:
    """
    Load model information from JSON file with validation.
    
    Args:
        file_path (str): Path to the model info JSON file
        
    Returns:
        Dict[str, Any]: Model information containing run_id and model_path
        
    Raises:
        FileNotFoundError: If the model info file doesn't exist
        json.JSONDecodeError: If the JSON file is malformed
        KeyError: If required keys are missing
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model info file not found: {file_path}")
        
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        
        # Validate required keys
        required_keys = ['run_id', 'model_path']
        missing_keys = [key for key in required_keys if key not in model_info]
        
        if missing_keys:
            error_msg = f"Model info file missing required keys: {missing_keys}"
            logger.error(error_msg)
            raise KeyError(error_msg)
        
        # Validate values
        if not model_info['run_id']:
            raise ValueError("run_id cannot be empty")
        if not model_info['model_path']:
            raise ValueError("model_path cannot be empty")
            
        logger.info(f"Successfully loaded model info from {file_path}")
        logger.debug(f"Model info: {model_info}")
        return model_info
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading model info from {file_path}: {e}")
        raise


def setup_mlflow_tracking(tracking_uri: str = "http://127.0.0.1:5000") -> bool:
    """
    Setup MLflow tracking with connection validation.
    
    Args:
        tracking_uri (str): MLflow tracking server URI
        
    Returns:
        bool: True if MLflow server is accessible, False otherwise
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        
        # Test connection by listing experiments (non-blocking)
        experiments = mlflow.search_experiments(max_results=1)
        logger.info(f"Successfully connected to MLflow tracking server: {tracking_uri}")
        logger.debug(f"Found {len(experiments)} experiments")
        return True
        
    except Exception as e:
        logger.warning(f"Could not connect to MLflow tracking server {tracking_uri}: {e}")
        logger.warning("Model registration will be attempted but may fail")
        return False


def register_model(model_name: str, model_info: Dict[str, Any]) -> str:
    """
    Register the model to the MLflow Model Registry with comprehensive error handling.
    
    Args:
        model_name (str): Name to register the model under
        model_info (Dict[str, Any]): Model information containing run_id and model_path
        
    Returns:
        str: Registered model version
        
    Raises:
        ValueError: If model_name is invalid
        KeyError: If required model_info keys are missing
        Exception: For MLflow registration failures
    """
    # Validate inputs
    if not model_name or not isinstance(model_name, str):
        raise ValueError("model_name must be a non-empty string")
    
    if not model_info:
        raise ValueError("model_info cannot be empty")
    
    required_keys = ['run_id', 'model_path']
    for key in required_keys:
        if key not in model_info:
            raise KeyError(f"model_info missing required key: {key}")
        if not model_info[key]:
            raise ValueError(f"model_info['{key}'] cannot be empty")
    
    try:
        # Construct model URI
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logger.info(f"Registering model '{model_name}' from URI: {model_uri}")
        
        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)
        logger.info(f"Successfully registered model '{model_name}' as version {model_version.version}")
        
        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        logger.info(f"Model '{model_name}' version {model_version.version} transitioned to Staging stage")
        
        return f"{model_version.version}"
        
    except mlflow.exceptions.MlflowException as e:
        error_msg = f"MLflow error during model registration: {e}"
        logger.error(error_msg)
        raise Exception(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error during model registration: {e}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


def validate_params(params: Dict[str, Any]) -> None:
    """
    Validate that all required parameters are present.
    
    Args:
        params (Dict[str, Any]): Loaded parameters
        
    Raises:
        KeyError: If required parameters are missing
    """
    if 'model_registry' not in params:
        raise KeyError("Missing 'model_registry' section in parameters")
    
    registry_params = params['model_registry']
    required_keys = ['registry_path', 'model_name']
    missing_keys = [key for key in required_keys if key not in registry_params]
    
    if missing_keys:
        raise KeyError(f"Missing required parameters in model_registry section: {missing_keys}")
    
    if not registry_params['registry_path']:
        raise ValueError("registry_path cannot be empty")
    if not registry_params['model_name']:
        raise ValueError("model_name cannot be empty")


def main():
    """
    Main function to run model registration pipeline.
    
    Raises:
        SystemExit: If the pipeline fails
    """
    try:
        logger.info("Starting model registration pipeline")
        
        # Load parameters
        params_path = 'params.yaml'
        logger.info(f"Loading parameters from {params_path}")
        params = params_load(params_path)
        
        # Validate parameters
        validate_params(params)
        
        registry_config = params["model_registry"]
        registry_path = registry_config["registry_path"]
        model_name = registry_config["model_name"]
        
        logger.info(f"Registry path: {registry_path}")
        logger.info(f"Model name: {model_name}")
        
        # Setup MLflow tracking
        mlflow_available = setup_mlflow_tracking("http://127.0.0.1:5000")
        
        if not mlflow_available:
            logger.warning("MLflow server not available. Registration may fail.")
        
        # Load model info
        logger.info(f"Loading model info from {registry_path}")
        model_info = load_model_info(registry_path)
        
        # Register model
        logger.info(f"Registering model '{model_name}'")
        version = register_model(model_name, model_info)
        
        logger.info(f"Model registration completed successfully. Version: {version}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except KeyError as e:
        logger.error(f"Missing required configuration: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Model registration pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()