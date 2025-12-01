import numpy as np
import pandas as pd
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
import logging
import mlflow
import mlflow.sklearn
import os
import joblib
import yaml
from mlflow.models.signature import infer_signature
from typing import Dict, Any  # Add this import

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Setup MLflow tracking with proper error handling"""
    try:
        # Try to connect to MLflow server
        mlflow.set_tracking_uri('http://127.0.0.1:5000')
        # Test connection by listing experiments (non-blocking)
        mlflow.search_experiments(max_results=1)
        logger.info("MLflow tracking server connected successfully")
        return True
    except Exception as e:
        logger.warning(f"MLflow tracking server not available: {e}")
        logger.warning("Running without MLflow tracking...")
        # Disable MLflow tracking for tests
        mlflow.set_tracking_uri(None)
        return False

def params_load(file_path: str) -> Dict[str, Any]:
    """Load parameters from YAML file with error handling."""
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

def load_data(xtest, ytest):
    """Load test data from CSV files"""
    xtest = pd.read_csv(xtest)
    ytest = pd.read_csv(ytest)
    return xtest, ytest

def save_metrics(metrics, path):
    """Save metrics dictionary to JSON file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file"""
    model_info = {'run_id': run_id, 'model_path': model_path}
    with open(file_path, 'w') as file:
        json.dump(model_info, file, indent=4)

def load_model(path):
    """Load a trained model from file"""
    with open(path, 'rb') as f:
        clf = joblib.load(f)
        return clf

def evaluate_model(clf, X_test, y_test):
    """Evaluate model performance and return metrics"""
    
    # Check for empty data
    if len(X_test) == 0 or len(y_test) == 0:
        logger.warning("Empty test data provided, returning default metrics")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'auc': 0.5,  # AUC is 0.5 for random classifier
            'f1_score': 0.0
        }
    
    # Extract target values if y_test is a DataFrame
    if isinstance(y_test, pd.DataFrame):
        if 'Default' in y_test.columns:
            y_test_values = y_test['Default'].values
        else:
            y_test_values = y_test.iloc[:, 0].values
    else:
        y_test_values = y_test.values if hasattr(y_test, 'values') else y_test
    
    # Predict
    y_pred = clf.predict(X_test)
    
    # Calculate metrics with zero_division parameter to handle warnings
    try:
        accuracy = accuracy_score(y_test_values, y_pred)
    except Exception:
        accuracy = 0.0
    
    try:
        precision = precision_score(y_test_values, y_pred, zero_division=0)
    except Exception:
        precision = 0.0
    
    try:
        recall = recall_score(y_test_values, y_pred, zero_division=0)
    except Exception:
        recall = 0.0
    
    # Handle AUC calculation for edge cases
    try:
        auc = roc_auc_score(y_test_values, y_pred)
    except ValueError:
        # If only one class is present, set AUC to 0.5 (random)
        auc = 0.5
    except Exception:
        auc = 0.5
    
    try:
        f1 = f1_score(y_test_values, y_pred, zero_division=0)
    except Exception:
        f1 = 0.0

    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'f1_score': f1
    }

    return metrics_dict

def main():
    """Main function to run model evaluation pipeline"""
    params_path = 'params.yaml'
    
    try:
        params = params_load(params_path)
        metrics_output = params["model_evaluation"]["metrics_output"]
    except FileNotFoundError:
        logger.error(f"Params file not found: {params_path}")
        raise
    except KeyError as e:
        logger.error(f"Missing key in params file: {e}")
        raise
    
    # Setup MLflow
    mlflow_available = setup_mlflow()
    
    # Set MLflow experiment
    mlflow.set_experiment("Loan-defaulter-prediction-evaluation")
    
    # Ensure no active run before starting
    if mlflow.active_run():
        mlflow.end_run()
    
    try:
        # Load model and test data
        clf = load_model('Models/model.pkl')
        X_test, y_test = load_data('RAW_DATA/Clearn_data/X_test.csv', 'RAW_DATA/Clearn_data/y_test.csv')
        
        # Evaluate model
        metrics = evaluate_model(clf, X_test, y_test)
        logger.info(f'Model Evaluation Metrics: {metrics}')

        # Save metrics to file
        save_metrics(metrics, metrics_output)

        # Only log to MLflow if it's available
        if mlflow_available:
            with mlflow.start_run() as run:
                # Log metrics to MLflow
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                # Log model parameters to MLflow if available
                if hasattr(clf, 'get_params'):
                    model_params = clf.get_params()
                    for param_name, param_value in model_params.items():
                        mlflow.log_param(param_name, param_value)

                # Log model to MLflow with signature - FIXED LINE
                signature = infer_signature(X_test, clf.predict(X_test))

                mlflow.sklearn.log_model(
                    sk_model=clf,  # Explicitly name the model parameter
                    artifact_path="loan-default-model",  # This is the artifact_path
                    input_example=X_test.iloc[:5] if len(X_test) >= 5 else X_test,
                    signature=signature
                )

                # Save model info
                save_model_info(run.info.run_id, "loan-default-model", 'report/experiment_info.json')

                # Log the metrics file to MLflow
                mlflow.log_artifact(metrics_output)

                logger.info("Model evaluation completed successfully")

        else:
            # Save basic experiment info even without MLflow
            save_model_info("no_mlflow_run", "model", 'report/experiment_info.json')

        logger.info("Model evaluation completed successfully")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        # End the run if there's an error
        if mlflow_available and mlflow.active_run():
            mlflow.end_run()
        raise

if __name__ == '__main__':
    main()