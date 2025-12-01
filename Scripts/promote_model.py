#!/usr/bin/env python3
"""
Model Promotion Script for CI/CD Pipeline
Evaluates model performance metrics and promotes to Staging/Production
based on predefined thresholds.
"""

import json
import mlflow
import logging
import os
import sys
import yaml
import pandas as pd
from typing import Dict, Any, Tuple
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('model_promotion.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'params.yaml') -> Dict[str, Any]:
    """
    Load promotion configuration from YAML file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set default thresholds if not specified
        if 'model_promotion' not in config:
            config['model_promotion'] = {
                'staging_thresholds': {
                    'accuracy': 0.75,
                    'precision': 0.70,
                    'recall': 0.65,
                    'f1_score': 0.70,
                    'auc': 0.75
                },
                'production_thresholds': {
                    'accuracy': 0.85,
                    'precision': 0.80,
                    'recall': 0.75,
                    'f1_score': 0.80,
                    'auc': 0.85
                },
                'min_samples': 100,
                'metrics_path': 'report/metrics.json',
                'experiment_name': 'Loan-defaulter-prediction-evaluation'
            }
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def load_metrics(metrics_path: str) -> Dict[str, float]:
    """
    Load model evaluation metrics from JSON file.
    
    Args:
        metrics_path (str): Path to metrics JSON file
        
    Returns:
        Dict[str, float]: Model metrics dictionary
    """
    try:
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
        
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        for metric in required_metrics:
            if metric not in metrics:
                raise ValueError(f"Missing required metric: {metric}")
        
        logger.info(f"Loaded metrics from {metrics_path}")
        logger.info(f"Metrics: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error loading metrics: {e}")
        raise


def setup_mlflow(tracking_uri: str = "http://127.0.0.1:5000") -> bool:
    """
    Setup MLflow connection.
    
    Args:
        tracking_uri (str): MLflow tracking URI
        
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        mlflow.set_tracking_uri(tracking_uri)
        # Test connection
        mlflow.search_experiments(max_results=1)
        logger.info(f"Connected to MLflow at {tracking_uri}")
        return True
    except Exception as e:
        logger.warning(f"Could not connect to MLflow: {e}")
        return False


def evaluate_model_metrics(
    metrics: Dict[str, float],
    thresholds: Dict[str, float]
) -> Tuple[bool, Dict[str, bool]]:
    """
    Evaluate if model metrics meet promotion thresholds.
    
    Args:
        metrics (Dict[str, float]): Model metrics
        thresholds (Dict[str, float]): Threshold values
        
    Returns:
        Tuple[bool, Dict[str, bool]]: (All thresholds met, individual results)
    """
    results = {}
    all_met = True
    
    for metric_name, threshold_value in thresholds.items():
        if metric_name in metrics:
            metric_value = metrics[metric_name]
            met_threshold = metric_value >= threshold_value
            results[metric_name] = {
                'value': metric_value,
                'threshold': threshold_value,
                'met': met_threshold,
                'difference': metric_value - threshold_value
            }
            
            if not met_threshold:
                all_met = False
                logger.warning(
                    f"Metric '{metric_name}' failed threshold: "
                    f"{metric_value:.4f} < {threshold_value:.4f}"
                )
            else:
                logger.info(
                    f"Metric '{metric_name}' passed threshold: "
                    f"{metric_value:.4f} >= {threshold_value:.4f}"
                )
    
    return all_met, results


def get_latest_model_version(model_name: str) -> Tuple[int, str]:
    """
    Get the latest version of a model from MLflow Model Registry.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        Tuple[int, str]: (version number, current stage)
    """
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Get all versions of the model
        model_versions = client.search_model_versions(f"name='{model_name}'")
        
        if not model_versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        
        # Find the latest version (highest version number)
        latest_version = max(model_versions, key=lambda x: x.version)
        
        return int(latest_version.version), latest_version.current_stage
        
    except Exception as e:
        logger.error(f"Error getting latest model version: {e}")
        raise


def promote_model(
    model_name: str,
    version: int,
    from_stage: str,
    to_stage: str,
    archive_existing: bool = False
) -> bool:
    """
    Promote a model version to a new stage in MLflow Model Registry.
    
    Args:
        model_name (str): Name of the model
        version (int): Model version number
        from_stage (str): Current stage (e.g., 'Staging')
        to_stage (str): Target stage (e.g., 'Production')
        archive_existing (bool): Whether to archive existing models in target stage
        
    Returns:
        bool: True if promotion successful, False otherwise
    """
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Archive existing models in target stage if requested
        if archive_existing and to_stage in ['Staging', 'Production']:
            existing_models = client.search_model_versions(
                f"name='{model_name}' and current_stage='{to_stage}'"
            )
            
            for model in existing_models:
                logger.info(f"Archiving version {model.version} from {to_stage} stage")
                client.transition_model_version_stage(
                    name=model_name,
                    version=model.version,
                    stage="Archived"
                )
        
        # Promote the model
        logger.info(f"Promoting model '{model_name}' version {version} from {from_stage} to {to_stage}")
        
        client.transition_model_version_stage(
            name=model_name,
            version=str(version),
            stage=to_stage,
            archive_existing_versions=archive_existing
        )
        
        logger.info(f"✅ Successfully promoted model to {to_stage} stage")
        return True
        
    except Exception as e:
        logger.error(f"Error promoting model: {e}")
        return False


def create_promotion_report(
    metrics: Dict[str, float],
    evaluation_results: Dict[str, Any],
    thresholds: Dict[str, float],
    promotion_decision: str,
    model_name: str,
    model_version: int
) -> None:
    """
    Create a detailed promotion report.
    
    Args:
        metrics (Dict[str, float]): Model metrics
        evaluation_results (Dict[str, Any]): Threshold evaluation results
        thresholds (Dict[str, float]): Threshold values used
        promotion_decision (str): Final promotion decision
        model_name (str): Name of the model
        model_version (int): Model version
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name,
        'model_version': model_version,
        'promotion_decision': promotion_decision,
        'metrics': metrics,
        'thresholds': thresholds,
        'evaluation_results': evaluation_results,
        'summary': {
            'total_metrics': len(metrics),
            'metrics_passed': sum(1 for r in evaluation_results.values() if r['met']),
            'metrics_failed': sum(1 for r in evaluation_results.values() if not r['met'])
        }
    }
    
    # Save report to file
    report_path = f'report/promotion_report_v{model_version}.json'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    logger.info(f"Promotion report saved to {report_path}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("MODEL PROMOTION REPORT")
    print("="*60)
    print(f"Model: {model_name} (v{model_version})")
    print(f"Decision: {promotion_decision}")
    print(f"Timestamp: {report['timestamp']}")
    print("\nMETRIC EVALUATION:")
    print("-"*60)
    
    for metric_name, result in evaluation_results.items():
        status = "✅ PASS" if result['met'] else "❌ FAIL"
        print(f"{metric_name:15} {result['value']:.4f} >= {result['threshold']:.4f} {status}")
    
    print(f"\n📊 Summary: {report['summary']['metrics_passed']}/{report['summary']['total_metrics']} metrics passed")
    print("="*60)


def main():
    """Main function to run model promotion pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Promote model based on performance metrics')
    parser.add_argument('--config', type=str, default='params.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--metrics', type=str, default=None,
                       help='Path to metrics JSON file (overrides config)')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Model name (overrides config)')
    parser.add_argument('--force-staging', action='store_true',
                       help='Force promotion to Staging regardless of metrics')
    parser.add_argument('--force-production', action='store_true',
                       help='Force promotion to Production regardless of metrics')
    parser.add_argument('--dry-run', action='store_true',
                       help='Evaluate metrics without actual promotion')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting model promotion pipeline")
        
        # Load configuration
        config = load_config(args.config)
        promotion_config = config.get('model_promotion', {})
        
        # Get paths and thresholds
        metrics_path = args.metrics or promotion_config.get('metrics_path', 'report/metrics.json')
        model_name = args.model_name or promotion_config.get('model_name', 'loan-default-model')
        staging_thresholds = promotion_config.get('staging_thresholds', {})
        production_thresholds = promotion_config.get('production_thresholds', {})
        
        # Load metrics
        metrics = load_metrics(metrics_path)
        
        # Setup MLflow
        tracking_uri = config.get('mlflow', {}).get('tracking_uri', 'http://127.0.0.1:5000')
        mlflow_connected = setup_mlflow(tracking_uri)
        
        if not mlflow_connected:
            logger.error("Cannot connect to MLflow. Promotion aborted.")
            sys.exit(1)
        
        # Get latest model version
        try:
            latest_version, current_stage = get_latest_model_version(model_name)
            logger.info(f"Latest model version: {latest_version}, Current stage: {current_stage}")
        except ValueError as e:
            logger.error(f"Cannot find model: {e}")
            sys.exit(1)
        
        # Force promotion flags
        if args.force_production:
            promotion_decision = "FORCED_PRODUCTION"
            target_stage = "Production"
            logger.warning("⚠️  Forcing promotion to Production (bypassing metrics check)")
            
        elif args.force_staging:
            promotion_decision = "FORCED_STAGING"
            target_stage = "Staging"
            logger.warning("⚠️  Forcing promotion to Staging (bypassing metrics check)")
            
        else:
            # Evaluate metrics against thresholds
            logger.info("Evaluating metrics against Staging thresholds...")
            staging_passed, staging_results = evaluate_model_metrics(metrics, staging_thresholds)
            
            logger.info("Evaluating metrics against Production thresholds...")
            production_passed, production_results = evaluate_model_metrics(metrics, production_thresholds)
            
            # Make promotion decision
            if production_passed:
                promotion_decision = "PRODUCTION_READY"
                target_stage = "Production"
                logger.info("✅ Model meets Production thresholds")
                
            elif staging_passed:
                promotion_decision = "STAGING_READY"
                target_stage = "Staging"
                logger.info("✅ Model meets Staging thresholds")
                
            else:
                promotion_decision = "REJECTED"
                target_stage = None
                logger.error("❌ Model does not meet minimum Staging thresholds")
        
        # Execute promotion if not in dry-run mode
        if args.dry_run:
            logger.info("🔄 Dry run mode - No actual promotion will be performed")
            evaluation_results = staging_results if staging_passed else production_results
            create_promotion_report(
                metrics, evaluation_results, 
                staging_thresholds if staging_passed else production_thresholds,
                f"DRY_RUN_{promotion_decision}",
                model_name, latest_version
            )
            sys.exit(0)
        
        if promotion_decision in ["REJECTED", None]:
            logger.error("Model promotion rejected due to failed metrics")
            create_promotion_report(
                metrics, staging_results, staging_thresholds,
                "REJECTED", model_name, latest_version
            )
            sys.exit(1)
        
        # Promote the model
        if target_stage and current_stage != target_stage:
            success = promote_model(
                model_name=model_name,
                version=latest_version,
                from_stage=current_stage,
                to_stage=target_stage,
                archive_existing=True if target_stage == "Production" else False
            )
            
            if not success:
                logger.error("Failed to promote model")
                sys.exit(1)
        else:
            logger.info(f"Model is already in {current_stage} stage. No promotion needed.")
        
        # Create promotion report
        evaluation_results = production_results if target_stage == "Production" else staging_results
        thresholds = production_thresholds if target_stage == "Production" else staging_thresholds
        create_promotion_report(
            metrics, evaluation_results, thresholds,
            promotion_decision, model_name, latest_version
        )
        
        logger.info(f"🎉 Model promotion pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Model promotion pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()