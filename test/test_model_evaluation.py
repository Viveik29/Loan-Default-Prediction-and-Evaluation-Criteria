import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import json
import yaml
import joblib
from unittest.mock import Mock, patch, MagicMock
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score

# Add the src directory to the path to import your modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the functions from your model evaluation module
try:
    from model_evaluation import (
        params_load,
        load_data,
        save_metrics,
        save_model_info,
        load_model,
        evaluate_model,
        main
    )
except ImportError as e:
    print(f"Import Error: {e}")
    raise


class TestParamsLoad:
    """Test cases for parameters loading function"""
    
    def test_params_load_valid_yaml(self):
        """Test loading valid YAML parameters"""
        yaml_content = """
        model_evaluation:
            metrics_output: "report/metrics.json"
            threshold: 0.5
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_file = f.name
        
        try:
            params = params_load(temp_file)
            
            assert isinstance(params, dict)
            assert params['model_evaluation']['metrics_output'] == "report/metrics.json"
            assert params['model_evaluation']['threshold'] == 0.5
        finally:
            os.unlink(temp_file)
    
    def test_params_load_invalid_file(self):
        """Test loading from non-existent file"""
        with pytest.raises(FileNotFoundError):
            params_load('non_existent_file.yaml')


class TestLoadData:
    """Test cases for data loading function"""
    
    def test_load_data_returns_correct_types(self):
        """Test that load_data returns correct object types"""
        # Create temporary test files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            X_test_data = pd.DataFrame({
                'feature1': [1, 2, 3, 4, 5],
                'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
            })
            y_test_data = pd.DataFrame({
                'Default': [0, 1, 0, 1, 0]
            })
            
            # Save test files
            X_test_path = os.path.join(temp_dir, "X_test.csv")
            y_test_path = os.path.join(temp_dir, "y_test.csv")
            X_test_data.to_csv(X_test_path, index=False)
            y_test_data.to_csv(y_test_path, index=False)
            
            # Load data
            X_test, y_test = load_data(X_test_path, y_test_path)
            
            assert isinstance(X_test, pd.DataFrame)
            assert isinstance(y_test, pd.DataFrame)
            assert len(X_test) == 5
            assert len(y_test) == 5
    
    def test_load_data_missing_files(self):
        """Test load_data with missing files"""
        with pytest.raises(FileNotFoundError):
            load_data('missing_X.csv', 'missing_y.csv')


class TestSaveMetrics:
    """Test cases for metrics saving function"""
    
    def test_save_metrics_creates_file(self):
        """Test that save_metrics creates a JSON file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = {
                'accuracy': 0.85,
                'precision': 0.78,
                'recall': 0.82,
                'auc': 0.88,
                'f1_score': 0.80
            }
            
            output_path = os.path.join(temp_dir, "metrics", "test_metrics.json")
            save_metrics(metrics, output_path)
            
            assert os.path.exists(output_path)
    
    def test_save_metrics_correct_content(self):
        """Test that saved metrics contain correct data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = {
                'accuracy': 0.85,
                'precision': 0.78,
                'recall': 0.82
            }
            
            output_path = os.path.join(temp_dir, "test_metrics.json")
            save_metrics(metrics, output_path)
            
            # Load and verify the saved metrics
            with open(output_path, 'r') as f:
                loaded_metrics = json.load(f)
            
            assert loaded_metrics == metrics


class TestSaveModelInfo:
    """Test cases for model info saving function"""
    
    def test_save_model_info_creates_file(self):
        """Test that save_model_info creates a JSON file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_id = "test_run_123"
            model_path = "models/test_model.pkl"
            output_path = os.path.join(temp_dir, "model_info.json")
            
            save_model_info(run_id, model_path, output_path)
            
            assert os.path.exists(output_path)
    
    def test_save_model_info_correct_content(self):
        """Test that saved model info contains correct data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_id = "test_run_123"
            model_path = "models/test_model.pkl"
            output_path = os.path.join(temp_dir, "model_info.json")
            
            save_model_info(run_id, model_path, output_path)
            
            # Load and verify the saved info
            with open(output_path, 'r') as f:
                loaded_info = json.load(f)
            
            expected_info = {
                'run_id': run_id,
                'model_path': model_path
            }
            assert loaded_info == expected_info


class TestLoadModel:
    """Test cases for model loading function"""
    
    def test_load_model_returns_model(self):
        """Test that load_model returns a model object"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple model and save it
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            X_dummy = np.random.rand(10, 3)
            y_dummy = np.random.randint(0, 2, 10)
            model.fit(X_dummy, y_dummy)
            
            model_path = os.path.join(temp_dir, "test_model.pkl")
            joblib.dump(model, model_path)
            
            # Load the model
            loaded_model = load_model(model_path)
            
            assert loaded_model is not None
            assert hasattr(loaded_model, 'predict')
    
    def test_load_model_invalid_file(self):
        """Test load_model with invalid file"""
        with pytest.raises(FileNotFoundError):
            load_model('non_existent_model.pkl')


class TestEvaluateModel:
    """Test cases for model evaluation function"""
    
    @pytest.fixture
    def sample_model(self):
        """Fixture providing a trained model for testing"""
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X = np.random.rand(100, 3)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        return model
    
    @pytest.fixture
    def sample_test_data(self):
        """Fixture providing sample test data with 3 features"""
        X_test = pd.DataFrame({
            'feature1': np.random.rand(20),
            'feature2': np.random.rand(20),
            'feature3': np.random.rand(20)
        })
        y_test = pd.Series(np.random.randint(0, 2, 20))
        return X_test, y_test
    
    def test_evaluate_model_returns_metrics(self, sample_model, sample_test_data):
        """Test that evaluate_model returns all expected metrics"""
        X_test, y_test = sample_test_data
        
        # Use the model directly without mocking joblib.load
        metrics = evaluate_model(sample_model, X_test, y_test)
        
        expected_metrics = ['accuracy', 'precision', 'recall', 'auc', 'f1_score']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
            # Metrics should be between 0 and 1
            assert 0 <= metrics[metric] <= 1
    
    def test_evaluate_model_correct_calculations(self):
        """Test that metrics are calculated correctly"""
        # Create a mock model that returns predictable outputs
        mock_model = Mock()
        
        # Simple case: perfect predictions with 2 features
        X_test = pd.DataFrame({'feature1': [1, 2, 3, 4], 'feature2': [1, 2, 3, 4]})
        y_test = pd.Series([0, 1, 0, 1])
        mock_model.predict.return_value = np.array([0, 1, 0, 1])  # Perfect predictions
        
        metrics = evaluate_model(mock_model, X_test, y_test)
        
        # With perfect predictions, all metrics should be 1.0
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_evaluate_model_with_dataframe_target(self):
        """Test evaluation when y_test is a DataFrame"""
        mock_model = Mock()
        X_test = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [1, 2, 3]})
        y_test_df = pd.DataFrame({'Default': [0, 1, 0]})  # DataFrame target
        
        mock_model.predict.return_value = np.array([0, 1, 0])
        
        metrics = evaluate_model(mock_model, X_test, y_test_df)
        
        # Should handle DataFrame target correctly
        assert 'accuracy' in metrics
    
    def test_evaluate_model_empty_data(self):
        """Test evaluation with empty test data"""
        mock_model = Mock()
        X_test = pd.DataFrame()
        y_test = pd.Series([])
        
        mock_model.predict.return_value = np.array([])
        
        # The function should handle empty data gracefully and return metrics with default values
        # or raise an exception. Let's test what actually happens.
        try:
            metrics = evaluate_model(mock_model, X_test, y_test)
            # If it doesn't raise, check that it returns valid metrics structure
            assert isinstance(metrics, dict)
            expected_metrics = ['accuracy', 'precision', 'recall', 'auc', 'f1_score']
            for metric in expected_metrics:
                assert metric in metrics
        except (ValueError, IndexError, ZeroDivisionError) as e:
            # It's also acceptable for it to raise an exception with empty data
            assert isinstance(e, (ValueError, IndexError, ZeroDivisionError))
    
    def test_evaluate_model_handles_warnings(self):
        """Test that evaluate_model handles sklearn warnings gracefully"""
        mock_model = Mock()
        
        # Create data that might cause warnings (e.g., single class)
        X_test = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [1, 2, 3]})
        y_test = pd.Series([0, 0, 0])  # All same class - might cause warnings
        
        mock_model.predict.return_value = np.array([0, 0, 0])
        
        # This should not crash even with warnings
        metrics = evaluate_model(mock_model, X_test, y_test)
        
        assert 'accuracy' in metrics


class TestMainFunction:
    """Test cases for the main function"""
    
    @pytest.fixture
    def sample_params_content(self):
        """Fixture providing sample params content"""
        return """
        model_evaluation:
            metrics_output: "report/metrics.json"
            threshold: 0.5
        """
    
    @pytest.fixture
    def sample_test_files(self):
        """Fixture creating sample test files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data with consistent feature dimensions
            X_test_data = pd.DataFrame({
                'feature1': np.random.rand(10),
                'feature2': np.random.rand(10),
                'feature3': np.random.rand(10)
            })
            y_test_data = pd.DataFrame({
                'Default': np.random.randint(0, 2, 10)
            })
            
            # Create directories and save files
            os.makedirs(os.path.join(temp_dir, "RAW_DATA", "Clearn_data"), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, "Models"), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, "report"), exist_ok=True)
            
            X_test_data.to_csv(os.path.join(temp_dir, "RAW_DATA", "Clearn_data", "X_test.csv"), index=False)
            y_test_data.to_csv(os.path.join(temp_dir, "RAW_DATA", "Clearn_data", "y_test.csv"), index=False)
            
            # Create a simple model with matching feature dimensions
            model = RandomForestClassifier(n_estimators=5, random_state=42)
            X_dummy = np.random.rand(20, 3)
            y_dummy = np.random.randint(0, 2, 20)
            model.fit(X_dummy, y_dummy)
            
            joblib.dump(model, os.path.join(temp_dir, "Models", "model.pkl"))
            
            yield temp_dir
    
    @patch('model_evaluation.mlflow.set_tracking_uri')
    @patch('model_evaluation.mlflow.set_experiment')
    @patch('model_evaluation.mlflow.start_run')
    @patch('model_evaluation.mlflow.log_metric')
    @patch('model_evaluation.mlflow.sklearn.log_model')
    @patch('model_evaluation.mlflow.log_artifact')
    @patch('model_evaluation.mlflow.active_run')
    def test_main_creates_output_files(self, mock_active_run, mock_log_artifact, mock_log_model, 
                                     mock_log_metric, mock_start_run, mock_set_experiment, 
                                     mock_set_tracking_uri, sample_params_content, sample_test_files, 
                                     monkeypatch):
        """Test that main function creates expected output files"""
        # Mock MLflow run
        mock_run = Mock()
        mock_run.info.run_id = "test_run_123"
        mock_start_run.return_value.__enter__ = Mock(return_value=mock_run)
        mock_start_run.return_value.__exit__ = Mock(return_value=None)
        
        # Mock active_run to return None (no active run)
        mock_active_run.return_value = None
        
        # Create params file
        params_path = os.path.join(sample_test_files, "params.yaml")
        with open(params_path, 'w') as f:
            f.write(sample_params_content)
        
        original_cwd = os.getcwd()
        monkeypatch.chdir(sample_test_files)
        
        try:
            # Run main function
            main()
            
            # Check if metrics file was created
            metrics_path = os.path.join("report", "metrics.json")
            assert os.path.exists(metrics_path)
            
            # Check if experiment info file was created
            experiment_info_path = os.path.join("report", "experiment_info.json")
            assert os.path.exists(experiment_info_path)
            
            # Verify metrics content
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            expected_metrics = ['accuracy', 'precision', 'recall', 'auc', 'f1_score']
            for metric in expected_metrics:
                assert metric in metrics
                assert 0 <= metrics[metric] <= 1
            
            # Verify experiment info content
            with open(experiment_info_path, 'r') as f:
                experiment_info = json.load(f)
            
            assert 'run_id' in experiment_info
            assert 'model_path' in experiment_info
            
        finally:
            monkeypatch.chdir(original_cwd)
    
    @patch('model_evaluation.mlflow.set_tracking_uri')
    @patch('model_evaluation.mlflow.set_experiment')
    @patch('model_evaluation.mlflow.active_run')
    def test_main_missing_model_file(self, mock_active_run, mock_set_experiment, 
                                   mock_set_tracking_uri, sample_params_content, 
                                   sample_test_files, monkeypatch):
        """Test main function when model file is missing"""
        # Mock active_run to return None (no active run)
        mock_active_run.return_value = None
        
        # Remove model file
        model_path = os.path.join(sample_test_files, "Models", "model.pkl")
        if os.path.exists(model_path):
            os.remove(model_path)
        
        # Create params file
        params_path = os.path.join(sample_test_files, "params.yaml")
        with open(params_path, 'w') as f:
            f.write(sample_params_content)
        
        original_cwd = os.getcwd()
        monkeypatch.chdir(sample_test_files)
        
        try:
            with pytest.raises(FileNotFoundError):
                main()
        finally:
            monkeypatch.chdir(original_cwd)
    
    @patch('model_evaluation.mlflow.set_tracking_uri')
    @patch('model_evaluation.mlflow.set_experiment')
    @patch('model_evaluation.mlflow.active_run')
    def test_main_missing_test_files(self, mock_active_run, mock_set_experiment, 
                                   mock_set_tracking_uri, sample_params_content, 
                                   sample_test_files, monkeypatch):
        """Test main function when test files are missing"""
        # Mock active_run to return None (no active run)
        mock_active_run.return_value = None
        
        # Remove test files
        X_test_path = os.path.join(sample_test_files, "RAW_DATA", "Clearn_data", "X_test.csv")
        y_test_path = os.path.join(sample_test_files, "RAW_DATA", "Clearn_data", "y_test.csv")
        
        if os.path.exists(X_test_path):
            os.remove(X_test_path)
        if os.path.exists(y_test_path):
            os.remove(y_test_path)
        
        # Create params file
        params_path = os.path.join(sample_test_files, "params.yaml")
        with open(params_path, 'w') as f:
            f.write(sample_params_content)
        
        original_cwd = os.getcwd()
        monkeypatch.chdir(sample_test_files)
        
        try:
            with pytest.raises(FileNotFoundError):
                main()
        finally:
            monkeypatch.chdir(original_cwd)


class TestIntegration:
    """Integration tests for the entire model evaluation flow"""
    
    @patch('model_evaluation.mlflow.set_tracking_uri')
    @patch('model_evaluation.mlflow.set_experiment')
    @patch('model_evaluation.mlflow.start_run')
    @patch('model_evaluation.mlflow.log_metric')
    @patch('model_evaluation.mlflow.sklearn.log_model')
    @patch('model_evaluation.mlflow.log_artifact')
    @patch('model_evaluation.mlflow.active_run')
    def test_end_to_end_evaluation(self, mock_active_run, mock_log_artifact, mock_log_model, 
                                 mock_log_metric, mock_start_run, mock_set_experiment, 
                                 mock_set_tracking_uri):
        """Test complete model evaluation workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock MLflow run
            mock_run = Mock()
            mock_run.info.run_id = "test_run_123"
            mock_start_run.return_value.__enter__ = Mock(return_value=mock_run)
            mock_start_run.return_value.__exit__ = Mock(return_value=None)
            mock_active_run.return_value = None
            
            # Create test data with consistent dimensions
            X_test = pd.DataFrame({
                'feature1': np.random.rand(50),
                'feature2': np.random.rand(50),
                'feature3': np.random.rand(50)
            })
            y_test = pd.Series(np.random.randint(0, 2, 50))
            
            # Create and train a model with matching dimensions
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            X_train = np.random.rand(100, 3)
            y_train = np.random.randint(0, 2, 100)
            model.fit(X_train, y_train)
            
            # Save test data and model
            os.makedirs(os.path.join(temp_dir, "RAW_DATA", "Clearn_data"), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, "Models"), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, "report"), exist_ok=True)
            
            X_test.to_csv(os.path.join(temp_dir, "RAW_DATA", "Clearn_data", "X_test.csv"), index=False)
            y_test.to_csv(os.path.join(temp_dir, "RAW_DATA", "Clearn_data", "y_test.csv"), index=False)
            joblib.dump(model, os.path.join(temp_dir, "Models", "model.pkl"))
            
            # Create params file
            params_content = """
            model_evaluation:
                metrics_output: "report/metrics.json"
                threshold: 0.5
            """
            with open(os.path.join(temp_dir, "params.yaml"), 'w') as f:
                f.write(params_content)
            
            # Test individual functions
            # Load model
            loaded_model = load_model(os.path.join(temp_dir, "Models", "model.pkl"))
            assert loaded_model is not None
            
            # Load test data
            X_loaded, y_loaded = load_data(
                os.path.join(temp_dir, "RAW_DATA", "Clearn_data", "X_test.csv"),
                os.path.join(temp_dir, "RAW_DATA", "Clearn_data", "y_test.csv")
            )
            assert len(X_loaded) == 50
            assert len(y_loaded) == 50
            
            # Evaluate model
            metrics = evaluate_model(loaded_model, X_loaded, y_loaded)
            
            # Verify metrics
            expected_keys = ['accuracy', 'precision', 'recall', 'auc', 'f1_score']
            for key in expected_keys:
                assert key in metrics
                assert isinstance(metrics[key], float)


# Test configuration to handle MLflow cleanup
@pytest.fixture(autouse=True)
def cleanup_mlflow():
    """Cleanup MLflow runs after each test"""
    yield
    # Import mlflow here to avoid circular imports
    import mlflow
    try:
        # End any active run
        if mlflow.active_run():
            mlflow.end_run()
    except:
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])