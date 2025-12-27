import pytest
import json
import os
import tempfile
import yaml
from unittest.mock import Mock, patch, MagicMock
import sys
from src.model_registry import load_params   # or alias


# Add the src directory to the path to import your modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the functions from your model registry module
try:
    from src.model_registry import (
        params_load,
        load_model_info,
        register_model,
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
        model_registry:
            registry_path: "report/experiment_info.json"
            model_name: "loan-default-model"
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_file = f.name
        
        try:
            params = params_load(temp_file)
            
            assert isinstance(params, dict)
            assert params['model_registry']['registry_path'] == "report/experiment_info.json"
            assert params['model_registry']['model_name'] == "loan-default-model"
        finally:
            os.unlink(temp_file)
    
    def test_params_load_invalid_file(self):
        """Test loading from non-existent file"""
        with pytest.raises(FileNotFoundError):
            params_load('non_existent_file.yaml')
    
    def test_params_load_invalid_yaml(self):
        """Test loading invalid YAML content"""
        invalid_yaml_content = """
        model_registry:
            registry_path: "report/experiment_info.json"
            model_name: [unclosed list
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml_content)
            temp_file = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                params_load(temp_file)
        finally:
            os.unlink(temp_file)


class TestLoadModelInfo:
    """Test cases for model info loading function"""
    
    def test_load_model_info_valid_file(self):
        """Test loading valid model info from JSON file"""
        model_info_data = {
            'run_id': 'test_run_123',
            'model_path': 'model'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(model_info_data, f)
            temp_file = f.name
        
        try:
            model_info = load_model_info(temp_file)
            
            assert isinstance(model_info, dict)
            assert model_info['run_id'] == 'test_run_123'
            assert model_info['model_path'] == 'model'
        finally:
            os.unlink(temp_file)
    
    def test_load_model_info_invalid_file(self):
        """Test loading from non-existent file"""
        with pytest.raises(FileNotFoundError):
            load_model_info('non_existent_file.json')
    
    def test_load_model_info_invalid_json(self):
        """Test loading invalid JSON content"""
        invalid_json_content = "{'run_id': 'test', 'model_path': 'model'"  # Missing closing brace
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(invalid_json_content)
            temp_file = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_model_info(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_load_model_info_missing_keys(self):
        """Test loading model info with missing required keys"""
        model_info_data = {
            'run_id': 'test_run_123'
            # Missing 'model_path'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(model_info_data, f)
            temp_file = f.name
        
        try:
            with pytest.raises(KeyError) as exc_info:
                load_model_info(temp_file)
            assert "missing required keys" in str(exc_info.value)
        finally:
            os.unlink(temp_file)
    
    def test_load_model_info_empty_values(self):
        """Test loading model info with empty required values"""
        model_info_data = {
            'run_id': '',  # Empty run_id
            'model_path': 'model'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(model_info_data, f)
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError) as exc_info:
                load_model_info(temp_file)
            assert "cannot be empty" in str(exc_info.value)
        finally:
            os.unlink(temp_file)


class TestRegisterModel:
    """Test cases for model registration function"""
    
    @patch('model_registry.mlflow.tracking.MlflowClient')
    @patch('model_registry.mlflow.register_model')
    def test_register_model_success(self, mock_register_model, mock_mlflow_client):
        """Test successful model registration"""
        # Mock the model version
        mock_version = Mock()
        mock_version.version = 1
        mock_register_model.return_value = mock_version
        
        # Mock the client
        mock_client_instance = Mock()
        mock_mlflow_client.return_value = mock_client_instance
        
        model_name = "loan-default-model"
        model_info = {
            'run_id': 'test_run_123',
            'model_path': 'model'
        }
        
        # Call the function
        version = register_model(model_name, model_info)
        
        # Verify MLflow calls
        expected_model_uri = "runs:/test_run_123/model"
        mock_register_model.assert_called_once_with(expected_model_uri, model_name)
        
        mock_mlflow_client.assert_called_once()
        mock_client_instance.transition_model_version_stage.assert_called_once_with(
            name=model_name,
            version=1,
            stage="Staging"
        )
        
        assert version == "1"
    
    @patch('model_registry.mlflow.tracking.MlflowClient')
    @patch('model_registry.mlflow.register_model')
    def test_register_model_missing_run_id(self, mock_register_model, mock_mlflow_client):
        """Test model registration with missing run_id"""
        model_name = "loan-default-model"
        model_info = {
            'model_path': 'model'
            # Missing 'run_id'
        }
        
        with pytest.raises(KeyError):
            register_model(model_name, model_info)
        
        # Verify no MLflow calls were made
        mock_register_model.assert_not_called()
        mock_mlflow_client.assert_not_called()
    
    @patch('model_registry.mlflow.tracking.MlflowClient')
    @patch('model_registry.mlflow.register_model')
    def test_register_model_missing_model_path(self, mock_register_model, mock_mlflow_client):
        """Test model registration with missing model_path"""
        model_name = "loan-default-model"
        model_info = {
            'run_id': 'test_run_123'
            # Missing 'model_path'
        }
        
        with pytest.raises(KeyError):
            register_model(model_name, model_info)
        
        # Verify no MLflow calls were made
        mock_register_model.assert_not_called()
        mock_mlflow_client.assert_not_called()
    
    @patch('model_registry.mlflow.tracking.MlflowClient')
    @patch('model_registry.mlflow.register_model')
    def test_register_model_mlflow_error(self, mock_register_model, mock_mlflow_client):
        """Test model registration when MLflow fails"""
        # Mock MLflow to raise an exception
        mock_register_model.side_effect = Exception("MLflow server unavailable")
        
        model_name = "loan-default-model"
        model_info = {
            'run_id': 'test_run_123',
            'model_path': 'model'
        }
        
        with pytest.raises(Exception) as exc_info:
            register_model(model_name, model_info)
        
        assert "MLflow server unavailable" in str(exc_info.value)
    
    @patch('model_registry.mlflow.tracking.MlflowClient')
    @patch('model_registry.mlflow.register_model')
    def test_register_model_transition_error(self, mock_register_model, mock_mlflow_client):
        """Test model registration when stage transition fails"""
        # Mock the model version
        mock_version = Mock()
        mock_version.version = 1
        mock_register_model.return_value = mock_version
        
        # Mock the client to raise an exception during transition
        mock_client_instance = Mock()
        mock_client_instance.transition_model_version_stage.side_effect = Exception("Transition failed")
        mock_mlflow_client.return_value = mock_client_instance
        
        model_name = "loan-default-model"
        model_info = {
            'run_id': 'test_run_123',
            'model_path': 'model'
        }
        
        with pytest.raises(Exception) as exc_info:
            register_model(model_name, model_info)
        
        assert "Transition failed" in str(exc_info.value)
        
        # Verify register_model was called but transition failed
        expected_model_uri = "runs:/test_run_123/model"
        mock_register_model.assert_called_once_with(expected_model_uri, model_name)
    
    def test_register_model_invalid_model_name(self):
        """Test model registration with invalid model name"""
        model_name = ""  # Empty model name
        model_info = {
            'run_id': 'test_run_123',
            'model_path': 'model'
        }
        
        with pytest.raises(ValueError) as exc_info:
            register_model(model_name, model_info)
        
        assert "model_name must be a non-empty string" in str(exc_info.value)


class TestMainFunction:
    """Test cases for the main function"""
    
    @pytest.fixture
    def sample_params_content(self):
        """Fixture providing sample params content"""
        return """
        model_registry:
            registry_path: "report/experiment_info.json"
            model_name: "loan-default-model"
        """
    
    @pytest.fixture
    def sample_model_info(self):
        """Fixture providing sample model info"""
        return {
            'run_id': 'test_run_123',
            'model_path': 'model'
        }
    
    @patch('model_registry.mlflow.set_tracking_uri')
    @patch('model_registry.register_model')
    @patch('model_registry.load_model_info')
    @patch('model_registry.params_load')
    def test_main_success(self, mock_params_load, mock_load_model_info, 
                         mock_register_model, mock_set_tracking_uri,
                         sample_params_content, sample_model_info):
        """Test successful main function execution"""
        # Mock parameters
        mock_params = {
            'model_registry': {
                'registry_path': 'report/experiment_info.json',
                'model_name': 'loan-default-model'
            }
        }
        mock_params_load.return_value = mock_params
        
        # Mock model info loading
        mock_load_model_info.return_value = sample_model_info
        
        # Mock registration
        mock_register_model.return_value = "1"
        
        # Call main function - should not raise SystemExit
        try:
            main()
        except SystemExit as e:
            if e.code != 0:
                pytest.fail(f"main() exited with non-zero code: {e.code}")
        
        # Verify function calls
        mock_set_tracking_uri.assert_called_once_with("http://127.0.0.1:5000")
        mock_params_load.assert_called_once_with('params.yaml')
        mock_load_model_info.assert_called_once_with('report/experiment_info.json')
        mock_register_model.assert_called_once_with('loan-default-model', sample_model_info)
    
    @patch('model_registry.mlflow.set_tracking_uri')
    @patch('model_registry.params_load')
    def test_main_missing_params_file(self, mock_params_load, mock_set_tracking_uri):
        """Test main function when params file is missing"""
        # Mock params_load to raise FileNotFoundError
        mock_params_load.side_effect = FileNotFoundError("Params file not found")
        
        # Should exit with code 1
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1
        
        # MLflow setup should NOT be called when params loading fails early
        mock_set_tracking_uri.assert_not_called()
        mock_params_load.assert_called_once_with('params.yaml')
    
    @patch('model_registry.mlflow.set_tracking_uri')
    @patch('model_registry.load_model_info')
    @patch('model_registry.params_load')
    def test_main_missing_model_info_file(self, mock_params_load, mock_load_model_info, 
                                        mock_set_tracking_uri):
        """Test main function when model info file is missing"""
        # Mock parameters
        mock_params = {
            'model_registry': {
                'registry_path': 'report/experiment_info.json',
                'model_name': 'loan-default-model'
            }
        }
        mock_params_load.return_value = mock_params
        
        # Mock model info loading to raise FileNotFoundError
        mock_load_model_info.side_effect = FileNotFoundError("Model info file not found")
        
        # Should exit with code 1
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1
        
        # MLflow setup SHOULD be called since params loaded successfully
        mock_set_tracking_uri.assert_called_once_with("http://127.0.0.1:5000")
        mock_params_load.assert_called_once_with('params.yaml')
        mock_load_model_info.assert_called_once_with('report/experiment_info.json')
    
    @patch('model_registry.mlflow.set_tracking_uri')
    @patch('model_registry.register_model')
    @patch('model_registry.load_model_info')
    @patch('model_registry.params_load')
    def test_main_registration_failure(self, mock_params_load, mock_load_model_info, 
                                     mock_register_model, mock_set_tracking_uri,
                                     sample_model_info):
        """Test main function when model registration fails"""
        # Mock parameters
        mock_params = {
            'model_registry': {
                'registry_path': 'report/experiment_info.json',
                'model_name': 'loan-default-model'
            }
        }
        mock_params_load.return_value = mock_params
        
        # Mock model info loading
        mock_load_model_info.return_value = sample_model_info
        
        # Mock registration to fail
        mock_register_model.side_effect = Exception("Registration failed")
        
        # Should exit with code 1
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1
        
        # Verify all functions were called including MLflow setup
        mock_set_tracking_uri.assert_called_once_with("http://127.0.0.1:5000")
        mock_params_load.assert_called_once_with('params.yaml')
        mock_load_model_info.assert_called_once_with('report/experiment_info.json')
        mock_register_model.assert_called_once_with('loan-default-model', sample_model_info)
    
    @patch('model_registry.mlflow.set_tracking_uri')
    @patch('model_registry.params_load')
    def test_main_missing_model_registry_section(self, mock_params_load, mock_set_tracking_uri):
        """Test main function when model_registry section is missing"""
        # Mock parameters without model_registry section
        mock_params = {
            'other_section': {
                'some_key': 'some_value'
            }
        }
        mock_params_load.return_value = mock_params
        
        # Should exit with code 1 due to KeyError
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1
        
        # MLflow setup should NOT be called when validation fails immediately after params loading
        mock_set_tracking_uri.assert_not_called()
        mock_params_load.assert_called_once_with('params.yaml')


class TestIntegration:
    """Integration tests for the entire model registry flow"""
    
    @patch('model_registry.mlflow.set_tracking_uri')
    @patch('model_registry.mlflow.tracking.MlflowClient')
    @patch('model_registry.mlflow.register_model')
    def test_end_to_end_registration(self, mock_register_model, mock_mlflow_client, 
                                   mock_set_tracking_uri):
        """Test complete model registration workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create params file
            params_content = """
            model_registry:
                registry_path: "report/experiment_info.json"
                model_name: "loan-default-model"
            """
            params_path = os.path.join(temp_dir, "params.yaml")
            with open(params_path, 'w') as f:
                f.write(params_content)
            
            # Create model info file
            model_info_data = {
                'run_id': 'test_run_123',
                'model_path': 'model'
            }
            model_info_path = os.path.join(temp_dir, "report", "experiment_info.json")
            os.makedirs(os.path.dirname(model_info_path), exist_ok=True)
            with open(model_info_path, 'w') as f:
                json.dump(model_info_data, f, indent=4)
            
            # Mock MLflow
            mock_version = Mock()
            mock_version.version = 1
            mock_register_model.return_value = mock_version
            
            mock_client_instance = Mock()
            mock_mlflow_client.return_value = mock_client_instance
            
            # Test individual functions
            # Load parameters
            params = params_load(params_path)
            assert params['model_registry']['model_name'] == 'loan-default-model'
            
            # Load model info
            model_info = load_model_info(model_info_path)
            assert model_info['run_id'] == 'test_run_123'
            assert model_info['model_path'] == 'model'
            
            # Register model
            version = register_model('loan-default-model', model_info)
            
            # Verify MLflow calls
            expected_model_uri = "runs:/test_run_123/model"
            mock_register_model.assert_called_once_with(expected_model_uri, 'loan-default-model')
            mock_client_instance.transition_model_version_stage.assert_called_once_with(
                name='loan-default-model',
                version=1,
                stage="Staging"
            )
            
            assert version == "1"


# Test configuration
@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after tests"""
    yield
    # Cleanup code if needed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])