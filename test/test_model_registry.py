import pytest
import json
import os
import tempfile
import yaml
import sys
from unittest.mock import Mock, patch

# Add src directory to sys.path to import your modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import functions from src.model_registry
try:
    from src.model_registry import params_load, load_model_info, register_model, main
except ImportError as e:
    print(f"Import Error: {e}")
    raise


class TestParamsLoad:
    """Test cases for parameters loading function."""

    def test_params_load_valid_yaml(self):
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
        with pytest.raises(FileNotFoundError):
            params_load('non_existent_file.yaml')

    def test_params_load_invalid_yaml(self):
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
    """Test cases for loading model info JSON."""

    def test_load_model_info_valid_file(self):
        model_info_data = {'run_id': 'test_run_123', 'model_path': 'model'}
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
        with pytest.raises(FileNotFoundError):
            load_model_info('non_existent_file.json')

    def test_load_model_info_invalid_json(self):
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
        model_info_data = {'run_id': 'test_run_123'}  # missing model_path
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
        model_info_data = {'run_id': '', 'model_path': 'model'}  # empty run_id
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
    """Tests for register_model function."""

    @patch('src.model_registry.mlflow.register_model')
    @patch('src.model_registry.mlflow.tracking.MlflowClient')
    def test_register_model_success(self, mock_mlflow_client, mock_register_model):
        mock_version = Mock()
        mock_version.version = 1
        mock_register_model.return_value = mock_version

        mock_client_instance = Mock()
        mock_mlflow_client.return_value = mock_client_instance

        model_name = "loan-default-model"
        model_info = {'run_id': 'test_run_123', 'model_path': 'model'}

        version = register_model(model_name, model_info)

        expected_model_uri = "runs:/test_run_123/model"
        mock_register_model.assert_called_once_with(expected_model_uri, model_name)
        mock_client_instance.transition_model_version_stage.assert_called_once_with(
            name=model_name,
            version=1,
            stage="Staging"
        )
        assert version == "1"

    @patch('src.model_registry.mlflow.register_model')
    @patch('src.model_registry.mlflow.tracking.MlflowClient')
    def test_register_model_missing_run_id(self, mock_mlflow_client, mock_register_model):
        model_name = "loan-default-model"
        model_info = {'model_path': 'model'}  # missing run_id

        with pytest.raises(KeyError):
            register_model(model_name, model_info)

        mock_register_model.assert_not_called()
        mock_mlflow_client.assert_not_called()

    @patch('src.model_registry.mlflow.register_model')
    @patch('src.model_registry.mlflow.tracking.MlflowClient')
    def test_register_model_missing_model_path(self, mock_mlflow_client, mock_register_model):
        model_name = "loan-default-model"
        model_info = {'run_id': 'test_run_123'}  # missing model_path

        with pytest.raises(KeyError):
            register_model(model_name, model_info)

        mock_register_model.assert_not_called()
        mock_mlflow_client.assert_not_called()

    @patch('src.model_registry.mlflow.register_model')
    @patch('src.model_registry.mlflow.tracking.MlflowClient')
    def test_register_model_mlflow_error(self, mock_mlflow_client, mock_register_model):
        mock_register_model.side_effect = Exception("MLflow server unavailable")

        model_name = "loan-default-model"
        model_info = {'run_id': 'test_run_123', 'model_path': 'model'}

        with pytest.raises(Exception) as exc_info:
            register_model(model_name, model_info)
        assert "MLflow server unavailable" in str(exc_info.value)

    @patch('src.model_registry.mlflow.register_model')
    @patch('src.model_registry.mlflow.tracking.MlflowClient')
    def test_register_model_transition_error(self, mock_mlflow_client, mock_register_model):
        mock_version = Mock()
        mock_version.version = 1
        mock_register_model.return_value = mock_version

        mock_client_instance = Mock()
        mock_client_instance.transition_model_version_stage.side_effect = Exception("Transition failed")
        mock_mlflow_client.return_value = mock_client_instance

        model_name = "loan-default-model"
        model_info = {'run_id': 'test_run_123', 'model_path': 'model'}

        with pytest.raises(Exception) as exc_info:
            register_model(model_name, model_info)
        assert "Transition failed" in str(exc_info.value)

        expected_model_uri = "runs:/test_run_123/model"
        mock_register_model.assert_called_once_with(expected_model_uri, model_name)

    def test_register_model_invalid_model_name(self):
        model_name = ""  # Empty
        model_info = {'run_id': 'test_run_123', 'model_path': 'model'}

        with pytest.raises(ValueError) as exc_info:
            register_model(model_name, model_info)
        assert "model_name must be a non-empty string" in str(exc_info.value)


class TestMainFunction:
    """Tests for main function."""

    @pytest.fixture
    def sample_params(self):
        return {
            'model_registry': {
                'registry_path': 'report/experiment_info.json',
                'model_name': 'loan-default-model'
            }
        }

    @pytest.fixture
    def sample_model_info(self):
        return {'run_id': 'test_run_123', 'model_path': 'model'}

    @patch('src.model_registry.mlflow.set_tracking_uri')
    @patch('src.model_registry.register_model')
    @patch('src.model_registry.load_model_info')
    @patch('src.model_registry.params_load')
    def test_main_success(self, mock_params_load, mock_load_model_info,
                          mock_register_model, mock_set_tracking_uri, sample_params, sample_model_info):
        mock_params_load.return_value = sample_params
        mock_load_model_info.return_value = sample_model_info
        mock_register_model.return_value = "1"

        try:
            main()
        except SystemExit as e:
            if e.code != 0:
                pytest.fail(f"main() exited with non-zero code: {e.code}")

        mock_set_tracking_uri.assert_called_once_with("http://127.0.0.1:5000")
        mock_params_load.assert_called_once_with('params.yaml')
        mock_load_model_info.assert_called_once_with('report/experiment_info.json')
        mock_register_model.assert_called_once_with('loan-default-model', sample_model_info)

    @patch('src.model_registry.mlflow.set_tracking_uri')
    @patch('src.model_registry.params_load')
    def test_main_missing_params_file(self, mock_params_load, mock_set_tracking_uri):
        mock_params_load.side_effect = FileNotFoundError("Params file not found")

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

        mock_set_tracking_uri.assert_not_called()
        mock_params_load.assert_called_once_with('params.yaml')

    @patch('src.model_registry.mlflow.set_tracking_uri')
    @patch('src.model_registry.load_model_info')
    @patch('src.model_registry.params_load')
    def test_main_missing_model_info_file(self, mock_params_load, mock_load_model_info, mock_set_tracking_uri):
        mock_params = {
            'model_registry': {
                'registry_path': 'report/experiment_info.json',
                'model_name': 'loan-default-model'
            }
        }
        mock_params_load.return_value = mock_params
        mock_load_model_info.side_effect = FileNotFoundError("Model info file not found")

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

        mock_set_tracking_uri.assert_called_once_with("http://127.0.0.1:5000")
        mock_params_load.assert_called_once_with('params.yaml')
        mock_load_model_info.assert_called_once_with('report/experiment_info.json')

    @patch('src.model_registry.mlflow.set_tracking_uri')
    @patch('src.model_registry.register_model')
    @patch('src.model_registry.load_model_info')
    @patch('src.model_registry.params_load')
    def test_main_registration_failure(self, mock_params_load, mock_load_model_info,
                                       mock_register_model, mock_set_tracking_uri, sample_model_info):
        mock_params = {
            'model_registry': {
                'registry_path': 'report/experiment_info.json',
                'model_name': 'loan-default-model'
            }
        }
        mock_params_load.return_value = mock_params
        mock_load_model_info.return_value = sample_model_info
        mock_register_model.side_effect = Exception("Registration failed")

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

        mock_set_tracking_uri.assert_called_once_with("http://127.0.0.1:5000")
        mock_params_load.assert_called_once_with('params.yaml')
        mock_load_model_info.assert_called_once_with('report/experiment_info.json')
        mock_register_model.assert_called_once_with('loan-default-model', sample_model_info)

    @patch('src.model_registry.mlflow.set_tracking_uri')
    @patch('src.model_registry.params_load')
    def test_main_missing_model_registry_section(self, mock_params_load, mock_set_tracking_uri):
        mock_params_load.return_value = {'other_section': {'some_key': 'some_value'}}

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

        mock_set_tracking_uri.assert_not_called()
        mock_params_load.assert_called_once_with('params.yaml')


@pytest.fixture(autouse=True)
def cleanup():
    """Run cleanup after each test if needed."""
    yield
    # Add any needed cleanup logic here


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
