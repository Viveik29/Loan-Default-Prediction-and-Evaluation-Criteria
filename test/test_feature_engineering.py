import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import yaml
from unittest.mock import mock_open, patch
import sys

# Add the src directory to the path to import your modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the functions from your feature engineering module
try:
    from feature_engineering import (
        params_load,
        data_split,
        main
    )
except ImportError as e:
    print(f"Import Error: {e}")
    raise


class TestParamsLoad:
    """Test cases for parameters loading function"""
    
    def test_params_load_valid_yaml(self):
        """Test loading valid YAML parameters"""
        # Create a mock YAML content
        yaml_content = """
        test_size: 0.2
        random_state: 42
        model_params:
            n_estimators: 100
            max_depth: 10
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_file = f.name
        
        try:
            params = params_load(temp_file)
            
            assert isinstance(params, dict)
            assert params['test_size'] == 0.2
            assert params['random_state'] == 42
            assert params['model_params']['n_estimators'] == 100
        finally:
            os.unlink(temp_file)
    
    def test_params_load_invalid_file(self):
        """Test loading from non-existent file"""
        with pytest.raises(FileNotFoundError):
            params_load('non_existent_file.yaml')
    
    def test_params_load_invalid_yaml(self):
        """Test loading invalid YAML content"""
        invalid_yaml_content = """
        test_size: 0.2
        random_state: 42
        invalid_yaml: [unclosed list
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml_content)
            temp_file = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                params_load(temp_file)
        finally:
            os.unlink(temp_file)


class TestDataSplit:
    """Test cases for data splitting function"""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample data for splitting tests"""
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            'feature3': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'Default': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Balanced target
        })
    
    @pytest.fixture
    def imbalanced_data(self):
        """Fixture providing imbalanced data for testing stratification"""
        return pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200),
            'Default': [0] * 90 + [1] * 10  # 90% class 0, 10% class 1
        })
    
    def test_data_split_returns_correct_types(self, sample_data):
        """Test that data_split returns correct object types"""
        X_train, X_test, y_train, y_test = data_split(sample_data)
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
    
    def test_data_split_excludes_target(self, sample_data):
        """Test that target column is excluded from features"""
        X_train, X_test, y_train, y_test = data_split(sample_data)
        
        assert 'Default' not in X_train.columns
        assert 'Default' not in X_test.columns
        assert 'feature1' in X_train.columns
        assert 'feature2' in X_train.columns
    
    def test_data_split_preserves_data_integrity(self, sample_data):
        """Test that no data is lost during splitting"""
        X_train, X_test, y_train, y_test = data_split(sample_data)
        
        total_original = len(sample_data)
        total_split = len(X_train) + len(X_test)
        
        assert total_original == total_split
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
    
    def test_data_split_correct_sizes(self, sample_data):
        """Test that train-test split has correct proportions"""
        X_train, X_test, y_train, y_test = data_split(sample_data)
        
        # Default test_size is 0.2
        expected_test_size = int(len(sample_data) * 0.2)
        expected_train_size = len(sample_data) - expected_test_size
        
        assert len(X_train) == expected_train_size
        assert len(X_test) == expected_test_size
        assert len(y_train) == expected_train_size
        assert len(y_test) == expected_test_size
    
    def test_data_split_stratification(self, imbalanced_data):
        """Test that stratification preserves class distribution"""
        X_train, X_test, y_train, y_test = data_split(imbalanced_data)
        
        # Calculate class distributions
        original_distribution = imbalanced_data['Default'].value_counts(normalize=True)
        train_distribution = y_train.value_counts(normalize=True)
        test_distribution = y_test.value_counts(normalize=True)
        
        # Stratification should preserve similar distributions
        assert abs(original_distribution[0] - train_distribution[0]) < 0.1
        assert abs(original_distribution[0] - test_distribution[0]) < 0.1
        assert abs(original_distribution[1] - train_distribution[1]) < 0.1
        assert abs(original_distribution[1] - test_distribution[1]) < 0.1
    
    def test_data_split_deterministic(self, sample_data):
        """Test that splitting is deterministic with same random state"""
        # First split
        X_train1, X_test1, y_train1, y_test1 = data_split(sample_data)
        
        # Second split with same data
        X_train2, X_test2, y_train2, y_test2 = data_split(sample_data)
        
        # Should be identical due to fixed random state
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_test1, y_test2)
    
    def test_data_split_no_data_leakage(self, sample_data):
        """Test that there's no data leakage between train and test"""
        X_train, X_test, y_train, y_test = data_split(sample_data)
        
        # Check that train and test sets are disjoint
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        
        assert train_indices.isdisjoint(test_indices)
    
    def test_data_split_empty_dataframe(self):
        """Test data splitting with empty DataFrame"""
        empty_df = pd.DataFrame()
        
        with pytest.raises(KeyError):
            data_split(empty_df)
    
    def test_data_split_missing_target(self):
        """Test data splitting when target column is missing"""
        df_no_target = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        with pytest.raises(KeyError):
            data_split(df_no_target)


class TestMainFunction:
    """Test cases for the main function"""
    
    @pytest.fixture
    def sample_clean_data(self):
        """Fixture providing sample clean data for main function tests"""
        return pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200),
            'feature3': ['A', 'B'] * 50,
            'Default': [0, 1] * 50  # Balanced
        })
    
    def test_main_creates_output_files(self, sample_clean_data, monkeypatch):
        """Test that main function creates expected output files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create RAW_DATA directory and clean data file
            raw_data_dir = os.path.join(temp_dir, "RAW_DATA")
            os.makedirs(raw_data_dir, exist_ok=True)
            
            clean_data_path = os.path.join(raw_data_dir, "df_clean.csv")
            sample_clean_data.to_csv(clean_data_path, index=False)
            
            # Create params.yaml file
            params_content = """
            test_size: 0.2
            random_state: 42
            """
            
            params_path = os.path.join(temp_dir, "params.yaml")
            with open(params_path, 'w') as f:
                f.write(params_content)
            
            # Monkeypatch to use our temp directory
            original_cwd = os.getcwd()
            monkeypatch.chdir(temp_dir)
            
            try:
                # Run main function
                main()
                
                # Check if output directory was created
                output_dir = os.path.join("RAW_DATA", "Clearn_data")
                assert os.path.exists(output_dir)
                
                # Check if all output files were created
                expected_files = [
                    "X_train.csv",
                    "X_test.csv", 
                    "y_train.csv",
                    "y_test.csv"
                ]
                
                for file_name in expected_files:
                    file_path = os.path.join(output_dir, file_name)
                    assert os.path.exists(file_path), f"File {file_path} was not created"
                
                # Verify file contents
                X_train = pd.read_csv(os.path.join(output_dir, "X_train.csv"))
                X_test = pd.read_csv(os.path.join(output_dir, "X_test.csv"))
                y_train = pd.read_csv(os.path.join(output_dir, "y_train.csv"))
                y_test = pd.read_csv(os.path.join(output_dir, "y_test.csv"))
                
                # Verify data integrity
                assert len(X_train) == len(y_train)
                assert len(X_test) == len(y_test)
                assert len(X_train) + len(X_test) == len(sample_clean_data)
                
                # Verify target column is excluded from features
                assert 'Default' not in X_train.columns
                assert 'Default' not in X_test.columns
                
            finally:
                monkeypatch.chdir(original_cwd)
    
    def test_main_missing_clean_data(self, monkeypatch):
        """Test main function when clean data file is missing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Don't create the clean data file
            
            # Create params.yaml file
            params_content = "test_size: 0.2"
            params_path = os.path.join(temp_dir, "params.yaml")
            with open(params_path, 'w') as f:
                f.write(params_content)
            
            original_cwd = os.getcwd()
            monkeypatch.chdir(temp_dir)
            
            try:
                with pytest.raises(FileNotFoundError):
                    main()
            finally:
                monkeypatch.chdir(original_cwd)
    
    def test_main_missing_params_file(self, sample_clean_data, monkeypatch):
        """Test main function when params file is missing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create clean data file but no params file
            raw_data_dir = os.path.join(temp_dir, "RAW_DATA")
            os.makedirs(raw_data_dir, exist_ok=True)
            
            clean_data_path = os.path.join(raw_data_dir, "df_clean.csv")
            sample_clean_data.to_csv(clean_data_path, index=False)
            
            original_cwd = os.getcwd()
            monkeypatch.chdir(temp_dir)
            
            try:
                # The main function should handle missing params file gracefully
                # It might use default values or raise a different exception
                try:
                    main()
                    # If it runs successfully, that's acceptable - the function might have defaults
                    output_dir = os.path.join("RAW_DATA", "Clearn_data")
                    assert os.path.exists(output_dir)
                except Exception as e:
                    # It might raise a different exception, which is also acceptable
                    assert isinstance(e, (FileNotFoundError, yaml.YAMLError))
                    
            finally:
                monkeypatch.chdir(original_cwd)


class TestIntegration:
    """Integration tests for the entire feature engineering flow"""
    
    def test_end_to_end_feature_engineering(self):
        """Test complete feature engineering workflow"""
        # Create realistic test data
        test_data = pd.DataFrame({
            'age': [25, 30, 35, 40, 25, 30, 35, 40, 25, 30],
            'income': [50000, 60000, 70000, 80000, 55000, 65000, 75000, 85000, 52000, 62000],
            'credit_score': [650, 700, 750, 680, 720, 690, 730, 710, 675, 685],
            'loan_amount': [10000, 15000, 20000, 18000, 12000, 17000, 22000, 19000, 11000, 16000],
            'employment_length': [2, 5, 3, 7, 4, 6, 8, 9, 1, 10],
            'debt_to_income': [0.35, 0.42, 0.28, 0.38, 0.45, 0.32, 0.40, 0.25, 0.35, 0.33],
            'Default': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save test data
            clean_data_path = os.path.join(temp_dir, "df_clean.csv")
            test_data.to_csv(clean_data_path, index=False)
            
            # Create params file
            params_content = """
            test_size: 0.2
            random_state: 42
            """
            params_path = os.path.join(temp_dir, "params.yaml")
            with open(params_path, 'w') as f:
                f.write(params_content)
            
            # Change to temp directory and run data_split manually
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                # Load and split data
                df = pd.read_csv("df_clean.csv")
                X_train, X_test, y_train, y_test = data_split(df)
                
                # Verify the split
                assert len(X_train) == 8  # 80% of 10
                assert len(X_test) == 2   # 20% of 10
                assert len(y_train) == 8
                assert len(y_test) == 2
                
                # Verify no data leakage
                assert set(X_train.index).isdisjoint(set(X_test.index))
                
                # Verify stratification (roughly equal class distribution)
                original_class_ratio = test_data['Default'].mean()
                train_class_ratio = y_train.mean()
                test_class_ratio = y_test.mean()
                
                # Should be roughly similar due to stratification
                assert abs(original_class_ratio - train_class_ratio) < 0.3
                assert abs(original_class_ratio - test_class_ratio) < 0.3
                
            finally:
                os.chdir(original_cwd)


# Test configuration
@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after tests"""
    yield
    # Cleanup code if needed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])