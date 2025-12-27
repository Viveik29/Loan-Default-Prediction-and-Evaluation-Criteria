import pytest
import pandas as pd
import numpy as np
import pickle
import os
import tempfile
import sys
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.Data_preprocessing import winsorize


# Add the src directory to the path to import your modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the functions from your preprocessing module
try:
    from Data_preprocessing import (
        winsorize_array, 
        log_transform_array, 
        build_pipeline, 
        save_pickle, 
        main,
        TARGET_COL
    )
except ImportError as e:
    print(f"Import Error: {e}")
    # If you can't import, the tests will fail - this helps identify the issue
    raise


class TestWinsorizeArray:
    """Test cases for winsorization function"""
    
    def test_winsorize_array_handles_outliers(self):
        """Test that extreme values are properly clipped"""
        # Create data with obvious outliers - use a single column for clearer test
        data = np.array([1, 2, 3, 4, 5, 100]).reshape(-1, 1)  # Single column with outlier
        winsorized = winsorize_array(data)
        
        # Calculate expected bounds for this data
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        
        # Check that outliers are clipped to the upper bound
        assert np.max(winsorized) <= upper_bound
    
    def test_winsorize_array_preserves_shape(self):
        """Test that output shape matches input shape"""
        data = np.random.rand(10, 3)
        winsorized = winsorize_array(data)
        
        assert winsorized.shape == data.shape
    
    def test_winsorize_array_with_normal_data(self):
        """Test winsorization with normally distributed data"""
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 2))
        winsorized = winsorize_array(data)
        
        # For normal data, winsorization should not change much
        assert winsorized.shape == data.shape


class TestLogTransformArray:
    """Test cases for log transformation function"""
    
    def test_log_transform_positive_values(self):
        """Test log transform with positive values"""
        data = np.array([[1, 10, 100], [2, 20, 200]]).T
        transformed = log_transform_array(data)
        
        # Check shape preserved
        assert transformed.shape == data.shape
        # Check all values are finite (no inf or NaN)
        assert np.all(np.isfinite(transformed))
    
    def test_log_transform_with_negative_values(self):
        """Test log transform skips columns with negative values"""
        data = np.array([[1, 2, 3], [-1, -2, -3]]).T
        transformed = log_transform_array(data)
        
        # Should not apply log to columns with negatives
        assert transformed.shape == data.shape
    
    def test_log_transform_with_zero_values(self):
        """Test log transform with zero values using log1p"""
        data = np.array([[0, 1, 2], [0, 10, 100]]).T
        transformed = log_transform_array(data)
        
        # Should handle zeros gracefully with log1p
        assert transformed.shape == data.shape
        assert np.all(np.isfinite(transformed))


class TestBuildPipeline:
    """Test cases for pipeline building function"""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample data for pipeline tests"""
        return pd.DataFrame({
            'age': [25, 30, 35, 40, 45, None],
            'income': [50000, 60000, 70000, None, 90000, 100000],
            'credit_score': [650, 700, 750, 680, 720, 690],
            'employment_type': ['Salaried', 'Self-Employed', 'Salaried', 'Business', 'Salaried', 'Self-Employed'],
            'loan_purpose': ['Car', 'Home', 'Education', 'Car', 'Business', 'Education'],
            'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005', 'C006'],
            'Default': [0, 1, 0, 1, 0, 0]
        })
    
    def test_build_pipeline_returns_correct_types(self, sample_data):
        """Test that build_pipeline returns correct object types"""
        preprocessor, feature_cols = build_pipeline(sample_data)
        
        assert isinstance(preprocessor, ColumnTransformer)
        assert isinstance(feature_cols, list)
        assert all(isinstance(col, str) for col in feature_cols)
    
    def test_build_pipeline_excludes_target(self, sample_data):
        """Test that target column is excluded from feature columns"""
        preprocessor, feature_cols = build_pipeline(sample_data)
        
        assert TARGET_COL not in feature_cols
        assert 'age' in feature_cols
        assert 'income' in feature_cols
        assert 'employment_type' in feature_cols
    
    def test_build_pipeline_handles_numeric_columns(self, sample_data):
        """Test that numeric columns are properly identified"""
        preprocessor, feature_cols = build_pipeline(sample_data)
        
        expected_numeric = ['age', 'income', 'credit_score']
        for col in expected_numeric:
            assert col in feature_cols
    
    def test_build_pipeline_handles_categorical_columns(self, sample_data):
        """Test that categorical columns are properly identified and split by cardinality"""
        preprocessor, feature_cols = build_pipeline(sample_data)
        
        # All categorical columns should be in feature_cols
        assert 'employment_type' in feature_cols
        assert 'loan_purpose' in feature_cols
        assert 'customer_id' in feature_cols
    
    def test_build_pipeline_can_fit_and_transform(self, sample_data):
        """Test that the built pipeline can be fitted and used for transformation"""
        preprocessor, feature_cols = build_pipeline(sample_data)
        
        # Fit the pipeline
        preprocessor.fit(sample_data[feature_cols], sample_data[TARGET_COL])
        
        # Transform data
        transformed = preprocessor.transform(sample_data[feature_cols])
        
        assert transformed.shape[0] == len(sample_data)
        assert transformed.shape[1] > 0
    
    def test_build_pipeline_with_missing_values(self, sample_data):
        """Test pipeline handles missing values correctly"""
        preprocessor, feature_cols = build_pipeline(sample_data)
        
        preprocessor.fit(sample_data[feature_cols], sample_data[TARGET_COL])
        transformed = preprocessor.transform(sample_data[feature_cols])
        
        # Check no NaN values in output
        assert not np.any(np.isnan(transformed))
    
    def test_build_pipeline_empty_dataframe(self):
        """Test pipeline building with empty DataFrame"""
        empty_df = pd.DataFrame()
        
        # The function should handle empty DataFrame
        preprocessor, feature_cols = build_pipeline(empty_df)
        assert isinstance(preprocessor, ColumnTransformer)
        assert feature_cols == []


class TestSavePickle:
    """Test cases for pickle saving function"""
    
    def test_save_pickle_creates_file(self):
        """Test that save_pickle creates a file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_obj = {"key": "value", "number": 42}
            file_path = os.path.join(temp_dir, "test.pkl")
            
            save_pickle(test_obj, file_path)
            
            assert os.path.exists(file_path)
    
    def test_save_pickle_can_be_loaded(self):
        """Test that saved pickle can be loaded back"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_obj = {"model": "random_forest", "accuracy": 0.85}
            file_path = os.path.join(temp_dir, "test.pkl")
            
            save_pickle(test_obj, file_path)
            
            with open(file_path, "rb") as f:
                loaded_obj = pickle.load(f)
            
            assert loaded_obj == test_obj


class TestMainFunction:
    """Test cases for the main function"""
    
    def test_main_creates_output_files(self, monkeypatch):
        """Test that main function creates expected output files"""
        # Create a mock DataFrame to avoid file dependencies
        mock_data = pd.DataFrame({
            'age': [25, 30, 35],
            'income': [50000, 60000, 70000],
            'employment_type': ['A', 'B', 'A'],
            'Default': [0, 1, 0]
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directories
            raw_data_dir = os.path.join(temp_dir, "RAW_DATA", "Raw")
            os.makedirs(raw_data_dir, exist_ok=True)
            models_dir = os.path.join(temp_dir, "Models")
            os.makedirs(models_dir, exist_ok=True)
            
            # Save mock data
            mock_data_path = os.path.join(raw_data_dir, "df_train.csv")
            mock_data.to_csv(mock_data_path, index=False)
            
            # Monkeypatch to use our temp directory
            original_cwd = os.getcwd()
            monkeypatch.chdir(temp_dir)
            
            try:
                # Run main function
                main()
                
                # Check if files were created
                assert os.path.exists("Models/preprocessor.pkl")
                assert os.path.exists("RAW_DATA/df_clean.csv")
                
            finally:
                monkeypatch.chdir(original_cwd)


class TestIntegration:
    """Integration tests for the entire preprocessing flow"""
    
    def test_end_to_end_preprocessing(self):
        """Test complete preprocessing workflow"""
        test_data = pd.DataFrame({
            'age': [25, 30, 35, 40, 25, 30, None, 45, 35, 40],
            'income': [50000, 75000, 120000, 80000, 45000, 95000, 60000, 110000, None, 70000],
            'credit_score': [650, 720, 800, 680, 630, 710, 690, 740, 700, 675],
            'loan_amount': [10000, 25000, 50000, 15000, 8000, 30000, 12000, 45000, 20000, 18000],
            'employment_length': [2, 5, 10, 3, 1, 7, 4, 12, 6, 8],
            'debt_to_income': [0.35, 0.42, 0.28, 0.38, 0.45, 0.32, 0.40, 0.25, 0.35, 0.33],
            'loan_purpose': ['car', 'home', 'education', 'car', 'personal', 'home', 'car', 'business', 'personal', 'home'],
            'employment_type': ['salaried', 'self-employed', 'salaried', 'business', 'salaried', 'self-employed', 'salaried', 'business', 'salaried', 'self-employed'],
            'Default': [0, 1, 0, 1, 1, 0, 0, 0, 1, 0]
        })
        
        # Build and test pipeline
        preprocessor, feature_cols = build_pipeline(test_data)
        
        # Verify feature columns
        expected_features = ['age', 'income', 'credit_score', 'loan_amount', 
                           'employment_length', 'debt_to_income', 'loan_purpose', 'employment_type']
        for feature in expected_features:
            assert feature in feature_cols
        
        # Fit and transform
        preprocessor.fit(test_data[feature_cols], test_data[TARGET_COL])
        transformed = preprocessor.transform(test_data[feature_cols])
        
        # Verify output
        assert isinstance(transformed, np.ndarray)
        assert transformed.shape[0] == len(test_data)
        assert transformed.shape[1] >= len(feature_cols)
        assert not np.any(np.isnan(transformed))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])