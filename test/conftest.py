import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_loan_data():
    """Sample loan data for multiple tests"""
    return pd.DataFrame({
        'age': [25, 30, 35, 40, 45, None],
        'income': [50000, 60000, 70000, None, 90000, 100000],
        'credit_score': [650, 700, 750, 680, 720, 690],
        'employment_type': ['Salaried', 'Self-Employed', 'Salaried', 'Business', 'Salaried', 'Self-Employed'],
        'loan_purpose': ['Car', 'Home', 'Education', 'Car', 'Business', 'Education'],
        'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005', 'C006'],
        'Default': [0, 1, 0, 1, 0, 0]
    })