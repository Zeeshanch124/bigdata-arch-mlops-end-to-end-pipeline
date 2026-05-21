import pandas as pd
import pytest

from data_pipeline.validation import (
    DataValidator,
    DataValidationError
)



def test_valid_dataset():

    data = {
        'Age': [30],
        'Income': [50000],
        'LoanAmount': [10000],
        'CreditScore': [700],
        'MonthsEmployed': [12],
        'InterestRate': [5.5],
        'DTIRatio': [0.3],
        'Default': [0]
    }

    df = pd.DataFrame(data)

    DataValidator.run_all_validations(df)



def test_invalid_credit_score():
 
    data = {
        'Age': [30],
        'Income': [50000],
        'LoanAmount': [10000],
        'CreditScore': [950],
        'MonthsEmployed': [12],
        'InterestRate': [5.5],
        'DTIRatio': [0.3],
        'Default': [0]
    }

    df = pd.DataFrame(data)
 
    with pytest.raises(DataValidationError):
        DataValidator.validate_credit_score(df)