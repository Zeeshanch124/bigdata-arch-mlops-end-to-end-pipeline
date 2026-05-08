import pandas as pd


REQUIRED_COLUMNS = [
    'Age',
    'Income',
    'LoanAmount',
    'CreditScore',
    'MonthsEmployed',
    'InterestRate',
    'DTIRatio',
    'Default'
]


class DataValidationError(Exception):
    pass


class DataValidator:

    @staticmethod
    def validate_columns(df: pd.DataFrame):
        missing_columns = [
            col for col in REQUIRED_COLUMNS if col not in df.columns
        ]

        if missing_columns:
            raise DataValidationError(
                f"Missing columns: {missing_columns}"
            )

    @staticmethod
    def validate_nulls(df: pd.DataFrame):
        null_counts = df.isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]

        if not columns_with_nulls.empty:
            raise DataValidationError(
                f"Null values detected:\n{columns_with_nulls}"
            )

    @staticmethod
    def validate_credit_score(df: pd.DataFrame):
        invalid_scores = df[
            (df['CreditScore'] < 300) |
            (df['CreditScore'] > 850)
        ]

        if len(invalid_scores) > 0:
            raise DataValidationError(
                'Invalid credit score values found.'
            )

    @staticmethod
    def validate_target(df: pd.DataFrame):
        invalid_targets = df[
            ~df['Default'].isin([0, 1])
        ]

        if len(invalid_targets) > 0:
            raise DataValidationError(
                'Target column must contain only 0 or 1.'
            )

    @classmethod
    def run_all_validations(cls, df: pd.DataFrame):
        cls.validate_columns(df)
        cls.validate_nulls(df)
        cls.validate_credit_score(df)
        cls.validate_target(df)

        print('All validation checks passed.')