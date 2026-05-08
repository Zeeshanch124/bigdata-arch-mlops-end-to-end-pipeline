import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


class DataPreprocessor:

    def __init__(self):

        self.numeric_features = [
            'Age',
            'Income',
            'LoanAmount',
            'CreditScore',
            'MonthsEmployed',
            'NumCreditLines',
            'InterestRate',
            'LoanTerm',
            'DTIRatio'
        ]

        self.categorical_features = [
            'Education',
            'EmploymentType',
            'MaritalStatus',
            'HasMortgage',
            'HasDependents',
            'LoanPurpose',
            'HasCoSigner'
        ]

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    'num',
                    Pipeline([
                        ('scaler', StandardScaler())
                    ]),
                    self.numeric_features
                ),
                (
                    'cat',
                    Pipeline([
                        (
                            'encoder',
                            OneHotEncoder(handle_unknown='ignore')
                        )
                    ]),
                    self.categorical_features
                )
            ]
        )

    def feature_engineering(self, df: pd.DataFrame):

        df['IncomeToLoanRatio'] = (
            df['Income'] / (df['LoanAmount'] + 1)
        )

        return df

    def preprocess(self, df: pd.DataFrame):

        df = self.feature_engineering(df)

        X = df.drop(columns=['LoanID', 'Default'])
        y = df['Default']

        X_processed = self.preprocessor.fit_transform(X)

        return X_processed, y