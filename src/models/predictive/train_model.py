import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor

class ModelPipeline:
    """
    Class to build the SGD pipeline for model traning and prediction
    """
    def __init__(self, numerical_columns, categorical_columns):
        self.numeric = numerical_columns
        self.categorical = categorical_columns

    def build_preprocessor(self):
        # Preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric),
                ('cat', OneHotEncoder(drop='first', handle_unknown = 'infrequent_if_exist'), self.categorical),
            ])
        return preprocessor

    def build_pipeline(self):
        # Model pipeline
        pipeline = Pipeline([
            ('preprocessor', self.build_preprocessor()),
            ('regressor', SGDRegressor(loss='squared_error', penalty = "elasticnet", 
                                    early_stopping=True,  # Enable early stopping
                                    max_iter=10000, tol=1e-3, alpha=0.001,
                                    validation_fraction=0.2,  # Fraction of training data used for validation
                                    n_iter_no_change=10  # Number of iterations with no improvement on validation set
            ))
            
        ],
        verbose = True)
        return pipeline

    def train(self, X_train, y_train):
        # Fit the model
        pipeline = self.build_pipeline()
        return pipeline.fit(X_train, y_train)
