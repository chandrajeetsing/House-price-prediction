# src/components/data_preprocessing.py

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils import save_object # Use the save_object function

class DataPreprocessing:
    def get_preprocessing_pipeline(self, train_data_path, test_data_path):
        """
        Creates and returns the preprocessing object (ColumnTransformer) 
        and the split, prepared dataframes.
        """
        try:
            # --- 1. Load Data ---
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            # Separate features (X) and target (y)
            target_column_name = 'price'
            
            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]
            
            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]

            # --- 2. Define Feature Types ---
            numerical_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
            
            # Categorical features that are binary (Yes/No)
            binary_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
            
            # Categorical feature that is nominal (Multi-class)
            nominal_features = ['furnishingstatus']
            
            # --- 3. Create Preprocessing Pipelines for each type ---
            
            # A. Numerical Pipeline (Imputation + Scaling)
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')), # In case of missing values
                ('scaler', StandardScaler())
            ])

            # B. Binary Pipeline (Mapping Yes/No to 1/0)
            # We can map 'yes' to 1 and 'no' to 0 directly, 
            # but for a generic ML pipeline, we'll convert them to 0/1 using OneHotEncoder
            # and dropping one to avoid multicollinearity.
            binary_pipeline = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False))
            ])
            
            # C. Nominal Pipeline (One-Hot Encoding)
            nominal_pipeline = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            # --- 4. Combine Pipelines using ColumnTransformer ---
            preprocessor = ColumnTransformer(
                [
                    ("Numerical_Pipeline", numerical_pipeline, numerical_features),
                    ("Binary_Pipeline", binary_pipeline, binary_features),
                    ("Nominal_Pipeline", nominal_pipeline, nominal_features)
                ],
                remainder='passthrough' # Keep any other columns if they exist
            )

            print("Preprocessing object (ColumnTransformer) created.")

            # Optional: Save the preprocessor object
            save_object(
                file_path=os.path.join('models', 'preprocessor.pkl'),
                obj=preprocessor
            )

            # Return the preprocessor and the data splits
            return preprocessor, X_train, y_train, X_test, y_test

        except Exception as e:
            raise e