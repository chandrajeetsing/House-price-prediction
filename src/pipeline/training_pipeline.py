# src/pipeline/training_pipeline.py

import os
import sys
# Import the components defined in the sibling directory 'components'
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        # Initialize the components
        self.data_ingestion = DataIngestion()
        self.data_preprocessing = DataPreprocessing()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        """
        Executes the entire Machine Learning pipeline sequentially.
        """
        try:
            print("\n==================== Stage 1: Data Ingestion ====================")
            # Stage 1: Ingests data and returns the paths to the split train/test sets
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            print("Stage 1: Data Ingestion completed. Data paths secured.")
            
            print("\n==================== Stage 2: Data Preprocessing ====================")
            # Stage 2: Creates the preprocessor object and returns the split dataframes
            preprocessor_obj, X_train, y_train, X_test, y_test = self.data_preprocessing.get_preprocessing_pipeline(
                train_data_path, test_data_path
            )
            print("Stage 2: Data Preprocessing completed. Preprocessor saved.")

            print("\n==================== Stage 3: Model Training ====================")
            # Stage 3: Trains models, selects the best one, and saves it
            r2_score = self.model_trainer.initiate_model_training(
                X_train, y_train, X_test, y_test, preprocessor_obj
            )
            print(f"Stage 3: Model Training completed. Best R2 Score achieved: {r2_score:.4f}")
            print("=================================================================\n")


        except Exception as e:
            # For debugging, re-raise the error
            raise Exception(f"Pipeline failed during execution: {e}")