# main.py

import os
import sys
from src.pipeline.training_pipeline import TrainingPipeline

def run_ml_pipeline():
    """
    Sets up the environment and executes the end-to-end Machine Learning training pipeline.
    """
    
    # 1. Directory Setup: Ensure all necessary folders exist
    try:
        os.makedirs(os.path.join(os.getcwd(), 'data', 'raw'), exist_ok=True)
        os.makedirs(os.path.join(os.getcwd(), 'data', 'processed'), exist_ok=True)
        os.makedirs(os.path.join(os.getcwd(), 'models'), exist_ok=True)
        os.makedirs(os.path.join(os.getcwd(), 'app'), exist_ok=True)
        print("Required directories created successfully.")
    except Exception as e:
        print(f"Error creating directories: {e}")
        sys.exit(1)


    # 2. Pipeline Execution
    print("\n--- Starting House Price Prediction Pipeline ---")
    
    try:
        # Initialize and run the training pipeline
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        
        print("\n--- Pipeline completed successfully. Model and artifacts saved. ---")
        
    except Exception as e:
        print(f"\nFATAL ERROR during pipeline execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure the Housing.csv file is placed in the data/raw/ directory
    if not os.path.exists(os.path.join('data', 'raw', 'Housing.csv')):
        print("ERROR: Housing.csv not found in 'data/raw/'. Please place your dataset there before running.")
        sys.exit(1)
        
    run_ml_pipeline()