# src/components/data_ingestion.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils import save_object # This function will be defined in utils.py

class DataIngestion:
    def __init__(self):
        # Define file paths for the raw and processed data
        self.raw_data_path = os.path.join('data', 'raw', 'Housing.csv')
        self.train_data_path = os.path.join('data', 'processed', 'train.csv')
        self.test_data_path = os.path.join('data', 'processed', 'test.csv')

    def initiate_data_ingestion(self):
        try:
            print("Reading data from Housing.csv...")
            # 1. Read the dataset
            df = pd.read_csv(self.raw_data_path)
            print("Data read successfully.")

            # Optional: Check for duplicates or initial missing values
            # print(df.info()) 
            # print(df.duplicated().sum())

            # 2. Split the data into training and testing sets
            print("Splitting data into train and test sets (80/20 split)...")
            train_set, test_set = train_test_split(
                df, 
                test_size=0.2, 
                random_state=42 # Ensures reproducibility
            )

            # 3. Save the split datasets
            train_set.to_csv(self.train_data_path, index=False, header=True)
            test_set.to_csv(self.test_data_path, index=False, header=True)

            print(f"Train data saved to: {self.train_data_path}")
            print(f"Test data saved to: {self.test_data_path}")

            # Return the paths for the next component (Data Preprocessing)
            return (
                self.train_data_path,
                self.test_data_path
            )

        except Exception as e:
            raise e # Handle the exception appropriately in a real-world scenario