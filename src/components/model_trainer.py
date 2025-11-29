# src/components/model_trainer.py

import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor # Consider installing catboost if you use it
from src.utils import save_object

class ModelTrainer:
    def initiate_model_training(self, X_train, y_train, X_test, y_test, preprocessor_obj):
        try:
            # 1. Define Models
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest Regressor": RandomForestRegressor(random_state=42),
                "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
                "XGBoost Regressor": XGBRegressor(random_state=42),
                # You can add more complex models here
            }

            # 2. Train and Evaluate Models
            # Dictionary to store evaluation metrics
            model_report = {}

            # Apply preprocessing to training and testing features
            X_train_processed = preprocessor_obj.fit_transform(X_train)
            X_test_processed = preprocessor_obj.transform(X_test)
            
            # Note: We are using the preprocessor saved in the previous step
            # For simplicity here, we apply it directly.

            for model_name, model in models.items():
                
                # Train model
                model.fit(X_train_processed, y_train)

                # Make predictions
                y_train_pred = model.predict(X_train_processed)
                y_test_pred = model.predict(X_test_processed)

                # Evaluate model on the test set
                test_model_score = r2_score(y_test, y_test_pred)

                # Store the score
                model_report[model_name] = test_model_score

                print(f"Model: {model_name} | R2 Score: {test_model_score:.4f}")

            # 3. Select Best Model
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            print(f"\n--- Best Model Found ---")
            print(f"Model Name: {best_model_name}")
            print(f"R2 Score: {best_model_score:.4f}")

            # 4. Save the Best Model
            # Note: For deployment, it's often best practice to save the preprocessor 
            # and the model *combined* into a single pipeline object.
            # However, for simplicity, we'll save the model object separately.

            save_object(
                file_path=os.path.join("models", "model.pkl"),
                obj=best_model
            )

            print(f"Best model ({best_model_name}) saved to models/model.pkl")

            # Return the R2 score of the best model for logging
            return best_model_score

        except Exception as e:
            raise e