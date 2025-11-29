# House-price-prediction

# 🏡 House Price Prediction Project

## 🌟 Project Overview

This project implements an end-to-end Machine Learning pipeline to predict residential house prices. Utilizing various features such as location, size, and physical characteristics, the goal is to develop a highly accurate regression model and deploy it as a practical prediction service.

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

* Python (3.8+)
* `pip` (Python package installer)

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourUsername/House-Price-Prediction.git](https://github.com/YourUsername/House-Price-Prediction.git)
    cd House-Price-Prediction
    ```

2.  **Create and Activate a Virtual Environment:** (Highly Recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # Use 'venv\Scripts\activate' on Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Training Pipeline:**
    ```bash
    python main.py
    ```

---

## 🗂️ Project Structure

The project follows a modular structure for maintainability and scalability, separating concerns into data handling, model training, and deployment.


---

## ⚙️ ML Pipeline Summary

The ML pipeline is executed sequentially by `training_pipeline.py`.

1.  **Data Ingestion:**
    * Reads the raw data.
    * Splits the data into **training** and **testing** sets.

2.  **Data Preprocessing & Feature Engineering:**
    * Applies **handling for missing values** (imputation).
    * Performs **feature transformations** (e.g., log transformations for skewed features, deriving `Age_of_House`).
    * Encodes **categorical features** (e.g., One-Hot Encoding).
    * Scales **numerical features** (e.g., StandardScaler).
    * A Scikit-learn **ColumnTransformer** is used to encapsulate these steps.

3.  **Model Training & Evaluation:**
    * Trains several **Regression Models** (e.g., Linear Regression, Random Forest, XGBoost).
    * Evaluates models using key metrics like **RMSE** and **$R^2$ score**.
    * The best-performing model is saved to the `models/` directory alongside the preprocessing pipeline.

4.  **Model Deployment (API):**
    * The `app/app.py` uses the saved model and preprocessor to create a **REST API**.
    * The `/predict` endpoint accepts new house features and returns a predicted price.

---

## 🛠️ Technologies & Libraries

* **Python:** Core programming language.
* **Pandas/NumPy:** Data manipulation and numerical operations.
* **Scikit-learn:** Comprehensive library for preprocessing, feature engineering, and model training.
* **XGBoost/LightGBM:** High-performance gradient boosting models (often providing the best results).
* **Joblib/Pickle:** For serializing and saving the model and preprocessor.
* **Flask/FastAPI:** For creating the prediction API.

---
