# Placeholder README for models directory

This directory contains the trained machine learning models for the Customer Churn project.

## Models

After running `04_modeling.ipynb`, you will find:

- `logreg_model.pkl` - Logistic Regression model (recommended for deployment)
- `xgb_model.pkl` - XGBoost model
- `rf_model.pkl` - Random Forest model

## Usage

The API (`deployment_api.py`) loads the Logistic Regression model by default.
To use a different model, update the model loading line in the API file.

## Training

To generate these models:
1. Ensure you have run notebooks 01-03 to prepare the data
2. Open and run `04_modeling.ipynb`
3. The models will be automatically saved to this directory