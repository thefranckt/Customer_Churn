# Telco Customer Churn Prediction 🚀

Welcome to the Telco Customer Churn Prediction project!  
This repository guides you through a complete data science workflow: from raw data to a deployed machine learning API.

---

## 📦 Project Structure

```
Customer_Churn/
│
├── data/                # Raw and processed datasets
├── notebooks/           # Jupyter notebooks for each step
├── models/              # Saved machine learning models
├── api/                 # FastAPI deployment code
├── requirements.txt     # Python dependencies
├── Dockerfile           # Containerization instructions
└── README.md            # Project documentation
```

---

## 🗂️ Steps Overview

### 1. Data Loading & Inspection
- **Notebook:** `01_data_loading.ipynb`
- Load the Telco churn dataset.
- Inspect shape, data types, missing values, and basic statistics.
- Save a clean working copy for further analysis.

### 2. Exploratory Data Analysis (EDA)
- **Notebook:** `02_eda.ipynb`
- Visualize churn distribution and key features.
- Analyze categorical and numerical variables.
- Discover patterns and correlations that drive customer churn.

### 3. Data Preprocessing
- **Notebook:** `03_preprocessing.ipynb`
- Handle missing values, encode categorical variables, and scale features.
- Prepare the dataset for modeling.

### 4. Modeling
- **Notebook:** `04_modeling.ipynb`
- Train machine learning models (e.g., Logistic Regression, XGBoost).
- Evaluate performance using metrics like accuracy, ROC-AUC, and F1-score.
- Save the best model for deployment.

### 5. Deployment
- **Notebook:** `06_deployment.ipynb`
- Build a REST API using FastAPI to serve churn predictions.
- Define input schema and prediction endpoint.
- Test the API locally and prepare for containerization.

### 6. Containerization
- **File:** `Dockerfile`
- Package the API and model into a Docker container for easy deployment.
- Run the API anywhere with Docker.

---

## 🚀 Quickstart

1. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run notebooks**
    - Open and execute each notebook in order for a full walkthrough.

3. **Deploy the API locally**
    ```bash
    uvicorn api.deployment_api:api --reload
    ```
    - Access interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs)

4. **Containerize with Docker**
    ```bash
    docker build -t churn-api .
    docker run -p 8000:8000 churn-api
    ```

---

## 🎯 Key Features

- End-to-end workflow: data, EDA, modeling, deployment
- Clean, modular code and notebooks
- Interactive API for real-time predictions
- Ready for production with Docker

---

## 📊 About the Data

- **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Goal:** Predict which customers are likely to churn based on demographics and service usage.

---

## 🤝 Contributing

Feel free to fork, open issues, or submit pull requests to improve the project!

---

## 📬 Contact

Questions or suggestions?  
Open an issue or reach out via GitHub!

---

Enjoy exploring and deploying churn prediction! 🚀