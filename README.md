# 🏦 Bank Customer Churn Decision Support System

An end-to-end machine learning solution designed to predict customer churn in the banking sector and provide actionable business insights through explainable AI, model serving APIs, and interactive analytics dashboards.

---

# 🎯 Project Objective

This project is designed to demonstrate competencies across three key professional roles:

* **Data Scientist** → Advanced predictive modeling, explainability, and model evaluation
* **Machine Learning Engineer** → Model deployment, API development, and containerization
* **Data Analyst** → Business intelligence, visualization, and decision support dashboards

---

# 🏗️ System Architecture

```text
Dataset → Preprocessing → Feature Engineering → Model Training (XGBoost)
                                                          ↓
PostgreSQL ← FastAPI (Model Serving) ← Streamlit Dashboard (Analytics & UI)
```

## Architecture Layers

### 1. Machine Learning Layer

* Data preprocessing
* Feature engineering
* Model training and optimization
* Explainable AI (SHAP)

### 2. FastAPI Backend

* RESTful API services
* Model inference endpoints
* PostgreSQL integration
* Prediction history management

### 3. Streamlit Dashboard

* Interactive customer churn predictions
* Model performance monitoring
* Business analytics and reporting

### 4. PostgreSQL Database

* Prediction records
* Model performance metrics
* Historical data storage

---

# 🛠️ Technology Stack

## Machine Learning & Data Science

* **Python** – Core programming language
* **Pandas & NumPy** – Data manipulation and numerical computing
* **Scikit-Learn** – Data preprocessing and ML utilities
* **XGBoost** – Gradient boosting model for churn prediction
* **SHAP** – Model interpretability and explainability
* **Imbalanced-Learn** – Class imbalance handling using SMOTE

## Backend Development

* **FastAPI** – High-performance asynchronous web framework
* **Uvicorn** – ASGI server
* **SQLAlchemy** – ORM and database management
* **Pydantic** – Data validation and serialization
* **psycopg2** – PostgreSQL adapter

## Dashboard & Visualization

* **Streamlit** – Interactive web dashboard framework
* **Plotly** – Interactive visualizations
* **Matplotlib & Seaborn** – Statistical data visualization

## Deployment & Infrastructure

* **Docker** – Containerization
* **Docker Compose** – Multi-container orchestration
* **PostgreSQL** – Relational database management system

---

# 📁 Project Structure

```text
bank-customer-churn-prediction/
│
├── data/
│   └── raw/
│       └── ChurnModel.csv
│
├── models/
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   └── shap_explainer.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── explain.py
│   └── db.py
│
├── api/
│   └── main.py
│
├── dashboard/
│   └── app.py
│
├── notebooks/
│   └── customerPrediction.ipynb
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd bank-customer-churn-prediction
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Create a `.env` file based on `.env.example`:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=churn_db
DB_USER=postgres
DB_PASSWORD=postgres

MODEL_PATH=models/xgboost_model.pkl
SCALER_PATH=models/scaler.pkl
EXPLAINER_PATH=models/shap_explainer.pkl

API_URL=http://localhost:8000
```

---

# 📊 Machine Learning Workflow

## 1. Exploratory Data Analysis (EDA)

* Data profiling and descriptive statistics
* Missing value analysis
* Categorical feature distribution analysis
* Target variable distribution assessment
* Class imbalance investigation

## 2. Feature Engineering

* Removal of non-informative columns:

  * RowNumber
  * CustomerId
  * Surname
* One-Hot Encoding:

  * Geography
  * Gender
* Optional engineered features:

  * age_group
  * balance_ratio
  * activity_score

## 3. Class Imbalance Handling

* Synthetic Minority Oversampling Technique (SMOTE)
* Balanced training dataset generation

## 4. Baseline Model

**Logistic Regression**

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-Score

## 5. Final Model

**XGBoost Classifier**

Optimization techniques:

* RandomizedSearchCV
* Hyperparameter tuning
* Stratified Cross-Validation
* ROC-AUC optimization

## 6. Model Evaluation

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* Classification Report

## 7. Explainable AI

Using SHAP:

* Global feature importance analysis
* Local prediction explanations
* Individual customer-level interpretability

---

# 🎯 Usage

## Train the Model

```bash
python src/train.py
```

This process will:

* Load and preprocess the dataset
* Train a Logistic Regression baseline model
* Train and optimize the XGBoost model
* Persist trained artifacts to the `models/` directory
* Optionally store evaluation metrics in PostgreSQL

---

## Run the API

### Local Environment

```bash
uvicorn api.main:app --reload
```

API Endpoint:

```text
http://localhost:8000
```

### Docker Deployment

```bash
docker-compose up --build
```

---

## Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard URL:

```text
http://localhost:8501
```

---

# 📡 API Endpoints

| Method | Endpoint   | Description                       |
| ------ | ---------- | --------------------------------- |
| GET    | `/`        | API information                   |
| GET    | `/health`  | Health check                      |
| POST   | `/predict` | Customer churn prediction         |
| GET    | `/metrics` | Model performance metrics         |
| POST   | `/explain` | SHAP-based prediction explanation |
| GET    | `/docs`    | Swagger/OpenAPI documentation     |

---

## Sample Prediction Request

### POST /predict

```json
{
  "creditScore": 619,
  "age": 42,
  "tenure": 2,
  "balance": 0.0,
  "numOfProducts": 1,
  "hasCrCard": 1,
  "isActiveMember": 1,
  "estimatedSalary": 101348.88,
  "geography": "France",
  "gender": "Female"
}
```

### Response

```json
{
  "prediction": 1,
  "probability": 0.87,
  "riskScore": "HIGH"
}
```

---

# 📊 Dashboard Features

## Customer Prediction

* Individual customer churn prediction
* Risk score classification (Low / Medium / High)
* Churn probability visualization
* SHAP waterfall plots for prediction explanation

## Model Performance Monitoring

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* ROC Curve
* Confusion Matrix

## Business Analytics

* Historical prediction records
* Risk distribution analysis
* Feature importance visualization
* Customer behavior insights

---

# 🗄️ PostgreSQL Database

## predictions

| Column         | Type      |
| -------------- | --------- |
| id             | PK        |
| input_features | JSON      |
| prediction     | INT       |
| probability    | FLOAT     |
| created_at     | TIMESTAMP |

## model_metrics

| Column     | Type      |
| ---------- | --------- |
| id         | PK        |
| accuracy   | FLOAT     |
| precision  | FLOAT     |
| recall     | FLOAT     |
| f1_score   | FLOAT     |
| roc_auc    | FLOAT     |
| created_at | TIMESTAMP |

### Database Initialization

```python
from src.db import initDatabase

initDatabase()
```

---

# 🐳 Docker Deployment

## Docker Compose

```bash
docker-compose up --build
```

This command:

* Starts PostgreSQL
* Starts FastAPI
* Creates an internal network between services

## Manual Docker Deployment

### Build

```bash
docker build -t churn-api .
```

### Run

```bash
docker run -p 8000:8000 churn-api
```

---

# 📈 Expected Model Performance

### XGBoost (Reference Results)

| Metric    | Score |
| --------- | ----- |
| Accuracy  | ~0.85 |
| Precision | ~0.85 |
| Recall    | ~0.87 |
| F1-Score  | ~0.86 |
| ROC-AUC   | ~0.92 |

> Actual results may vary depending on dataset splits, feature engineering strategies, and hyperparameter configurations.

---

# 🔧 Technical Highlights

## Data Preprocessing Pipeline

1. Removal of irrelevant features
2. Categorical encoding
3. Class imbalance mitigation using SMOTE
4. Feature scaling with StandardScaler

## Training Strategy

* 80/20 Train-Test Split
* Stratified Sampling
* 5-Fold Cross Validation
* Randomized Hyperparameter Search

---

# 🎯 Skills Demonstrated

* ✅ End-to-End Machine Learning Pipeline Development
* ✅ Explainable AI (SHAP)
* ✅ Production-Ready REST API Development
* ✅ Model Serving and Deployment
* ✅ Docker Containerization
* ✅ PostgreSQL Integration
* ✅ Interactive Business Intelligence Dashboard
* ✅ Applied MLOps Foundations

---

# 🔮 Future Enhancements

* MLflow integration for model versioning
* CI/CD pipelines with GitHub Actions
* Model drift detection and monitoring
* A/B testing framework
* Real-time prediction streaming
* Advanced feature engineering pipelines
* Feature Store integration
* Cloud-native deployment architecture
