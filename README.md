# 🏦 Bank Customer Churn Decision Support System

Bu proje, banka müşterilerinin churn (bankadan ayrılma) olasılığını tahmin eden ve karar destek sağlayan uçtan uca bir makine öğrenmesi sistemidir.

## 🎯 Proje Amacı

Bu proje üç farklı role hitap edecek şekilde tasarlanmıştır:

- **Data Scientist** → Güçlü modelleme ve explainability (SHAP)
- **ML Engineer** → API ve Docker ile model serving
- **Data Analyst** → Dashboard ve iş içgörüleri

## 🏗️ Sistem Mimarisi

```
Dataset → Preprocessing → Feature Engineering → Model Training (XGBoost)
                                                          ↓
PostgreSQL ← FastAPI (Model Serving) ← Streamlit Dashboard (Analytics & UI)
```

### Mimari Katmanlar

1. **ML Layer**: Model training, evaluation, explainability (SHAP)
2. **FastAPI Backend**: RESTful API, model serving, PostgreSQL entegrasyonu
3. **Streamlit Dashboard**: Interaktif analitik ve tahmin arayüzü
4. **PostgreSQL**: Müşteri verileri, tahmin geçmişi, model metrikleri

## 🛠️ Kullanılan Teknolojiler

### ML & Data
- **Python**: Ana programlama dili
- **pandas/numpy**: Veri manipülasyonu ve sayısal hesaplamalar
- **scikit-learn**: ML algoritmaları ve preprocessing
- **XGBoost**: Final model (gradient boosting)
- **SHAP**: Model explainability ve feature importance
- **imbalanced-learn**: SMOTE ile class imbalance handling

### Backend
- **FastAPI**: Modern, hızlı web framework (async desteği, otomatik dokümantasyon)
- **Uvicorn**: ASGI server
- **SQLAlchemy**: ORM ve veritabanı yönetimi
- **Pydantic**: Veri validasyonu ve serialization
- **psycopg2**: PostgreSQL adapter

### Dashboard
- **Streamlit**: Hızlı dashboard geliştirme
- **Plotly**: İnteraktif görselleştirmeler
- **Seaborn/Matplotlib**: İstatistiksel grafikler

### Deployment
- **Docker**: Containerization
- **docker-compose**: Multi-container orchestration (API + PostgreSQL)

## 📁 Proje Yapısı

```
bank-customer-churn-prediction/
│
├── data/
│   └── raw/
│       └── ChurnModel.csv          # Dataset
│
├── models/
│   ├── xgboost_model.pkl           # XGBoost model
│   ├── scaler.pkl                  # StandardScaler
│   └── shap_explainer.pkl          # SHAP explainer
│
├── src/
│   ├── preprocessing.py            # Veri ön işleme
│   ├── feature_engineering.py     # Feature engineering
│   ├── train.py                    # Model eğitimi
│   ├── explain.py                  # SHAP explainability
│   └── db.py                       # PostgreSQL entegrasyonu
│
├── api/
│   └── main.py                     # FastAPI uygulama
│
├── dashboard/
│   └── app.py                      # Streamlit dashboard
│
├── notebooks/
│   └── customerPrediction.ipynb   # Mevcut analiz notebook'u
│
├── Dockerfile                      # API container
├── docker-compose.yml              # Docker orchestration
├── requirements.txt                # Python dependencies
└── README.md                       # Bu dosya
```

## 🚀 Kurulum

### 1. Repository'yi Klonlayın

```bash
git clone <repository-url>
cd bank-customer-churn-prediction
```

### 2. Virtual Environment Oluşturun

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

### 3. Dependencies Yükleyin

```bash
pip install -r requirements.txt
```

### 4. Environment Variables Ayarlayın

`.env` dosyası oluşturun (`.env.example` dosyasını referans alın):

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

## 📊 Makine Öğrenmesi Süreci

### 1. Exploratory Data Analysis (EDA)
- Veri yapısı ve istatistiksel özet
- Eksik değer analizi
- Kategorik değişken dağılımları
- Target variable dağılımı (class imbalance)

### 2. Feature Engineering
- Gereksiz kolonların kaldırılması (RowNumber, CustomerId, Surname)
- Kategorik encoding (One-Hot Encoding: Geography, Gender)
- Opsiyonel yeni feature'lar (age_group, balance_ratio, activity_score)

### 3. Class Imbalance Handling
- SMOTE ile oversampling
- Balanced dataset oluşturma

### 4. Baseline Model
- Logistic Regression (baseline)
- Performans metrikleri: Accuracy, Precision, Recall, F1-Score

### 5. Final Model (XGBoost)
- XGBoostClassifier ile eğitim
- RandomizedSearchCV ile hyperparameter tuning
- Cross-validation
- ROC-AUC optimizasyonu

### 6. Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC score
- Confusion Matrix
- Classification Report

### 7. Explainability (SHAP)
- SHAP TreeExplainer oluşturma
- Global feature importance
- Local explanation (tek örnek için)

## 🎯 Kullanım

### Model Eğitimi

```bash
python src/train.py
```

Bu komut:
- Veriyi yükler ve ön işler
- Baseline (Logistic Regression) modelini eğitir
- XGBoost modelini eğitir ve hyperparameter tuning yapar
- Modeli `models/` klasörüne kaydeder
- Metrikleri PostgreSQL'e kaydeder (opsiyonel)

### API'yi Çalıştırma

#### Local

```bash
uvicorn api.main:app --reload
```

API şu adreste çalışacak: `http://localhost:8000`

#### Docker

```bash
docker-compose up --build
```

### Dashboard'u Çalıştırma

```bash
streamlit run dashboard/app.py
```

Dashboard şu adreste çalışacak: `http://localhost:8501`

## 📡 API Endpoint'leri

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/` | API bilgileri |
| GET | `/health` | Health check |
| POST | `/predict` | Tek müşteri için churn tahmini |
| GET | `/metrics` | Model performans metrikleri |
| POST | `/explain` | SHAP ile tahmin açıklaması |
| GET | `/docs` | Swagger UI (otomatik) |

### Örnek Request

**POST /predict**

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

**Response**

```json
{
  "prediction": 1,
  "probability": 0.87,
  "riskScore": "HIGH"
}
```

## 📊 Dashboard Özellikleri

### 1. Prediction Page
- Tekil müşteri tahmini formu
- Risk skoru gösterimi (LOW/MEDIUM/HIGH)
- Tahmin olasılığı (probability)
- SHAP waterfall plot (individual explanation)

### 2. Model Performance Page
- Model performans metrikleri (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- ROC eğrisi
- Confusion matrix

### 3. Analytics Page
- Geçmiş tahminler
- Risk dağılım grafiği
- Feature importance grafiği

## 🗄️ PostgreSQL Veritabanı

### Tablolar

1. **predictions**
   - id (PK)
   - input_features (JSON)
   - prediction (INT)
   - probability (FLOAT)
   - created_at (TIMESTAMP)

2. **model_metrics**
   - id (PK)
   - accuracy, precision, recall, f1_score, roc_auc (FLOAT)
   - created_at (TIMESTAMP)

### Veritabanı Başlatma

```python
from src.db import initDatabase
initDatabase()
```

## 🐳 Docker Deployment

### Docker Compose ile Çalıştırma

```bash
docker-compose up --build
```

Bu komut:
- PostgreSQL container'ını başlatır
- FastAPI container'ını başlatır
- İki servis arasında network oluşturur

### Manuel Docker

```bash
# Build
docker build -t churn-api .

# Run
docker run -p 8000:8000 churn-api
```

## 📈 Model Performans

### XGBoost Model Metrikleri (Örnek)

- **Accuracy**: ~0.85
- **Precision**: ~0.85
- **Recall**: ~0.87
- **F1-Score**: ~0.86
- **ROC-AUC**: ~0.92

*Not: Gerçek metrikler model eğitimi sonrası belirlenir.*

## 🔧 Teknik Detaylar

### Data Preprocessing
1. Gereksiz kolonların kaldırılması (RowNumber, CustomerId, Surname)
2. Kategorik encoding (One-Hot Encoding: Geography, Gender)
3. SMOTE ile class imbalance çözümü
4. Feature scaling (StandardScaler)

### Model Training
- **Data Split**: 80% training, 20% test
- **Stratified Split**: Class distribution korunur
- **Cross-Validation**: 5-fold CV
- **Hyperparameter Tuning**: RandomizedSearchCV

## 🎯 Projenin Sağladığı Yetkinlikler

- ✅ End-to-End ML Pipeline
- ✅ Explainable AI Implementation (SHAP)
- ✅ REST API Development (FastAPI)
- ✅ Model Serving with Docker
- ✅ Business-Oriented Dashboard (Streamlit)
- ✅ PostgreSQL Integration
- ✅ Production-Ready Code Structure

## 🔮 Gelecek Geliştirmeler

- Model versioning (MLflow entegrasyonu)
- CI/CD pipeline (GitHub Actions)
- Model drift monitoring
- A/B testing framework
- Real-time prediction streaming
- Advanced feature engineering (feature stores)


