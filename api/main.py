"""
FastAPI backend uygulaması.
Model serving ve prediction endpoints.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import logging

# src modülünü path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db import savePrediction, getLatestMetrics, initDatabase
from src.explain import loadExplainer, getLocalExplanation

# Logging yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI uygulaması
app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description="Müşteri churn tahmini için REST API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global değişkenler (startup'ta yüklenecek)
model = None
scaler = None
explainer = None
featureNames = None


# Pydantic modelleri
class CustomerInput(BaseModel):
    """Müşteri girdi şeması"""
    creditScore: int = Field(..., ge=0, le=850, description="Kredi skoru (0-850)")
    age: int = Field(..., ge=18, le=100, description="Yaş")
    tenure: int = Field(..., ge=0, le=10, description="Banka ile yıl sayısı")
    balance: float = Field(..., ge=0, description="Hesap bakiyesi")
    numOfProducts: int = Field(..., ge=1, le=4, description="Kullanılan ürün sayısı")
    hasCrCard: int = Field(..., ge=0, le=1, description="Kredi kartı var mı (0/1)")
    isActiveMember: int = Field(..., ge=0, le=1, description="Aktif üye mi (0/1)")
    estimatedSalary: float = Field(..., ge=0, description="Tahmini maaş")
    geography: str = Field(..., description="Coğrafya (France, Spain, Germany)")
    gender: str = Field(..., description="Cinsiyet (Male, Female)")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Tahmin yanıt şeması"""
    prediction: int = Field(..., description="Tahmin (0: Ayrılmayacak, 1: Ayrılacak)")
    probability: float = Field(..., description="Ayrılma olasılığı (0-1)")
    riskScore: str = Field(..., description="Risk seviyesi (LOW, MEDIUM, HIGH)")


class MetricsResponse(BaseModel):
    """Metrikler yanıt şeması"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    created_at: Optional[str] = None


def preprocessInput(inputData: CustomerInput) -> np.ndarray:
    """
    Kullanıcı girdisini model için hazırlar.
    
    Args:
        inputData: CustomerInput objesi
        
    Returns:
        np.ndarray: Preprocessed features
    """
    # Geography encoding
    geographyGermany = 1 if inputData.geography == "Germany" else 0
    geographySpain = 1 if inputData.geography == "Spain" else 0
    
    # Gender encoding
    genderMale = 1 if inputData.gender == "Male" else 0
    
    # Feature array
    features = np.array([[
        inputData.creditScore,
        inputData.age,
        inputData.tenure,
        inputData.balance,
        inputData.numOfProducts,
        inputData.hasCrCard,
        inputData.isActiveMember,
        inputData.estimatedSalary,
        geographyGermany,
        geographySpain,
        genderMale
    ]])
    
    return features


def getRiskScore(probability: float) -> str:
    """
    Olasılığa göre risk skoru döndürür.
    
    Args:
        probability: Tahmin olasılığı
        
    Returns:
        str: Risk skoru
    """
    if probability < 0.3:
        return "LOW"
    elif probability < 0.7:
        return "MEDIUM"
    else:
        return "HIGH"


@app.on_event("startup")
async def startupEvent():
    """Uygulama başlangıcında model ve scaler'ı yükle"""
    global model, scaler, explainer, featureNames
    
    modelPath = os.getenv('MODEL_PATH', 'models/xgboost_model.pkl')
    scalerPath = os.getenv('SCALER_PATH', 'models/scaler.pkl')
    explainerPath = os.getenv('EXPLAINER_PATH', 'models/shap_explainer.pkl')
    
    try:
        # Model yükle
        if os.path.exists(modelPath):
            model = joblib.load(modelPath)
            logger.info(f"Model yüklendi: {modelPath}")
        else:
            logger.warning(f"Model bulunamadı: {modelPath}")
        
        # Scaler yükle
        if os.path.exists(scalerPath):
            scaler = joblib.load(scalerPath)
            logger.info(f"Scaler yüklendi: {scalerPath}")
        else:
            logger.warning(f"Scaler bulunamadı: {scalerPath}")
        
        # Explainer yükle (opsiyonel)
        if os.path.exists(explainerPath):
            try:
                explainer = loadExplainer(explainerPath)
                logger.info(f"SHAP explainer yüklendi: {explainerPath}")
            except Exception as e:
                logger.warning(f"Explainer yüklenemedi: {e}")
        
        # Feature names
        featureNames = [
            'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
            'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
            'Geography_Germany', 'Geography_Spain', 'Gender_Male'
        ]
        
        # Veritabanı başlat (opsiyonel)
        try:
            initDatabase()
            logger.info("Veritabanı başlatıldı")
        except Exception as e:
            logger.warning(f"Veritabanı başlatılamadı (opsiyonel, devam ediliyor): {e}")
            
    except Exception as e:
        logger.error(f"Startup hatası: {e}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Bank Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(inputData: CustomerInput):
    """
    Tek müşteri için churn tahmini yapar.
    
    Args:
        inputData: Müşteri bilgileri
        
    Returns:
        PredictionResponse: Tahmin sonucu
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model veya scaler yüklenemedi. Lütfen model dosyalarını kontrol edin."
        )
    
    try:
        # Preprocessing
        features = preprocessInput(inputData)
        featuresScaled = scaler.transform(features)
        
        # Tahmin
        prediction = model.predict(featuresScaled)[0]
        probability = model.predict_proba(featuresScaled)[0][1]
        
        # Risk skoru
        riskScore = getRiskScore(probability)
        
        # Veritabanına kaydet
        try:
            inputDict = inputData.dict()
            savePrediction(inputDict, int(prediction), float(probability))
        except Exception as e:
            logger.warning(f"Veritabanı kayıt hatası: {e}")
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            riskScore=riskScore
        )
        
    except Exception as e:
        logger.error(f"Tahmin hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse)
async def getMetrics():
    """
    Model performans metriklerini getirir.
    
    Returns:
        MetricsResponse: Model metrikleri
    """
    try:
        # Önce JSON dosyasından oku (veritabanı opsiyonel)
        metricsPath = os.path.join('models', 'model_metrics.json')
        metrics = None
        
        if os.path.exists(metricsPath):
            import json
            with open(metricsPath, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            logger.info("Metrikler JSON dosyasından yüklendi.")
        else:
            # JSON yoksa veritabanından dene
            try:
                metrics = getLatestMetrics()
                logger.info("Metrikler veritabanından yüklendi.")
            except Exception as dbError:
                logger.warning(f"Veritabanından metrikler alınamadı: {dbError}")
        
        if metrics is None:
            raise HTTPException(
                status_code=404,
                detail="Metrikler bulunamadı. Önce model eğitimi yapılmalı."
            )
        
        # Sadece temel metrikleri döndür (response model için)
        return MetricsResponse(
            accuracy=metrics.get('accuracy'),
            precision=metrics.get('precision'),
            recall=metrics.get('recall'),
            f1_score=metrics.get('f1_score'),
            roc_auc=metrics.get('roc_auc'),
            created_at=metrics.get('created_at')
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrikler getirme hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Hata: {str(e)}")


@app.get("/metrics/full")
async def getFullMetrics():
    """
    Tüm metrik verilerini getirir (ROC ve Confusion Matrix dahil).
    
    Returns:
        dict: Tüm metrik verileri
    """
    try:
        metricsPath = os.path.join('models', 'model_metrics.json')
        
        if os.path.exists(metricsPath):
            import json
            with open(metricsPath, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            logger.info("Tüm metrikler JSON dosyasından yüklendi.")
            return metrics
        else:
            raise HTTPException(
                status_code=404,
                detail="Metrikler bulunamadı. Önce model eğitimi yapılmalı."
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrikler getirme hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Hata: {str(e)}")


@app.post("/explain")
async def explainPrediction(inputData: CustomerInput):
    """
    SHAP ile tahmin açıklaması yapar.
    
    Args:
        inputData: Müşteri bilgileri
        
    Returns:
        dict: SHAP explanation
    """
    if explainer is None:
        raise HTTPException(
            status_code=503,
            detail="SHAP explainer yüklenemedi."
        )
    
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model veya scaler yüklenemedi."
        )
    
    try:
        # Preprocessing
        features = preprocessInput(inputData)
        featuresScaled = scaler.transform(features)
        
        # SHAP explanation
        explanation = getLocalExplanation(
            explainer,
            featuresScaled[0],
            featureNames=featureNames
        )
        
        return explanation
        
    except Exception as e:
        logger.error(f"Explanation hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Hata: {str(e)}")


@app.get("/feature-importance")
async def getFeatureImportance():
    """
    Global feature importance değerlerini getirir (SHAP tabanlı).
    
    Returns:
        dict: Feature importance dict'i
    """
    try:
        # Önce JSON dosyasından oku
        # API'nin çalışma dizinini kontrol et
        baseDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        metricsPath = os.path.join(baseDir, 'models', 'model_metrics.json')
        
        # Alternatif yol (mevcut dizin)
        if not os.path.exists(metricsPath):
            metricsPath = os.path.join('models', 'model_metrics.json')
        
        if os.path.exists(metricsPath):
            import json
            with open(metricsPath, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            if 'feature_importance' in metrics:
                return {
                    'feature_importance': metrics['feature_importance'],
                    'source': 'model_metrics.json'
                }
        
        # JSON'da yoksa hata döndür
        raise HTTPException(
            status_code=404,
            detail="Feature importance bulunamadı. Model eğitimi yapılmalı ve SHAP feature importance hesaplanmalı."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature importance hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Hata: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
