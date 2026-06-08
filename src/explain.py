"""
SHAP explainability modülü.
Model açıklanabilirliği için SHAP değerleri hesaplar.
"""
import os
import joblib
import numpy as np
import pandas as pd
import shap
from typing import Optional


def createSHAPExplainer(model, X_train: np.ndarray, modelType: str = 'xgboost'):
    """
    SHAP explainer oluşturur.
    
    Args:
        model: Eğitilmiş model
        X_train: Eğitim verisi (background data)
        modelType: Model tipi ('xgboost' veya 'tree')
        
    Returns:
        SHAP explainer
    """
    if modelType == 'xgboost' or hasattr(model, 'get_booster'):
        # XGBoost için TreeExplainer
        explainer = shap.TreeExplainer(model)
    else:
        # Diğer tree-based modeller için
        explainer = shap.TreeExplainer(model)
    
    return explainer


def calculateSHAPValues(explainer, X: np.ndarray, maxSamples: int = 100):
    """
    SHAP değerlerini hesaplar.
    
    Args:
        explainer: SHAP explainer
        X: Tahmin yapılacak veri
        maxSamples: Maksimum örnek sayısı (performans için)
        
    Returns:
        SHAP values
    """
    if len(X) > maxSamples:
        # Büyük veri setleri için örnekleme
        indices = np.random.choice(len(X), maxSamples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    shapValues = explainer.shap_values(X_sample)
    
    return shapValues, X_sample


def getGlobalFeatureImportance(explainer, X_train: np.ndarray, 
                               featureNames: Optional[list] = None) -> dict:
    """
    Global feature importance hesaplar.
    
    Args:
        explainer: SHAP explainer
        X_train: Eğitim verisi
        featureNames: Feature isimleri
        
    Returns:
        dict: Feature importance dict'i
    """
    # SHAP değerlerini hesapla
    shapValues, _ = calculateSHAPValues(explainer, X_train, maxSamples=100)
    
    # Ortalama mutlak SHAP değerleri
    if isinstance(shapValues, list):
        # Binary classification için
        shapValues = shapValues[1]  # Positive class
    
    meanAbsShap = np.abs(shapValues).mean(axis=0)
    
    if featureNames is None:
        featureNames = [f'Feature_{i}' for i in range(len(meanAbsShap))]
    
    featureImportance = dict(zip(featureNames, meanAbsShap))
    
    # Sırala
    featureImportance = dict(
        sorted(featureImportance.items(), key=lambda x: x[1], reverse=True)
    )
    
    return featureImportance


def getLocalExplanation(explainer, X_instance: np.ndarray, 
                      featureNames: Optional[list] = None) -> dict:
    """
    Tek bir örnek için local explanation hesaplar.
    
    Args:
        explainer: SHAP explainer
        X_instance: Tek örnek (1D array veya 2D array)
        featureNames: Feature isimleri
        
    Returns:
        dict: Local explanation (SHAP values, base value, prediction)
    """
    # 2D array'e çevir
    if X_instance.ndim == 1:
        X_instance = X_instance.reshape(1, -1)
    
    shapValues = explainer.shap_values(X_instance)
    baseValue = explainer.expected_value
    
    # Binary classification için
    if isinstance(shapValues, list):
        shapValues = shapValues[1]  # Positive class
        if isinstance(baseValue, np.ndarray):
            baseValue = baseValue[1]
    
    shapValues = shapValues[0]  # İlk örnek
    
    if featureNames is None:
        featureNames = [f'Feature_{i}' for i in range(len(shapValues))]
    
    return {
        'shap_values': dict(zip(featureNames, shapValues)),
        'base_value': float(baseValue),
        'prediction': float(baseValue + shapValues.sum())
    }


def saveExplainer(explainer, filePath: str):
    """
    SHAP explainer'ı kaydeder.
    
    Args:
        explainer: SHAP explainer
        filePath: Kayıt yolu
    """
    os.makedirs(os.path.dirname(filePath), exist_ok=True)
    joblib.dump(explainer, filePath)
    print(f"SHAP explainer kaydedildi: {filePath}")


def loadExplainer(filePath: str):
    """
    SHAP explainer'ı yükler.
    
    Args:
        filePath: Dosya yolu
        
    Returns:
        SHAP explainer
    """
    if not os.path.exists(filePath):
        raise FileNotFoundError(f"Explainer bulunamadı: {filePath}")
    
    explainer = joblib.load(filePath)
    return explainer


def createExplainerFromModel(modelPath: str, X_train: np.ndarray,
                            explainerSavePath: Optional[str] = None) -> object:
    """
    Model dosyasından SHAP explainer oluşturur.
    
    Args:
        modelPath: Model dosya yolu
        X_train: Eğitim verisi
        explainerSavePath: Explainer kayıt yolu (opsiyonel)
        
    Returns:
        SHAP explainer
    """
    # Model yükle
    model = joblib.load(modelPath)
    
    # Explainer oluştur
    explainer = createSHAPExplainer(model, X_train)
    
    # Kaydet
    if explainerSavePath:
        saveExplainer(explainer, explainerSavePath)
    
    return explainer
