"""
Model eğitim modülü.
Baseline (Logistic Regression) ve Final Model (XGBoost) eğitimi.
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report)
import xgboost as xgb

# Proje root'unu path'e ekle
projectRoot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if projectRoot not in sys.path:
    sys.path.insert(0, projectRoot)

from src.preprocessing import preprocessPipeline
from src.feature_engineering import applyFeatureEngineering

# db import'u opsiyonel (veritabanı yoksa hata vermesin)
try:
    from src.db import saveModelMetrics
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("Veritabanı modülü yüklenemedi, metrikler veritabanına kaydedilmeyecek.")


def trainBaselineModel(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Baseline Logistic Regression modelini eğitir.
    
    Args:
        X_train: Eğitim özellikleri
        y_train: Eğitim etiketleri
        X_test: Test özellikleri
        y_test: Test etiketleri
        
    Returns:
        dict: Model ve metrikler
    """
    print("Baseline model (Logistic Regression) eğitiliyor...")
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Tahminler
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrikler
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"Baseline Model Metrikleri:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return {
        'model': model,
        'metrics': metrics,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def trainXGBoostModel(X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     useHyperparameterTuning: bool = True) -> dict:
    """
    XGBoost modelini eğitir ve hyperparameter tuning yapar.
    
    Args:
        X_train: Eğitim özellikleri
        y_train: Eğitim etiketleri
        X_test: Test özellikleri
        y_test: Test etiketleri
        useHyperparameterTuning: Hyperparameter tuning yapılacak mı
        
    Returns:
        dict: Model ve metrikler
    """
    print("XGBoost modeli eğitiliyor...")
    
    if useHyperparameterTuning:
        # Hyperparameter grid
        paramGrid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Base model
        baseModel = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # RandomizedSearchCV
        print("Hyperparameter tuning yapılıyor...")
        randomSearch = RandomizedSearchCV(
            estimator=baseModel,
            param_distributions=paramGrid,
            n_iter=20,
            cv=5,
            scoring='roc_auc',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        randomSearch.fit(X_train, y_train)
        model = randomSearch.best_estimator_
        
        print(f"En iyi parametreler: {randomSearch.best_params_}")
    else:
        # Default parametrelerle
        model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1
        )
        model.fit(X_train, y_train)
    
    # Cross-validation
    print("Cross-validation yapılıyor...")
    cvScores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"CV ROC-AUC: {cvScores.mean():.4f} (+/- {cvScores.std() * 2:.4f})")
    
    # Tahminler
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrikler
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\nXGBoost Model Metrikleri:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'model': model,
        'metrics': metrics,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm
    }


def saveModel(model: object, scaler: object, modelPath: str, scalerPath: str):
    """
    Model ve scaler'ı kaydeder.
    
    Args:
        model: Eğitilmiş model
        scaler: Eğitilmiş scaler
        modelPath: Model kayıt yolu
        scalerPath: Scaler kayıt yolu
    """
    os.makedirs(os.path.dirname(modelPath), exist_ok=True)
    os.makedirs(os.path.dirname(scalerPath), exist_ok=True)
    
    joblib.dump(model, modelPath)
    joblib.dump(scaler, scalerPath)
    
    print(f"Model kaydedildi: {modelPath}")
    print(f"Scaler kaydedildi: {scalerPath}")


def trainPipeline(dataPath: str, modelSavePath: str = 'models/xgboost_model.pkl',
                 scalerSavePath: str = 'models/scaler.pkl',
                 useFeatureEngineering: bool = False,
                 useHyperparameterTuning: bool = True,
                 saveToDatabase: bool = True) -> dict:
    """
    Tüm eğitim pipeline'ını çalıştırır.
    
    Args:
        dataPath: Dataset dosya yolu
        modelSavePath: Model kayıt yolu
        scalerSavePath: Scaler kayıt yolu
        useFeatureEngineering: Feature engineering uygulanacak mı
        useHyperparameterTuning: Hyperparameter tuning yapılacak mı
        saveToDatabase: Metrikleri veritabanına kaydedecek mi
        
    Returns:
        dict: Eğitim sonuçları
    """
    print("=" * 60)
    print("MODEL EĞİTİMİ BAŞLIYOR")
    print("=" * 60)
    
    # Preprocessing
    print("\n1. Preprocessing...")
    preprocessResult = preprocessPipeline(dataPath, applySmote=True, doScaling=True)
    
    X_train = preprocessResult['X_train']
    X_test = preprocessResult['X_test']
    y_train = preprocessResult['y_train']
    y_test = preprocessResult['y_test']
    scaler = preprocessResult['scaler']
    featureNames = preprocessResult['feature_names']
    
    # Feature Engineering (opsiyonel)
    if useFeatureEngineering:
        print("\n2. Feature Engineering...")
        # DataFrame'e dönüştür (feature engineering için)
        if featureNames:
            X_train_df = pd.DataFrame(X_train, columns=featureNames)
            X_test_df = pd.DataFrame(X_test, columns=featureNames)
            
            X_train_df = applyFeatureEngineering(X_train_df)
            X_test_df = applyFeatureEngineering(X_test_df)
            
            # Feature names güncelle (yeni feature'lar eklendi)
            featureNames = list(X_train_df.columns)
            
            # Tekrar scaling
            X_train = scaler.fit_transform(X_train_df)
            X_test = scaler.transform(X_test_df)
    
    # Baseline Model
    print("\n3. Baseline Model Eğitimi...")
    baselineResult = trainBaselineModel(X_train, y_train, X_test, y_test)
    
    # XGBoost Model
    print("\n4. XGBoost Model Eğitimi...")
    xgboostResult = trainXGBoostModel(
        X_train, y_train, X_test, y_test,
        useHyperparameterTuning=useHyperparameterTuning
    )
    
    # Model Kaydetme
    print("\n5. Model Kaydediliyor...")
    saveModel(xgboostResult['model'], scaler, modelSavePath, scalerSavePath)
    
    # SHAP Global Feature Importance hesapla ve kaydet
    featureImportance = None
    explainer = None
    try:
        from src.explain import createSHAPExplainer, getGlobalFeatureImportance, saveExplainer
        
        print("\n6. SHAP Global Feature Importance Hesaplanıyor...")
        explainer = createSHAPExplainer(xgboostResult['model'], X_train)
        featureImportance = getGlobalFeatureImportance(
            explainer, 
            X_train, 
            featureNames=featureNames
        )
        
        # Explainer'ı kaydet
        explainerPath = os.path.join(os.path.dirname(modelSavePath), 'shap_explainer.pkl')
        saveExplainer(explainer, explainerPath)
        print(f"SHAP explainer kaydedildi: {explainerPath}")
        print(f"Feature importance hesaplandı: {len(featureImportance)} özellik")
        
    except Exception as e:
        print(f"SHAP feature importance hesaplanamadı (devam ediliyor): {e}")
    
    # Metrikleri JSON dosyasına kaydet (ROC ve Confusion Matrix dahil)
    metricsPath = os.path.join(os.path.dirname(modelSavePath), 'model_metrics.json')
    print(f"\n7. Metrikler JSON dosyasına kaydediliyor: {metricsPath}")
    
    # ROC ve Confusion Matrix için gerekli verileri ekle
    metricsData = {
        **xgboostResult['metrics'],
        'y_test': y_test.tolist(),  # Gerçek değerler
        'y_pred': xgboostResult['y_pred'].tolist(),  # Tahminler
        'y_pred_proba': xgboostResult['y_pred_proba'].tolist(),  # Tahmin olasılıkları
        'confusion_matrix': xgboostResult['confusion_matrix'].tolist()  # Confusion matrix
    }
    
    # Feature importance varsa ekle (numpy tiplerini Python tiplerine çevir)
    if featureImportance is not None:
        # numpy float32/float64'ü Python float'a çevir
        metricsData['feature_importance'] = {
            k: float(v) for k, v in featureImportance.items()
        }
    
    with open(metricsPath, 'w', encoding='utf-8') as f:
        json.dump(metricsData, f, indent=2)
    print("Metrikler başarıyla kaydedildi (ROC, Confusion Matrix ve Feature Importance dahil).")
    
    # Veritabanına Kaydetme (opsiyonel)
    if saveToDatabase and DB_AVAILABLE:
        print("\n8. Metrikler Veritabanına Kaydediliyor...")
        try:
            saveModelMetrics(xgboostResult['metrics'])
            print("Metrikler veritabanına başarıyla kaydedildi.")
        except Exception as e:
            print(f"Veritabanı kayıt hatası (devam ediliyor): {e}")
    elif saveToDatabase and not DB_AVAILABLE:
        print("\n8. Veritabanı modülü mevcut değil, metrikler veritabanına kaydedilmedi.")
    
    print("\n" + "=" * 60)
    print("MODEL EĞİTİMİ TAMAMLANDI")
    print("=" * 60)
    
    return {
        'baseline': baselineResult,
        'xgboost': xgboostResult,
        'scaler': scaler
    }


if __name__ == '__main__':
    # Eğitim pipeline'ını çalıştır
    dataPath = 'data/raw/ChurnModel.csv'
    
    trainPipeline(
        dataPath=dataPath,
        modelSavePath='models/xgboost_model.pkl',
        scalerSavePath='models/scaler.pkl',
        useFeatureEngineering=False,
        useHyperparameterTuning=True,
        saveToDatabase=False  # DB henüz hazır değil
    )
