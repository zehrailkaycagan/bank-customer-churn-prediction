"""
Veri ön işleme modülü.
Dataset yükleme, temizleme, encoding ve scaling işlemlerini içerir.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os


def loadData(dataPath: str) -> pd.DataFrame:
    """
    CSV dosyasından veri yükler.
    
    Args:
        dataPath: Dataset dosya yolu
        
    Returns:
        DataFrame: Yüklenen veri
    """
    if not os.path.exists(dataPath):
        raise FileNotFoundError(f"Dataset bulunamadı: {dataPath}")
    
    df = pd.read_csv(dataPath)
    return df


def cleanData(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gereksiz kolonları kaldırır.
    
    Args:
        df: Ham veri DataFrame
        
    Returns:
        DataFrame: Temizlenmiş veri
    """
    columnsToDrop = ['RowNumber', 'CustomerId', 'Surname']
    df = df.drop(columns=columnsToDrop, errors='ignore')
    return df


def encodeCategorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kategorik değişkenleri One-Hot Encoding ile kodlar.
    
    Args:
        df: Veri DataFrame
        
    Returns:
        DataFrame: Kodlanmış veri
    """
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
    df = df.astype(int)
    return df


def splitData(df: pd.DataFrame, targetColumn: str = 'Exited', 
              testSize: float = 0.2, randomState: int = 42) -> tuple:
    """
    Veriyi train ve test setlerine ayırır.
    
    Args:
        df: Veri DataFrame
        targetColumn: Hedef değişken adı
        testSize: Test set oranı
        randomState: Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[targetColumn])
    y = df[targetColumn]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testSize, random_state=randomState, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def applySMOTE(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    SMOTE ile class imbalance sorununu çözer.
    
    Args:
        X_train: Eğitim özellikleri
        y_train: Eğitim etiketleri
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def scaleFeatures(X_train: np.ndarray, X_test: np.ndarray, 
                 scaler: StandardScaler = None) -> tuple:
    """
    Özellikleri standardize eder.
    
    Args:
        X_train: Eğitim özellikleri
        X_test: Test özellikleri
        scaler: Önceden eğitilmiş scaler (opsiyonel)
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = scaler.transform(X_train)
    
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def preprocessPipeline(dataPath: str, applySmote: bool = True, 
                      doScaling: bool = True) -> dict:
    """
    Tüm preprocessing pipeline'ını çalıştırır.
    
    Args:
        dataPath: Dataset dosya yolu
        applySmote: SMOTE uygulanacak mı
        doScaling: Feature scaling uygulanacak mı
        
    Returns:
        dict: Preprocessing sonuçları ve scaler
    """
    # Veri yükleme
    df = loadData(dataPath)
    
    # Temizleme
    df = cleanData(df)
    
    # Encoding
    df = encodeCategorical(df)
    
    # Train/Test split
    X_train, X_test, y_train, y_test = splitData(df)
    
    # Feature isimlerini scaling'den ÖNCE kaydet (scaling sonrası numpy array olur)
    feature_names = list(X_train.columns)
    
    # SMOTE
    if applySmote:
        X_train, y_train = applySMOTE(X_train, y_train)
    
    # Scaling
    scaler = None
    if doScaling:
        X_train, X_test, scaler = scaleFeatures(X_train, X_test, scaler=None)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': feature_names  # Scaling öncesi kaydedilen gerçek feature isimleri
    }
