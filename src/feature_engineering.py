"""
Feature engineering modülü.
Yeni feature'lar oluşturur: age_group, balance_ratio, activity_score
"""
import pandas as pd
import numpy as np


def createAgeGroup(df: pd.DataFrame, ageColumn: str = 'Age') -> pd.DataFrame:
    """
    Yaş grupları oluşturur.
    
    Args:
        df: Veri DataFrame
        ageColumn: Yaş kolonu adı
        
    Returns:
        DataFrame: age_group kolonu eklenmiş DataFrame
    """
    df = df.copy()
    
    def categorizeAge(age):
        if age < 30:
            return 'Young'
        elif age < 50:
            return 'Middle'
        else:
            return 'Senior'
    
    df['age_group'] = df[ageColumn].apply(categorizeAge)
    
    # One-Hot Encoding
    df = pd.get_dummies(df, columns=['age_group'], drop_first=True, prefix='age_group')
    
    return df


def createBalanceRatio(df: pd.DataFrame, balanceColumn: str = 'Balance', 
                      salaryColumn: str = 'EstimatedSalary') -> pd.DataFrame:
    """
    Balance/Salary oranı oluşturur.
    
    Args:
        df: Veri DataFrame
        balanceColumn: Bakiye kolonu adı
        salaryColumn: Maaş kolonu adı
        
    Returns:
        DataFrame: balance_ratio kolonu eklenmiş DataFrame
    """
    df = df.copy()
    
    # Sıfıra bölme hatasını önlemek için küçük bir değer ekle
    df['balance_ratio'] = df[balanceColumn] / (df[salaryColumn] + 1)
    
    return df


def createActivityScore(df: pd.DataFrame, hasCrCardColumn: str = 'HasCrCard',
                        isActiveMemberColumn: str = 'IsActiveMember',
                        numOfProductsColumn: str = 'NumOfProducts') -> pd.DataFrame:
    """
    Aktivite skoru oluşturur (kredi kartı + aktif üyelik + ürün sayısı).
    
    Args:
        df: Veri DataFrame
        hasCrCardColumn: Kredi kartı kolonu adı
        isActiveMemberColumn: Aktif üyelik kolonu adı
        numOfProductsColumn: Ürün sayısı kolonu adı
        
    Returns:
        DataFrame: activity_score kolonu eklenmiş DataFrame
    """
    df = df.copy()
    
    df['activity_score'] = (
        df[hasCrCardColumn] * 1 +
        df[isActiveMemberColumn] * 2 +
        df[numOfProductsColumn] * 1
    )
    
    return df


def applyFeatureEngineering(df: pd.DataFrame, useAgeGroup: bool = True,
                           useBalanceRatio: bool = True,
                           useActivityScore: bool = True) -> pd.DataFrame:
    """
    Tüm feature engineering işlemlerini uygular.
    
    Args:
        df: Veri DataFrame
        useAgeGroup: age_group feature'ı eklenecek mi
        useBalanceRatio: balance_ratio feature'ı eklenecek mi
        useActivityScore: activity_score feature'ı eklenecek mi
        
    Returns:
        DataFrame: Yeni feature'lar eklenmiş DataFrame
    """
    df = df.copy()
    
    if useAgeGroup:
        df = createAgeGroup(df)
    
    if useBalanceRatio:
        df = createBalanceRatio(df)
    
    if useActivityScore:
        df = createActivityScore(df)
    
    return df
