"""
PostgreSQL veritabanı entegrasyonu.
Tablolar: predictions, model_metrics
"""
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, JSON, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()


class Prediction(Base):
    """Predictions tablosu modeli"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, index=True)
    input_features = Column(JSON)
    prediction = Column(Integer)
    probability = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelMetric(Base):
    """Model metrics tablosu modeli"""
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True, index=True)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


def getDatabaseUrl() -> str:
    """
    Veritabanı URL'ini environment variable'lardan alır.
    
    Returns:
        str: Database URL
    """
    dbHost = os.getenv('DB_HOST', 'localhost')
    dbPort = os.getenv('DB_PORT', '5432')
    dbName = os.getenv('DB_NAME', 'churn_db')
    dbUser = os.getenv('DB_USER', 'postgres')
    dbPassword = os.getenv('DB_PASSWORD', 'postgres')
    
    return f"postgresql://{dbUser}:{dbPassword}@{dbHost}:{dbPort}/{dbName}"


def createEngine():
    """
    SQLAlchemy engine oluşturur.
    
    Returns:
        Engine: SQLAlchemy engine
    """
    databaseUrl = getDatabaseUrl()
    engine = create_engine(databaseUrl, echo=False)
    return engine


def createTables(engine):
    """
    Veritabanı tablolarını oluşturur.
    
    Args:
        engine: SQLAlchemy engine
    """
    Base.metadata.create_all(bind=engine)
    print("Veritabanı tabloları oluşturuldu.")


def getSession() -> Session:
    """
    Database session oluşturur.
    
    Returns:
        Session: SQLAlchemy session
    """
    engine = createEngine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()


def initDatabase():
    """
    Veritabanını başlatır (tabloları oluşturur).
    """
    engine = createEngine()
    createTables(engine)
    return engine


def savePrediction(inputFeatures: dict, prediction: int, probability: float):
    """
    Tahmin sonucunu veritabanına kaydeder.
    
    Args:
        inputFeatures: Girdi özellikleri (dict)
        prediction: Tahmin (0 veya 1)
        probability: Tahmin olasılığı (0-1 arası)
    """
    session = getSession()
    try:
        predictionRecord = Prediction(
            input_features=inputFeatures,
            prediction=prediction,
            probability=probability
        )
        session.add(predictionRecord)
        session.commit()
        return predictionRecord.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def saveModelMetrics(metrics: dict):
    """
    Model metriklerini veritabanına kaydeder.
    
    Args:
        metrics: Metrikler dict'i (accuracy, precision, recall, f1_score, roc_auc)
    """
    session = getSession()
    try:
        metricRecord = ModelMetric(
            accuracy=metrics.get('accuracy'),
            precision=metrics.get('precision'),
            recall=metrics.get('recall'),
            f1_score=metrics.get('f1_score'),
            roc_auc=metrics.get('roc_auc')
        )
        session.add(metricRecord)
        session.commit()
        return metricRecord.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def getLatestMetrics() -> dict:
    """
    En son model metriklerini getirir.
    
    Returns:
        dict: Model metrikleri
    """
    session = getSession()
    try:
        latestMetric = session.query(ModelMetric).order_by(
            ModelMetric.created_at.desc()
        ).first()
        
        if latestMetric:
            return {
                'accuracy': latestMetric.accuracy,
                'precision': latestMetric.precision,
                'recall': latestMetric.recall,
                'f1_score': latestMetric.f1_score,
                'roc_auc': latestMetric.roc_auc,
                'created_at': latestMetric.created_at.isoformat()
            }
        return None
    finally:
        session.close()


def getPredictions(limit: int = 100) -> list:
    """
    Son tahminleri getirir.
    
    Args:
        limit: Getirilecek kayıt sayısı
        
    Returns:
        list: Tahmin kayıtları
    """
    session = getSession()
    try:
        predictions = session.query(Prediction).order_by(
            Prediction.created_at.desc()
        ).limit(limit).all()
        
        return [
            {
                'id': p.id,
                'input_features': p.input_features,
                'prediction': p.prediction,
                'probability': p.probability,
                'created_at': p.created_at.isoformat()
            }
            for p in predictions
        ]
    finally:
        session.close()
