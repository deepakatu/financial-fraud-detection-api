
"""
Database models for fraud detection
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.sql import func
from .connection import Base

class Transaction(Base):
    """Transaction model"""
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, unique=True, index=True)
    user_id = Column(String, index=True)
    account_id = Column(String, index=True)
    merchant_id = Column(String, index=True)
    
    amount = Column(Float)
    transaction_type = Column(String)
    merchant_category = Column(String)
    
    is_fraud = Column(Boolean, default=False)
    fraud_probability = Column(Float)
    risk_score = Column(Float)
    
    location = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    
    device_id = Column(String)
    ip_address = Column(String)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class FraudAnalysis(Base):
    """Fraud analysis results model"""
    __tablename__ = "fraud_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, index=True)
    
    model_version = Column(String)
    fraud_probability = Column(Float)
    risk_score = Column(Float)
    risk_level = Column(String)
    
    risk_factors = Column(JSON)
    feature_importance = Column(JSON)
    recommendations = Column(JSON)
    
    processing_time_ms = Column(Float)
    models_used = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class UserProfile(Base):
    """User profile model"""
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, index=True)
    
    account_age_days = Column(Integer)
    total_transactions = Column(Integer, default=0)
    avg_transaction_amount = Column(Float, default=0.0)
    
    failed_transactions_count = Column(Integer, default=0)
    disputed_transactions_count = Column(Integer, default=0)
    risk_score = Column(Float, default=0.0)
    
    preferred_merchants = Column(JSON)
    typical_locations = Column(JSON)
    active_hours = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class MerchantProfile(Base):
    """Merchant profile model"""
    __tablename__ = "merchant_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    merchant_id = Column(String, unique=True, index=True)
    merchant_name = Column(String)
    category = Column(String)
    
    registration_date = Column(DateTime(timezone=True))
    location = Column(String)
    
    total_transactions = Column(Integer, default=0)
    total_volume = Column(Float, default=0.0)
    avg_transaction_amount = Column(Float, default=0.0)
    
    fraud_rate = Column(Float, default=0.0)
    chargeback_rate = Column(Float, default=0.0)
    risk_score = Column(Float, default=0.0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Alert(Base):
    """Alert model"""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, index=True)
    user_id = Column(String, index=True)
    
    alert_type = Column(String)
    severity = Column(String)
    message = Column(Text)
    
    is_resolved = Column(Boolean, default=False)
    resolved_by = Column(String)
    resolution_notes = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    resolved_at = Column(DateTime(timezone=True))

class ModelPerformance(Base):
    """Model performance tracking"""
    __tablename__ = "model_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String)
    version = Column(String)
    
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_roc = Column(Float)
    
    avg_prediction_time_ms = Column(Float)
    throughput_per_second = Column(Float)
    
    training_samples = Column(Integer)
    validation_samples = Column(Integer)
    feature_count = Column(Integer)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
