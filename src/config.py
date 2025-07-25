
"""
Configuration settings for the Fraud Detection API
"""

import os
from typing import List
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_TITLE: str = "Financial Fraud Detection API"
    API_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://user:password@localhost:5432/fraud_detection"
    )
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    BATCH_RATE_LIMIT_PER_MINUTE: int = 10
    MAX_BATCH_SIZE: int = 1000
    
    # ML Models
    MODEL_PATH: str = "models/"
    MODEL_UPDATE_INTERVAL_HOURS: int = 24
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Feature Engineering
    FEATURE_STORE_URL: str = os.getenv("FEATURE_STORE_URL", "")
    
    # External APIs
    EXTERNAL_RISK_API_URL: str = os.getenv("EXTERNAL_RISK_API_URL", "")
    EXTERNAL_RISK_API_KEY: str = os.getenv("EXTERNAL_RISK_API_KEY", "")
    
    # Performance
    MAX_WORKERS: int = 4
    WORKER_TIMEOUT: int = 120
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Model configurations
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "random_state": 42
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    },
    "neural_network": {
        "hidden_layers": [128, 64, 32],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 50
    }
}

# Feature configurations
FEATURE_CONFIG = {
    "transaction_features": [
        "amount",
        "merchant_category",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "amount_zscore",
        "frequency_1h",
        "frequency_24h",
        "avg_amount_7d",
        "location_risk_score"
    ],
    "user_features": [
        "user_age_days",
        "avg_transaction_amount",
        "transaction_count_7d",
        "unique_merchants_7d",
        "failed_transactions_7d",
        "user_risk_score"
    ],
    "merchant_features": [
        "merchant_risk_score",
        "merchant_transaction_count",
        "merchant_avg_amount",
        "merchant_fraud_rate"
    ]
}

# Risk thresholds
RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.8,
    "critical": 0.95
}

# Alert configurations
ALERT_CONFIG = {
    "high_risk_threshold": 0.8,
    "critical_risk_threshold": 0.95,
    "notification_channels": ["email", "slack", "webhook"],
    "escalation_rules": {
        "amount_threshold": 10000,
        "velocity_threshold": 5,
        "time_window_minutes": 60
    }
}

