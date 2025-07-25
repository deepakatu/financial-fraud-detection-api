
"""
API routes for fraud detection service
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
import logging

from ..database.connection import get_db
from ..models.schemas import (
    TransactionRequest, 
    FraudAnalysisResponse,
    ModelPerformanceMetrics,
    HealthCheckResponse
)
from ..services.fraud_detector import FraudDetectionService
from ..utils.monitoring import metrics

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize fraud detection service
fraud_service = FraudDetectionService()

@router.get("/statistics")
async def get_statistics(db: Session = Depends(get_db)):
    """Get fraud detection statistics"""
    try:
        # Mock statistics (would query database in production)
        stats = {
            "total_transactions_analyzed": 150000,
            "fraud_detected": 1250,
            "fraud_rate": 0.83,
            "average_processing_time_ms": 45.2,
            "models_active": 4,
            "accuracy": 99.2,
            "precision": 98.7,
            "recall": 97.3,
            "uptime_hours": 720,
            "requests_per_minute": 125.5
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

@router.get("/models/performance")
async def get_model_performance():
    """Get model performance metrics"""
    try:
        # Mock performance data
        performance_data = [
            {
                "model_name": "Random Forest",
                "version": "1.0.0",
                "accuracy": 0.942,
                "precision": 0.938,
                "recall": 0.951,
                "f1_score": 0.944,
                "auc_roc": 0.987,
                "avg_prediction_time_ms": 12.5,
                "throughput_per_second": 850.0,
                "training_samples": 100000,
                "validation_samples": 20000,
                "feature_count": 45
            },
            {
                "model_name": "XGBoost",
                "version": "1.0.0",
                "accuracy": 0.948,
                "precision": 0.945,
                "recall": 0.952,
                "f1_score": 0.948,
                "auc_roc": 0.991,
                "avg_prediction_time_ms": 15.2,
                "throughput_per_second": 720.0,
                "training_samples": 100000,
                "validation_samples": 20000,
                "feature_count": 45
            },
            {
                "model_name": "Neural Network",
                "version": "1.0.0",
                "accuracy": 0.951,
                "precision": 0.948,
                "recall": 0.955,
                "f1_score": 0.951,
                "auc_roc": 0.993,
                "avg_prediction_time_ms": 28.7,
                "throughput_per_second": 420.0,
                "training_samples": 100000,
                "validation_samples": 20000,
                "feature_count": 45
            }
        ]
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model performance")

@router.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    try:
        model_info = fraud_service.get_model_info()
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")

@router.post("/models/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    """Trigger model retraining"""
    try:
        # Add retraining task to background
        background_tasks.add_task(fraud_service.update_models)
        
        return {
            "message": "Model retraining initiated",
            "status": "in_progress",
            "estimated_completion": "30-60 minutes"
        }
        
    except Exception as e:
        logger.error(f"Error initiating model retraining: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initiate model retraining")

@router.get("/alerts")
async def get_alerts(limit: int = 100, db: Session = Depends(get_db)):
    """Get recent fraud alerts"""
    try:
        # Mock alerts data
        alerts = [
            {
                "id": 1,
                "transaction_id": "txn_20240125_143022_001",
                "user_id": "user_12345",
                "alert_type": "high_risk_transaction",
                "severity": "high",
                "message": "Transaction amount significantly higher than user's typical spending pattern",
                "is_resolved": False,
                "created_at": "2024-01-25T14:30:22Z"
            },
            {
                "id": 2,
                "transaction_id": "txn_20240125_142015_002",
                "user_id": "user_67890",
                "alert_type": "unusual_location",
                "severity": "medium",
                "message": "Transaction from unusual geographic location",
                "is_resolved": True,
                "resolved_by": "analyst_001",
                "created_at": "2024-01-25T14:20:15Z"
            }
        ]
        
        return alerts[:limit]
        
    except Exception as e:
        logger.error(f"Error getting alerts: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")

@router.get("/users/{user_id}/profile")
async def get_user_profile(user_id: str, db: Session = Depends(get_db)):
    """Get user risk profile"""
    try:
        # Mock user profile
        profile = {
            "user_id": user_id,
            "account_age_days": 365,
            "total_transactions": 1250,
            "avg_transaction_amount": 125.50,
            "risk_score": 0.15,
            "preferred_merchants": ["Amazon", "Starbucks", "Shell"],
            "typical_locations": ["New York, NY", "Brooklyn, NY"],
            "active_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "failed_transactions_count": 3,
            "disputed_transactions_count": 1,
            "last_updated": "2024-01-25T14:30:00Z"
        }
        
        return profile
        
    except Exception as e:
        logger.error(f"Error getting user profile: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user profile")

@router.get("/merchants/{merchant_id}/profile")
async def get_merchant_profile(merchant_id: str, db: Session = Depends(get_db)):
    """Get merchant risk profile"""
    try:
        # Mock merchant profile
        profile = {
            "merchant_id": merchant_id,
            "merchant_name": "Sample Merchant Inc.",
            "category": "retail",
            "total_transactions": 50000,
            "total_volume": 2500000.00,
            "avg_transaction_amount": 50.00,
            "fraud_rate": 0.02,
            "chargeback_rate": 0.01,
            "risk_score": 0.25,
            "registration_date": "2020-01-15T00:00:00Z",
            "location": "New York, NY",
            "last_updated": "2024-01-25T14:30:00Z"
        }
        
        return profile
        
    except Exception as e:
        logger.error(f"Error getting merchant profile: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve merchant profile")

@router.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        # Return current metrics
        return {
            "requests_per_minute": 125.5,
            "average_response_time_ms": 45.2,
            "error_rate_percent": 0.1,
            "fraud_detection_rate": 0.83,
            "model_accuracy": 99.2,
            "active_connections": 15,
            "memory_usage_mb": 512,
            "cpu_usage_percent": 25.5
        }
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")
