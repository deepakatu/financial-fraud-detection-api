
"""
Pydantic schemas for request/response validation
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

class TransactionType(str, Enum):
    """Transaction types"""
    PURCHASE = "purchase"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    DEPOSIT = "deposit"
    REFUND = "refund"

class MerchantCategory(str, Enum):
    """Merchant categories"""
    GROCERY = "grocery"
    GAS_STATION = "gas_station"
    RESTAURANT = "restaurant"
    RETAIL = "retail"
    ONLINE = "online"
    ATM = "atm"
    PHARMACY = "pharmacy"
    OTHER = "other"

class RiskLevel(str, Enum):
    """Risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TransactionRequest(BaseModel):
    """Transaction analysis request schema"""
    
    # Transaction details
    amount: float = Field(..., gt=0, description="Transaction amount")
    transaction_type: TransactionType = Field(..., description="Type of transaction")
    merchant_category: MerchantCategory = Field(..., description="Merchant category")
    merchant_id: Optional[str] = Field(None, description="Merchant identifier")
    
    # User details
    user_id: str = Field(..., description="User identifier")
    account_id: str = Field(..., description="Account identifier")
    
    # Location and time
    location: Optional[str] = Field(None, description="Transaction location")
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Latitude")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Transaction timestamp")
    
    # Payment details
    payment_method: Optional[str] = Field(None, description="Payment method")
    card_type: Optional[str] = Field(None, description="Card type")
    
    # Additional context
    device_id: Optional[str] = Field(None, description="Device identifier")
    ip_address: Optional[str] = Field(None, description="IP address")
    user_agent: Optional[str] = Field(None, description="User agent")
    
    # Custom fields
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        if v > 1000000:  # 1M limit
            raise ValueError('Amount exceeds maximum limit')
        return v
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        # Don't allow future timestamps
        if v > datetime.utcnow():
            raise ValueError('Timestamp cannot be in the future')
        return v

class BatchTransactionRequest(BaseModel):
    """Batch transaction analysis request"""
    
    transactions: List[TransactionRequest] = Field(..., min_items=1, max_items=1000)
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    priority: Optional[str] = Field("normal", description="Processing priority")
    
    @validator('transactions')
    def validate_transactions(cls, v):
        if len(v) > 1000:
            raise ValueError('Batch size cannot exceed 1000 transactions')
        return v

class FraudAnalysisResponse(BaseModel):
    """Fraud analysis response schema"""
    
    transaction_id: str = Field(..., description="Unique transaction identifier")
    is_fraud: bool = Field(..., description="Fraud prediction")
    fraud_probability: float = Field(..., ge=0, le=1, description="Fraud probability score")
    risk_score: float = Field(..., ge=0, le=100, description="Risk score (0-100)")
    risk_level: RiskLevel = Field(..., description="Risk level category")
    
    # Model information
    model_version: str = Field(..., description="Model version used")
    model_confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    
    # Risk factors
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list, description="Recommended actions")
    
    # Processing details
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")

class BatchAnalysisResponse(BaseModel):
    """Batch analysis response schema"""
    
    batch_id: str = Field(..., description="Batch identifier")
    total_transactions: int = Field(..., description="Total transactions processed")
    fraud_detected: int = Field(..., description="Number of fraudulent transactions")
    processing_time_ms: float = Field(..., description="Total processing time")
    
    results: List[FraudAnalysisResponse] = Field(..., description="Individual transaction results")
    
    # Summary statistics
    summary: Optional[Dict[str, Any]] = Field(None, description="Batch summary statistics")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Batch completion timestamp")

class UserProfile(BaseModel):
    """User profile for risk assessment"""
    
    user_id: str = Field(..., description="User identifier")
    account_age_days: int = Field(..., ge=0, description="Account age in days")
    total_transactions: int = Field(..., ge=0, description="Total transaction count")
    avg_transaction_amount: float = Field(..., ge=0, description="Average transaction amount")
    
    # Behavioral patterns
    preferred_merchants: List[str] = Field(default_factory=list, description="Frequently used merchants")
    typical_locations: List[str] = Field(default_factory=list, description="Typical transaction locations")
    active_hours: List[int] = Field(default_factory=list, description="Typical active hours")
    
    # Risk indicators
    failed_transactions_count: int = Field(default=0, ge=0, description="Failed transaction count")
    disputed_transactions_count: int = Field(default=0, ge=0, description="Disputed transaction count")
    risk_score: float = Field(default=0.0, ge=0, le=100, description="User risk score")
    
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Profile last updated")

class MerchantProfile(BaseModel):
    """Merchant profile for risk assessment"""
    
    merchant_id: str = Field(..., description="Merchant identifier")
    merchant_name: str = Field(..., description="Merchant name")
    category: MerchantCategory = Field(..., description="Merchant category")
    
    # Business details
    registration_date: Optional[datetime] = Field(None, description="Merchant registration date")
    location: Optional[str] = Field(None, description="Merchant location")
    
    # Transaction statistics
    total_transactions: int = Field(default=0, ge=0, description="Total transactions")
    total_volume: float = Field(default=0.0, ge=0, description="Total transaction volume")
    avg_transaction_amount: float = Field(default=0.0, ge=0, description="Average transaction amount")
    
    # Risk indicators
    fraud_rate: float = Field(default=0.0, ge=0, le=1, description="Historical fraud rate")
    chargeback_rate: float = Field(default=0.0, ge=0, le=1, description="Chargeback rate")
    risk_score: float = Field(default=0.0, ge=0, le=100, description="Merchant risk score")
    
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Profile last updated")

class AlertRequest(BaseModel):
    """Alert configuration request"""
    
    user_id: str = Field(..., description="User to alert")
    alert_type: str = Field(..., description="Type of alert")
    threshold: float = Field(..., ge=0, le=1, description="Alert threshold")
    channels: List[str] = Field(..., description="Notification channels")
    
    # Alert conditions
    conditions: Dict[str, Any] = Field(default_factory=dict, description="Alert conditions")
    active: bool = Field(default=True, description="Alert active status")

class ModelPerformanceMetrics(BaseModel):
    """Model performance metrics"""
    
    model_name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    
    # Performance metrics
    accuracy: float = Field(..., ge=0, le=1, description="Model accuracy")
    precision: float = Field(..., ge=0, le=1, description="Precision score")
    recall: float = Field(..., ge=0, le=1, description="Recall score")
    f1_score: float = Field(..., ge=0, le=1, description="F1 score")
    auc_roc: float = Field(..., ge=0, le=1, description="AUC-ROC score")
    
    # Operational metrics
    avg_prediction_time_ms: float = Field(..., description="Average prediction time")
    throughput_per_second: float = Field(..., description="Predictions per second")
    
    # Data metrics
    training_samples: int = Field(..., description="Training sample count")
    validation_samples: int = Field(..., description="Validation sample count")
    feature_count: int = Field(..., description="Number of features")
    
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Metrics last updated")

class HealthCheckResponse(BaseModel):
    """Health check response schema"""
    
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    version: str = Field(..., description="API version")
    
    components: Dict[str, str] = Field(..., description="Component health status")
    uptime: Optional[str] = Field(None, description="Service uptime")
    
    # Performance indicators
    response_time_ms: Optional[float] = Field(None, description="Average response time")
    requests_per_minute: Optional[float] = Field(None, description="Requests per minute")
    error_rate: Optional[float] = Field(None, description="Error rate percentage")

