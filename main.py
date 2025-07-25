
"""
Financial Fraud Detection API - Main FastAPI Application
High-performance real-time fraud detection service
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Optional
import asyncio
import uvicorn

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import redis
from sqlalchemy.orm import Session

from src.api.routes import router as api_router
from src.database.connection import get_db, init_database
from src.services.fraud_detector import FraudDetectionService
from src.services.rate_limiter import RateLimiter
from src.models.schemas import TransactionRequest, BatchTransactionRequest
from src.utils.monitoring import setup_monitoring, metrics
from src.utils.auth import verify_token
from src.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Fraud Detection API",
    description="High-performance real-time fraud detection service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security
security = HTTPBearer()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Initialize services
fraud_service = FraudDetectionService()
rate_limiter = RateLimiter()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        # Initialize database
        init_database()
        
        # Load ML models
        await fraud_service.load_models()
        
        # Setup monitoring
        setup_monitoring(app)
        
        logger.info("Fraud Detection API started successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        await fraud_service.cleanup()
        logger.info("Fraud Detection API shutdown complete")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Financial Fraud Detection API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "analyze": "/api/v1/analyze",
            "batch": "/api/v1/batch",
            "statistics": "/api/v1/statistics"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        db_status = "healthy"
        try:
            # This would check actual DB connection
            pass
        except Exception:
            db_status = "unhealthy"
        
        # Check Redis connection
        redis_status = "healthy"
        try:
            # This would check actual Redis connection
            pass
        except Exception:
            redis_status = "unhealthy"
        
        # Check model status
        model_status = "loaded" if fraud_service.models_loaded else "not_loaded"
        
        health_data = {
            "status": "healthy" if all([
                db_status == "healthy",
                redis_status == "healthy",
                model_status == "loaded"
            ]) else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "database": db_status,
                "redis": redis_status,
                "ml_models": model_status
            },
            "uptime": "N/A",  # Would calculate actual uptime
            "version": "1.0.0"
        }
        
        status_code = 200 if health_data["status"] == "healthy" else 503
        return JSONResponse(content=health_data, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            },
            status_code=503
        )

@app.post("/api/v1/analyze")
async def analyze_transaction(
    transaction: TransactionRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Analyze single transaction for fraud"""
    try:
        # Verify authentication
        user_id = verify_token(credentials.credentials)
        
        # Rate limiting
        client_ip = request.client.host
        if not await rate_limiter.check_rate_limit(client_ip, user_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Record metrics
        metrics.transaction_requests.inc()
        
        start_time = datetime.utcnow()
        
        # Perform fraud analysis
        result = await fraud_service.analyze_transaction(transaction.dict())
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Log transaction (background task)
        background_tasks.add_task(
            log_transaction,
            transaction.dict(),
            result,
            user_id,
            processing_time
        )
        
        # Update metrics
        metrics.processing_time.observe(processing_time)
        if result["is_fraud"]:
            metrics.fraud_detected.inc()
        
        response = {
            "transaction_id": result["transaction_id"],
            "is_fraud": result["is_fraud"],
            "fraud_probability": result["fraud_probability"],
            "risk_score": result["risk_score"],
            "risk_factors": result["risk_factors"],
            "model_version": result["model_version"],
            "processing_time_ms": round(processing_time * 1000, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "recommendations": result.get("recommendations", [])
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transaction analysis failed: {str(e)}")
        metrics.errors.inc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/v1/batch")
async def batch_analyze(
    batch_request: BatchTransactionRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Analyze multiple transactions in batch"""
    try:
        # Verify authentication
        user_id = verify_token(credentials.credentials)
        
        # Rate limiting (stricter for batch)
        client_ip = request.client.host
        if not await rate_limiter.check_batch_rate_limit(client_ip, user_id, len(batch_request.transactions)):
            raise HTTPException(status_code=429, detail="Batch rate limit exceeded")
        
        start_time = datetime.utcnow()
        
        # Process batch
        results = await fraud_service.batch_analyze(
            [t.dict() for t in batch_request.transactions]
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Background logging
        background_tasks.add_task(
            log_batch_analysis,
            batch_request.dict(),
            results,
            user_id,
            processing_time
        )
        
        # Update metrics
        metrics.batch_requests.inc()
        metrics.batch_processing_time.observe(processing_time)
        fraud_count = sum(1 for r in results if r["is_fraud"])
        metrics.fraud_detected.inc(fraud_count)
        
        response = {
            "batch_id": f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "total_transactions": len(results),
            "fraud_detected": fraud_count,
            "processing_time_ms": round(processing_time * 1000, 2),
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        metrics.errors.inc()
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

async def log_transaction(transaction_data, result, user_id, processing_time):
    """Background task to log transaction"""
    try:
        # This would log to database
        logger.info(f"Transaction logged: {result['transaction_id']}")
    except Exception as e:
        logger.error(f"Failed to log transaction: {str(e)}")

async def log_batch_analysis(batch_data, results, user_id, processing_time):
    """Background task to log batch analysis"""
    try:
        # This would log to database
        logger.info(f"Batch analysis logged: {len(results)} transactions")
    except Exception as e:
        logger.error(f"Failed to log batch analysis: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=4,
        log_level="info"
    )

