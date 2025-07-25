
"""
Monitoring and metrics utilities
"""

import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from fastapi import FastAPI

logger = logging.getLogger(__name__)

class Metrics:
    """Prometheus metrics"""
    
    def __init__(self):
        # Request metrics
        self.transaction_requests = Counter(
            'fraud_detection_transaction_requests_total',
            'Total transaction analysis requests'
        )
        
        self.batch_requests = Counter(
            'fraud_detection_batch_requests_total',
            'Total batch analysis requests'
        )
        
        self.fraud_detected = Counter(
            'fraud_detection_fraud_detected_total',
            'Total fraud cases detected'
        )
        
        self.errors = Counter(
            'fraud_detection_errors_total',
            'Total errors encountered'
        )
        
        # Performance metrics
        self.processing_time = Histogram(
            'fraud_detection_processing_seconds',
            'Transaction processing time in seconds'
        )
        
        self.batch_processing_time = Histogram(
            'fraud_detection_batch_processing_seconds',
            'Batch processing time in seconds'
        )
        
        # System metrics
        self.active_connections = Gauge(
            'fraud_detection_active_connections',
            'Number of active connections'
        )
        
        self.model_load_time = Histogram(
            'fraud_detection_model_load_seconds',
            'Model loading time in seconds'
        )

# Global metrics instance
metrics = Metrics()

def setup_monitoring(app: FastAPI):
    """Setup monitoring for the application"""
    try:
        # Start Prometheus metrics server
        from ..config import settings
        if settings.ENABLE_METRICS:
            start_http_server(settings.METRICS_PORT)
            logger.info(f"Metrics server started on port {settings.METRICS_PORT}")
        
        # Add middleware for request tracking
        @app.middleware("http")
        async def track_requests(request, call_next):
            metrics.active_connections.inc()
            try:
                response = await call_next(request)
                return response
            finally:
                metrics.active_connections.dec()
        
        logger.info("Monitoring setup completed")
        
    except Exception as e:
        logger.error(f"Failed to setup monitoring: {str(e)}")
