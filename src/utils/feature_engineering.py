
"""
Feature engineering utilities for fraud detection
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import hashlib

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering for fraud detection"""
    
    def __init__(self):
        self.user_profiles = {}  # In-memory user profiles (would use database in production)
        self.merchant_profiles = {}  # In-memory merchant profiles
        self.location_cache = {}  # Location risk cache
        
    async def extract_features(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive features from transaction data"""
        try:
            features = {}
            
            # Basic transaction features
            features.update(self._extract_basic_features(transaction_data))
            
            # Temporal features
            features.update(self._extract_temporal_features(transaction_data))
            
            # Amount-based features
            features.update(self._extract_amount_features(transaction_data))
            
            # User behavioral features
            features.update(await self._extract_user_features(transaction_data))
            
            # Merchant features
            features.update(await self._extract_merchant_features(transaction_data))
            
            # Location features
            features.update(await self._extract_location_features(transaction_data))
            
            # Device and session features
            features.update(self._extract_device_features(transaction_data))
            
            # Velocity features
            features.update(await self._extract_velocity_features(transaction_data))
            
            logger.debug(f"Extracted {len(features)} features")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return self._get_default_features()
    
    def _extract_basic_features(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic transaction features"""
        features = {}
        
        # Transaction amount
        features['amount'] = float(transaction_data.get('amount', 0))
        features['amount_log'] = np.log1p(features['amount'])
        
        # Transaction type
        transaction_type = transaction_data.get('transaction_type', 'purchase')
        features['transaction_type'] = transaction_type
        features['is_withdrawal'] = 1 if transaction_type == 'withdrawal' else 0
        features['is_transfer'] = 1 if transaction_type == 'transfer' else 0
        
        # Merchant category
        merchant_category = transaction_data.get('merchant_category', 'other')
        features['merchant_category'] = merchant_category
        features['is_online'] = 1 if merchant_category == 'online' else 0
        features['is_atm'] = 1 if merchant_category == 'atm' else 0
        
        return features
    
    def _extract_temporal_features(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract time-based features"""
        features = {}
        
        # Get timestamp
        timestamp = transaction_data.get('timestamp', datetime.utcnow())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        # Time features
        features['hour_of_day'] = timestamp.hour
        features['day_of_week'] = timestamp.weekday()
        features['day_of_month'] = timestamp.day
        features['month'] = timestamp.month
        
        # Binary time features
        features['is_weekend'] = 1 if timestamp.weekday() >= 5 else 0
        features['is_night'] = 1 if timestamp.hour < 6 or timestamp.hour > 22 else 0
        features['is_business_hours'] = 1 if 9 <= timestamp.hour <= 17 else 0
        
        # Cyclical encoding for time features
        features['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)
        features['day_sin'] = np.sin(2 * np.pi * timestamp.weekday() / 7)
        features['day_cos'] = np.cos(2 * np.pi * timestamp.weekday() / 7)
        
        return features
    
    def _extract_amount_features(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract amount-based features"""
        features = {}
        
        amount = float(transaction_data.get('amount', 0))
        
        # Amount categories
        features['is_small_amount'] = 1 if amount < 10 else 0
        features['is_medium_amount'] = 1 if 10 <= amount <= 1000 else 0
        features['is_large_amount'] = 1 if amount > 1000 else 0
        features['is_round_amount'] = 1 if amount % 10 == 0 else 0
        
        # Amount patterns
        features['amount_digits'] = len(str(int(amount)))
        features['has_cents'] = 1 if amount % 1 != 0 else 0
        
        return features
    
    async def _extract_user_features(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user behavioral features"""
        features = {}
        
        user_id = transaction_data.get('user_id', 'unknown')
        
        # Get or create user profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = self._create_user_profile(user_id)
        
        profile = self.user_profiles[user_id]
        
        # User age features
        features['user_age_days'] = profile.get('age_days', 0)
        features['is_new_user'] = 1 if profile.get('age_days', 0) < 30 else 0
        
        # Transaction history features
        features['user_total_transactions'] = profile.get('total_transactions', 0)
        features['user_avg_amount'] = profile.get('avg_amount', 0)
        features['user_std_amount'] = profile.get('std_amount', 0)
        
        # Calculate z-score for current amount
        if profile.get('std_amount', 0) > 0:
            features['amount_zscore'] = abs(
                (float(transaction_data.get('amount', 0)) - profile.get('avg_amount', 0)) / 
                profile.get('std_amount', 1)
            )
        else:
            features['amount_zscore'] = 0
        
        # User risk indicators
        features['user_failed_transactions'] = profile.get('failed_transactions', 0)
        features['user_disputed_transactions'] = profile.get('disputed_transactions', 0)
        features['user_risk_score'] = profile.get('risk_score', 0)
        
        return features
    
    async def _extract_merchant_features(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract merchant-based features"""
        features = {}
        
        merchant_id = transaction_data.get('merchant_id', 'unknown')
        
        # Get or create merchant profile
        if merchant_id not in self.merchant_profiles:
            self.merchant_profiles[merchant_id] = self._create_merchant_profile(merchant_id)
        
        profile = self.merchant_profiles[merchant_id]
        
        # Merchant features
        features['merchant_total_transactions'] = profile.get('total_transactions', 0)
        features['merchant_avg_amount'] = profile.get('avg_amount', 0)
        features['merchant_fraud_rate'] = profile.get('fraud_rate', 0)
        features['merchant_risk_score'] = profile.get('risk_score', 0)
        features['is_new_merchant'] = 1 if profile.get('total_transactions', 0) < 100 else 0
        
        return features
    
    async def _extract_location_features(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract location-based features"""
        features = {}
        
        location = transaction_data.get('location', '')
        latitude = transaction_data.get('latitude')
        longitude = transaction_data.get('longitude')
        
        # Location risk score (mock implementation)
        location_hash = hashlib.md5(location.encode()).hexdigest() if location else 'unknown'
        features['location_risk_score'] = hash(location_hash) % 100 / 100.0
        
        # Geographic features
        if latitude is not None and longitude is not None:
            features['latitude'] = float(latitude)
            features['longitude'] = float(longitude)
            
            # Distance from user's typical locations (mock)
            features['distance_from_home'] = np.random.uniform(0, 1000)  # Mock distance
            features['is_foreign_location'] = 1 if features['distance_from_home'] > 500 else 0
        else:
            features['latitude'] = 0.0
            features['longitude'] = 0.0
            features['distance_from_home'] = 0.0
            features['is_foreign_location'] = 0
        
        return features
    
    def _extract_device_features(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract device and session features"""
        features = {}
        
        device_id = transaction_data.get('device_id', 'unknown')
        ip_address = transaction_data.get('ip_address', '0.0.0.0')
        user_agent = transaction_data.get('user_agent', '')
        
        # Device features
        features['is_new_device'] = 1 if device_id == 'unknown' else 0
        features['device_risk_score'] = hash(device_id) % 100 / 100.0
        
        # IP features
        features['ip_risk_score'] = hash(ip_address) % 100 / 100.0
        features['is_tor_ip'] = 0  # Would check against Tor exit node list
        features['is_vpn_ip'] = 0  # Would check against VPN IP ranges
        
        # User agent features
        features['is_mobile'] = 1 if 'mobile' in user_agent.lower() else 0
        features['is_bot'] = 1 if 'bot' in user_agent.lower() else 0
        
        return features
    
    async def _extract_velocity_features(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract velocity-based features"""
        features = {}
        
        user_id = transaction_data.get('user_id', 'unknown')
        
        # Mock velocity calculations (would use real transaction history)
        features['frequency_1h'] = np.random.poisson(2)  # Transactions in last hour
        features['frequency_24h'] = np.random.poisson(10)  # Transactions in last 24h
        features['frequency_7d'] = np.random.poisson(50)  # Transactions in last 7 days
        
        # Amount velocity
        features['amount_1h'] = np.random.uniform(0, 5000)  # Total amount in last hour
        features['amount_24h'] = np.random.uniform(0, 20000)  # Total amount in last 24h
        
        # Merchant velocity
        features['unique_merchants_1h'] = min(features['frequency_1h'], 3)
        features['unique_merchants_24h'] = min(features['frequency_24h'], 10)
        
        return features
    
    def _create_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Create mock user profile"""
        return {
            'age_days': np.random.randint(1, 3650),  # 1 day to 10 years
            'total_transactions': np.random.randint(0, 10000),
            'avg_amount': np.random.uniform(10, 1000),
            'std_amount': np.random.uniform(5, 500),
            'failed_transactions': np.random.randint(0, 50),
            'disputed_transactions': np.random.randint(0, 10),
            'risk_score': np.random.uniform(0, 1)
        }
    
    def _create_merchant_profile(self, merchant_id: str) -> Dict[str, Any]:
        """Create mock merchant profile"""
        return {
            'total_transactions': np.random.randint(100, 100000),
            'avg_amount': np.random.uniform(20, 2000),
            'fraud_rate': np.random.uniform(0, 0.1),  # 0-10% fraud rate
            'risk_score': np.random.uniform(0, 1)
        }
    
    def _get_default_features(self) -> Dict[str, Any]:
        """Return default features in case of error"""
        return {f'feature_{i}': 0.0 for i in range(20)}
