
"""
Unit tests for fraud detection service
"""

import unittest
import asyncio
from unittest.mock import patch, MagicMock
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.fraud_detector import FraudDetectionService

class TestFraudDetectionService(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.fraud_service = FraudDetectionService()
        
    def test_service_initialization(self):
        """Test service initialization"""
        self.assertIsNotNone(self.fraud_service)
        self.assertFalse(self.fraud_service.models_loaded)
        self.assertEqual(self.fraud_service.model_version, "1.0.0")
        self.assertIsInstance(self.fraud_service.models, dict)
    
    @patch('os.path.exists')
    def test_load_models(self, mock_exists):
        """Test model loading"""
        mock_exists.return_value = False  # Force model creation
        
        # Run async method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.fraud_service.load_models())
            self.assertTrue(self.fraud_service.models_loaded)
            self.assertIn('random_forest', self.fraud_service.models)
            self.assertIn('xgboost', self.fraud_service.models)
            self.assertIn('neural_network', self.fraud_service.models)
            self.assertIn('isolation_forest', self.fraud_service.models)
        finally:
            loop.close()
    
    def test_generate_dummy_training_data(self):
        """Test dummy data generation"""
        X, y = self.fraud_service._generate_dummy_training_data(100)
        
        self.assertEqual(X.shape, (100, 20))
        self.assertEqual(y.shape, (100,))
        self.assertTrue(np.all(np.isin(y, [0, 1])))
    
    def test_create_label_encoders(self):
        """Test label encoder creation"""
        encoders = self.fraud_service._create_label_encoders()
        
        self.assertIn('merchant_category', encoders)
        self.assertIn('transaction_type', encoders)
        
        # Test encoding
        encoded = encoders['merchant_category'].transform(['grocery'])
        self.assertIsInstance(encoded[0], (int, np.integer))
    
    def test_preprocess_features(self):
        """Test feature preprocessing"""
        # Load models first
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.fraud_service.load_models())
            
            # Test preprocessing
            features = {'amount': 100.0, 'hour_of_day': 14}
            processed = self.fraud_service._preprocess_features(features)
            
            self.assertEqual(len(processed), 20)
            self.assertIsInstance(processed, np.ndarray)
        finally:
            loop.close()
    
    def test_calculate_ensemble_probability(self):
        """Test ensemble probability calculation"""
        predictions = {
            'random_forest': 0.8,
            'xgboost': 0.7,
            'neural_network': 0.9,
            'isolation_forest': 0.6
        }
        
        prob = self.fraud_service._calculate_ensemble_probability(predictions)
        
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
    
    def test_get_risk_level(self):
        """Test risk level determination"""
        self.assertEqual(self.fraud_service._get_risk_level(0.1), 'low')
        self.assertEqual(self.fraud_service._get_risk_level(0.5), 'medium')
        self.assertEqual(self.fraud_service._get_risk_level(0.7), 'high')
        self.assertEqual(self.fraud_service._get_risk_level(0.95), 'critical')
    
    def test_identify_risk_factors(self):
        """Test risk factor identification"""
        transaction_data = {'amount': 10000}
        features = {'hour_of_day': 2, 'is_weekend': True}
        predictions = {'isolation_forest': 0.8}
        
        risk_factors = self.fraud_service._identify_risk_factors(
            transaction_data, features, predictions
        )
        
        self.assertIsInstance(risk_factors, list)
        self.assertIn('High transaction amount', risk_factors)
        self.assertIn('Unusual transaction time', risk_factors)
    
    def test_generate_recommendations(self):
        """Test recommendation generation"""
        recommendations = self.fraud_service._generate_recommendations(
            0.9, ['High transaction amount']
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertTrue(len(recommendations) > 0)
        self.assertIn('Block transaction immediately', recommendations)
    
    def test_analyze_transaction(self):
        """Test transaction analysis"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Load models first
            loop.run_until_complete(self.fraud_service.load_models())
            
            # Test transaction
            transaction_data = {
                'amount': 500.0,
                'transaction_type': 'purchase',
                'merchant_category': 'grocery',
                'user_id': 'test_user',
                'account_id': 'test_account'
            }
            
            result = loop.run_until_complete(
                self.fraud_service.analyze_transaction(transaction_data)
            )
            
            # Verify result structure
            self.assertIn('transaction_id', result)
            self.assertIn('is_fraud', result)
            self.assertIn('fraud_probability', result)
            self.assertIn('risk_score', result)
            self.assertIn('risk_factors', result)
            self.assertIn('recommendations', result)
            
            # Verify data types
            self.assertIsInstance(result['is_fraud'], bool)
            self.assertIsInstance(result['fraud_probability'], float)
            self.assertIsInstance(result['risk_score'], float)
            self.assertIsInstance(result['risk_factors'], list)
            self.assertIsInstance(result['recommendations'], list)
            
        finally:
            loop.close()
    
    def test_batch_analyze(self):
        """Test batch analysis"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Load models first
            loop.run_until_complete(self.fraud_service.load_models())
            
            # Test batch
            transactions = [
                {
                    'amount': 100.0,
                    'transaction_type': 'purchase',
                    'merchant_category': 'grocery',
                    'user_id': 'user1',
                    'account_id': 'account1'
                },
                {
                    'amount': 5000.0,
                    'transaction_type': 'withdrawal',
                    'merchant_category': 'atm',
                    'user_id': 'user2',
                    'account_id': 'account2'
                }
            ]
            
            results = loop.run_until_complete(
                self.fraud_service.batch_analyze(transactions)
            )
            
            self.assertEqual(len(results), 2)
            for result in results:
                self.assertIn('transaction_id', result)
                self.assertIn('is_fraud', result)
                self.assertIn('fraud_probability', result)
                
        finally:
            loop.close()
    
    def test_get_model_info(self):
        """Test model info retrieval"""
        info = self.fraud_service.get_model_info()
        
        self.assertIn('models_loaded', info)
        self.assertIn('model_version', info)
        self.assertIn('available_models', info)
        self.assertIn('feature_count', info)
        self.assertIn('last_updated', info)

if __name__ == '__main__':
    unittest.main()
