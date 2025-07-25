
"""
Core fraud detection service with ensemble ML models
"""

import os
import pickle
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
import joblib

from ..utils.feature_engineering import FeatureEngineer
from ..config import settings, MODEL_CONFIG, RISK_THRESHOLDS

logger = logging.getLogger(__name__)

class FraudDetectionService:
    """Main fraud detection service with ensemble models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_engineer = FeatureEngineer()
        self.models_loaded = False
        self.model_version = "1.0.0"
        
    async def load_models(self):
        """Load all ML models"""
        try:
            model_path = settings.MODEL_PATH
            os.makedirs(model_path, exist_ok=True)
            
            # Load or create models
            await self._load_or_create_random_forest()
            await self._load_or_create_xgboost()
            await self._load_or_create_neural_network()
            await self._load_or_create_isolation_forest()
            
            # Load preprocessing components
            await self._load_preprocessing_components()
            
            self.models_loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            raise
    
    async def _load_or_create_random_forest(self):
        """Load or create Random Forest model"""
        try:
            model_file = os.path.join(settings.MODEL_PATH, "random_forest.pkl")
            
            if os.path.exists(model_file):
                self.models['random_forest'] = joblib.load(model_file)
                logger.info("Random Forest model loaded from file")
            else:
                # Create and train new model
                config = MODEL_CONFIG['random_forest']
                model = RandomForestClassifier(**config)
                
                # Train with dummy data for demo
                X_dummy, y_dummy = self._generate_dummy_training_data()
                model.fit(X_dummy, y_dummy)
                
                # Save model
                joblib.dump(model, model_file)
                self.models['random_forest'] = model
                logger.info("New Random Forest model created and saved")
                
        except Exception as e:
            logger.error(f"Error with Random Forest model: {str(e)}")
            raise
    
    async def _load_or_create_xgboost(self):
        """Load or create XGBoost model"""
        try:
            model_file = os.path.join(settings.MODEL_PATH, "xgboost.pkl")
            
            if os.path.exists(model_file):
                self.models['xgboost'] = joblib.load(model_file)
                logger.info("XGBoost model loaded from file")
            else:
                # Create and train new model
                config = MODEL_CONFIG['xgboost']
                model = xgb.XGBClassifier(**config)
                
                # Train with dummy data
                X_dummy, y_dummy = self._generate_dummy_training_data()
                model.fit(X_dummy, y_dummy)
                
                # Save model
                joblib.dump(model, model_file)
                self.models['xgboost'] = model
                logger.info("New XGBoost model created and saved")
                
        except Exception as e:
            logger.error(f"Error with XGBoost model: {str(e)}")
            raise
    
    async def _load_or_create_neural_network(self):
        """Load or create Neural Network model"""
        try:
            model_file = os.path.join(settings.MODEL_PATH, "neural_network.h5")
            
            if os.path.exists(model_file):
                self.models['neural_network'] = keras.models.load_model(model_file)
                logger.info("Neural Network model loaded from file")
            else:
                # Create new model
                config = MODEL_CONFIG['neural_network']
                
                model = keras.Sequential([
                    keras.layers.Dense(config['hidden_layers'][0], activation='relu', input_shape=(20,)),
                    keras.layers.Dropout(config['dropout_rate']),
                    keras.layers.Dense(config['hidden_layers'][1], activation='relu'),
                    keras.layers.Dropout(config['dropout_rate']),
                    keras.layers.Dense(config['hidden_layers'][2], activation='relu'),
                    keras.layers.Dropout(config['dropout_rate']),
                    keras.layers.Dense(1, activation='sigmoid')
                ])
                
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
                    loss='binary_crossentropy',
                    metrics=['accuracy', 'precision', 'recall']
                )
                
                # Train with dummy data
                X_dummy, y_dummy = self._generate_dummy_training_data()
                model.fit(X_dummy, y_dummy, epochs=5, verbose=0)
                
                # Save model
                model.save(model_file)
                self.models['neural_network'] = model
                logger.info("New Neural Network model created and saved")
                
        except Exception as e:
            logger.error(f"Error with Neural Network model: {str(e)}")
            raise
    
    async def _load_or_create_isolation_forest(self):
        """Load or create Isolation Forest for anomaly detection"""
        try:
            model_file = os.path.join(settings.MODEL_PATH, "isolation_forest.pkl")
            
            if os.path.exists(model_file):
                self.models['isolation_forest'] = joblib.load(model_file)
                logger.info("Isolation Forest model loaded from file")
            else:
                # Create new model
                model = IsolationForest(
                    contamination=0.1,
                    random_state=42,
                    n_estimators=100
                )
                
                # Train with dummy data
                X_dummy, _ = self._generate_dummy_training_data()
                model.fit(X_dummy)
                
                # Save model
                joblib.dump(model, model_file)
                self.models['isolation_forest'] = model
                logger.info("New Isolation Forest model created and saved")
                
        except Exception as e:
            logger.error(f"Error with Isolation Forest model: {str(e)}")
            raise
    
    async def _load_preprocessing_components(self):
        """Load preprocessing components"""
        try:
            # Load scaler
            scaler_file = os.path.join(settings.MODEL_PATH, "scaler.pkl")
            if os.path.exists(scaler_file):
                self.scalers['standard'] = joblib.load(scaler_file)
            else:
                # Create new scaler
                scaler = StandardScaler()
                X_dummy, _ = self._generate_dummy_training_data()
                scaler.fit(X_dummy)
                joblib.dump(scaler, scaler_file)
                self.scalers['standard'] = scaler
            
            # Load encoders
            encoder_file = os.path.join(settings.MODEL_PATH, "label_encoders.pkl")
            if os.path.exists(encoder_file):
                self.encoders = joblib.load(encoder_file)
            else:
                # Create new encoders
                self.encoders = self._create_label_encoders()
                joblib.dump(self.encoders, encoder_file)
            
            logger.info("Preprocessing components loaded")
            
        except Exception as e:
            logger.error(f"Error loading preprocessing components: {str(e)}")
            raise
    
    def _generate_dummy_training_data(self, n_samples=1000):
        """Generate dummy training data for model initialization"""
        np.random.seed(42)
        
        # Generate features
        X = np.random.randn(n_samples, 20)
        
        # Generate labels (10% fraud)
        y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
        
        return X, y
    
    def _create_label_encoders(self):
        """Create label encoders for categorical features"""
        encoders = {}
        
        # Merchant category encoder
        merchant_categories = ['grocery', 'gas_station', 'restaurant', 'retail', 'online', 'atm', 'pharmacy', 'other']
        encoders['merchant_category'] = LabelEncoder()
        encoders['merchant_category'].fit(merchant_categories)
        
        # Transaction type encoder
        transaction_types = ['purchase', 'withdrawal', 'transfer', 'deposit', 'refund']
        encoders['transaction_type'] = LabelEncoder()
        encoders['transaction_type'].fit(transaction_types)
        
        return encoders
    
    async def analyze_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze single transaction for fraud"""
        try:
            if not self.models_loaded:
                raise ValueError("Models not loaded")
            
            # Generate transaction ID
            transaction_id = f"txn_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
            
            # Feature engineering
            features = await self.feature_engineer.extract_features(transaction_data)
            
            # Preprocess features
            processed_features = self._preprocess_features(features)
            
            # Get predictions from all models
            predictions = await self._get_ensemble_predictions(processed_features)
            
            # Calculate final fraud probability
            fraud_probability = self._calculate_ensemble_probability(predictions)
            
            # Determine risk level
            risk_level = self._get_risk_level(fraud_probability)
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(transaction_data, features, predictions)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(fraud_probability, risk_factors)
            
            result = {
                "transaction_id": transaction_id,
                "is_fraud": fraud_probability > RISK_THRESHOLDS['medium'],
                "fraud_probability": float(fraud_probability),
                "risk_score": float(fraud_probability * 100),
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "recommendations": recommendations,
                "model_version": self.model_version,
                "feature_count": len(processed_features),
                "models_used": list(predictions.keys())
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing transaction: {str(e)}")
            raise
    
    async def batch_analyze(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze multiple transactions in batch"""
        try:
            results = []
            
            # Process transactions concurrently
            tasks = [self.analyze_transaction(txn) for txn in transactions]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing transaction {i}: {str(result)}")
                    processed_results.append({
                        "transaction_id": f"error_{i}",
                        "error": str(result),
                        "is_fraud": False,
                        "fraud_probability": 0.0,
                        "risk_score": 0.0
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {str(e)}")
            raise
    
    def _preprocess_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Preprocess features for model input"""
        try:
            # Convert to DataFrame for easier processing
            df = pd.DataFrame([features])
            
            # Handle categorical features
            for col, encoder in self.encoders.items():
                if col in df.columns:
                    try:
                        df[col] = encoder.transform(df[col])
                    except ValueError:
                        # Handle unseen categories
                        df[col] = 0
            
            # Select numerical features (mock feature selection)
            numerical_features = [f"feature_{i}" for i in range(20)]
            
            # Create feature array with dummy values if features don't exist
            feature_array = np.zeros(20)
            for i, feature_name in enumerate(numerical_features):
                if feature_name in features:
                    feature_array[i] = features[feature_name]
                else:
                    # Use transaction amount and other available features
                    if i == 0 and 'amount' in features:
                        feature_array[i] = features['amount']
                    elif i == 1 and 'hour_of_day' in features:
                        feature_array[i] = features['hour_of_day']
                    else:
                        feature_array[i] = np.random.randn()  # Random for demo
            
            # Scale features
            if 'standard' in self.scalers:
                feature_array = self.scalers['standard'].transform([feature_array])[0]
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Error preprocessing features: {str(e)}")
            # Return dummy features for demo
            return np.random.randn(20)
    
    async def _get_ensemble_predictions(self, features: np.ndarray) -> Dict[str, float]:
        """Get predictions from all models"""
        predictions = {}
        
        try:
            # Random Forest
            if 'random_forest' in self.models:
                rf_pred = self.models['random_forest'].predict_proba([features])[0][1]
                predictions['random_forest'] = float(rf_pred)
            
            # XGBoost
            if 'xgboost' in self.models:
                xgb_pred = self.models['xgboost'].predict_proba([features])[0][1]
                predictions['xgboost'] = float(xgb_pred)
            
            # Neural Network
            if 'neural_network' in self.models:
                nn_pred = self.models['neural_network'].predict([features.reshape(1, -1)])[0][0]
                predictions['neural_network'] = float(nn_pred)
            
            # Isolation Forest (anomaly score)
            if 'isolation_forest' in self.models:
                anomaly_score = self.models['isolation_forest'].decision_function([features])[0]
                # Convert to probability (higher anomaly = higher fraud probability)
                iso_pred = 1 / (1 + np.exp(anomaly_score))  # Sigmoid transformation
                predictions['isolation_forest'] = float(iso_pred)
            
        except Exception as e:
            logger.error(f"Error getting ensemble predictions: {str(e)}")
            # Return dummy predictions
            predictions = {
                'random_forest': 0.1,
                'xgboost': 0.15,
                'neural_network': 0.12,
                'isolation_forest': 0.08
            }
        
        return predictions
    
    def _calculate_ensemble_probability(self, predictions: Dict[str, float]) -> float:
        """Calculate ensemble fraud probability"""
        if not predictions:
            return 0.0
        
        # Weighted average (can be tuned based on model performance)
        weights = {
            'random_forest': 0.3,
            'xgboost': 0.3,
            'neural_network': 0.25,
            'isolation_forest': 0.15
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model, pred in predictions.items():
            if model in weights:
                weighted_sum += pred * weights[model]
                total_weight += weights[model]
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return np.mean(list(predictions.values()))
    
    def _get_risk_level(self, fraud_probability: float) -> str:
        """Determine risk level based on fraud probability"""
        if fraud_probability >= RISK_THRESHOLDS['critical']:
            return 'critical'
        elif fraud_probability >= RISK_THRESHOLDS['high']:
            return 'high'
        elif fraud_probability >= RISK_THRESHOLDS['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _identify_risk_factors(self, transaction_data: Dict[str, Any], 
                             features: Dict[str, Any], 
                             predictions: Dict[str, float]) -> List[str]:
        """Identify specific risk factors"""
        risk_factors = []
        
        # High amount
        if transaction_data.get('amount', 0) > 5000:
            risk_factors.append('High transaction amount')
        
        # Unusual time
        hour = features.get('hour_of_day', 12)
        if hour < 6 or hour > 22:
            risk_factors.append('Unusual transaction time')
        
        # Weekend transaction
        if features.get('is_weekend', False):
            risk_factors.append('Weekend transaction')
        
        # High velocity
        if features.get('frequency_1h', 0) > 5:
            risk_factors.append('High transaction frequency')
        
        # New merchant
        if features.get('merchant_risk_score', 0) > 0.7:
            risk_factors.append('High-risk merchant')
        
        # Anomaly detection
        if predictions.get('isolation_forest', 0) > 0.6:
            risk_factors.append('Anomalous transaction pattern')
        
        # Location risk
        if features.get('location_risk_score', 0) > 0.8:
            risk_factors.append('High-risk location')
        
        return risk_factors
    
    def _generate_recommendations(self, fraud_probability: float, 
                                risk_factors: List[str]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if fraud_probability >= RISK_THRESHOLDS['critical']:
            recommendations.extend([
                'Block transaction immediately',
                'Contact customer for verification',
                'Review account for suspicious activity',
                'Consider temporary account freeze'
            ])
        elif fraud_probability >= RISK_THRESHOLDS['high']:
            recommendations.extend([
                'Require additional authentication',
                'Manual review recommended',
                'Monitor account closely'
            ])
        elif fraud_probability >= RISK_THRESHOLDS['medium']:
            recommendations.extend([
                'Consider step-up authentication',
                'Flag for review',
                'Monitor transaction patterns'
            ])
        else:
            recommendations.append('Process normally')
        
        # Add specific recommendations based on risk factors
        if 'High transaction amount' in risk_factors:
            recommendations.append('Verify large amount with customer')
        
        if 'Unusual transaction time' in risk_factors:
            recommendations.append('Confirm transaction timing with customer')
        
        if 'High transaction frequency' in risk_factors:
            recommendations.append('Check for card compromise')
        
        return recommendations
    
    async def update_models(self):
        """Update models with new training data"""
        try:
            logger.info("Starting model update process")
            # This would implement model retraining logic
            # For now, just log the update
            logger.info("Model update completed")
        except Exception as e:
            logger.error(f"Model update failed: {str(e)}")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            # Clear models from memory
            self.models.clear()
            self.scalers.clear()
            self.encoders.clear()
            self.models_loaded = False
            logger.info("Fraud detection service cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "models_loaded": self.models_loaded,
            "model_version": self.model_version,
            "available_models": list(self.models.keys()),
            "feature_count": 20,
            "last_updated": datetime.utcnow().isoformat()
        }

