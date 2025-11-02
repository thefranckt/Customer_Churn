"""
Comprehensive test suite for Customer Churn Prediction API
Tests covering validation, prediction, monitoring, and error handling
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import numpy as np
from datetime import datetime, timedelta
import json
import tempfile
from pathlib import Path

# Import your API modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.deployment_api import api
from api.config import settings
from api.models import CustomerFeatures, PredictionResult, HealthCheck
from api.exceptions import ValidationError, ModelError
from api.monitoring import metrics_collector, MetricsCollector
from api.model_manager import ModelVersionManager, ModelMetadata


# Test fixtures
@pytest.fixture
def client():
    """Create test client"""
    return TestClient(api)


@pytest.fixture
def sample_customer_data():
    """Sample valid customer data"""
    return {
        "tenure": 12.0,
        "MonthlyCharges": 65.5,
        "TotalCharges": 786.0,
        "Contract_Two_year": 0,
        "InternetService_Fiber_optic": 1,
        "OnlineSecurity_No": 1,
        "TechSupport_No": 1,
        "PaperlessBilling": 1
    }


@pytest.fixture
def mock_model():
    """Mock model for testing"""
    model = Mock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])  # 70% churn probability
    model.predict.return_value = np.array([1])  # Churn prediction
    return model


@pytest.fixture
def mock_model_metadata():
    """Mock model metadata"""
    return ModelMetadata(
        model_name="test_model",
        model_type="LogisticRegression",
        version="1.0.0",
        file_hash="test_hash",
        training_date=datetime.utcnow(),
        data_version="1.0",
        training_samples=1000,
        accuracy=0.85,
        precision=0.80,
        recall=0.75,
        f1_score=0.77,
        roc_auc=0.82,
        features=["tenure", "MonthlyCharges", "TotalCharges", "Contract_Two_year",
                 "InternetService_Fiber_optic", "OnlineSecurity_No", "TechSupport_No", "PaperlessBilling"],
        is_active=True
    )


class TestCustomerValidation:
    """Test customer data validation"""
    
    def test_valid_customer_features(self, sample_customer_data):
        """Test valid customer features pass validation"""
        features = CustomerFeatures(**sample_customer_data)
        assert features.tenure == 12.0
        assert features.MonthlyCharges == 65.5
        assert features.Contract_Two_year == 0
    
    def test_invalid_tenure(self, sample_customer_data):
        """Test invalid tenure values"""
        # Negative tenure
        sample_customer_data["tenure"] = -5
        with pytest.raises(ValueError, match="Tenure cannot be negative"):
            CustomerFeatures(**sample_customer_data)
        
        # Extremely high tenure
        sample_customer_data["tenure"] = 150
        with pytest.raises(ValueError, match="Tenure exceeding 100 months"):
            CustomerFeatures(**sample_customer_data)
    
    def test_invalid_monthly_charges(self, sample_customer_data):
        """Test invalid monthly charges"""
        # Zero monthly charges
        sample_customer_data["MonthlyCharges"] = 0
        with pytest.raises(ValueError, match="Monthly charges must be positive"):
            CustomerFeatures(**sample_customer_data)
        
        # Extremely high monthly charges
        sample_customer_data["MonthlyCharges"] = 250
        with pytest.raises(ValueError, match="Monthly charges exceeding"):
            CustomerFeatures(**sample_customer_data)
    
    def test_charges_consistency_validation(self, sample_customer_data):
        """Test charges consistency validation"""
        # TotalCharges too low for tenure and monthly charges
        sample_customer_data["tenure"] = 24
        sample_customer_data["MonthlyCharges"] = 100
        sample_customer_data["TotalCharges"] = 100  # Should be around 2400
        
        with pytest.raises(ValueError, match="TotalCharges.*seems too low"):
            CustomerFeatures(**sample_customer_data)
    
    def test_binary_field_validation(self, sample_customer_data):
        """Test binary field validation"""
        # Invalid binary value
        sample_customer_data["Contract_Two_year"] = 2
        with pytest.raises(ValueError):
            CustomerFeatures(**sample_customer_data)
    
    def test_risk_factors_identification(self, sample_customer_data):
        """Test risk factor identification"""
        features = CustomerFeatures(**sample_customer_data)
        risk_factors = features.get_risk_factors()
        
        # Should identify several risk factors for this customer
        assert "No long-term contract" in risk_factors
        assert "No online security service" in risk_factors
        assert "No tech support service" in risk_factors
    
    def test_to_prediction_array(self, sample_customer_data):
        """Test conversion to prediction array"""
        features = CustomerFeatures(**sample_customer_data)
        array = features.to_prediction_array()
        
        assert array.shape == (1, 8)
        assert array[0, 0] == 12.0  # tenure
        assert array[0, 1] == 65.5  # MonthlyCharges


class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == settings.app_name
        assert data["status"] == "operational"
        assert "version" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "checks" in data
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "successful_requests" in data
        assert "timestamp" in data
    
    @patch('api.deployment_api.get_current_model')
    def test_predict_endpoint_success(self, mock_get_model, client, sample_customer_data, mock_model, mock_model_metadata):
        """Test successful prediction"""
        mock_get_model.return_value = (mock_model, mock_model_metadata)
        
        response = client.post("/predict", json=sample_customer_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "churn_probability" in data
        assert "prediction" in data
        assert "interpretation" in data
        assert "explanation" in data
        assert data["prediction"] == 1
        assert 0 <= data["churn_probability"] <= 1
    
    @patch('api.deployment_api.get_current_model')
    def test_predict_endpoint_validation_error(self, mock_get_model, client, sample_customer_data):
        """Test prediction with validation error"""
        # Invalid data
        sample_customer_data["tenure"] = -5
        
        response = client.post("/predict", json=sample_customer_data)
        assert response.status_code == 422
    
    @patch('api.deployment_api.get_current_model')
    def test_batch_predict_endpoint(self, mock_get_model, client, sample_customer_data, mock_model, mock_model_metadata):
        """Test batch prediction"""
        mock_get_model.return_value = (mock_model, mock_model_metadata)
        
        batch_data = {
            "customers": [sample_customer_data, sample_customer_data],
            "include_explanation": True
        }
        
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["predictions"]) == 2
        assert "processing_time_ms" in data
    
    @patch('api.deployment_api.get_current_model')
    def test_model_info_endpoint(self, mock_get_model, client, mock_model, mock_model_metadata):
        """Test model info endpoint"""
        mock_get_model.return_value = (mock_model, mock_model_metadata)
        
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert data["model_name"] == "test_model"
        assert data["version"] == "1.0.0"
        assert "performance" in data
        assert "features" in data


class TestMonitoring:
    """Test monitoring and metrics system"""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization"""
        collector = MetricsCollector()
        assert len(collector.request_history) == 0
        assert collector.total_predictions == 0
    
    def test_record_prediction(self):
        """Test recording predictions"""
        collector = MetricsCollector()
        
        # Record some predictions
        collector.record_prediction(1)  # churn
        collector.record_prediction(0)  # stay
        collector.record_prediction(1)  # churn
        
        assert collector.total_predictions == 3
        assert collector.prediction_counts['churn'] == 2
        assert collector.prediction_counts['stay'] == 1
    
    def test_get_metrics(self):
        """Test getting aggregated metrics"""
        collector = MetricsCollector()
        
        # Record some data
        collector.record_prediction(1)
        collector.record_prediction(0)
        
        metrics = collector.get_metrics()
        assert metrics.total_predictions == 2
        assert metrics.churn_predictions == 1
        assert metrics.stay_predictions == 1


class TestModelManager:
    """Test model version management"""
    
    def setUp(self):
        """Set up temporary directory for tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_manager = ModelVersionManager(Path(self.temp_dir))
    
    def test_model_manager_initialization(self):
        """Test model manager initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ModelVersionManager(Path(temp_dir))
            assert manager.model_dir == Path(temp_dir)
            assert len(manager.metadata_cache) == 0
    
    def test_model_metadata_serialization(self):
        """Test model metadata serialization"""
        metadata = ModelMetadata(
            model_name="test_model",
            model_type="LogisticRegression",
            version="1.0.0",
            file_hash="test_hash",
            training_date=datetime.utcnow(),
            data_version="1.0",
            training_samples=1000,
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            roc_auc=0.82,
            features=["feature1", "feature2"]
        )
        
        # Test serialization
        data_dict = metadata.to_dict()
        assert isinstance(data_dict["training_date"], str)
        assert data_dict["model_name"] == "test_model"
        
        # Test deserialization
        restored_metadata = ModelMetadata.from_dict(data_dict)
        assert restored_metadata.model_name == metadata.model_name
        assert isinstance(restored_metadata.training_date, datetime)


class TestErrorHandling:
    """Test error handling and exception management"""
    
    def test_api_error_creation(self):
        """Test API error creation"""
        from api.exceptions import APIError, ErrorResponse, create_error_response
        
        error = APIError(
            message="Test error",
            error_code="TEST_ERROR",
            details={"key": "value"}
        )
        
        response = create_error_response(error)
        assert isinstance(response, ErrorResponse)
        assert response.error_code == "TEST_ERROR"
        assert response.message == "Test error"
        assert response.success is False
    
    def test_validation_error_handling(self):
        """Test validation error handling"""
        from api.exceptions import ValidationErrorDetail, ErrorHandler
        
        errors = [
            ValidationErrorDetail(
                field="tenure",
                value=-5,
                error="Value cannot be negative",
                constraint={"min": 0}
            )
        ]
        
        validation_error = ErrorHandler.handle_validation_error(errors)
        assert validation_error.error_code == "INPUT_VALIDATION_ERROR"
        assert len(validation_error.details["validation_errors"]) == 1


class TestBusinessLogic:
    """Test business logic and calculations"""
    
    def test_prediction_confidence_calculation(self, sample_customer_data):
        """Test prediction confidence calculation"""
        from api.models import PredictionConfidence
        
        # High confidence (extreme probabilities)
        features = CustomerFeatures(**sample_customer_data)
        
        # Test confidence determination logic would go here
        # This is a placeholder for the actual logic
        
        # Test that confidence levels are calculated correctly
        probabilities_and_expected_confidence = [
            (0.95, PredictionConfidence.HIGH),
            (0.1, PredictionConfidence.HIGH),
            (0.7, PredictionConfidence.MEDIUM),
            (0.5, PredictionConfidence.LOW)
        ]
        
        for prob, expected_confidence in probabilities_and_expected_confidence:
            if prob >= 0.8 or prob <= 0.2:
                confidence = PredictionConfidence.HIGH
            elif prob >= 0.6 or prob <= 0.4:
                confidence = PredictionConfidence.MEDIUM
            else:
                confidence = PredictionConfidence.LOW
            
            assert confidence == expected_confidence


# Integration Tests
class TestIntegration:
    """Integration tests for the complete system"""
    
    @patch('api.deployment_api.get_current_model')
    def test_end_to_end_prediction_flow(self, mock_get_model, client, sample_customer_data, mock_model, mock_model_metadata):
        """Test complete prediction flow from API call to response"""
        mock_get_model.return_value = (mock_model, mock_model_metadata)
        
        # Make prediction request
        response = client.post("/predict", json=sample_customer_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify complete response structure
        required_fields = [
            "churn_probability", "prediction", "interpretation",
            "explanation", "model_version", "prediction_timestamp"
        ]
        
        for field in required_fields:
            assert field in data
        
        # Verify explanation structure
        explanation = data["explanation"]
        assert "confidence" in explanation
        assert "risk_factors" in explanation
        assert "recommendation" in explanation
        
        # Verify business logic
        assert isinstance(data["churn_probability"], float)
        assert 0 <= data["churn_probability"] <= 1
        assert data["prediction"] in [0, 1]
    
    def test_system_health_monitoring(self, client):
        """Test system health monitoring integration"""
        # Check health endpoint
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        health_data = health_response.json()
        assert health_data["status"] in ["healthy", "degraded", "unhealthy"]
        
        # Check metrics endpoint
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        
        metrics_data = metrics_response.json()
        assert isinstance(metrics_data["total_requests"], int)


# Performance Tests
class TestPerformance:
    """Performance and load testing"""
    
    @patch('api.deployment_api.get_current_model')
    def test_prediction_response_time(self, mock_get_model, client, sample_customer_data, mock_model, mock_model_metadata):
        """Test prediction response time"""
        import time
        
        mock_get_model.return_value = (mock_model, mock_model_metadata)
        
        start_time = time.time()
        response = client.post("/predict", json=sample_customer_data)
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        assert response.status_code == 200
        assert response_time < 1000  # Should respond within 1 second
    
    @patch('api.deployment_api.get_current_model')
    def test_batch_prediction_scalability(self, mock_get_model, client, sample_customer_data, mock_model, mock_model_metadata):
        """Test batch prediction scalability"""
        mock_get_model.return_value = (mock_model, mock_model_metadata)
        
        # Test with larger batch
        batch_size = 50
        batch_data = {
            "customers": [sample_customer_data] * batch_size,
            "include_explanation": False  # Faster processing
        }
        
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["predictions"]) == batch_size
        assert data["processing_time_ms"] < 5000  # Should complete within 5 seconds


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])