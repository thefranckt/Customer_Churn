"""
Enterprise-grade Customer Churn Prediction API
Production-ready FastAPI with comprehensive monitoring, validation, and error handling
"""

import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import logging

# Internal imports
from .config import settings, logger
from .models import (
    CustomerFeatures, PredictionResult, PredictionRequest,
    BatchPredictionResponse, HealthCheck, MetricsResponse,
    PredictionExplanation, PredictionConfidence
)
from .exceptions import (
    APIError, ModelError, ValidationError, ErrorHandler,
    setup_exception_handlers, validate_business_rules
)
from .model_manager import model_manager
from .monitoring import metrics_collector, health_monitor, track_request_time

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))

# Initialize FastAPI app with comprehensive configuration
api = FastAPI(
    title=settings.app_name,
    description="Enterprise Customer Churn Prediction API with advanced monitoring and validation",
    version=settings.app_version,
    debug=settings.debug,
    docs_url="/docs" if settings.debug else None,  # Disable docs in production
    redoc_url="/redoc" if settings.debug else None
)

# Add middleware
if settings.enable_cors:
    api.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

# Add trusted host middleware for security
api.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.debug else ["localhost", "127.0.0.1"]
)

# Setup exception handlers
setup_exception_handlers(api)

# Global model cache
_model_cache = {}
_model_metadata = {}


async def get_current_model():
    """Dependency to get the current active model"""
    try:
        if "active" not in _model_cache:
            model, metadata = model_manager.get_active_model()
            _model_cache["active"] = model
            _model_metadata["active"] = metadata
            logger.info(f"Loaded active model: {metadata.model_name} v{metadata.version}")
        
        return _model_cache["active"], _model_metadata["active"]
    
    except Exception as e:
        # Fallback to direct model loading for backward compatibility
        try:
            import joblib
            model = joblib.load(settings.model_dir / settings.default_model)
            
            # Create minimal metadata
            from .model_manager import ModelMetadata
            metadata = ModelMetadata(
                model_name=settings.default_model.replace('.pkl', ''),
                model_type="LogisticRegression",
                version=settings.model_version,
                file_hash="unknown",
                training_date=datetime.utcnow(),
                data_version="unknown",
                training_samples=0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                roc_auc=0.0,
                features=list(settings.feature_ranges.keys())
            )
            
            _model_cache["active"] = model
            _model_metadata["active"] = metadata
            logger.warning("Loaded model without registry - consider registering models for better management")
            return model, metadata
            
        except Exception as fallback_error:
            raise ErrorHandler.handle_model_loading_error(
                str(settings.model_dir / settings.default_model),
                fallback_error
            )


def create_prediction_explanation(
    features: CustomerFeatures,
    probability: float,
    model_metadata
) -> PredictionExplanation:
    """Create detailed prediction explanation"""
    
    # Determine confidence
    if probability >= 0.8 or probability <= 0.2:
        confidence = PredictionConfidence.HIGH
    elif probability >= 0.6 or probability <= 0.4:
        confidence = PredictionConfidence.MEDIUM
    else:
        confidence = PredictionConfidence.LOW
    
    # Get risk factors
    risk_factors = features.get_risk_factors()
    
    # Determine protective factors
    protective_factors = []
    if features.Contract_Two_year == 1:
        protective_factors.append("Long-term contract commitment")
    if features.OnlineSecurity_No == 0:
        protective_factors.append("Has online security service")
    if features.TechSupport_No == 0:
        protective_factors.append("Has tech support service")
    if features.tenure >= 24:
        protective_factors.append("Long-term customer (2+ years)")
    if features.MonthlyCharges < 50:
        protective_factors.append("Lower monthly charges")
    
    # Create recommendation
    if probability > 0.7:
        recommendation = "High churn risk - immediate retention action recommended"
    elif probability > 0.5:
        recommendation = "Moderate churn risk - proactive engagement suggested"
    elif probability > 0.3:
        recommendation = "Low churn risk - routine monitoring sufficient"
    else:
        recommendation = "Very low churn risk - customer likely to stay"
    
    return PredictionExplanation(
        confidence=confidence,
        risk_factors=risk_factors,
        protective_factors=protective_factors,
        recommendation=recommendation,
        confidence_score=max(probability, 1 - probability)
    )


@api.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Add request ID to response headers
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


# API Endpoints

@api.get("/", response_model=Dict[str, Any], tags=["Health"])
async def root():
    """Root endpoint with basic API information"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "documentation": "/docs" if settings.debug else "Contact administrator",
    }


@api.get("/health", response_model=HealthCheck, tags=["Health"])
async def health_check():
    """Comprehensive health check endpoint"""
    async with track_request_time(metrics_collector, "/health", "GET") as request_metrics:
        try:
            health_status = await health_monitor.get_health_status()
            request_metrics.status_code = 200
            return health_status
        except Exception as e:
            request_metrics.status_code = 500
            request_metrics.error_type = type(e).__name__
            raise HTTPException(status_code=500, detail="Health check failed")


@api.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """Get API performance metrics"""
    async with track_request_time(metrics_collector, "/metrics", "GET") as request_metrics:
        try:
            metrics = metrics_collector.get_metrics()
            request_metrics.status_code = 200
            return metrics
        except Exception as e:
            request_metrics.status_code = 500
            request_metrics.error_type = type(e).__name__
            raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@api.post("/predict", response_model=PredictionResult, tags=["Prediction"])
async def predict_churn(
    features: CustomerFeatures,
    request: Request,
    model_data: tuple = Depends(get_current_model)
):
    """Single customer churn prediction with detailed explanation"""
    
    async with track_request_time(metrics_collector, "/predict", "POST") as request_metrics:
        try:
            model, metadata = model_data
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            logger.info(f"Processing prediction request {request_id}")
            
            # Validate business rules
            business_errors = validate_business_rules(
                features.dict(), 
                settings.feature_ranges
            )
            
            if business_errors:
                raise ErrorHandler.handle_validation_error(business_errors)
            
            # Convert to model input
            input_data = features.to_prediction_array()
            
            # Make prediction
            start_time = time.time()
            probability = float(model.predict_proba(input_data)[0][1])
            prediction = int(model.predict(input_data)[0])
            prediction_time = (time.time() - start_time) * 1000
            
            logger.debug(f"Prediction completed in {prediction_time:.2f}ms")
            
            # Record prediction for metrics
            metrics_collector.record_prediction(prediction)
            
            # Create explanation
            explanation = create_prediction_explanation(features, probability, metadata)
            
            # Determine interpretation
            if prediction == 1:
                interpretation = f"Customer is likely to churn (confidence: {explanation.confidence.value})"
            else:
                interpretation = f"Customer is likely to stay (confidence: {explanation.confidence.value})"
            
            result = PredictionResult(
                churn_probability=round(probability, 4),
                prediction=prediction,
                interpretation=interpretation,
                explanation=explanation,
                model_version=metadata.version,
                prediction_timestamp=datetime.utcnow()
            )
            
            request_metrics.status_code = 200
            logger.info(f"Prediction request {request_id} completed successfully")
            
            return result
            
        except APIError:
            request_metrics.status_code = 422
            request_metrics.error_type = "APIError"
            raise
        except Exception as e:
            request_metrics.status_code = 500
            request_metrics.error_type = type(e).__name__
            logger.error(f"Prediction failed: {str(e)}")
            raise ErrorHandler.handle_prediction_error(features.dict(), e)


@api.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request_data: PredictionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    model_data: tuple = Depends(get_current_model)
):
    """Batch prediction for multiple customers"""
    
    async with track_request_time(metrics_collector, "/predict/batch", "POST") as request_metrics:
        try:
            model, metadata = model_data
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            logger.info(f"Processing batch prediction request {request_id} for {len(request_data.customers)} customers")
            
            start_time = time.time()
            predictions = []
            
            for i, customer in enumerate(request_data.customers):
                try:
                    # Validate business rules
                    business_errors = validate_business_rules(
                        customer.dict(), 
                        settings.feature_ranges
                    )
                    
                    if business_errors:
                        # For batch processing, we can continue with warnings
                        logger.warning(f"Customer {i} has validation warnings: {len(business_errors)} issues")
                    
                    # Make prediction
                    input_data = customer.to_prediction_array()
                    probability = float(model.predict_proba(input_data)[0][1])
                    prediction = int(model.predict(input_data)[0])
                    
                    # Record prediction for metrics
                    metrics_collector.record_prediction(prediction)
                    
                    # Create explanation if requested
                    explanation = None
                    if request_data.include_explanation:
                        explanation = create_prediction_explanation(customer, probability, metadata)
                    
                    interpretation = ("Likely to churn" if prediction == 1 else "Likely to stay")
                    
                    result = PredictionResult(
                        customer_id=str(i),  # Could be enhanced to accept actual customer IDs
                        churn_probability=round(probability, 4),
                        prediction=prediction,
                        interpretation=interpretation,
                        explanation=explanation,
                        model_version=metadata.version,
                        prediction_timestamp=datetime.utcnow()
                    )
                    
                    predictions.append(result)
                    
                except Exception as e:
                    logger.error(f"Failed to process customer {i}: {str(e)}")
                    # Continue with other customers in batch
                    continue
            
            processing_time = (time.time() - start_time) * 1000
            
            response = BatchPredictionResponse(
                predictions=predictions,
                processing_time_ms=processing_time,
                model_info={
                    "name": metadata.model_name,
                    "version": metadata.version,
                    "type": metadata.model_type
                },
                metadata={
                    "total_customers": len(request_data.customers),
                    "successful_predictions": len(predictions),
                    "failed_predictions": len(request_data.customers) - len(predictions),
                    "request_id": request_id
                }
            )
            
            request_metrics.status_code = 200
            logger.info(f"Batch prediction request {request_id} completed: {len(predictions)}/{len(request_data.customers)} successful")
            
            return response
            
        except Exception as e:
            request_metrics.status_code = 500
            request_metrics.error_type = type(e).__name__
            logger.error(f"Batch prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Batch prediction failed")


@api.get("/model/info", response_model=Dict[str, Any], tags=["Model"])
async def get_model_info(model_data: tuple = Depends(get_current_model)):
    """Get information about the current model"""
    try:
        model, metadata = model_data
        
        return {
            "model_name": metadata.model_name,
            "version": metadata.version,
            "type": metadata.model_type,
            "training_date": metadata.training_date.isoformat(),
            "features": metadata.features,
            "performance": {
                "accuracy": metadata.accuracy,
                "precision": metadata.precision,
                "recall": metadata.recall,
                "f1_score": metadata.f1_score,
                "roc_auc": metadata.roc_auc
            },
            "is_active": metadata.is_active,
            "model_size_mb": metadata.model_size_mb
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model information")


# Startup and shutdown events
@api.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    try:
        # Initialize model (this will cache it)
        model, metadata = await get_current_model()
        logger.info(f"Successfully loaded model: {metadata.model_name} v{metadata.version}")
        
    except Exception as e:
        logger.error(f"Failed to load model during startup: {str(e)}")
        # Don't fail startup - let the dependency handle errors per request


@api.on_event("shutdown")
async def shutdown_event():
    """Clean up resources"""
    logger.info("Shutting down API")
    
    # Clear model cache
    _model_cache.clear()
    _model_metadata.clear()
    
    logger.info("API shutdown complete")
