"""
Exception handling and validation for Customer Churn API
Comprehensive error management for production reliability
"""

from typing import Dict, Any, List, Optional
from fastapi import HTTPException, status
from pydantic import BaseModel, validator, ValidationError
import logging
import traceback
from datetime import datetime

logger = logging.getLogger("churn_api.exceptions")


class APIError(Exception):
    """Base exception for API errors"""
    
    def __init__(self, message: str, error_code: str = "GENERIC_ERROR", details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        super().__init__(self.message)


class ModelError(APIError):
    """Exceptions related to model operations"""
    pass


class ValidationError(APIError):
    """Exceptions related to input validation"""
    pass


class ConfigurationError(APIError):
    """Exceptions related to configuration issues"""
    pass


class ErrorResponse(BaseModel):
    """Standardized error response format"""
    
    success: bool = False
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str
    request_id: Optional[str] = None


class ValidationErrorDetail(BaseModel):
    """Detailed validation error information"""
    
    field: str
    value: Any
    error: str
    constraint: Optional[Dict[str, Any]] = None


def create_error_response(
    error: Exception,
    request_id: Optional[str] = None
) -> ErrorResponse:
    """Create standardized error response"""
    
    if isinstance(error, APIError):
        return ErrorResponse(
            error_code=error.error_code,
            message=error.message,
            details=error.details,
            timestamp=error.timestamp.isoformat(),
            request_id=request_id
        )
    elif isinstance(error, ValidationError):
        return ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="Input validation failed",
            details={"validation_errors": str(error)},
            timestamp=datetime.utcnow().isoformat(),
            request_id=request_id
        )
    else:
        # Log unexpected errors
        logger.error(f"Unexpected error: {str(error)}\n{traceback.format_exc()}")
        
        return ErrorResponse(
            error_code="INTERNAL_ERROR",
            message="An unexpected error occurred",
            details={"type": type(error).__name__},
            timestamp=datetime.utcnow().isoformat(),
            request_id=request_id
        )


def validate_business_rules(data: Dict[str, Any], feature_ranges: Dict[str, Dict[str, float]]) -> List[ValidationErrorDetail]:
    """Validate business rules for input data"""
    
    errors = []
    
    for field, value in data.items():
        if field in feature_ranges:
            constraints = feature_ranges[field]
            
            # Check range constraints
            if "min" in constraints and value < constraints["min"]:
                errors.append(ValidationErrorDetail(
                    field=field,
                    value=value,
                    error=f"Value {value} is below minimum allowed value",
                    constraint={"min": constraints["min"]}
                ))
            
            if "max" in constraints and value > constraints["max"]:
                errors.append(ValidationErrorDetail(
                    field=field,
                    value=value,
                    error=f"Value {value} is above maximum allowed value",
                    constraint={"max": constraints["max"]}
                ))
    
    # Business logic validations
    
    # TotalCharges should be reasonable given MonthlyCharges and tenure
    if all(key in data for key in ["tenure", "MonthlyCharges", "TotalCharges"]):
        expected_min_total = data["tenure"] * data["MonthlyCharges"] * 0.5  # Allow 50% variance
        expected_max_total = data["tenure"] * data["MonthlyCharges"] * 2.0  # Allow 100% variance
        
        if data["TotalCharges"] < expected_min_total or data["TotalCharges"] > expected_max_total:
            errors.append(ValidationErrorDetail(
                field="TotalCharges",
                value=data["TotalCharges"],
                error=f"TotalCharges {data['TotalCharges']} seems inconsistent with tenure ({data['tenure']}) and MonthlyCharges ({data['MonthlyCharges']})",
                constraint={
                    "expected_range": [expected_min_total, expected_max_total],
                    "actual_range": [data["TotalCharges"], data["TotalCharges"]]
                }
            ))
    
    # Binary features should be 0 or 1
    binary_fields = [
        "Contract_Two_year", "InternetService_Fiber_optic", 
        "OnlineSecurity_No", "TechSupport_No", "PaperlessBilling"
    ]
    
    for field in binary_fields:
        if field in data and data[field] not in [0, 1]:
            errors.append(ValidationErrorDetail(
                field=field,
                value=data[field],
                error=f"Binary field must be 0 or 1",
                constraint={"allowed_values": [0, 1]}
            ))
    
    return errors


class ErrorHandler:
    """Centralized error handling utilities"""
    
    @staticmethod
    def handle_model_loading_error(model_path: str, original_error: Exception) -> ModelError:
        """Handle model loading failures"""
        
        logger.error(f"Failed to load model from {model_path}: {str(original_error)}")
        
        return ModelError(
            message=f"Failed to load prediction model",
            error_code="MODEL_LOADING_ERROR",
            details={
                "model_path": model_path,
                "original_error": str(original_error),
                "error_type": type(original_error).__name__
            }
        )
    
    @staticmethod
    def handle_prediction_error(input_data: Dict[str, Any], original_error: Exception) -> ModelError:
        """Handle prediction failures"""
        
        logger.error(f"Prediction failed for input {input_data}: {str(original_error)}")
        
        return ModelError(
            message="Failed to generate prediction",
            error_code="PREDICTION_ERROR",
            details={
                "input_data": input_data,
                "original_error": str(original_error),
                "error_type": type(original_error).__name__
            }
        )
    
    @staticmethod
    def handle_validation_error(validation_errors: List[ValidationErrorDetail]) -> ValidationError:
        """Handle validation failures"""
        
        logger.warning(f"Input validation failed with {len(validation_errors)} errors")
        
        return ValidationError(
            message=f"Input validation failed with {len(validation_errors)} error(s)",
            error_code="INPUT_VALIDATION_ERROR",
            details={
                "validation_errors": [error.dict() for error in validation_errors],
                "error_count": len(validation_errors)
            }
        )
    
    @staticmethod
    def convert_to_http_exception(error: APIError) -> HTTPException:
        """Convert API error to HTTP exception"""
        
        status_code_map = {
            "VALIDATION_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "INPUT_VALIDATION_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "MODEL_LOADING_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
            "PREDICTION_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "CONFIGURATION_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
        }
        
        status_code = status_code_map.get(error.error_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return HTTPException(
            status_code=status_code,
            detail=create_error_response(error).dict()
        )


# Custom exception handlers for FastAPI
def setup_exception_handlers(app):
    """Setup custom exception handlers for the FastAPI app"""
    
    @app.exception_handler(APIError)
    async def api_error_handler(request, exc: APIError):
        """Handle custom API errors"""
        http_exc = ErrorHandler.convert_to_http_exception(exc)
        return http_exc
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc: Exception):
        """Handle unexpected exceptions"""
        error_response = create_error_response(exc)
        
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response.dict()
        )
    
    return app