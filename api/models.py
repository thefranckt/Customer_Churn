"""
Enhanced input validation and data models for Customer Churn API
Robust validation with business rules and data quality checks
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime
import numpy as np
from enum import Enum

from .config import settings, model_config
from .exceptions import ValidationError, validate_business_rules


class PredictionConfidence(str, Enum):
    """Prediction confidence levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CustomerFeatures(BaseModel):
    """Enhanced customer features with comprehensive validation"""
    
    tenure: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Number of months the customer has been with the company",
        example=12.0
    )
    
    MonthlyCharges: float = Field(
        ..., 
        ge=0, 
        le=200, 
        description="Monthly charges for the customer",
        example=65.5
    )
    
    TotalCharges: float = Field(
        ..., 
        ge=0, 
        le=10000, 
        description="Total charges accumulated by the customer",
        example=786.0
    )
    
    Contract_Two_year: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Whether customer has a two-year contract (0=No, 1=Yes)",
        example=0
    )
    
    InternetService_Fiber_optic: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Whether customer has fiber optic internet (0=No, 1=Yes)",
        example=1
    )
    
    OnlineSecurity_No: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Whether customer has no online security (0=Has security, 1=No security)",
        example=1
    )
    
    TechSupport_No: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Whether customer has no tech support (0=Has support, 1=No support)",
        example=1
    )
    
    PaperlessBilling: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Whether customer uses paperless billing (0=No, 1=Yes)",
        example=1
    )
    
    @validator('tenure')
    def validate_tenure(cls, v):
        """Validate tenure business rules"""
        if v < 0:
            raise ValueError("Tenure cannot be negative")
        if v > 100:
            raise ValueError("Tenure exceeding 100 months is unusual, please verify")
        return v
    
    @validator('MonthlyCharges')
    def validate_monthly_charges(cls, v):
        """Validate monthly charges business rules"""
        if v <= 0:
            raise ValueError("Monthly charges must be positive")
        if v > 200:
            raise ValueError("Monthly charges exceeding $200 is unusual, please verify")
        return v
    
    @validator('TotalCharges')
    def validate_total_charges(cls, v):
        """Validate total charges business rules"""
        if v < 0:
            raise ValueError("Total charges cannot be negative")
        return v
    
    @root_validator
    def validate_charges_consistency(cls, values):
        """Validate consistency between charges and tenure"""
        tenure = values.get('tenure')
        monthly = values.get('MonthlyCharges')
        total = values.get('TotalCharges')
        
        if all(x is not None for x in [tenure, monthly, total]):
            # Calculate expected range (allowing for promotions, upgrades, etc.)
            min_expected = tenure * monthly * 0.3  # 70% discount allowance
            max_expected = tenure * monthly * 1.5   # 50% premium allowance
            
            if total < min_expected:
                raise ValueError(
                    f"TotalCharges ({total}) seems too low for tenure ({tenure}) months "
                    f"and MonthlyCharges (${monthly}). Expected minimum: ${min_expected:.2f}"
                )
            
            if total > max_expected:
                raise ValueError(
                    f"TotalCharges ({total}) seems too high for tenure ({tenure}) months "
                    f"and MonthlyCharges (${monthly}). Expected maximum: ${max_expected:.2f}"
                )
        
        return values
    
    @root_validator
    def validate_service_logic(cls, values):
        """Validate service-related business logic"""
        
        # If customer has fiber optic, they're more likely to need tech support
        fiber = values.get('InternetService_Fiber_optic', 0)
        no_tech_support = values.get('TechSupport_No', 0)
        
        # This is just a warning, not a hard validation error
        if fiber == 1 and no_tech_support == 1:
            # Could log this for data quality monitoring
            pass
        
        return values
    
    def to_prediction_array(self) -> np.ndarray:
        """Convert to numpy array for model prediction"""
        return np.array([[
            self.tenure,
            self.MonthlyCharges,
            self.TotalCharges,
            self.Contract_Two_year,
            self.InternetService_Fiber_optic,
            self.OnlineSecurity_No,
            self.TechSupport_No,
            self.PaperlessBilling
        ]])
    
    def get_risk_factors(self) -> List[str]:
        """Identify potential churn risk factors"""
        risk_factors = []
        
        # High risk patterns
        if self.Contract_Two_year == 0:
            risk_factors.append("No long-term contract")
        
        if self.OnlineSecurity_No == 1:
            risk_factors.append("No online security service")
        
        if self.TechSupport_No == 1:
            risk_factors.append("No tech support service")
        
        if self.MonthlyCharges > 80:
            risk_factors.append("High monthly charges")
        
        if self.tenure < 12:
            risk_factors.append("New customer (less than 1 year)")
        
        if self.InternetService_Fiber_optic == 1 and self.MonthlyCharges > 70:
            risk_factors.append("High-cost fiber customer")
        
        return risk_factors


class PredictionRequest(BaseModel):
    """Request model for batch predictions"""
    
    customers: List[CustomerFeatures] = Field(
        ..., 
        min_items=1, 
        max_items=100,  # Limit batch size for performance
        description="List of customers for prediction"
    )
    
    include_explanation: bool = Field(
        default=True,
        description="Whether to include prediction explanation"
    )
    
    model_version: Optional[str] = Field(
        default=None,
        description="Specific model version to use (defaults to latest)"
    )


class PredictionExplanation(BaseModel):
    """Explanation of prediction results"""
    
    confidence: PredictionConfidence
    risk_factors: List[str]
    protective_factors: List[str]
    recommendation: str
    confidence_score: float = Field(ge=0.0, le=1.0)


class PredictionResult(BaseModel):
    """Enhanced prediction result with explanation"""
    
    customer_id: Optional[str] = None
    churn_probability: float = Field(ge=0.0, le=1.0, description="Probability of customer churn")
    prediction: int = Field(ge=0, le=1, description="Binary prediction (0=Stay, 1=Churn)")
    interpretation: str = Field(description="Human-readable interpretation")
    
    explanation: Optional[PredictionExplanation] = None
    model_version: str = Field(description="Model version used for prediction")
    prediction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    def determine_confidence(self) -> PredictionConfidence:
        """Determine confidence level based on probability"""
        if self.churn_probability >= 0.8 or self.churn_probability <= 0.2:
            return PredictionConfidence.HIGH
        elif self.churn_probability >= 0.6 or self.churn_probability <= 0.4:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    
    success: bool = True
    predictions: List[PredictionResult]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: float
    model_info: Dict[str, Any]


class HealthStatus(str, Enum):
    """Health check status options"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthCheck(BaseModel):
    """Health check response model"""
    
    status: HealthStatus
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    uptime_seconds: float
    
    checks: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Performance metrics
    total_predictions: int = 0
    avg_response_time_ms: float = 0.0
    error_rate_percent: float = 0.0
    
    # Resource usage
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None


class MetricsResponse(BaseModel):
    """API metrics response"""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Performance metrics
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    
    # Prediction metrics
    total_predictions: int = 0
    churn_predictions: int = 0
    stay_predictions: int = 0
    
    # Error tracking
    validation_errors: int = 0
    model_errors: int = 0
    system_errors: int = 0


class DataQualityReport(BaseModel):
    """Data quality assessment for input validation"""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_samples: int
    
    # Quality scores (0-1)
    overall_quality_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    consistency_score: float = Field(ge=0.0, le=1.0)
    validity_score: float = Field(ge=0.0, le=1.0)
    
    # Detailed findings
    missing_values: Dict[str, int] = Field(default_factory=dict)
    outliers: Dict[str, List[float]] = Field(default_factory=dict)
    inconsistencies: List[str] = Field(default_factory=list)
    
    recommendations: List[str] = Field(default_factory=list)