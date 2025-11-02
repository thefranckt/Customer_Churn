"""
Configuration management for Customer Churn Prediction API
Centralized settings for consistency and reliability across environments
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
from pydantic import BaseSettings, validator
import logging


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application Info
    app_name: str = "Customer Churn Prediction API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # API Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # Model Configuration
    model_dir: Path = Path("models")
    default_model: str = "logreg_model.pkl"
    model_version: str = "1.0"
    
    # Data Configuration
    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = "logs/app.log"
    
    # Performance Configuration
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    
    # Security Configuration
    enable_cors: bool = True
    allowed_origins: List[str] = ["*"]
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    
    # Monitoring Configuration
    enable_metrics: bool = True
    metrics_port: int = 9090
    health_check_interval: int = 60
    
    # Feature Configuration - Business Rules
    feature_ranges: Dict[str, Dict[str, float]] = {
        "tenure": {"min": 0, "max": 100},
        "MonthlyCharges": {"min": 0, "max": 200},
        "TotalCharges": {"min": 0, "max": 10000},
        "Contract_Two_year": {"min": 0, "max": 1},
        "InternetService_Fiber_optic": {"min": 0, "max": 1},
        "OnlineSecurity_No": {"min": 0, "max": 1},
        "TechSupport_No": {"min": 0, "max": 1},
        "PaperlessBilling": {"min": 0, "max": 1}
    }
    
    @validator("model_dir", "data_dir", "raw_data_dir", "processed_data_dir")
    def validate_paths(cls, v):
        """Ensure paths exist or can be created"""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("log_file")
    def validate_log_file(cls, v):
        """Ensure log directory exists"""
        if v:
            log_path = Path(v)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_prefix = "CHURN_API_"


class ModelConfig:
    """Model-specific configuration and metadata"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_metadata = self._load_model_metadata()
    
    def _load_model_metadata(self) -> Dict[str, Any]:
        """Load model metadata from YAML file"""
        metadata_file = self.settings.model_dir / "model_metadata.yaml"
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return yaml.safe_load(f)
        
        # Default metadata if file doesn't exist
        return {
            "model_type": "LogisticRegression",
            "version": self.settings.model_version,
            "features": list(self.settings.feature_ranges.keys()),
            "target": "Churn",
            "performance": {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "roc_auc": None
            },
            "training_date": None,
            "data_version": None
        }
    
    def get_model_path(self, model_name: Optional[str] = None) -> Path:
        """Get full path to model file"""
        model_name = model_name or self.settings.default_model
        return self.settings.model_dir / model_name
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names in correct order"""
        return self.model_metadata.get("features", list(self.settings.feature_ranges.keys()))
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get model performance metrics"""
        return self.model_metadata.get("performance", {})


class LoggingConfig:
    """Centralized logging configuration"""
    
    @staticmethod
    def setup_logging(settings: Settings):
        """Configure application logging"""
        
        # Create logs directory if it doesn't exist
        if settings.log_file:
            log_path = Path(settings.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, settings.log_level.upper()),
            format=settings.log_format,
            handlers=[
                logging.StreamHandler(),  # Console output
                logging.FileHandler(settings.log_file) if settings.log_file else logging.NullHandler()
            ]
        )
        
        # Configure specific loggers
        loggers = {
            "uvicorn": logging.INFO,
            "fastapi": logging.INFO,
            "churn_api": getattr(logging, settings.log_level.upper()),
        }
        
        for logger_name, level in loggers.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(level)
        
        return logging.getLogger("churn_api")


# Global settings instance
settings = Settings()
model_config = ModelConfig(settings)

# Setup logging
logger = LoggingConfig.setup_logging(settings)