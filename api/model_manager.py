"""
Model management system with versioning and metadata tracking
Enterprise-grade model lifecycle management
"""

import json
import pickle
import joblib
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import yaml

from .config import settings, model_config
from .exceptions import ModelError, ErrorHandler

logger = logging.getLogger("churn_api.model_manager")


@dataclass
class ModelMetadata:
    """Comprehensive model metadata"""
    
    # Basic info
    model_name: str
    model_type: str
    version: str
    file_hash: str
    
    # Training info
    training_date: datetime
    data_version: str
    training_samples: int
    
    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    
    # Feature info
    features: List[str]
    feature_importance: Optional[Dict[str, float]] = None
    
    # Model configuration
    hyperparameters: Dict[str, Any] = None
    preprocessing_steps: List[str] = None
    
    # Validation info
    validation_method: str = "train_test_split"
    test_size: float = 0.2
    cross_validation_scores: Optional[List[float]] = None
    
    # Deployment info
    deployment_date: Optional[datetime] = None
    is_active: bool = False
    model_size_mb: float = 0.0
    
    # Business metrics
    business_impact: Optional[Dict[str, Any]] = None
    approval_status: str = "pending"  # pending, approved, rejected
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        # Convert ISO strings back to datetime objects
        datetime_fields = ['training_date', 'deployment_date']
        for field in datetime_fields:
            if field in data and data[field]:
                data[field] = datetime.fromisoformat(data[field])
        
        return cls(**data)


class ModelVersionManager:
    """Manage model versions and metadata"""
    
    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir or settings.model_dir
        self.metadata_file = self.model_dir / "model_registry.yaml"
        self.models_cache = {}
        self.metadata_cache = {}
        
        # Ensure directory exists
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing metadata
        self._load_metadata()
    
    def _load_metadata(self):
        """Load model metadata from registry file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = yaml.safe_load(f) or {}
                
                for model_name, metadata_dict in data.items():
                    self.metadata_cache[model_name] = ModelMetadata.from_dict(metadata_dict)
                    
                logger.info(f"Loaded metadata for {len(self.metadata_cache)} models")
                
            except Exception as e:
                logger.error(f"Failed to load model metadata: {e}")
                self.metadata_cache = {}
    
    def _save_metadata(self):
        """Save model metadata to registry file"""
        try:
            data = {}
            for model_name, metadata in self.metadata_cache.items():
                data[model_name] = metadata.to_dict()
            
            with open(self.metadata_file, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, indent=2)
                
            logger.info(f"Saved metadata for {len(self.metadata_cache)} models")
            
        except Exception as e:
            logger.error(f"Failed to save model metadata: {e}")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of model file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def register_model(
        self,
        model_file: Union[str, Path],
        metadata: ModelMetadata,
        overwrite: bool = False
    ) -> bool:
        """Register a new model with metadata"""
        
        model_path = Path(model_file) if isinstance(model_file, str) else model_file
        
        if not model_path.exists():
            raise ModelError(f"Model file not found: {model_path}")
        
        # Calculate file hash for integrity checking
        file_hash = self._calculate_file_hash(model_path)
        metadata.file_hash = file_hash
        
        # Calculate file size
        metadata.model_size_mb = model_path.stat().st_size / (1024 * 1024)
        
        # Check if model already exists
        if metadata.model_name in self.metadata_cache and not overwrite:
            raise ModelError(f"Model {metadata.model_name} already exists. Use overwrite=True to replace.")
        
        # Store metadata
        self.metadata_cache[metadata.model_name] = metadata
        self._save_metadata()
        
        logger.info(f"Registered model: {metadata.model_name} v{metadata.version}")
        return True
    
    def load_model(self, model_name: str, validate_hash: bool = True) -> Tuple[Any, ModelMetadata]:
        """Load model with metadata validation"""
        
        if model_name not in self.metadata_cache:
            raise ModelError(f"Model {model_name} not found in registry")
        
        metadata = self.metadata_cache[model_name]
        model_path = self.model_dir / f"{model_name}"
        
        if not model_path.exists():
            raise ModelError(f"Model file not found: {model_path}")
        
        # Validate file integrity
        if validate_hash:
            current_hash = self._calculate_file_hash(model_path)
            if current_hash != metadata.file_hash:
                raise ModelError(f"Model file integrity check failed for {model_name}")
        
        # Load from cache if available
        if model_name in self.models_cache:
            logger.debug(f"Loading model {model_name} from cache")
            return self.models_cache[model_name], metadata
        
        try:
            # Load model
            model = joblib.load(model_path)
            
            # Cache the loaded model
            self.models_cache[model_name] = model
            
            logger.info(f"Loaded model: {model_name} v{metadata.version}")
            return model, metadata
            
        except Exception as e:
            raise ErrorHandler.handle_model_loading_error(str(model_path), e)
    
    def get_active_model(self) -> Tuple[Any, ModelMetadata]:
        """Get the currently active model"""
        
        active_models = [
            (name, meta) for name, meta in self.metadata_cache.items() 
            if meta.is_active
        ]
        
        if not active_models:
            # Fall back to default model if no active model set
            default_model_name = settings.default_model.replace('.pkl', '')
            if default_model_name in self.metadata_cache:
                return self.load_model(default_model_name)
            else:
                raise ModelError("No active model found and default model not registered")
        
        if len(active_models) > 1:
            logger.warning(f"Multiple active models found: {[name for name, _ in active_models]}")
        
        # Use the most recently deployed model
        active_model = max(active_models, key=lambda x: x[1].deployment_date or datetime.min)
        return self.load_model(active_model[0])
    
    def set_active_model(self, model_name: str) -> bool:
        """Set a model as active"""
        
        if model_name not in self.metadata_cache:
            raise ModelError(f"Model {model_name} not found in registry")
        
        # Deactivate all other models
        for meta in self.metadata_cache.values():
            meta.is_active = False
        
        # Activate the specified model
        self.metadata_cache[model_name].is_active = True
        self.metadata_cache[model_name].deployment_date = datetime.utcnow()
        
        self._save_metadata()
        
        logger.info(f"Set active model: {model_name}")
        return True
    
    def list_models(self) -> List[ModelMetadata]:
        """List all registered models"""
        return list(self.metadata_cache.values())
    
    def get_model_info(self, model_name: str) -> ModelMetadata:
        """Get metadata for a specific model"""
        if model_name not in self.metadata_cache:
            raise ModelError(f"Model {model_name} not found in registry")
        
        return self.metadata_cache[model_name]
    
    def compare_models(self, model_names: List[str]) -> pd.DataFrame:
        """Compare performance metrics of multiple models"""
        
        comparison_data = []
        
        for model_name in model_names:
            if model_name in self.metadata_cache:
                meta = self.metadata_cache[model_name]
                comparison_data.append({
                    'model_name': meta.model_name,
                    'version': meta.version,
                    'accuracy': meta.accuracy,
                    'precision': meta.precision,
                    'recall': meta.recall,
                    'f1_score': meta.f1_score,
                    'roc_auc': meta.roc_auc,
                    'training_date': meta.training_date,
                    'is_active': meta.is_active
                })
        
        return pd.DataFrame(comparison_data)
    
    def cleanup_old_models(self, keep_versions: int = 3) -> List[str]:
        """Clean up old model versions, keeping only the most recent"""
        
        # Group models by base name
        model_groups = {}
        for name, meta in self.metadata_cache.items():
            base_name = name.split('_v')[0]  # Assuming version format: model_v1.0
            if base_name not in model_groups:
                model_groups[base_name] = []
            model_groups[base_name].append((name, meta))
        
        removed_models = []
        
        for base_name, models in model_groups.items():
            # Sort by training date (newest first)
            models.sort(key=lambda x: x[1].training_date, reverse=True)
            
            # Remove old versions beyond keep_versions
            for model_name, meta in models[keep_versions:]:
                if not meta.is_active:  # Don't remove active models
                    model_path = self.model_dir / model_name
                    if model_path.exists():
                        model_path.unlink()
                    
                    del self.metadata_cache[model_name]
                    if model_name in self.models_cache:
                        del self.models_cache[model_name]
                    
                    removed_models.append(model_name)
                    logger.info(f"Removed old model: {model_name}")
        
        self._save_metadata()
        return removed_models


# Global model manager instance
model_manager = ModelVersionManager()