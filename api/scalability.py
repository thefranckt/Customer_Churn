"""
Scalability and performance optimizations
Caching, async processing, and performance enhancements
"""

import asyncio
import time
import os
from typing import Dict, Any, Optional, List
from functools import lru_cache
import hashlib
import json
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict

import redis
from cachetools import TTLCache, LRUCache
import numpy as np

from .config import settings
from .models import CustomerFeatures, PredictionResult

logger = logging.getLogger("churn_api.scalability")


class CacheManager:
    """Intelligent caching system for predictions and model operations"""
    
    def __init__(self):
        self.memory_cache = TTLCache(maxsize=1000, ttl=settings.cache_ttl)
        self.model_cache = LRUCache(maxsize=10)  # Cache for loaded models
        self.prediction_cache = TTLCache(maxsize=5000, ttl=1800)  # 30 minutes for predictions
        
        # Try to connect to Redis for distributed caching
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache connected successfully")
        except Exception as e:
            logger.warning(f"Redis not available, using memory cache only: {e}")
            self.redis_client = None
    
    def _generate_cache_key(self, prefix: str, data: Dict[str, Any]) -> str:
        """Generate consistent cache key for data"""
        # Sort keys for consistent hashing
        sorted_data = json.dumps(data, sort_keys=True)
        hash_object = hashlib.md5(sorted_data.encode())
        return f"{prefix}:{hash_object.hexdigest()}"
    
    async def get_prediction(self, features: CustomerFeatures, model_version: str) -> Optional[PredictionResult]:
        """Get cached prediction result"""
        cache_key = self._generate_cache_key(
            f"prediction:{model_version}", 
            features.dict()
        )
        
        # Try Redis first
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    result_dict = json.loads(cached_data)
                    logger.debug(f"Cache hit (Redis): {cache_key}")
                    return PredictionResult(**result_dict)
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")
        
        # Fallback to memory cache
        if cache_key in self.prediction_cache:
            logger.debug(f"Cache hit (Memory): {cache_key}")
            return self.prediction_cache[cache_key]
        
        return None
    
    async def set_prediction(
        self, 
        features: CustomerFeatures, 
        model_version: str, 
        result: PredictionResult
    ):
        """Cache prediction result"""
        cache_key = self._generate_cache_key(
            f"prediction:{model_version}", 
            features.dict()
        )
        
        # Store in Redis
        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key, 
                    1800,  # 30 minutes TTL
                    json.dumps(result.dict(), default=str)
                )
                logger.debug(f"Cached in Redis: {cache_key}")
            except Exception as e:
                logger.warning(f"Redis cache write failed: {e}")
        
        # Store in memory cache as backup
        self.prediction_cache[cache_key] = result
        logger.debug(f"Cached in memory: {cache_key}")
    
    def get_model(self, model_name: str):
        """Get cached model"""
        return self.model_cache.get(model_name)
    
    def set_model(self, model_name: str, model):
        """Cache loaded model"""
        self.model_cache[model_name] = model
        logger.info(f"Cached model: {model_name}")
    
    def clear_cache(self, pattern: Optional[str] = None):
        """Clear cache entries"""
        if pattern and self.redis_client:
            try:
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    logger.info(f"Cleared Redis cache for pattern: {pattern}")
            except Exception as e:
                logger.error(f"Failed to clear Redis cache: {e}")
        
        # Clear memory caches
        self.memory_cache.clear()
        self.prediction_cache.clear()
        if not pattern:  # Only clear model cache if clearing all
            self.model_cache.clear()
        
        logger.info("Memory cache cleared")


class AsyncProcessingManager:
    """Manage asynchronous and background processing"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.processing_queue = asyncio.Queue()
        self.results_cache = {}
        
        logger.info(f"Async processing manager initialized with {self.max_workers} workers")
    
    async def process_batch_async(
        self, 
        customers: List[CustomerFeatures], 
        model, 
        model_metadata
    ) -> List[PredictionResult]:
        """Process batch predictions asynchronously"""
        
        # Split into chunks for parallel processing
        chunk_size = max(1, len(customers) // self.max_workers)
        chunks = [
            customers[i:i + chunk_size] 
            for i in range(0, len(customers), chunk_size)
        ]
        
        # Process chunks in parallel
        tasks = []
        for chunk in chunks:
            task = asyncio.create_task(
                self._process_chunk(chunk, model, model_metadata)
            )
            tasks.append(task)
        
        # Wait for all chunks to complete
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        all_results = []
        for result in chunk_results:
            if isinstance(result, Exception):
                logger.error(f"Chunk processing failed: {result}")
                continue
            all_results.extend(result)
        
        return all_results
    
    async def _process_chunk(
        self, 
        chunk: List[CustomerFeatures], 
        model, 
        model_metadata
    ) -> List[PredictionResult]:
        """Process a chunk of customers"""
        
        loop = asyncio.get_event_loop()
        
        # Run prediction in thread pool to avoid blocking
        def predict_chunk():
            results = []
            for customer in chunk:
                try:
                    # Convert to numpy array
                    input_data = customer.to_prediction_array()
                    
                    # Make prediction
                    probability = float(model.predict_proba(input_data)[0][1])
                    prediction = int(model.predict(input_data)[0])
                    
                    # Create result
                    result = PredictionResult(
                        churn_probability=round(probability, 4),
                        prediction=prediction,
                        interpretation="Likely to churn" if prediction == 1 else "Likely to stay",
                        model_version=model_metadata.version,
                        prediction_timestamp=datetime.utcnow()
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Chunk prediction failed for customer: {e}")
                    continue
            
            return results
        
        # Execute in thread pool
        return await loop.run_in_executor(self.executor, predict_chunk)


class ConnectionPoolManager:
    """Manage database and external service connections"""
    
    def __init__(self):
        self.connection_pools = {}
        self.max_connections = settings.max_concurrent_requests
        
    async def get_connection(self, service_name: str):
        """Get connection from pool"""
        if service_name not in self.connection_pools:
            # Create new pool for service
            # This would be implemented based on specific service requirements
            pass
        
        return self.connection_pools.get(service_name)


class RateLimiter:
    """Rate limiting for API endpoints"""
    
    def __init__(self):
        self.requests = defaultdict(list)
        self.blocked_ips = set()
        self.cleanup_interval = 60  # seconds
        self.last_cleanup = time.time()
    
    def is_allowed(self, client_ip: str, endpoint: str) -> bool:
        """Check if request is allowed based on rate limits"""
        
        if not settings.enable_rate_limiting:
            return True
        
        current_time = time.time()
        
        # Cleanup old requests periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_requests()
            self.last_cleanup = current_time
        
        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            return False
        
        # Get recent requests for this IP and endpoint
        key = f"{client_ip}:{endpoint}"
        recent_requests = self.requests[key]
        
        # Remove requests older than the window
        window_start = current_time - settings.rate_limit_window
        recent_requests[:] = [req_time for req_time in recent_requests if req_time > window_start]
        
        # Check if limit exceeded
        if len(recent_requests) >= settings.rate_limit_requests:
            logger.warning(f"Rate limit exceeded for {client_ip} on {endpoint}")
            # Optionally block the IP for repeated violations
            violation_count = len([req for req in recent_requests if req > current_time - 300])  # Last 5 minutes
            if violation_count > settings.rate_limit_requests * 2:
                self.blocked_ips.add(client_ip)
                logger.warning(f"IP blocked for repeated violations: {client_ip}")
            return False
        
        # Record this request
        recent_requests.append(current_time)
        return True
    
    def _cleanup_old_requests(self):
        """Clean up old request records"""
        current_time = time.time()
        cutoff_time = current_time - settings.rate_limit_window * 2  # Keep some extra history
        
        for key in list(self.requests.keys()):
            self.requests[key] = [
                req_time for req_time in self.requests[key] 
                if req_time > cutoff_time
            ]
            
            # Remove empty entries
            if not self.requests[key]:
                del self.requests[key]
        
        logger.debug(f"Cleaned up rate limiter, tracking {len(self.requests)} keys")


class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def cached_feature_validation(features_hash: str) -> bool:
        """Cache feature validation results"""
        # This would contain the actual validation logic
        # For now, it's a placeholder
        return True
    
    @staticmethod
    def optimize_numpy_operations():
        """Optimize NumPy operations for performance"""
        # Set optimal thread count for NumPy operations
        import os
        os.environ['OMP_NUM_THREADS'] = str(min(4, os.cpu_count() or 1))
        os.environ['MKL_NUM_THREADS'] = str(min(4, os.cpu_count() or 1))
        
        logger.info("NumPy operations optimized for performance")
    
    @staticmethod
    def precompute_common_features():
        """Precompute common feature combinations"""
        # This could precompute common customer profiles
        # and cache their predictions for faster response
        pass


class MemoryManager:
    """Monitor and manage memory usage"""
    
    def __init__(self):
        self.memory_threshold = 0.85  # 85% memory usage threshold
        self.cleanup_handlers = []
    
    def add_cleanup_handler(self, handler):
        """Add handler to be called during memory cleanup"""
        self.cleanup_handlers.append(handler)
    
    def check_memory_usage(self) -> float:
        """Check current memory usage"""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            logger.warning("psutil not available for memory monitoring")
            return 0.0
    
    def cleanup_if_needed(self):
        """Perform cleanup if memory usage is high"""
        current_usage = self.check_memory_usage()
        
        if current_usage > self.memory_threshold:
            logger.warning(f"High memory usage detected: {current_usage:.1%}")
            
            # Run cleanup handlers
            for handler in self.cleanup_handlers:
                try:
                    handler()
                except Exception as e:
                    logger.error(f"Cleanup handler failed: {e}")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("Memory cleanup completed")


# Global instances
cache_manager = CacheManager()
async_processor = AsyncProcessingManager()
rate_limiter = RateLimiter()
memory_manager = MemoryManager()

# Setup cleanup handlers
memory_manager.add_cleanup_handler(lambda: cache_manager.clear_cache())

# Optimize performance on startup
PerformanceOptimizer.optimize_numpy_operations()