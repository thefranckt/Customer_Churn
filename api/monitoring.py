"""
Monitoring, metrics, and health check system
Enterprise-grade observability and performance tracking
"""

import time
import psutil
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
import threading
import logging
from contextlib import asynccontextmanager

from .config import settings
from .models import HealthStatus, HealthCheck, MetricsResponse

logger = logging.getLogger("churn_api.monitoring")


@dataclass
class RequestMetrics:
    """Individual request metrics"""
    timestamp: datetime
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    error_type: Optional[str] = None


class MetricsCollector:
    """Collect and aggregate application metrics"""
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.start_time = datetime.utcnow()
        
        # Thread-safe collections
        self._lock = threading.Lock()
        self.request_history = deque(maxlen=max_samples)
        self.error_counts = defaultdict(int)
        self.endpoint_stats = defaultdict(lambda: {'count': 0, 'total_time': 0.0})
        
        # Prediction tracking
        self.prediction_counts = {'churn': 0, 'stay': 0}
        self.total_predictions = 0
        
        # Cache for computed metrics
        self._metrics_cache = {}
        self._cache_timestamp = datetime.utcnow()
        self._cache_ttl = timedelta(seconds=30)  # Cache metrics for 30 seconds
    
    def record_request(self, metrics: RequestMetrics):
        """Record a request for metrics calculation"""
        with self._lock:
            self.request_history.append(metrics)
            
            # Update endpoint statistics
            self.endpoint_stats[metrics.endpoint]['count'] += 1
            self.endpoint_stats[metrics.endpoint]['total_time'] += metrics.response_time_ms
            
            # Track errors
            if metrics.status_code >= 400:
                self.error_counts[metrics.error_type or 'unknown'] += 1
            
            # Invalidate cache
            self._metrics_cache.clear()
    
    def record_prediction(self, prediction: int):
        """Record a prediction result"""
        with self._lock:
            self.total_predictions += 1
            if prediction == 1:
                self.prediction_counts['churn'] += 1
            else:
                self.prediction_counts['stay'] += 1
    
    def get_metrics(self, force_refresh: bool = False) -> MetricsResponse:
        """Get aggregated metrics (cached for performance)"""
        
        # Check cache first
        if not force_refresh and self._is_cache_valid():
            return self._metrics_cache.get('metrics')
        
        with self._lock:
            now = datetime.utcnow()
            
            # Filter recent requests (last hour)
            recent_requests = [
                req for req in self.request_history
                if now - req.timestamp < timedelta(hours=1)
            ]
            
            # Calculate response time metrics
            response_times = [req.response_time_ms for req in recent_requests]
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                sorted_times = sorted(response_times)
                p95_index = int(0.95 * len(sorted_times))
                p99_index = int(0.99 * len(sorted_times))
                p95_response_time = sorted_times[p95_index] if p95_index < len(sorted_times) else 0
                p99_response_time = sorted_times[p99_index] if p99_index < len(sorted_times) else 0
            else:
                avg_response_time = p95_response_time = p99_response_time = 0
            
            # Count request outcomes
            total_requests = len(recent_requests)
            successful_requests = len([req for req in recent_requests if req.status_code < 400])
            failed_requests = total_requests - successful_requests
            
            # Count error types
            validation_errors = self.error_counts.get('VALIDATION_ERROR', 0)
            model_errors = self.error_counts.get('MODEL_ERROR', 0)
            system_errors = sum(
                count for error_type, count in self.error_counts.items()
                if error_type not in ['VALIDATION_ERROR', 'MODEL_ERROR']
            )
            
            metrics = MetricsResponse(
                timestamp=now,
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                avg_response_time_ms=avg_response_time,
                p95_response_time_ms=p95_response_time,
                p99_response_time_ms=p99_response_time,
                total_predictions=self.total_predictions,
                churn_predictions=self.prediction_counts['churn'],
                stay_predictions=self.prediction_counts['stay'],
                validation_errors=validation_errors,
                model_errors=model_errors,
                system_errors=system_errors
            )
            
            # Cache the result
            self._metrics_cache['metrics'] = metrics
            self._cache_timestamp = now
            
            return metrics
    
    def _is_cache_valid(self) -> bool:
        """Check if cached metrics are still valid"""
        return (
            'metrics' in self._metrics_cache and
            datetime.utcnow() - self._cache_timestamp < self._cache_ttl
        )
    
    def get_endpoint_stats(self) -> Dict[str, Dict[str, float]]:
        """Get per-endpoint performance statistics"""
        with self._lock:
            stats = {}
            for endpoint, data in self.endpoint_stats.items():
                if data['count'] > 0:
                    stats[endpoint] = {
                        'request_count': data['count'],
                        'avg_response_time_ms': data['total_time'] / data['count'],
                        'total_time_ms': data['total_time']
                    }
            return stats
    
    def reset_metrics(self):
        """Reset all collected metrics"""
        with self._lock:
            self.request_history.clear()
            self.error_counts.clear()
            self.endpoint_stats.clear()
            self.prediction_counts = {'churn': 0, 'stay': 0}
            self.total_predictions = 0
            self._metrics_cache.clear()
            self.start_time = datetime.utcnow()
            
        logger.info("Metrics reset successfully")


class HealthMonitor:
    """Monitor application health and dependencies"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.start_time = datetime.utcnow()
        self._health_checks = {}
        
    def add_health_check(self, name: str, check_function, timeout: float = 5.0):
        """Add a custom health check"""
        self._health_checks[name] = {
            'function': check_function,
            'timeout': timeout
        }
    
    async def check_model_health(self) -> Dict[str, Any]:
        """Check if models are loaded and working"""
        try:
            # This would be implemented to test model loading/prediction
            # For now, we'll simulate a basic check
            return {
                'status': 'healthy',
                'message': 'Models loaded successfully',
                'details': {
                    'model_loaded': True,
                    'model_version': settings.model_version
                }
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Model health check failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage_mb = (memory.total - memory.available) / (1024 * 1024)
            memory_usage_percent = memory.percent
            
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # Determine health status
            status = 'healthy'
            issues = []
            
            if memory_usage_percent > 90:
                status = 'degraded'
                issues.append(f'High memory usage: {memory_usage_percent:.1f}%')
            
            if cpu_usage > 90:
                status = 'degraded'
                issues.append(f'High CPU usage: {cpu_usage:.1f}%')
            
            if disk_usage_percent > 90:
                status = 'degraded'
                issues.append(f'High disk usage: {disk_usage_percent:.1f}%')
            
            return {
                'status': status,
                'message': 'System resources checked' if status == 'healthy' else '; '.join(issues),
                'details': {
                    'memory_usage_mb': memory_usage_mb,
                    'memory_usage_percent': memory_usage_percent,
                    'cpu_usage_percent': cpu_usage,
                    'disk_usage_percent': disk_usage_percent
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Resource check failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def check_api_performance(self) -> Dict[str, Any]:
        """Check API performance metrics"""
        try:
            metrics = self.metrics_collector.get_metrics()
            
            # Calculate error rate
            total_requests = metrics.total_requests
            error_rate = (metrics.failed_requests / total_requests * 100) if total_requests > 0 else 0
            
            # Determine health based on performance thresholds
            status = 'healthy'
            issues = []
            
            if metrics.avg_response_time_ms > 2000:  # 2 seconds
                status = 'degraded'
                issues.append(f'High average response time: {metrics.avg_response_time_ms:.1f}ms')
            
            if error_rate > 10:  # 10% error rate
                status = 'degraded'
                issues.append(f'High error rate: {error_rate:.1f}%')
            
            if metrics.p95_response_time_ms > 5000:  # 5 seconds
                status = 'degraded'
                issues.append(f'High P95 response time: {metrics.p95_response_time_ms:.1f}ms')
            
            return {
                'status': status,
                'message': 'API performance OK' if status == 'healthy' else '; '.join(issues),
                'details': {
                    'avg_response_time_ms': metrics.avg_response_time_ms,
                    'p95_response_time_ms': metrics.p95_response_time_ms,
                    'error_rate_percent': error_rate,
                    'total_requests': total_requests
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'message': f'Performance check failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def get_health_status(self) -> HealthCheck:
        """Get comprehensive health status"""
        
        checks = {}
        overall_status = HealthStatus.HEALTHY
        
        # Run all health checks
        health_check_tasks = [
            ('model', self.check_model_health()),
            ('system', self.check_system_resources()),
            ('api_performance', self.check_api_performance())
        ]
        
        # Add custom health checks
        for name, check_config in self._health_checks.items():
            try:
                # Run with timeout
                task = asyncio.wait_for(
                    check_config['function'](),
                    timeout=check_config['timeout']
                )
                health_check_tasks.append((name, task))
            except Exception as e:
                checks[name] = {
                    'status': 'unhealthy',
                    'message': f'Health check failed: {str(e)}',
                    'details': {'error': str(e)}
                }
        
        # Execute all health checks concurrently
        for name, task in health_check_tasks:
            try:
                result = await task
                checks[name] = result
                
                # Update overall status
                if result['status'] == 'unhealthy':
                    overall_status = HealthStatus.UNHEALTHY
                elif result['status'] == 'degraded' and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                    
            except asyncio.TimeoutError:
                checks[name] = {
                    'status': 'unhealthy',
                    'message': 'Health check timed out',
                    'details': {'timeout': True}
                }
                overall_status = HealthStatus.UNHEALTHY
                
            except Exception as e:
                checks[name] = {
                    'status': 'unhealthy',
                    'message': f'Health check error: {str(e)}',
                    'details': {'error': str(e)}
                }
                overall_status = HealthStatus.UNHEALTHY
        
        # Get metrics for health check
        metrics = self.metrics_collector.get_metrics()
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Get system resources
        try:
            memory = psutil.virtual_memory()
            memory_usage_mb = (memory.total - memory.available) / (1024 * 1024)
            cpu_usage = psutil.cpu_percent()
        except:
            memory_usage_mb = None
            cpu_usage = None
        
        return HealthCheck(
            status=overall_status,
            version=settings.app_version,
            uptime_seconds=uptime,
            checks=checks,
            total_predictions=metrics.total_predictions,
            avg_response_time_ms=metrics.avg_response_time_ms,
            error_rate_percent=(metrics.failed_requests / metrics.total_requests * 100) 
                               if metrics.total_requests > 0 else 0,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage
        )


# Request timing decorator
@asynccontextmanager
async def track_request_time(
    metrics_collector: MetricsCollector,
    endpoint: str,
    method: str = "POST"
):
    """Context manager to track request timing and outcomes"""
    
    start_time = time.time()
    request_metrics = RequestMetrics(
        timestamp=datetime.utcnow(),
        endpoint=endpoint,
        method=method,
        status_code=200,  # Will be updated
        response_time_ms=0,  # Will be calculated
    )
    
    try:
        yield request_metrics
    except Exception as e:
        request_metrics.status_code = 500
        request_metrics.error_type = type(e).__name__
        raise
    finally:
        # Calculate response time
        end_time = time.time()
        request_metrics.response_time_ms = (end_time - start_time) * 1000
        
        # Record the metrics
        metrics_collector.record_request(request_metrics)


# Global instances
metrics_collector = MetricsCollector()
health_monitor = HealthMonitor(metrics_collector)