"""
Security enhancements for Customer Churn Prediction API
Authentication, authorization, and security middleware
"""

import hashlib
import hmac
import jwt
import time
import secrets
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from functools import wraps

from fastapi import HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
import logging
import numpy as np

from .config import settings

logger = logging.getLogger("churn_api.security")

# Security configuration
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


class SecurityConfig:
    """Security configuration and secrets management"""
    
    def __init__(self):
        # Generate secure secret key if not provided
        self.secret_key = os.getenv("SECRET_KEY", self._generate_secret_key())
        self.api_keys = self._load_api_keys()
        self.trusted_ips = set(os.getenv("TRUSTED_IPS", "").split(",")) if os.getenv("TRUSTED_IPS") else set()
        
        # Security headers
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        }
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key"""
        return secrets.token_urlsafe(32)
    
    def _load_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load API keys from environment or configuration"""
        # In production, this would load from a secure key management system
        api_keys = {}
        
        # Example API keys (in production, load from secure storage)
        demo_keys = {
            "demo_key_123": {
                "name": "Demo Client",
                "permissions": ["predict", "batch_predict"],
                "rate_limit": 1000,
                "expires": None
            }
        }
        
        return demo_keys
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return associated metadata"""
        key_info = self.api_keys.get(api_key)
        
        if not key_info:
            return None
        
        # Check expiration
        if key_info.get("expires") and datetime.utcnow() > key_info["expires"]:
            logger.warning(f"Expired API key used: {api_key[:8]}...")
            return None
        
        return key_info


class TokenManager:
    """JWT token management"""
    
    def __init__(self, security_config: SecurityConfig):
        self.security_config = security_config
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "iat": datetime.utcnow()})
        
        encoded_jwt = jwt.encode(
            to_encode, 
            self.security_config.secret_key, 
            algorithm=ALGORITHM
        )
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.security_config.secret_key, 
                algorithms=[ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Expired token used")
            return None
        except jwt.JWTError as e:
            logger.warning(f"Invalid token: {e}")
            return None


class AuthenticationService:
    """Handle authentication and authorization"""
    
    def __init__(self):
        self.security_config = SecurityConfig()
        self.token_manager = TokenManager(self.security_config)
        self.bearer_scheme = HTTPBearer(auto_error=False)
        
        # Failed attempt tracking
        self.failed_attempts = {}
        self.blocked_ips = set()
    
    async def verify_api_key(self, request: Request) -> Optional[Dict[str, Any]]:
        """Verify API key from request headers"""
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            return None
        
        key_info = self.security_config.validate_api_key(api_key)
        if key_info:
            logger.info(f"Valid API key used: {key_info['name']}")
            return key_info
        
        # Track failed attempts
        client_ip = self._get_client_ip(request)
        self._track_failed_attempt(client_ip)
        
        return None
    
    async def verify_bearer_token(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))) -> Optional[Dict[str, Any]]:
        """Verify Bearer token"""
        if not credentials:
            return None
        
        payload = self.token_manager.verify_token(credentials.credentials)
        if payload:
            return payload
        
        return None
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request"""
        # Check for forwarded IP first (for reverse proxy setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        # Check for real IP (some proxy configurations)
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        return request.client.host if request.client else "unknown"
    
    def _track_failed_attempt(self, client_ip: str):
        """Track failed authentication attempts"""
        current_time = time.time()
        
        if client_ip not in self.failed_attempts:
            self.failed_attempts[client_ip] = []
        
        # Add current attempt
        self.failed_attempts[client_ip].append(current_time)
        
        # Clean old attempts (last 15 minutes)
        cutoff_time = current_time - 900
        self.failed_attempts[client_ip] = [
            attempt_time for attempt_time in self.failed_attempts[client_ip]
            if attempt_time > cutoff_time
        ]
        
        # Block IP if too many failed attempts
        if len(self.failed_attempts[client_ip]) > 10:  # 10 failed attempts in 15 minutes
            self.blocked_ips.add(client_ip)
            logger.warning(f"IP blocked due to repeated authentication failures: {client_ip}")
    
    def is_ip_blocked(self, client_ip: str) -> bool:
        """Check if IP is blocked"""
        return client_ip in self.blocked_ips


class SecurityMiddleware:
    """Security middleware for request processing"""
    
    def __init__(self, auth_service: AuthenticationService):
        self.auth_service = auth_service
    
    async def process_request(self, request: Request) -> Optional[HTTPException]:
        """Process request for security checks"""
        
        client_ip = self.auth_service._get_client_ip(request)
        
        # Check if IP is blocked
        if self.auth_service.is_ip_blocked(client_ip):
            logger.warning(f"Blocked IP attempted access: {client_ip}")
            return HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Check IP whitelist for sensitive endpoints
        if self._is_sensitive_endpoint(request.url.path):
            if self.auth_service.security_config.trusted_ips:
                if client_ip not in self.auth_service.security_config.trusted_ips:
                    logger.warning(f"Untrusted IP accessing sensitive endpoint: {client_ip}")
                    return HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Access denied from untrusted IP"
                    )
        
        # Validate request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
            return HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Request too large"
            )
        
        return None
    
    def _is_sensitive_endpoint(self, path: str) -> bool:
        """Check if endpoint is sensitive and requires additional security"""
        sensitive_endpoints = ["/model/info", "/metrics", "/health"]
        return any(path.startswith(endpoint) for endpoint in sensitive_endpoints)


class InputSanitizer:
    """Sanitize and validate input data"""
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 100) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            return ""
        
        # Remove potentially dangerous characters
        sanitized = "".join(char for char in value if char.isprintable())
        
        # Limit length
        return sanitized[:max_length]
    
    @staticmethod
    def validate_numeric_input(value: float, min_val: float = None, max_val: float = None) -> bool:
        """Validate numeric input for safety"""
        if not isinstance(value, (int, float)):
            return False
        
        # Check for NaN or infinity
        if not np.isfinite(value):
            return False
        
        # Check bounds
        if min_val is not None and value < min_val:
            return False
        
        if max_val is not None and value > max_val:
            return False
        
        return True


class AuditLogger:
    """Security audit logging"""
    
    def __init__(self):
        self.logger = logging.getLogger("security_audit")
    
    def log_authentication_success(self, client_ip: str, user_info: Dict[str, Any]):
        """Log successful authentication"""
        self.logger.info(f"Authentication success - IP: {client_ip}, User: {user_info.get('name', 'unknown')}")
    
    def log_authentication_failure(self, client_ip: str, reason: str):
        """Log failed authentication"""
        self.logger.warning(f"Authentication failure - IP: {client_ip}, Reason: {reason}")
    
    def log_prediction_request(self, client_ip: str, user_info: Dict[str, Any], endpoint: str):
        """Log prediction requests for audit trail"""
        self.logger.info(f"Prediction request - IP: {client_ip}, User: {user_info.get('name', 'unknown')}, Endpoint: {endpoint}")
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security events"""
        self.logger.warning(f"Security event - Type: {event_type}, Details: {details}")


def require_auth(permissions: List[str] = None):
    """Decorator to require authentication for endpoints"""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            auth_service = AuthenticationService()
            
            # Check for API key first
            api_key_info = await auth_service.verify_api_key(request)
            if api_key_info:
                # Check permissions
                if permissions:
                    user_permissions = api_key_info.get("permissions", [])
                    if not any(perm in user_permissions for perm in permissions):
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="Insufficient permissions"
                        )
                
                # Add user info to request state
                request.state.user = api_key_info
                return await func(request, *args, **kwargs)
            
            # Check for Bearer token
            try:
                credentials = await auth_service.bearer_scheme(request)
                if credentials:
                    token_payload = await auth_service.verify_bearer_token(credentials)
                    if token_payload:
                        request.state.user = token_payload
                        return await func(request, *args, **kwargs)
            except Exception:
                pass
            
            # No valid authentication found
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return wrapper
    return decorator


def add_security_headers(response, security_config: SecurityConfig):
    """Add security headers to response"""
    for header, value in security_config.security_headers.items():
        response.headers[header] = value
    
    return response


# Global instances
auth_service = AuthenticationService()
security_middleware = SecurityMiddleware(auth_service)
audit_logger = AuditLogger()
input_sanitizer = InputSanitizer()