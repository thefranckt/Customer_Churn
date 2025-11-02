# Production Deployment Guide

This guide covers deploying the Customer Churn Prediction API in a production environment with enterprise-grade features.

## 🏗️ Architecture Overview

The enhanced API now includes:

- **Configuration Management**: Centralized settings with environment variable support
- **Error Handling**: Comprehensive exception handling with structured error responses
- **Input Validation**: Business rule validation with detailed error reporting
- **Model Versioning**: Advanced model lifecycle management with metadata tracking
- **Monitoring**: Health checks, metrics collection, and performance tracking
- **Scalability**: Caching, async processing, and connection pooling
- **Security**: Authentication, rate limiting, and security headers

## 🚀 Quick Start (Development)

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   # Create .env file
   echo "CHURN_API_LOG_LEVEL=INFO" > .env
   echo "CHURN_API_DEBUG=true" >> .env
   ```

3. **Run the API**
   ```bash
   uvicorn api.deployment_api:api --reload
   ```

## 🏭 Production Deployment

### Prerequisites

- Docker and Docker Compose
- Redis (for caching and session management)
- Load balancer (nginx, HAProxy, or cloud load balancer)
- SSL/TLS certificates
- Monitoring infrastructure (Prometheus, Grafana)

### Environment Configuration

Create a production environment file:

```bash
# .env.production
CHURN_API_LOG_LEVEL=INFO
CHURN_API_DEBUG=false
CHURN_API_SECRET_KEY=your-super-secure-secret-key-here
CHURN_API_ENABLE_RATE_LIMITING=true
CHURN_API_RATE_LIMIT_REQUESTS=100
CHURN_API_RATE_LIMIT_WINDOW=3600
REDIS_HOST=redis
REDIS_PORT=6379
TRUSTED_IPS=10.0.0.0/8,172.16.0.0/12,192.168.0.0/16
```

### Docker Compose Setup

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CHURN_API_LOG_LEVEL=INFO
      - CHURN_API_DEBUG=false
      - REDIS_HOST=redis
    depends_on:
      - redis
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis_data:/data
    deploy:
      resources:
        limits:
          memory: 256M
          cpus: '0.1'

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped

volumes:
  redis_data:
```

### Nginx Configuration

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        server api:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

    server {
        listen 80;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
        
        location / {
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://api_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Security headers
            add_header X-Frame-Options DENY;
            add_header X-Content-Type-Options nosniff;
            add_header X-XSS-Protection "1; mode=block";
            add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
        }

        location /health {
            access_log off;
            proxy_pass http://api_backend;
        }
    }
}
```

## 🔐 Security Configuration

### API Key Management

1. **Generate API Keys**
   ```python
   import secrets
   api_key = secrets.token_urlsafe(32)
   ```

2. **Configure API Keys** in your environment:
   ```bash
   # Add to environment variables or secure key store
   API_KEYS='{"key123": {"name": "Production Client", "permissions": ["predict", "batch_predict"]}}'
   ```

### SSL/TLS Setup

1. **Obtain SSL certificates** (Let's Encrypt recommended):
   ```bash
   certbot certonly --webroot -w /var/www/html -d your-domain.com
   ```

2. **Configure certificate paths** in nginx configuration

### Firewall Configuration

```bash
# Allow only necessary ports
ufw allow 22    # SSH
ufw allow 80    # HTTP (redirect to HTTPS)
ufw allow 443   # HTTPS
ufw enable
```

## 📊 Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'churn-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Grafana Dashboard

Import the provided dashboard configuration for monitoring:
- API response times
- Error rates
- Prediction volumes
- System resources
- Cache hit rates

### Log Management

Configure log aggregation:

```yaml
# docker-compose.logging.yml
version: '3.8'
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
    volumes:
      - ./logs:/app/logs
```

## 🔧 Performance Tuning

### Database Optimization

For production workloads, consider:

1. **Connection Pooling**: Configure appropriate pool sizes
2. **Query Optimization**: Monitor and optimize slow queries
3. **Indexing**: Ensure proper indexing for lookup operations

### Caching Strategy

1. **Redis Configuration**:
   ```redis
   # redis.conf
   maxmemory 256mb
   maxmemory-policy allkeys-lru
   save 900 1
   save 300 10
   save 60 10000
   ```

2. **Cache Tuning**:
   - Adjust TTL values based on your use case
   - Monitor cache hit rates
   - Consider cache warming strategies

### Auto-scaling

Configure horizontal pod autoscaling (Kubernetes):

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: churn-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: churn-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 🚨 Incident Response

### Health Check Endpoints

- **Basic Health**: `GET /health`
- **Detailed Metrics**: `GET /metrics`
- **Model Status**: `GET /model/info`

### Troubleshooting

1. **High Response Times**:
   - Check cache hit rates
   - Monitor database performance
   - Verify resource allocation

2. **High Error Rates**:
   - Check application logs
   - Verify model availability
   - Review input validation errors

3. **Memory Issues**:
   - Monitor model cache size
   - Check for memory leaks
   - Adjust cache TTL settings

### Backup and Recovery

1. **Model Backups**:
   ```bash
   # Backup models directory
   tar -czf models-backup-$(date +%Y%m%d).tar.gz models/
   ```

2. **Configuration Backups**:
   ```bash
   # Backup configuration
   cp .env.production .env.backup-$(date +%Y%m%d)
   ```

## 📈 Scaling Considerations

### Horizontal Scaling

1. **Load Balancing**: Use multiple API instances
2. **Session Affinity**: Not required (stateless design)
3. **Database Scaling**: Consider read replicas for model metadata

### Vertical Scaling

1. **Memory**: Increase for larger model caches
2. **CPU**: Increase for higher prediction throughput
3. **Storage**: Monitor log and cache storage needs

### Geographic Distribution

For global deployments:

1. **CDN**: Use CDN for static content
2. **Regional Deployments**: Deploy in multiple regions
3. **Data Locality**: Consider data residency requirements

## 🔍 Maintenance

### Regular Tasks

1. **Model Updates**: Regular model retraining and deployment
2. **Dependency Updates**: Keep dependencies current
3. **Security Patches**: Apply security updates promptly
4. **Log Rotation**: Implement log rotation policies

### Monitoring Checklist

- [ ] API response times < 500ms (95th percentile)
- [ ] Error rate < 1%
- [ ] Cache hit rate > 80%
- [ ] Memory usage < 80%
- [ ] CPU usage < 70%
- [ ] Disk usage < 80%
- [ ] SSL certificate validity > 30 days

## 📞 Support

For production support:

1. **Documentation**: Refer to API documentation at `/docs`
2. **Logs**: Check application logs in `/app/logs`
3. **Metrics**: Monitor dashboard for performance metrics
4. **Health Checks**: Use automated health monitoring

---

This production setup provides enterprise-grade reliability, security, and scalability for your Customer Churn Prediction API.