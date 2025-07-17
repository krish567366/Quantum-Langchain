# Deployment Guide

Complete guide for deploying QuantumLangChain applications in production environments.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Configuration Management](#configuration-management)
3. [Container Deployment](#container-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Cloud Deployment](#cloud-deployment)

## Environment Setup

### Production Environment Requirements

**System Requirements:**

- **CPU:** 8+ cores (Intel/AMD x64 or ARM64)
- **Memory:** 32GB+ RAM (16GB minimum)
- **Storage:** 100GB+ SSD for quantum state persistence
- **Network:** High-bandwidth, low-latency connection for quantum backends
- **GPU:** Optional, CUDA-compatible for accelerated classical ML

**Python Environment:**

```bash
# Create dedicated Python environment
python -m venv quantum-langchain-prod
source quantum-langchain-prod/bin/activate  # Linux/Mac
# or
quantum-langchain-prod\Scripts\activate  # Windows

# Install production dependencies
pip install quantumlangchain[production]
pip install gunicorn uvicorn fastapi
pip install prometheus-client structlog
```

### Environment Variables

**Production Environment Configuration:**

```bash
# Core Configuration
export QUANTUMLANGCHAIN_ENV=production
export QUANTUMLANGCHAIN_LOG_LEVEL=INFO
export QUANTUMLANGCHAIN_CONFIG_PATH=/etc/quantumlangchain/config.yml

# Quantum Backend Configuration
export QISKIT_IBM_TOKEN=your_ibm_quantum_token
export PENNYLANE_DEVICE=default.qubit
export BRAKET_S3_BUCKET=your-braket-bucket

# Database Configuration
export QUANTUM_DB_URL=postgresql://user:pass@host:5432/quantum_db
export REDIS_URL=redis://redis-host:6379/0

# Security Configuration
export QUANTUM_SECRET_KEY=your-secret-key
export QUANTUM_ENCRYPTION_KEY=your-encryption-key
export JWT_SECRET_KEY=your-jwt-secret

# Monitoring Configuration
export PROMETHEUS_PORT=9090
export GRAFANA_URL=https://grafana.example.com
export SENTRY_DSN=your-sentry-dsn
```

### Dependencies and Requirements

**Production requirements.txt:**

```txt
# Core QuantumLangChain
quantumlangchain[production]==1.0.0

# Quantum Computing Backends
qiskit>=0.45.0
qiskit-aer>=0.12.0
qiskit-ibmq-provider>=0.20.0
pennylane>=0.32.0
amazon-braket-sdk>=1.50.0

# Web Framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
gunicorn>=21.2.0

# Database and Storage
sqlalchemy>=2.0.0
alembic>=1.12.0
redis>=5.0.0
psycopg2-binary>=2.9.0

# Monitoring and Logging
prometheus-client>=0.19.0
structlog>=23.2.0
sentry-sdk[fastapi]>=1.38.0

# Security
cryptography>=41.0.0
python-jose[cryptography]>=3.3.0
bcrypt>=4.1.0

# Performance
numpy>=1.24.0
scipy>=1.11.0
numba>=0.58.0
```

## Configuration Management

### Configuration Structure

**config/production.yml:**

```yaml
# QuantumLangChain Production Configuration

app:
  name: "QuantumLangChain Production"
  version: "1.0.0"
  environment: "production"
  debug: false

quantum:
  # Quantum Computing Configuration
  backends:
    primary: "qiskit"
    fallback: "pennylane"
    simulator: "qiskit_aer"
  
  qiskit:
    provider: "IBMQ"
    backend: "ibmq_qasm_simulator"
    shots: 4096
    optimization_level: 3
    max_experiments: 10
  
  pennylane:
    device: "default.qubit"
    shots: 2048
    interface: "torch"
  
  # Quantum State Management
  state:
    num_qubits: 8
    circuit_depth: 10
    decoherence_threshold: 0.05
    max_entanglements: 20
    coherence_monitoring_interval: 30

  # Resource Limits
  resources:
    max_concurrent_operations: 4
    memory_limit_gb: 8
    operation_timeout_seconds: 60
    circuit_cache_size: 1000

# Database Configuration
database:
  url: "${QUANTUM_DB_URL}"
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
  pool_recycle: 3600
  
  # Quantum State Storage
  quantum_storage:
    provider: "postgresql"
    table_prefix: "quantum_"
    compression: true
    encryption: true

# Cache Configuration
cache:
  redis:
    url: "${REDIS_URL}"
    max_connections: 50
    socket_keepalive: true
    socket_keepalive_options: {}
  
  # Quantum Circuit Cache
  circuit_cache:
    ttl_seconds: 3600
    max_size_mb: 512
    compression: true

# Security Configuration
security:
  secret_key: "${QUANTUM_SECRET_KEY}"
  encryption_key: "${QUANTUM_ENCRYPTION_KEY}"
  jwt_secret: "${JWT_SECRET_KEY}"
  
  # API Security
  api:
    rate_limit: "100/minute"
    cors_origins: ["https://app.example.com"]
    trusted_proxies: ["10.0.0.0/8"]
  
  # Quantum Operation Security
  quantum:
    enable_audit_logging: true
    require_authentication: true
    operation_timeout: 300

# Monitoring Configuration
monitoring:
  metrics:
    enabled: true
    port: 9090
    path: "/metrics"
  
  logging:
    level: "INFO"
    format: "json"
    structured: true
  
  alerting:
    enabled: true
    webhook_url: "${ALERT_WEBHOOK_URL}"
    
# Performance Tuning
performance:
  # Classical ML Optimization
  classical:
    numpy_threads: 4
    torch_threads: 4
    enable_mkl: true
  
  # Quantum Optimization
  quantum:
    circuit_optimization: true
    gate_fusion: true
    measurement_optimization: true
    
  # Memory Management
  memory:
    garbage_collection_threshold: 0.8
    quantum_state_compression: true
    classical_cache_limit_gb: 4
```

### Configuration Loading

**config_loader.py:**

```python
import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class QuantumConfig:
    """Production quantum configuration."""
    
    # Quantum parameters
    num_qubits: int = 8
    circuit_depth: int = 10
    decoherence_threshold: float = 0.05
    backend_type: str = "qiskit"
    
    # Performance settings
    shots: int = 4096
    optimization_level: int = 3
    max_concurrent_operations: int = 4
    
    # Resource limits
    memory_limit_gb: int = 8
    operation_timeout: int = 60
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'QuantumConfig':
        """Load configuration from YAML file."""
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Expand environment variables
        config_data = cls._expand_env_vars(config_data)
        
        # Extract quantum configuration
        quantum_config = config_data.get('quantum', {})
        
        return cls(
            num_qubits=quantum_config.get('state', {}).get('num_qubits', 8),
            circuit_depth=quantum_config.get('state', {}).get('circuit_depth', 10),
            decoherence_threshold=quantum_config.get('state', {}).get('decoherence_threshold', 0.05),
            backend_type=quantum_config.get('backends', {}).get('primary', 'qiskit'),
            shots=quantum_config.get('qiskit', {}).get('shots', 4096),
            optimization_level=quantum_config.get('qiskit', {}).get('optimization_level', 3),
            max_concurrent_operations=quantum_config.get('resources', {}).get('max_concurrent_operations', 4),
            memory_limit_gb=quantum_config.get('resources', {}).get('memory_limit_gb', 8),
            operation_timeout=quantum_config.get('resources', {}).get('operation_timeout_seconds', 60)
        )
    
    @staticmethod
    def _expand_env_vars(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively expand environment variables in configuration."""
        
        if isinstance(config_data, dict):
            return {k: QuantumConfig._expand_env_vars(v) for k, v in config_data.items()}
        elif isinstance(config_data, list):
            return [QuantumConfig._expand_env_vars(item) for item in config_data]
        elif isinstance(config_data, str) and config_data.startswith("${") and config_data.endswith("}"):
            env_var = config_data[2:-1]
            return os.getenv(env_var, config_data)
        else:
            return config_data

class ConfigManager:
    """Production configuration manager."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.getenv('QUANTUMLANGCHAIN_CONFIG_PATH', 'config/production.yml')
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate configuration."""
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required configuration sections
        required_sections = ['quantum', 'database', 'security']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        return config
    
    def get_quantum_config(self) -> QuantumConfig:
        """Get quantum-specific configuration."""
        return QuantumConfig.from_yaml(self.config_path)
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.config.get('database', {})
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return self.config.get('security', {})
```

## Container Deployment

### Dockerfile

**Dockerfile:**

```dockerfile
# Multi-stage build for QuantumLangChain production deployment
FROM python:3.11-slim as builder

# Install system dependencies for quantum computing
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash quantum
USER quantum
WORKDIR /home/quantum

# Copy application code
COPY --chown=quantum:quantum . /home/quantum/app/

# Set up configuration directory
RUN mkdir -p /home/quantum/config /home/quantum/logs /home/quantum/data

# Copy configuration files
COPY --chown=quantum:quantum config/ /home/quantum/config/

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["gunicorn", "--config", "config/gunicorn.conf.py", "app.main:app"]
```

### Docker Compose

**docker-compose.production.yml:**

```yaml
version: '3.8'

services:
  quantumlangchain:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    
    container_name: quantumlangchain-app
    
    environment:
      - QUANTUMLANGCHAIN_ENV=production
      - QUANTUMLANGCHAIN_CONFIG_PATH=/home/quantum/config/production.yml
      - QUANTUM_DB_URL=postgresql://quantum:${DB_PASSWORD}@postgres:5432/quantum_db
      - REDIS_URL=redis://redis:6379/0
    
    ports:
      - "8000:8000"
      - "9090:9090"
    
    volumes:
      - quantum_data:/home/quantum/data
      - quantum_logs:/home/quantum/logs
      - ./config:/home/quantum/config:ro
    
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    
    restart: unless-stopped
    
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
        reservations:
          memory: 4G
          cpus: '2.0'

  postgres:
    image: postgres:15-alpine
    container_name: quantumlangchain-postgres
    
    environment:
      - POSTGRES_DB=quantum_db
      - POSTGRES_USER=quantum
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    
    ports:
      - "5432:5432"
    
    restart: unless-stopped
    
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U quantum -d quantum_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: quantumlangchain-redis
    
    command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
    
    volumes:
      - redis_data:/data
    
    ports:
      - "6379:6379"
    
    restart: unless-stopped
    
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    container_name: quantumlangchain-prometheus
    
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    
    ports:
      - "9091:9090"
    
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: quantumlangchain-grafana
    
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    
    ports:
      - "3000:3000"
    
    restart: unless-stopped

volumes:
  quantum_data:
  quantum_logs:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: quantumlangchain-network
```

### Gunicorn Configuration

**config/gunicorn.conf.py:**

```python
"""Gunicorn configuration for QuantumLangChain production deployment."""

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = min(multiprocessing.cpu_count() * 2 + 1, 8)  # Cap at 8 workers
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Timeout settings
timeout = 120  # Longer timeout for quantum operations
keepalive = 2
graceful_timeout = 30

# Memory management
preload_app = True
max_requests_jitter = 100

# Process naming
proc_name = "quantumlangchain"

# Logging
accesslog = "/home/quantum/logs/access.log"
errorlog = "/home/quantum/logs/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# SSL (if using HTTPS)
if os.getenv('SSL_CERT_PATH') and os.getenv('SSL_KEY_PATH'):
    keyfile = os.getenv('SSL_KEY_PATH')
    certfile = os.getenv('SSL_CERT_PATH')
    ssl_version = 2
    ciphers = 'TLSv1.2'

# Worker process hooks
def when_ready(server):
    """Called when the server is ready to accept connections."""
    server.log.info("QuantumLangChain server is ready. Workers: %s", server.cfg.workers)

def worker_int(worker):
    """Called when a worker receives the INT or QUIT signal."""
    worker.log.info("Worker received INT or QUIT signal, shutting down gracefully")

def pre_fork(server, worker):
    """Called before a worker is forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    """Called after a worker has been forked."""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def worker_abort(worker):
    """Called when a worker times out."""
    worker.log.warning("Worker timeout (pid: %s)", worker.pid)
```

## Kubernetes Deployment

### Kubernetes Manifests

**k8s/namespace.yaml:**

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: quantumlangchain
  labels:
    name: quantumlangchain
    environment: production
```

**k8s/configmap.yaml:**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: quantumlangchain-config
  namespace: quantumlangchain
data:
  production.yml: |
    app:
      name: "QuantumLangChain Production"
      version: "1.0.0"
      environment: "production"
    
    quantum:
      backends:
        primary: "qiskit"
        fallback: "pennylane"
      
      state:
        num_qubits: 8
        circuit_depth: 10
        decoherence_threshold: 0.05
      
      resources:
        max_concurrent_operations: 4
        memory_limit_gb: 8
        operation_timeout_seconds: 60
    
    database:
      url: "postgresql://quantum:${QUANTUM_DB_PASSWORD}@postgres-service:5432/quantum_db"
      pool_size: 20
    
    cache:
      redis:
        url: "redis://redis-service:6379/0"
    
    monitoring:
      metrics:
        enabled: true
        port: 9090
      
      logging:
        level: "INFO"
        format: "json"
```

**k8s/secret.yaml:**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: quantumlangchain-secrets
  namespace: quantumlangchain
type: Opaque
data:
  quantum-secret-key: <base64-encoded-secret>
  quantum-encryption-key: <base64-encoded-encryption-key>
  jwt-secret-key: <base64-encoded-jwt-secret>
  quantum-db-password: <base64-encoded-db-password>
  qiskit-ibm-token: <base64-encoded-ibm-token>
```

**k8s/deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantumlangchain-app
  namespace: quantumlangchain
  labels:
    app: quantumlangchain
    component: app
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  
  selector:
    matchLabels:
      app: quantumlangchain
      component: app
  
  template:
    metadata:
      labels:
        app: quantumlangchain
        component: app
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    
    spec:
      containers:
      - name: quantumlangchain
        image: quantumlangchain:1.0.0
        imagePullPolicy: Always
        
        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        - name: metrics
          containerPort: 9090
          protocol: TCP
        
        env:
        - name: QUANTUMLANGCHAIN_ENV
          value: "production"
        - name: QUANTUMLANGCHAIN_CONFIG_PATH
          value: "/etc/quantumlangchain/production.yml"
        - name: QUANTUM_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: quantumlangchain-secrets
              key: quantum-secret-key
        - name: QUANTUM_ENCRYPTION_KEY
          valueFrom:
            secretKeyRef:
              name: quantumlangchain-secrets
              key: quantum-encryption-key
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: quantumlangchain-secrets
              key: jwt-secret-key
        - name: QUANTUM_DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: quantumlangchain-secrets
              key: quantum-db-password
        - name: QISKIT_IBM_TOKEN
          valueFrom:
            secretKeyRef:
              name: quantumlangchain-secrets
              key: qiskit-ibm-token
        
        volumeMounts:
        - name: config-volume
          mountPath: /etc/quantumlangchain
          readOnly: true
        - name: data-volume
          mountPath: /home/quantum/data
        - name: logs-volume
          mountPath: /home/quantum/logs
        
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
      
      volumes:
      - name: config-volume
        configMap:
          name: quantumlangchain-config
      - name: data-volume
        persistentVolumeClaim:
          claimName: quantumlangchain-data-pvc
      - name: logs-volume
        persistentVolumeClaim:
          claimName: quantumlangchain-logs-pvc
      
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - quantumlangchain
              topologyKey: kubernetes.io/hostname
```

**k8s/service.yaml:**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: quantumlangchain-service
  namespace: quantumlangchain
  labels:
    app: quantumlangchain
    component: app
spec:
  type: ClusterIP
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
  selector:
    app: quantumlangchain
    component: app
```

**k8s/ingress.yaml:**

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: quantumlangchain-ingress
  namespace: quantumlangchain
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.quantumlangchain.example.com
    secretName: quantumlangchain-tls
  
  rules:
  - host: api.quantumlangchain.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: quantumlangchain-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler

**k8s/hpa.yaml:**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: quantumlangchain-hpa
  namespace: quantumlangchain
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: quantumlangchain-app
  
  minReplicas: 3
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
  
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

## Cloud Deployment

### AWS Deployment

**terraform/aws/main.tf:**

```hcl
# AWS Infrastructure for QuantumLangChain

provider "aws" {
  region = var.aws_region
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "quantumlangchain-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true
  
  tags = {
    Environment = "production"
    Project     = "quantumlangchain"
  }
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = "quantumlangchain-eks"
  cluster_version = "1.28"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  enable_irsa = true
  
  node_groups = {
    quantum_nodes = {
      desired_capacity = 3
      max_capacity     = 10
      min_capacity     = 3
      
      instance_types = ["m5.2xlarge"]
      
      k8s_labels = {
        Environment = "production"
        NodeGroup   = "quantum-compute"
      }
      
      additional_tags = {
        ExtraTag = "quantum-langchain"
      }
    }
  }
  
  tags = {
    Environment = "production"
    Project     = "quantumlangchain"
  }
}

# RDS for PostgreSQL
resource "aws_db_instance" "quantum_db" {
  identifier = "quantumlangchain-db"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.r6g.xlarge"
  
  allocated_storage     = 200
  max_allocated_storage = 1000
  storage_type          = "gp3"
  storage_encrypted     = true
  
  db_name  = "quantum_db"
  username = "quantum"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.quantum.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "quantumlangchain-final-snapshot"
  
  tags = {
    Name        = "quantumlangchain-db"
    Environment = "production"
  }
}

# ElastiCache for Redis
resource "aws_elasticache_subnet_group" "quantum" {
  name       = "quantumlangchain-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_replication_group" "quantum_cache" {
  replication_group_id       = "quantumlangchain-cache"
  description                = "QuantumLangChain Redis Cache"
  
  node_type          = "cache.r6g.large"
  port               = 6379
  parameter_group_name = "default.redis7"
  
  num_cache_clusters = 3
  
  subnet_group_name  = aws_elasticache_subnet_group.quantum.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  automatic_failover_enabled = true
  multi_az_enabled          = true
  
  tags = {
    Name        = "quantumlangchain-cache"
    Environment = "production"
  }
}

# Application Load Balancer
resource "aws_lb" "quantum_alb" {
  name               = "quantumlangchain-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = module.vpc.public_subnets
  
  enable_deletion_protection = true
  
  tags = {
    Name        = "quantumlangchain-alb"
    Environment = "production"
  }
}

# S3 Bucket for Quantum State Storage
resource "aws_s3_bucket" "quantum_storage" {
  bucket = "quantumlangchain-quantum-states-${random_id.bucket_suffix.hex}"
  
  tags = {
    Name        = "quantumlangchain-quantum-storage"
    Environment = "production"
  }
}

resource "aws_s3_bucket_encryption_configuration" "quantum_storage" {
  bucket = aws_s3_bucket.quantum_storage.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "quantum_storage" {
  bucket = aws_s3_bucket.quantum_storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# Security Groups
resource "aws_security_group" "rds" {
  name_prefix = "quantumlangchain-rds-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "quantumlangchain-rds-sg"
  }
}

resource "aws_security_group" "redis" {
  name_prefix = "quantumlangchain-redis-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "quantumlangchain-redis-sg"
  }
}

resource "aws_security_group" "alb" {
  name_prefix = "quantumlangchain-alb-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "quantumlangchain-alb-sg"
  }
}

# Outputs
output "eks_cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "rds_endpoint" {
  value = aws_db_instance.quantum_db.endpoint
}

output "redis_endpoint" {
  value = aws_elasticache_replication_group.quantum_cache.configuration_endpoint_address
}

output "alb_dns_name" {
  value = aws_lb.quantum_alb.dns_name
}
```

This comprehensive deployment guide covers all aspects of production deployment for QuantumLangChain applications, from basic environment setup to advanced cloud infrastructure with monitoring and security.
