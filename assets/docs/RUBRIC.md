# Final Course Project Rubric Assessment
## Building an MLOps System

**Project**: Toxic Comment Analysis - AWS Deployment  
**Dataset**: Jigsaw Toxic Comment Classification
**Due Date**: August 26th, 2025

---

## Project Overview
- [x] **Problem Selection**: Toxic Comment Moderation
  - **Implementation**: Multi-label classification system to detect toxic, severe_toxic, obscene, threat, insult, and identity_hate in user comments
  - **Dataset**: Jigsaw Toxic Comment Classification Challenge dataset from Kaggle
  - **Files**: `config.yaml` (kaggle dataset configuration), `src/sklearn_training/utils/data_loader.py`

---

## Core System Requirements

### Multi-Component Application Deployed on AWS
- [x] **Experiment Tracking & Model Registry**
  - **Implementation**: MLflow server with PostgreSQL backend on dedicated EC2 instance
  - **Files**: `infrastructure/ec2.tf` (MLflow server), `src/sklearn_training/utils/experiment_tracking.py`
  - **Details**: Logs parameters, metrics, model artifacts; Model Registry for versioning and promotion

- [x] **ML Model Backend**
  - **Implementation**: FastAPI application serving registered models with prediction endpoints
  - **Files**: `src/fastapi_backend/main.py`, `src/fastapi_backend/utils/`
  - **Details**: `/predict`, `/predict_proba`, `/health` endpoints; loads models from MLflow registry

- [x] **Persistent Data Store**
  - **Implementation**: DynamoDB for prediction logs and PostgreSQL (RDS) for MLflow backend
  - **Files**: `infrastructure/dynamodb.tf`, `infrastructure/rds.tf`
  - **Details**: Stores prediction requests, outputs, timestamps, and user feedback

- [x] **Frontend Interface**
  - **Implementation**: Streamlit user-facing application for comment toxicity testing
  - **Files**: `src/streamlit_frontend/app.py`
  - **Details**: Submit comments, view predictions, provide feedback on model accuracy

- [x] **Model Monitoring Dashboard**
  - **Implementation**: Separate Streamlit monitoring dashboard on dedicated EC2 instance
  - **Files**: `src/streamlit_monitoring/app.py`, `src/streamlit_monitoring/utils/`
  - **Details**: Visualizes prediction latency, class distributions, live accuracy from user feedback

- [x] **CI/CD Pipeline**
  - **Implementation**: GitHub Actions workflow with automated testing and linting
  - **Files**: `.github/workflows/ci.yml`, `.github/actions/setup/action.yml`
  - **Details**: Triggers on PR to main; runs ruff linting and pytest unit tests

---

## Phase 1: Experimentation and Model Management

### 1.1 Model Development
- [x] **Baseline Model Training**
  - **Implementation**: Multi-output logistic regression with TF-IDF features
  - **Files**: `src/sklearn_training/train_model.py`, `src/sklearn_training/utils/`
  - **Details**: Trains on 6 toxicity labels simultaneously using MultiOutputClassifier

### 1.2 Experiment Tracking
- [x] **MLflow Integration**
  - **Implementation**: Complete MLflow setup with experiment logging
  - **Files**: `src/sklearn_training/utils/experiment_tracking.py`
  - **Details**: Logs Git commit hashes, hyperparameters, metrics, data versions

- [x] **Logging**
  - **Parameters**: Model type, hyperparameters, feature engineering settings
  - **Metrics**: ROC-AUC per label, precision, recall, F1-scores
  - **Artifacts**: Model files, preprocessing pipelines, feature extractors
  - **Code Version**: Git commit SHA tracking

### 1.3 Model Versioning & Registry
- [x] **Model Registry Implementation**
  - **Implementation**: MLflow Model Registry with versioning and staging
  - **Files**: `src/sklearn_training/utils/experiment_tracking.py` (promote_model_to_production function)
  - **Details**: Models promoted to "Production" stage; backend loads latest production model

---

## Phase 2: Backend API and Database Integration

### 2.1 FastAPI Backend
- [x] **FastAPI Application**
  - **Implementation**: API with toxic sentiment prediction, feedback, moderation, and example endpoints
  - **Files**: `src/fastapi_backend/main.py`
  - **Endpoints**:
    - `/predict` - Multi-label toxicity classification
    - `/predict_proba` - Probability scores for all toxicity categories
    - `/health` - Health check endpoint
    - `/example` - Sample request for testing
    - `/moderation/*` - Human-in-the-loop moderation workflow

- [x] **Model Registry Integration**
  - **Implementation**: Loads latest "Production" model from MLflow registry
  - **Files**: `src/fastapi_backend/utils/model_loader.py`
  - **Details**: Environment-aware loading (local files in dev, S3/MLflow in prod)

### 2.2 Cloud Database
- [x] **AWS DynamoDB Implementation**
  - **Implementation**: NoSQL database for prediction logging and monitoring
  - **Files**: `infrastructure/dynamodb.tf`
  - **Schema**: partition_key, sort_key, with GSI for timestamp-based queries
  - **Logging**: Every prediction request, output, timestamp, user feedback

- [x] **PostgreSQL RDS for MLflow**
  - **Implementation**: Managed PostgreSQL database for MLflow backend store
  - **Files**: `infrastructure/rds.tf`
  - **Details**: Stores experiment metadata, runs, parameters, and metrics

---

## Phase 3: Frontend and Live Monitoring

### 3.1 User Interface
- [x] **Streamlit Frontend Application**
  - **Implementation**: Interactive web application for toxicity testing
  - **Files**: `src/streamlit_frontend/app.py`
  - **Features**: 
    - Submit text for toxicity analysis
    - View multi-label predictions with confidence scores
    - Provide feedback on prediction accuracy
    - Real-time interaction with FastAPI backend

### 3.2 Model Monitoring Dashboard
- [x] **Separate Monitoring Application**
  - **Implementation**: Dedicated Streamlit dashboard on separate EC2 instance
  - **Files**: `src/streamlit_monitoring/app.py`
  - **Data Source**: DynamoDB prediction logs (database-driven, not file-based)

- [x] **Key Monitoring Metrics**
  - **Prediction Latency**: Response time tracking over time
  - **Target Drift**: Distribution of predicted toxicity classes
  - **Live Accuracy**: User feedback collection and accuracy calculation
  - **Volume Metrics**: Prediction request counts and patterns
  - **Files**: `src/streamlit_monitoring/utils/`

---

## Phase 4: Testing and CI/CD Automation

### 4.1 Testing
- [x] **Unit Tests**
  - **Implementation**: Isolated function testing with comprehensive mocking framework
  - **Files**: `tests/`

- [x] **Integration Tests**
  - **Implementation**: Full API endpoint testing with realistic request/response cycles
  - **Files**: `tests/`


### 4.2 CI/CD Pipeline
- [x] **GitHub Actions Workflow**
  - **Implementation**: Automated CI pipeline with comprehensive checks
  - **Files**: `.github/workflows/ci.yml`
  - **Triggers**: Pull requests to main branch with path filtering

- [x] **Quality Checks**
  - **Linting**: ruff check and format verification
  - **Testing**: Full pytest test suite execution
  - **Merge Protection**: PRs cannot merge if checks fail

---

## Phase 5: Containerization and Deployment

### 5.1 Docker Packaging
- [x] **Service Containerization**
  - **Implementation**: Individual Dockerfiles for each service
  - **Files**: 
    - `src/fastapi_backend/Dockerfile`
    - `src/streamlit_frontend/Dockerfile` 
    - `src/streamlit_monitoring/Dockerfile`
    - `src/sklearn_training/Dockerfile`
    - `docker-compose.yml` (local orchestration)

### 5.2 AWS Deployment
- [x] **EC2 Container Deployment**
  - **Implementation**: Terraform-provisioned EC2 instances with Docker
  - **Files**: `infrastructure/ec2.tf`, `infrastructure/modules/docker_deployment/`
  - **Architecture**: 5 separate EC2 instances, one per service

- [x] **Infrastructure as Code**
  - **Implementation**: Complete Terraform infrastructure automation
  - **Files**: `infrastructure/*.tf`
  - **Resources**: EC2, S3, DynamoDB, RDS, Security Groups, CloudWatch

### 5.3 Documentation
- [x] **High-Quality README.md**
  - **Implementation**: Comprehensive project documentation
  - **Files**: `README.md`
  - **Content**: 
    - Complete setup and deployment instructions
    - Local development with `task dev:up/down`
    - Production deployment with `task prod:init/apply/destroy`
    - Architecture diagrams and component descriptions
    - Example usage and API documentation

---

## Deliverables & Submission

### Required Deliverables
- [x] **GitHub Repository URL**
  - **Repository**: https://github.com/jairus-m/toxic-mlops
  - **Content**: All source code, configuration files, infrastructure, and documentation
  - **Structure**: Clean monorepo with uv workspaces

- [x] **Experiment Tracking Dashboard URL**
  - **Implementation**: Public MLflow server accessible via AWS deployment
  - **Access**: `http://<MLFLOW_SERVER_PUBLIC_IP>:5000` after `task prod:apply`
  - **Content**: All training experiments, model registry, and performance metrics

### Additional Features Implemented
- [x] **Human-in-the-Loop Moderation**
  - **Implementation**: Advanced moderation workflow with review queue
  - **Files**: `src/fastapi_backend/utils/moderation.py`
  - **Features**: Automatic content filtering, manual review system, decision tracking

- [x] **Multi-Environment Support**
  - **Implementation**: Environment-aware configuration system
  - **Files**: `config.yaml`, `src/core/load_config.py`
  - **Environments**: Development (local files) and Production (cloud resources)

- [x] **Monitoring**
  - **Implementation**: CloudWatch integration for centralized and exposed logging from EC2 instances
  - **Files**: `infrastructure/cloudwatch.tf`
  - **Features**: Service logs, error tracking, performance monitoring

---
