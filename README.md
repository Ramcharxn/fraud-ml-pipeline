# Fraud Detection MLOps Pipeline  
**Fully Automated End-to-End Fraud Detection System on AWS**  
Real-time inference • Continuous training • Drift detection • Auto-retraining • Zero-downtime deployment  

![AWS](https://img.shields.io/badge/AWS-FF9900?style=flat)
![SageMaker](https://img.shields.io/badge/SageMaker-FF6F00?style=flat)
![LightGBM](https://img.shields.io/badge/LightGBM-00A1E1?style=flat)
![Lambda](https://img.shields.io/badge/Lambda-FF9900?style=flat)
![CI/CD](https://img.shields.io/badge/CI%2FCD-4B0082?style=flat)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat)
![MLOps](https://img.shields.io/badge/MLOps-000000?style=flat)




## 🚀 Key Features
- **Automated CI/CD** via GitHub → CodePipeline → SageMaker
- **Real-time inference** through API Gateway + Lambda + SageMaker Endpoint
- **Live data capture** enabled on the endpoint (input + output)
- **Custom drift detection Lambda** (EventBridge scheduled)
- **Automatic retraining** when data drift exceeds threshold
- **Zero-downtime model updates** (`update_endpoint=True`)
- Combines historical + recent live data (with pseudo-labels) for retraining

## 🏗️ Architecture Overview
<img width="2730" height="798" alt="diagram-export-12-3-2025-11_44_23-AM" src="https://github.com/user-attachments/assets/7c18e9a9-56dc-4c23-9a4c-9ec944ff8dc8" />


## 📁 Project Structure
```text
## 📁 Project Structure
├── fraud_sagemaker.py          # LightGBM training script
├── start_training.py           # Orchestrates data merge + training + deployment
├── buildspec.yml               # CodeBuild configuration
├── lambda
│    ├──fraud-inference.py      # Real-time inference Lambda
│    └──fraud-drift-check.py    # Drift monitoring & pipeline trigger    
├── fraud-drift-check-lambda/   
└── README.md                   # This file
```


## 🧠 Model & Data
- **Model**: LightGBM binary classifier (`isFraud`)
- **Primary metric**: AUC ROC
- **Historical data**: `s3://<bucket>/raw/train_merged1.csv`
- **Live data**: Captured from endpoint → `s3://<bucket>/data-capture/`
- **Pseudo-labeling**: `prediction ≥ 0.5 → isFraud = 1`

## 🔄 Automated Training & Retraining Flow (`start_training.py`)
1. Load historical dataset  
2. Load recent captured inference data (last N days)  
3. Generate pseudo-labels from model predictions  
4. Merge datasets  
5. Upload combined dataset to S3  
6. Launch SageMaker Training Job (SKLearn container)  
7. Deploy updated model to existing endpoint (zero downtime)

## ⚡ Real-Time Inference
- **Public URL**: API Gateway → Lambda → SageMaker Endpoint  
- **Response**: JSON with `fraud_probability` (0.0 – 1.0)

## 📊 Data Capture
Enabled on SageMaker endpoint:  
- Captures both request payload and model output  
- Stored as JSONL in `s3://<bucket>/data-capture/...`

## ⚠️ Drift Detection (fraud-drift-check-lambda)
- Runs every X minutes via EventBridge  
- Compares key feature distributions (e.g., transaction amount mean)  
- Drift score = `|live_mean - train_mean| / train_mean`  
- If ≥ threshold (default 30%) → triggers CodePipeline retraining

## 🛠️ AWS Services Used
| Service             | Purpose                                      |
|---------------------|----------------------------------------------|
| SageMaker           | Training, hosting, data capture              |
| S3                  | Raw data, combined data, captured inferences |
| Lambda              | Inference & drift detection                  |
| API Gateway         | Public REST endpoint                         |
| CodePipeline        | End-to-end CI/CD orchestration               |
| CodeBuild           | Run training automation script apply         |
| EventBridge         | Schedule drift checks                        |
| IAM                 | Permissions & roles                          |

## 🚀 Deployment & CI/CD
- Push to `main` branch → automatically starts pipeline  
- CodeBuild runs `start_training.py` → new model deployed instantly  
- Manual retraining also triggered by drift Lambda

## 🔜 Future Improvements
- Add ground-truth labels feedback loop
- Use Feature Store (SageMaker Feature Store)
- Advanced drift detection (PSI, KS-test, model performance drift)
- A/B testing & canary deployments
- CloudWatch + Grafana monitoring dashboard

---

**This is a production-grade, self-healing MLOps fraud detection system that continuously adapts to changing transaction patterns without human intervention.**


