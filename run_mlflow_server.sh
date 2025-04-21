#!/bin/bash
echo "[MLflow Server] Dang khoi chay MLflow Tracking Server cuc bo..."
echo "[MLflow Server] Backend Store: sqlite:///mlflow.db"
echo "[MLflow Server] Artifact Store: ./mlartifacts"
echo "[MLflow Server] Truy cap UI tai: http://127.0.0.1:5000"
mkdir -p mlartifacts # Tao thu muc neu chua co
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts \
    --host 127.0.0.1 \
    --port 5000