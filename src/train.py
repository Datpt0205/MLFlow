# src/train.py
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient # Import client để tương tác registry
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import itertools
import pandas as pd
import os
import traceback
import time

from data_generator import get_data
from utils import evaluate_model
from config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MLFLOW_MODEL_NAME,
    MODEL_TYPE, MODEL_PARAMS, PRIMARY_METRIC, API_MODEL_STAGE # Thêm API_MODEL_STAGE
)

def train_and_log():
    print("--- Bat dau qua trinh huan luyen & logging (Local Server Mode) ---")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI) # Đặt URI kết nối

    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        print(f"Su dung MLflow Experiment: '{MLFLOW_EXPERIMENT_NAME}'")
    except Exception as e:
        print(f"!!! Loi khi set experiment '{MLFLOW_EXPERIMENT_NAME}'. Server MLflow co dang chay khong?")
        print(f"Loi chi tiet: {e}")
        print(traceback.format_exc())
        return

    X_train, X_test, y_train, y_test = get_data()

    if not MODEL_PARAMS:
        print(f"CANH BAO: Khong tim thay cau hinh sieu tham so cho model type '{MODEL_TYPE}'. Chay 1 lan voi tham so mac dinh.")
        param_combinations = [{}]
    else:
        param_names = list(MODEL_PARAMS.keys())
        param_values = list(MODEL_PARAMS.values())
        param_combinations = [dict(zip(param_names, prod)) for prod in itertools.product(*param_values)]

    best_metric_value = -float('inf')
    best_run_id = None
    latest_registered_version = None # Lưu thông tin version mới nhất được đăng ký
    run_results = []

    print(f"\nBat dau thu nghiem {len(param_combinations)} cau hinh cho model {MODEL_TYPE}...")

    for params in param_combinations:
        try:
            with mlflow.start_run() as run:
                run_id = run.info.run_id
                # ... (Phần log params, tạo model, train, evaluate, log metrics, log model artifact giữ nguyên như phiên bản trước) ...
                print(f"\n--- Dang chay MLflow Run ID: {run_id} ---")
                print(f"Sieu tham so: {params}")
                mlflow.log_params(params)
                mlflow.log_param("model_type", MODEL_TYPE)

                steps = []
                model_instance = None
                use_scaler = False
                if MODEL_TYPE == "LogisticRegression":
                    model_instance = LogisticRegression(**params, random_state=42, max_iter=1500)
                    if params.get('solver') == 'saga': use_scaler = True
                elif MODEL_TYPE == "RandomForestClassifier":
                    model_instance = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                elif MODEL_TYPE == "SVC":
                    model_instance = SVC(**params, probability=True, random_state=42)
                    use_scaler = True
                else: raise ValueError(f"Loai model khong duoc ho tro: {MODEL_TYPE}")

                if use_scaler: steps.append(('scaler', StandardScaler()))
                steps.append(('classifier', model_instance))
                pipeline = Pipeline(steps)

                print("Dang huan luyen...")
                pipeline.fit(X_train, y_train)
                print("Dang danh gia...")
                y_pred_test = pipeline.predict(X_test)
                metrics = evaluate_model(y_test, y_pred_test)
                print("Dang log metrics...")
                mlflow.log_metrics(metrics)
                print("Dang log model artifact...")
                signature = mlflow.models.infer_signature(X_train, pipeline.predict(X_train))
                mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model", signature=signature)
                print("Log model artifact thanh cong.")

                current_metric_value = metrics.get(PRIMARY_METRIC)
                if current_metric_value is not None:
                     run_results.append({
                        "run_id": run_id, "params": params, "metrics": metrics,
                        "primary_metric_value": current_metric_value
                     })
                     if current_metric_value > best_metric_value:
                         best_metric_value = current_metric_value
                         best_run_id = run_id # Lưu run_id tốt nhất tạm thời
                         print(f"*** Run {run_id}: Model tot hon -> {PRIMARY_METRIC} = {best_metric_value:.4f} ***")
                else:
                     print(f"CANH BAO: Metric chinh '{PRIMARY_METRIC}' khong tim thay trong run {run_id}.")

        except Exception as e_run:
            print(f"!!! Loi trong MLflow Run ID {run_id}: {e_run}")
            print(traceback.format_exc())
            if mlflow.active_run():
                 try:
                      mlflow.log_param("run_error", str(e_run))
                      mlflow.end_run(status="FAILED")
                 except Exception as e_log: print(f"Loi khi log loi/ket thuc run {run_id}: {e_log}")
            print("--- Ket thuc Run (FAILED) ---")
            continue

    print("\n--- Ket thuc tat ca thu nghiem ---")
    if not run_results:
        print("Khong co run nao hoan thanh thanh cong de dang ky model.")
        return

    # Tìm best run cuối cùng
    run_results.sort(key=lambda x: x['primary_metric_value'], reverse=True)
    best_run = run_results[0]
    best_run_id = best_run['run_id']
    best_metric_value = best_run['primary_metric_value']

    print(f"\nModel tot nhat (dựa trên '{PRIMARY_METRIC}'):")
    print(f"  Run ID: {best_run_id}")
    print(f"  Metrics: {best_run['metrics']}")

    # Đăng ký model tốt nhất vào Model Registry
    if best_run_id:
        print(f"\nDang dang ky model tu run {best_run_id} vao Registry '{MLFLOW_MODEL_NAME}'...")
        model_uri = f"runs:/{best_run_id}/model"
        try:
            registered_model_info = mlflow.register_model(
                model_uri=model_uri,
                name=MLFLOW_MODEL_NAME
            )
            latest_registered_version = registered_model_info.version
            print(f"Dang ky model thanh cong:")
            print(f"  Ten: {registered_model_info.name}")
            print(f"  Phien ban: {latest_registered_version}") # Lấy version mới nhất
            print(f"  Stage ban dau: {registered_model_info.current_stage}")

            # --- TỰ ĐỘNG THĂNG CẤP STAGE (Tùy chọn nhưng hữu ích cho API) ---
            if API_MODEL_STAGE and API_MODEL_STAGE.lower() != "none":
                print(f"\nDang thu chuyen phien ban {latest_registered_version} sang stage '{API_MODEL_STAGE}'...")
                client = MlflowClient()
                try:
                    # Chờ một chút để registry cập nhật (đôi khi cần thiết)
                    time.sleep(5)
                    client.transition_model_version_stage(
                        name=MLFLOW_MODEL_NAME,
                        version=latest_registered_version,
                        stage=API_MODEL_STAGE,
                        archive_existing_versions=True # Chuyển các version cũ cùng stage sang Archived
                    )
                    print(f"Chuyen sang stage '{API_MODEL_STAGE}' thanh cong.")
                except Exception as e_stage:
                    print(f"!!! Loi khi chuyen stage model: {e_stage}")
                    print(traceback.format_exc())
            # --- Kết thúc phần thăng cấp stage ---

        except Exception as e_reg:
            print(f"!!! Loi khi dang ky model: {e_reg}")
            print(traceback.format_exc())
    else:
        print("\nKhong tim thay best run ID hop le de dang ky model.")

    print("\n--- Hoan thanh ---")

if __name__ == "__main__":
    train_and_log()