# src/train.py
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature  # Import signature
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier  # Import XGBoost
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
)  # Import để in report nếu cần (utils đã làm)
import itertools
import pandas as pd
import os
import traceback
import time
import joblib  # Import joblib để lưu model cục bộ

from src.data_generator import get_data
from src.utils import evaluate_model
from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
    MODEL_TYPE,
    MODEL_PARAMS_GRID,
    PRIMARY_METRIC,
    API_MODEL_STAGE,
    USE_SCALER,  # Thêm USE_SCALER
    LOCAL_MODEL_OUTPUT_DIR,
    BEST_MODEL_FILENAME,  # Thêm cấu hình lưu model cục bộ
)


def train_and_log():
    print("--- Bat dau qua trinh huan luyen & logging ---")
    print(f"Cau hinh:")
    print(f"  - MLflow Server: {MLFLOW_TRACKING_URI}")
    print(f"  - Experiment: {MLFLOW_EXPERIMENT_NAME}")
    print(f"  - Registered Model Name: {MLFLOW_MODEL_NAME}")
    print(f"  - Model Type to Train: {MODEL_TYPE}")
    print(f"  - Use Scaler: {USE_SCALER}")
    print(f"  - Primary Metric: {PRIMARY_METRIC}")
    print(f"  - Target API Stage: {API_MODEL_STAGE}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        print(f"\nSu dung/Tao MLflow Experiment: '{MLFLOW_EXPERIMENT_NAME}'")
    except Exception as e:
        print(
            f"!!! Loi khi set experiment '{MLFLOW_EXPERIMENT_NAME}'. Server MLflow co dang chay khong?"
        )
        print(f"Loi chi tiet: {e}")
        print(traceback.format_exc())
        return

    print("\n--- Dang tai/tao du lieu ---")
    X_train, X_test, y_train, y_test = get_data()  # get_data đã xử lý augment/scale

    # Tạo các tổ hợp siêu tham số từ grid trong config
    if not MODEL_PARAMS_GRID:
        print(
            f"CANH BAO: Khong tim thay grid sieu tham so cho model type '{MODEL_TYPE}'. Chay 1 lan voi tham so mac dinh."
        )
        param_combinations = [{}]  # Chạy với một dictionary rỗng
    else:
        param_names = list(MODEL_PARAMS_GRID.keys())
        param_values = list(MODEL_PARAMS_GRID.values())
        # Tạo các dictionary tham số hoàn chỉnh
        param_combinations = [
            dict(zip(param_names, prod)) for prod in itertools.product(*param_values)
        ]

    print(
        f"\nSe thu nghiem {len(param_combinations)} cau hinh cho model '{MODEL_TYPE}'..."
    )

    best_metric_value = -float("inf")
    best_run_id = None
    best_pipeline = None  # Lưu pipeline tốt nhất
    run_results = []

    for params in param_combinations:
        run_id_current = None  # Khởi tạo để bắt lỗi
        try:
            with mlflow.start_run() as run:
                run_id_current = run.info.run_id
                print(f"\n--- Dang chay MLflow Run ID: {run_id_current} ---")
                print(f"Sieu tham so (cho Pipeline): {params}")

                # Log các tham số (bao gồm cả tiền tố 'classifier__')
                mlflow.log_params(params)
                mlflow.log_param("model_type", MODEL_TYPE)  # Log loại model chính

                # --- Xây dựng Pipeline ---
                steps = []
                # Thêm Scaler vào pipeline NẾU USE_SCALER là True
                if USE_SCALER:
                    steps.append(("scaler", StandardScaler()))
                    mlflow.log_param("pipeline_scaler", True)
                else:
                    mlflow.log_param("pipeline_scaler", False)

                # Chọn và khởi tạo model instance
                # Lưu ý: Tham số cho model bây giờ nằm trong `params` với tiền tố 'classifier__'
                # Chúng ta cần tách chúng ra hoặc truyền trực tiếp vào Pipeline
                model_instance = None
                model_specific_params = {
                    k.split("__", 1)[1]: v
                    for k, v in params.items()
                    if k.startswith("classifier__")
                }

                if MODEL_TYPE == "LogisticRegression":
                    # Các tham số không tune có thể thêm vào đây
                    model_instance = LogisticRegression(
                        random_state=42, max_iter=1500, **model_specific_params
                    )
                elif MODEL_TYPE == "RandomForestClassifier":
                    # n_jobs=-1 để dùng tất cả CPU cores
                    model_instance = RandomForestClassifier(
                        random_state=42, n_jobs=-1, **model_specific_params
                    )
                elif MODEL_TYPE == "XGBClassifier":
                    # Các tham số cố định như ví dụ mẫu
                    model_instance = XGBClassifier(
                        random_state=42,
                        use_label_encoder=False,  # Nên set False cho các phiên bản XGBoost mới
                        eval_metric="logloss",  # hoặc 'auc', 'error' tùy bài toán
                        tree_method="hist",  # thường nhanh hơn
                        **model_specific_params,
                    )
                elif MODEL_TYPE == "SVC":
                    # probability=True để có thể dùng predict_proba (cần cho 1 số metric/signature)
                    model_instance = SVC(
                        probability=True, random_state=42, **model_specific_params
                    )
                else:
                    raise ValueError(f"Loai model khong duoc ho tro: {MODEL_TYPE}")

                steps.append(("classifier", model_instance))
                pipeline = Pipeline(steps)

                # --- Huấn luyện & Đánh giá ---
                print("Dang huan luyen Pipeline...")
                start_time = time.time()
                pipeline.fit(X_train, y_train)
                train_time = time.time() - start_time
                print(f"Huan luyen xong trong {train_time:.2f} giay.")
                mlflow.log_metric("training_duration_seconds", train_time)

                print("Dang danh gia tren tap test...")
                y_pred_test = pipeline.predict(X_test)

                # Sử dụng utils.evaluate_model (đã bao gồm in classification_report)
                metrics = evaluate_model(y_test, y_pred_test)
                print(f"Metrics (Weighted Avg): {metrics}")

                # --- Logging MLflow ---
                print("Dang log metrics vao MLflow...")
                mlflow.log_metrics(metrics)

                print("Dang log model artifact (Pipeline)...")
                # Tạo signature: quan trọng cho việc serving và hiểu model input/output
                try:
                    # predict() thường trả về output đơn giản hơn cho signature
                    signature = infer_signature(X_train, pipeline.predict(X_train))
                    mlflow.sklearn.log_model(
                        sk_model=pipeline,
                        artifact_path="model",  # Thư mục con trong artifact của run
                        signature=signature,
                        input_example=X_train[:5],  # Lưu ví dụ input
                        pip_requirements="-r requirements.txt",  # Gắn requirements nếu có
                    )
                    print("Log model artifact thanh cong.")
                except Exception as e_sig:
                    print(f"!!! Loi khi log model/signature: {e_sig}")
                    mlflow.log_param(
                        "logging_error", "Failed to log model artifact/signature"
                    )

                # --- Theo dõi best run ---
                current_metric_value = metrics.get(PRIMARY_METRIC)
                if current_metric_value is not None:
                    run_results.append(
                        {
                            "run_id": run_id_current,
                            "params": params,
                            "metrics": metrics,
                            "primary_metric_value": current_metric_value,
                            "pipeline": pipeline,  # Lưu lại pipeline để có thể dùng lại
                        }
                    )
                    if current_metric_value > best_metric_value:
                        best_metric_value = current_metric_value
                        best_run_id = run_id_current  # Lưu run_id tốt nhất
                        best_pipeline = pipeline  # Lưu pipeline tốt nhất
                        print(
                            f"*** Run {run_id_current}: Model tot hon -> {PRIMARY_METRIC} = {best_metric_value:.4f} ***"
                        )
                else:
                    print(
                        f"CANH BAO: Metric chinh '{PRIMARY_METRIC}' khong tim thay trong run {run_id_current}."
                    )

                print(f"--- Ket thuc Run ID: {run_id_current} ---")

        except Exception as e_run:
            print(f"!!! Loi trong MLflow Run ID {run_id_current or 'UNKNOWN'}: {e_run}")
            print(traceback.format_exc())
            if mlflow.active_run():
                try:
                    mlflow.log_param(
                        "run_error", str(e_run)[:250]
                    )  # Giới hạn độ dài lỗi log
                    mlflow.end_run(status="FAILED")
                except Exception as e_log:
                    print(f"Loi khi log loi/ket thuc run {run_id_current}: {e_log}")
            print("--- Ket thuc Run (FAILED) ---")
            continue  # Chuyển sang vòng lặp params tiếp theo

    print("\n--- Ket thuc tat ca thu nghiem ---")

    # --- Xử lý kết quả ---
    if not run_results:
        print("Khong co run nao hoan thanh thanh cong de dang ky model.")
        return

    # Sắp xếp lại run_results để chắc chắn best_run là tốt nhất (dù đã theo dõi trong vòng lặp)
    run_results.sort(key=lambda x: x["primary_metric_value"], reverse=True)
    best_run_info = run_results[0]
    best_run_id = best_run_info["run_id"]
    best_metric_value = best_run_info["primary_metric_value"]
    best_pipeline = best_run_info["pipeline"]  # Lấy pipeline từ run tốt nhất đã lưu

    print(f"\nModel tot nhat (dựa trên '{PRIMARY_METRIC}'):")
    print(f"  Run ID: {best_run_id}")
    print(f"  Sieu tham so: {best_run_info['params']}")
    print(f"  Metrics: {best_run_info['metrics']}")

    # --- Đăng ký model tốt nhất vào MLflow Model Registry ---
    if best_run_id:
        print(
            f"\nDang dang ky model tu run {best_run_id} vao Registry '{MLFLOW_MODEL_NAME}'..."
        )
        model_uri = f"runs:/{best_run_id}/model"  # Đường dẫn tới artifact model đã log
        try:
            registered_model_info = mlflow.register_model(
                model_uri=model_uri,
                name=MLFLOW_MODEL_NAME,
                tags={  # Thêm tags để mô tả model version
                    "source_run_id": best_run_id,
                    "primary_metric": PRIMARY_METRIC,
                    "metric_value": f"{best_metric_value:.4f}",
                    "model_type": MODEL_TYPE,
                },
            )
            latest_registered_version = registered_model_info.version
            print(f"Dang ky model thanh cong:")
            print(f"  Ten: {registered_model_info.name}")
            print(f"  Phien ban moi nhat: {latest_registered_version}")
            print(f"  Stage ban dau: {registered_model_info.current_stage}")

            # --- Tự động chuyển Stage (Nếu được cấu hình) ---
            if API_MODEL_STAGE and API_MODEL_STAGE.lower() != "none":
                print(
                    f"\nDang thu chuyen phien ban {latest_registered_version} sang stage '{API_MODEL_STAGE}'..."
                )
                client = MlflowClient()
                try:
                    # Chờ một chút để registry xử lý (đôi khi cần thiết)
                    time.sleep(5)
                    client.transition_model_version_stage(
                        name=MLFLOW_MODEL_NAME,
                        version=latest_registered_version,
                        stage=API_MODEL_STAGE,
                        archive_existing_versions=True,  # Chuyển các version cũ cùng stage sang Archived
                    )
                    print(f"Chuyen sang stage '{API_MODEL_STAGE}' thanh cong.")

                    # Cập nhật description cho version mới chuyển stage
                    stage_description = f"Model {MODEL_TYPE} (run {best_run_id}) with {PRIMARY_METRIC}={best_metric_value:.4f}, automatically promoted to {API_MODEL_STAGE}."
                    client.update_model_version(
                        name=MLFLOW_MODEL_NAME,
                        version=latest_registered_version,
                        description=stage_description,
                    )
                    print(
                        f"Da cap nhat mo ta cho phien ban {latest_registered_version}."
                    )

                except Exception as e_stage:
                    print(
                        f"!!! Loi khi chuyen stage hoac cap nhat mo ta model: {e_stage}"
                    )
                    print(traceback.format_exc())
            # --- Kết thúc phần chuyển stage ---

        except Exception as e_reg:
            print(f"!!! Loi khi dang ky model vao Registry: {e_reg}")
            print(traceback.format_exc())
    else:
        print("\nKhong tim thay best run ID hop le de dang ky model.")

    # --- Lưu model tốt nhất ra file cục bộ (bằng joblib) ---
    if best_pipeline and LOCAL_MODEL_OUTPUT_DIR:
        try:
            output_dir = LOCAL_MODEL_OUTPUT_DIR
            os.makedirs(output_dir, exist_ok=True)  # Tạo thư mục nếu chưa có
            model_path = os.path.join(output_dir, BEST_MODEL_FILENAME)
            joblib.dump(best_pipeline, model_path)
            print(f"\nDa luu pipeline tot nhat vao file cuc bo: {model_path}")
        except Exception as e_save:
            print(f"!!! Loi khi luu model pipeline tot nhat ra file: {e_save}")

    print("\n--- Hoan thanh qua trinh ---")


if __name__ == "__main__":
    train_and_log()
