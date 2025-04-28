# src/train.py
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import itertools
import pandas as pd
import os
import traceback
import time
import joblib

from data_generator import get_data
from utils import evaluate_model
from config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME, # Tên model đăng ký chung
    HYPERPARAMS_TO_TUNE, # Dictionary chứa grids của tất cả models
    MODEL_TYPES_TO_RUN,  # Danh sách các model types cần chạy
    PRIMARY_METRIC,
    API_MODEL_STAGE,
    USE_SCALER,
    LOCAL_MODEL_OUTPUT_DIR,
    BEST_OVERALL_MODEL_FILENAME, # Tên file cố định cho model tốt nhất
)

def train_and_log():
    print("--- Bat dau qua trinh huan luyen & logging cho NHIEU LOAI MODEL ---")
    print(f"Models se duoc thu nghiem: {MODEL_TYPES_TO_RUN}")
    # ... (in các cấu hình khác giữ nguyên) ...
    print(f"Cau hinh:")
    print(f"  - MLflow Server: {MLFLOW_TRACKING_URI}")
    print(f"  - Experiment: {MLFLOW_EXPERIMENT_NAME}")
    print(f"  - Registered Model Name: {MLFLOW_MODEL_NAME}")
    print(f"  - Use Scaler: {USE_SCALER}")
    print(f"  - Primary Metric: {PRIMARY_METRIC}")
    print(f"  - Target API Stage: {API_MODEL_STAGE}")


    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        print(f"\nSu dung/Tao MLflow Experiment: '{MLFLOW_EXPERIMENT_NAME}'")
    except Exception as e:
        # ... (xử lý lỗi set experiment giữ nguyên) ...
        print(
            f"!!! Loi khi set experiment '{MLFLOW_EXPERIMENT_NAME}'. Server MLflow co dang chay khong?"
        )
        print(f"Loi chi tiet: {e}")
        print(traceback.format_exc())
        return

    print("\n--- Dang tai/tao du lieu ---")
    X_train, X_test, y_train, y_test = get_data()

    # Biến để theo dõi model tốt nhất TỔNG THỂ
    overall_best_metric_value = -float("inf")
    overall_best_run_id = None
    overall_best_pipeline = None
    overall_best_model_type = None
    overall_best_params = None
    all_run_results = [] # Lưu kết quả của tất cả các run thành công

    # --- Vòng lặp ngoài: Duyệt qua từng loại model ---
    for current_model_type in MODEL_TYPES_TO_RUN:
        print(f"\n===== Bat dau thu nghiem cho Model Type: {current_model_type} =====")

        # Lấy grid siêu tham số cho model type hiện tại
        current_param_grid = HYPERPARAMS_TO_TUNE.get(current_model_type, {})

        # Tạo các tổ hợp siêu tham số
        if not current_param_grid:
            print(f"CANH BAO: Khong tim thay grid sieu tham so cho '{current_model_type}'. Chay 1 lan voi tham so mac dinh.")
            param_combinations = [{}]
        else:
            param_names = list(current_param_grid.keys())
            param_values = list(current_param_grid.values())
            param_combinations = [dict(zip(param_names, prod)) for prod in itertools.product(*param_values)]

        print(f"Se thu nghiem {len(param_combinations)} cau hinh cho '{current_model_type}'...")

        # --- Vòng lặp trong: Duyệt qua từng bộ siêu tham số (Giữ nguyên logic cũ) ---
        for params in param_combinations:
            run_id_current = None
            try:
                # Mỗi tổ hợp siêu tham số là một run riêng biệt trong MLflow
                with mlflow.start_run() as run:
                    run_id_current = run.info.run_id
                    print(f"\n--- Dang chay MLflow Run ID: {run_id_current} ({current_model_type}) ---")
                    print(f"Sieu tham so (cho Pipeline): {params}")

                    # Log params và model type hiện tại
                    mlflow.log_params(params)
                    mlflow.log_param("model_type", current_model_type) # Log loại model của run này

                    # Xây dựng Pipeline (Giữ nguyên logic)
                    steps = []
                    if USE_SCALER:
                        steps.append(("scaler", StandardScaler()))
                        mlflow.log_param("pipeline_scaler", True)
                    else:
                         mlflow.log_param("pipeline_scaler", False)

                    model_instance = None
                    model_specific_params = {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith("classifier__")}

                    # Chọn model instance dựa trên current_model_type
                    if current_model_type == "LogisticRegression":
                        model_instance = LogisticRegression(random_state=42, max_iter=1500, **model_specific_params)
                    elif current_model_type == "RandomForestClassifier":
                        model_instance = RandomForestClassifier(random_state=42, n_jobs=-1, **model_specific_params)
                    elif current_model_type == "XGBClassifier":
                        model_instance = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', tree_method="hist", **model_specific_params)
                    elif current_model_type == "SVC":
                        model_instance = SVC(probability=True, random_state=42, **model_specific_params)
                    else:
                        # Nên báo lỗi nếu model type không nằm trong HYPERPARAMS_TO_TUNE
                         raise ValueError(f"Loai model khong duoc ho tro: {current_model_type}")


                    steps.append(("classifier", model_instance))
                    pipeline = Pipeline(steps)

                    # Huấn luyện & Đánh giá (Giữ nguyên logic)
                    print("Dang huan luyen Pipeline...")
                    start_time = time.time()
                    pipeline.fit(X_train, y_train)
                    train_time = time.time() - start_time
                    print(f"Huan luyen xong trong {train_time:.2f} giay.")
                    mlflow.log_metric("training_duration_seconds", train_time)

                    print("Dang danh gia tren tap test...")
                    y_pred_test = pipeline.predict(X_test)
                    metrics = evaluate_model(y_test, y_pred_test)
                    print(f"Metrics (Weighted Avg): {metrics}")

                    # Logging MLflow (Giữ nguyên logic)
                    print("Dang log metrics vao MLflow...")
                    mlflow.log_metrics(metrics)
                    print("Dang log model artifact (Pipeline)...")
                    try:
                        signature = infer_signature(X_train, pipeline.predict(X_train))
                        mlflow.sklearn.log_model(
                            sk_model=pipeline,
                            artifact_path="model",
                            signature=signature,
                            input_example=X_train[:5],
                            pip_requirements="-r requirements.txt"
                        )
                        print("Log model artifact thanh cong.")
                    except Exception as e_sig:
                        # ... (xử lý lỗi log model giữ nguyên) ...
                        print(f"!!! Loi khi log model/signature: {e_sig}")
                        mlflow.log_param(
                            "logging_error", "Failed to log model artifact/signature"
                        )


                    # --- Theo dõi best run TỔNG THỂ ---
                    current_metric_value = metrics.get(PRIMARY_METRIC)
                    if current_metric_value is not None:
                        # Lưu kết quả của run này vào danh sách chung
                        all_run_results.append({
                            "run_id": run_id_current,
                            "model_type": current_model_type, # Thêm model type vào kết quả
                            "params": params,
                            "metrics": metrics,
                            "primary_metric_value": current_metric_value,
                            "pipeline": pipeline
                        })
                        # So sánh với best tổng thể hiện tại
                        if current_metric_value > overall_best_metric_value:
                            print(f"*** Run {run_id_current}: Model TOT HON TONG THE -> {PRIMARY_METRIC} = {current_metric_value:.4f} ({current_model_type}) ***")
                            overall_best_metric_value = current_metric_value
                            overall_best_run_id = run_id_current
                            overall_best_pipeline = pipeline
                            overall_best_model_type = current_model_type
                            overall_best_params = params # Lưu lại cả params tốt nhất
                        # else:
                            # print(f"Run {run_id_current}: Ket qua {PRIMARY_METRIC} = {current_metric_value:.4f}, khong phai tot nhat.")

                    else:
                        # ... (cảnh báo metric không tìm thấy giữ nguyên) ...
                         print(f"CANH BAO: Metric chinh '{PRIMARY_METRIC}' khong tim thay trong run {run_id_current}.")

                    print(f"--- Ket thuc Run ID: {run_id_current} ---")

            except Exception as e_run:
                # ... (xử lý lỗi trong 1 run giữ nguyên) ...
                 print(f"!!! Loi trong MLflow Run ID {run_id_current or 'UNKNOWN'}: {e_run}")
                 print(traceback.format_exc())
                 if mlflow.active_run():
                      try:
                           mlflow.log_param("run_error", str(e_run)[:250])
                           mlflow.end_run(status="FAILED")
                      except Exception as e_log: print(f"Loi khi log loi/ket thuc run {run_id_current}: {e_log}")
                 print("--- Ket thuc Run (FAILED) ---")
                 continue # Chuyển sang siêu tham số tiếp theo

        print(f"===== Ket thuc thu nghiem cho Model Type: {current_model_type} =====")
    # --- Kết thúc vòng lặp ngoài ---

    print("\n--- Ket thuc tat ca thu nghiem cho tat ca Model Types ---")

    # --- Xử lý kết quả TỔNG THỂ ---
    if not all_run_results:
        print("Khong co run nao hoan thanh thanh cong de tim model tot nhat.")
        return

    # Tìm lại run tốt nhất từ danh sách all_run_results (mặc dù đã theo dõi) để chắc chắn
    # Hoặc đơn giản là sử dụng các biến overall_best_* đã lưu
    if overall_best_run_id:
        print(f"\nModel TOT NHAT TONG THE (dựa trên '{PRIMARY_METRIC}' = {overall_best_metric_value:.4f}):")
        print(f"  Model Type: {overall_best_model_type}")
        print(f"  Run ID: {overall_best_run_id}")
        # Tìm lại metrics và params đầy đủ từ all_run_results nếu cần in chi tiết hơn
        best_run_full_info = next((r for r in all_run_results if r['run_id'] == overall_best_run_id), None)
        if best_run_full_info:
             print(f"  Sieu tham so: {best_run_full_info['params']}")
             print(f"  Metrics: {best_run_full_info['metrics']}")
        else: # In thông tin đã lưu trực tiếp
             print(f"  Sieu tham so (da luu): {overall_best_params}")


        # --- Đăng ký model TỐT NHẤT TỔNG THỂ vào MLflow Model Registry ---
        print(f"\nDang dang ky model tot nhat tong the tu run {overall_best_run_id} vao Registry '{MLFLOW_MODEL_NAME}'...")
        model_uri = f"runs:/{overall_best_run_id}/model"
        try:
            registered_model_info = mlflow.register_model(
                model_uri=model_uri,
                name=MLFLOW_MODEL_NAME, # Dùng tên model chung đã định nghĩa
                tags={ # Cập nhật tags
                    "source_run_id": overall_best_run_id,
                    "primary_metric": PRIMARY_METRIC,
                    "metric_value": f"{overall_best_metric_value:.4f}",
                    "best_model_type": overall_best_model_type, # Tag loại model tốt nhất
                }
            )
            latest_registered_version = registered_model_info.version
            print(f"Dang ky model tot nhat tong the thanh cong:")
            print(f"  Ten: {registered_model_info.name}")
            print(f"  Phien ban moi nhat: {latest_registered_version}")
            print(f"  Stage ban dau: {registered_model_info.current_stage}")

            # --- Tự động chuyển Stage (Giữ nguyên logic) ---
            if API_MODEL_STAGE and API_MODEL_STAGE.lower() != "none":
                # ... (code chuyển stage giữ nguyên, chỉ cần đảm bảo client được tạo đúng) ...
                 print(
                     f"\nDang thu chuyen phien ban {latest_registered_version} sang stage '{API_MODEL_STAGE}'..."
                 )
                 client = MlflowClient()
                 try:
                     time.sleep(5)
                     client.transition_model_version_stage(
                         name=MLFLOW_MODEL_NAME,
                         version=latest_registered_version,
                         stage=API_MODEL_STAGE,
                         archive_existing_versions=True,
                     )
                     print(f"Chuyen sang stage '{API_MODEL_STAGE}' thanh cong.")
                     # Cập nhật description
                     stage_description = f"Overall Best Model: {overall_best_model_type} (run {overall_best_run_id}) with {PRIMARY_METRIC}={overall_best_metric_value:.4f}, promoted to {API_MODEL_STAGE}."
                     client.update_model_version(
                          name=MLFLOW_MODEL_NAME,
                          version=latest_registered_version,
                          description=stage_description
                     )
                     print(f"Da cap nhat mo ta cho phien ban {latest_registered_version}.")

                 except Exception as e_stage:
                      print(f"!!! Loi khi chuyen stage hoac cap nhat mo ta model: {e_stage}")
                      print(traceback.format_exc())


        except Exception as e_reg:
            # ... (xử lý lỗi đăng ký model giữ nguyên) ...
             print(f"!!! Loi khi dang ky model vao Registry: {e_reg}")
             print(traceback.format_exc())

    else:
        print("\nKhong tim thay run nao tot nhat de dang ky model.")


    # --- Lưu model TỐT NHẤT TỔNG THỂ ra file cục bộ ---
    if overall_best_pipeline and LOCAL_MODEL_OUTPUT_DIR:
        try:
            output_dir = LOCAL_MODEL_OUTPUT_DIR
            os.makedirs(output_dir, exist_ok=True)
            # Sử dụng tên file cố định từ config
            model_path = os.path.join(output_dir, BEST_OVERALL_MODEL_FILENAME)
            joblib.dump(overall_best_pipeline, model_path)
            print(f"\nDa luu pipeline tot nhat TONG THE vao file cuc bo: {model_path}")
            print(f"  (Loai model: {overall_best_model_type})")
        except Exception as e_save:
            print(f"!!! Loi khi luu model pipeline tot nhat tong the ra file: {e_save}")

    print("\n--- Hoan thanh toan bo qua trinh ---")


if __name__ == "__main__":
    train_and_log()