# src/config.py
import os

# --- Cấu hình MLflow ---
# URI trỏ đến MLflow Server cục bộ đang chạy (sử dụng SQLite backend)
# Bạn cần chạy server này trước! Xem run_mlflow_server.sh
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Phan Loai Local Flask")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "BestSimpleClassifierLocal") # Tên model đăng ký

# --- Cấu hình API (Flask) ---
# Stage của model mà API Flask sẽ tự động tải về và sử dụng
# Sau khi train.py chạy, nó sẽ cố gắng chuyển model mới nhất sang stage này
API_MODEL_STAGE = os.getenv("API_MODEL_STAGE", "Staging") # Ví dụ: Staging hoặc Production


# --- Cấu hình Huấn luyện & Dữ liệu (Giữ nguyên) ---
N_SAMPLES = int(os.getenv("N_SAMPLES", "1000"))
N_FEATURES = int(os.getenv("N_FEATURES", "10"))
N_CLASSES = int(os.getenv("N_CLASSES", "2"))
RANDOM_STATE_DATA = int(os.getenv("RANDOM_STATE_DATA", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE_SPLIT = int(os.getenv("RANDOM_STATE_SPLIT", "123"))

MODEL_TYPE = os.getenv("MODEL_TYPE", "LogisticRegression")
HYPERPARAMS_TO_TUNE = {
     "LogisticRegression": { "C": [0.1, 1.0, 10.0], "solver": ["liblinear"] },
     "RandomForestClassifier": { "n_estimators": [50, 100], "max_depth": [None, 10] }
}
MODEL_PARAMS = HYPERPARAMS_TO_TUNE.get(MODEL_TYPE, {})
PRIMARY_METRIC = os.getenv("PRIMARY_METRIC", "accuracy")
