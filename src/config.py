# src/config.py
import os

# --- Cấu hình MLflow ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Phan Loai Local Flask")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "BestSimpleClassifierLocal")

# --- Cấu hình API (Flask) ---
API_MODEL_STAGE = os.getenv("API_MODEL_STAGE", "Staging")

# --- Cấu hình Dữ liệu ---
# Dữ liệu gốc
N_SAMPLES_ORIGINAL = int(os.getenv("N_SAMPLES_ORIGINAL", "1000"))
N_FEATURES = int(os.getenv("N_FEATURES", "10"))
N_CLASSES = int(os.getenv("N_CLASSES", "2"))
RANDOM_STATE_DATA = int(os.getenv("RANDOM_STATE_DATA", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.3"))  # Giữ test_size 0.3 như ví dụ
RANDOM_STATE_SPLIT = int(
    os.getenv("RANDOM_STATE_SPLIT", "42")
)  # Giữ random_state 42 như ví dụ

# Tùy chọn Augmentation & Preprocessing (Thêm mới)
USE_AUGMENTATION = (
    os.getenv("USE_AUGMENTATION", "True").lower() == "true"
)  # Bật/tắt augmentation
N_SAMPLES_AUGMENTED = int(
    os.getenv("N_SAMPLES_AUGMENTED", "2000")
)  # Số mẫu sau augment
ADD_NOISE = os.getenv("ADD_NOISE", "True").lower() == "true"  # Thêm nhiễu Gaussian
NOISE_LEVEL = float(os.getenv("NOISE_LEVEL", "0.1"))  # Độ lớn nhiễu
USE_SCALER = os.getenv("USE_SCALER", "True").lower() == "true"  # Bật/tắt StandardScaler

# --- Cấu hình Huấn luyện ---
# Chọn loại mô hình để chạy thử nghiệm và tuning
# Các lựa chọn: "LogisticRegression", "RandomForestClassifier", "XGBClassifier", "SVC"
MODEL_TYPE = os.getenv("MODEL_TYPE", "XGBClassifier")  # Mặc định chạy XGBoost cuối cùng

# Siêu tham số để tune cho từng loại model (Mở rộng)
HYPERPARAMS_TO_TUNE = {
    "LogisticRegression": {
        "classifier__C": [
            0.01,
            0.1,
            1,
            10,
            100,
        ],  # Thêm tiền tố 'classifier__' vì dùng Pipeline
        "classifier__solver": ["liblinear", "lbfgs"],
        "classifier__max_iter": [100, 200, 500],
    },
    "RandomForestClassifier": {
        "classifier__n_estimators": [50, 100, 200],  # Thêm tiền tố 'classifier__'
        "classifier__max_depth": [10, 20, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
    },
    "XGBClassifier": {  # Thêm cấu hình cho XGBoost
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 6, 9],
        "classifier__learning_rate": [0.01, 0.1, 0.2],
        # Các tham số cố định có thể đặt trực tiếp khi khởi tạo model trong train.py
    },
    "SVC": {  # Giữ lại SVC nếu bạn vẫn muốn dùng
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__kernel": ["linear", "rbf"],
    },
}

# Lấy siêu tham số cho MODEL_TYPE đã chọn
MODEL_PARAMS_GRID = HYPERPARAMS_TO_TUNE.get(MODEL_TYPE, {})

# Metric chính để chọn model tốt nhất
PRIMARY_METRIC = os.getenv(
    "PRIMARY_METRIC", "accuracy"
)  # Có thể đổi thành f1_score_weighted nếu muốn

# --- Cấu hình Lưu Model (Thêm mới) ---
# Thư mục lưu model tốt nhất cục bộ (nếu cần, ngoài MLflow registry)
LOCAL_MODEL_OUTPUT_DIR = os.getenv("LOCAL_MODEL_OUTPUT_DIR", "models")
BEST_MODEL_FILENAME = os.getenv(
    "BEST_MODEL_FILENAME", f"best_{MODEL_TYPE.lower()}_model.pkl"
)
