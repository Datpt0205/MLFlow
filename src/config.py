# src/config.py
import os

# --- Cấu hình MLflow ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Phan Loai Local Flask")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "BestOverallClassifierLocal") # Đổi tên để phản ánh là best tổng thể

# --- Cấu hình API (Flask) ---
API_MODEL_STAGE = os.getenv("API_MODEL_STAGE", "Staging")

# --- Cấu hình Dữ liệu ---
# (Giữ nguyên các cấu hình dữ liệu: N_SAMPLES_ORIGINAL, N_FEATURES, ..., USE_SCALER)
N_SAMPLES_ORIGINAL = int(os.getenv("N_SAMPLES_ORIGINAL", "1000"))
N_FEATURES = int(os.getenv("N_FEATURES", "10"))
N_CLASSES = int(os.getenv("N_CLASSES", "2"))
RANDOM_STATE_DATA = int(os.getenv("RANDOM_STATE_DATA", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.3"))
RANDOM_STATE_SPLIT = int(os.getenv("RANDOM_STATE_SPLIT", "42"))
USE_AUGMENTATION = os.getenv("USE_AUGMENTATION", "True").lower() == "true"
N_SAMPLES_AUGMENTED = int(os.getenv("N_SAMPLES_AUGMENTED", "2000"))
ADD_NOISE = os.getenv("ADD_NOISE", "True").lower() == "true"
NOISE_LEVEL = float(os.getenv("NOISE_LEVEL", "0.1"))
USE_SCALER = os.getenv("USE_SCALER", "True").lower() == "true"


# --- Cấu hình Huấn luyện ---
# Danh sách các loại model muốn thử nghiệm và tune
# Lấy key từ HYPERPARAMS_TO_TUNE làm danh sách model types
# Bỏ biến MODEL_TYPE đơn lẻ

# Siêu tham số để tune cho từng loại model (Giữ nguyên cấu trúc này)
HYPERPARAMS_TO_TUNE = {
    "LogisticRegression": {
        "classifier__C": [0.1, 1, 10], # Giảm bớt để chạy nhanh hơn khi test
        "classifier__solver": ["liblinear"],
        "classifier__max_iter": [200]
    },
    "RandomForestClassifier": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10, None],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 3]
    },
    "XGBClassifier": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 6],
        "classifier__learning_rate": [0.1, 0.2]
    },
    # "SVC": { # Bỏ comment nếu muốn chạy cả SVC
    #     "classifier__C": [0.1, 1.0],
    #     "classifier__kernel": ["rbf"]
    # }
}

# Lấy danh sách các model types từ dictionary trên
MODEL_TYPES_TO_RUN = list(HYPERPARAMS_TO_TUNE.keys())

# Bỏ MODEL_PARAMS_GRID vì sẽ lấy grid theo từng model type trong vòng lặp

# Metric chính để chọn model tốt nhất tổng thể
PRIMARY_METRIC = os.getenv("PRIMARY_METRIC", "accuracy")

# --- Cấu hình Lưu Model ---
LOCAL_MODEL_OUTPUT_DIR = os.getenv("LOCAL_MODEL_OUTPUT_DIR", "models")
# Đặt tên file cố định cho model tốt nhất tổng thể
BEST_OVERALL_MODEL_FILENAME = os.getenv("BEST_OVERALL_MODEL_FILENAME", "best_overall_model.pkl")
# Bỏ BEST_MODEL_FILENAME cũ phụ thuộc vào MODEL_TYPE