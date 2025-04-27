#!/bin/bash

# --- Thêm đoạn code kill tiến trình cũ ---
MLFLOW_PORT=5000 # Port mặc định của MLflow server

echo "[Setup] Kiem tra va dung MLflow server cu neu co tren port ${MLFLOW_PORT}..."

# Tìm PID của tiến trình đang listen trên port 5000
# Sử dụng lsof tương tự như với Flask app
PID_TO_KILL=$(lsof -t -i:${MLFLOW_PORT} -sTCP:LISTEN)

# Kiểm tra xem có tìm thấy PID không
if [ -n "$PID_TO_KILL" ]; then
  echo "[Setup] Tim thay tien trinh MLflow server cu (PID: ${PID_TO_KILL}) dang chay tren port ${MLFLOW_PORT}. Dang dung..."
  # Kill tiến trình
  kill ${PID_TO_KILL}
  # Chờ một chút để đảm bảo tiến trình đã được kill
  sleep 2
  echo "[Setup] Da dung tien trinh MLflow server cu."
else
  echo "[Setup] Khong tim thay tien trinh MLflow server nao tren port ${MLFLOW_PORT}."
fi
# --- Kết thúc đoạn code kill ---


# --- Phần khởi động MLflow server (giữ nguyên logic đường dẫn tuyệt đối) ---
# Lấy đường dẫn tuyệt đối đến thư mục chứa script này
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
# Đường dẫn tuyệt đối đến thư mục mlartifacts
ARTIFACT_ROOT="file://${SCRIPT_DIR}/mlartifacts" # Sử dụng file:// và đường dẫn tuyệt đối

echo "[MLflow Server] Dang khoi chay MLflow Tracking Server cuc bo..."
echo "[MLflow Server] Backend Store: sqlite:///mlflow.db (relative to execution dir)"
echo "[MLflow Server] Artifact Store: ${ARTIFACT_ROOT}" # Hiển thị đường dẫn tuyệt đối
echo "[MLflow Server] Truy cap UI tai: http://127.0.0.1:${MLFLOW_PORT}" # Dùng biến port

# Tạo thư mục mlartifacts trong cùng thư mục với script nếu chưa có
mkdir -p "${SCRIPT_DIR}/mlartifacts"

# Khởi động server mới
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root "${ARTIFACT_ROOT}" \
    --host 127.0.0.1 \
    --port ${MLFLOW_PORT} # Dùng biến port