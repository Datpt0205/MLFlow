#!/bin/bash

# Hàm dọn dẹp: Dùng để dừng các tiến trình nền khi script kết thúc (ví dụ: bằng Ctrl+C)
cleanup() {
    echo -e "\n[run_all] Nhan duoc tin hieu dung... Dang dung cac tien trinh con..."
    # Kiểm tra xem biến PID có tồn tại và khác rỗng không
    if [ ! -z "$MLFLOW_SERVER_PID" ]; then
        # Gửi tín hiệu TERM trước, đợi một chút, sau đó gửi KILL nếu chưa dừng
        kill -TERM $MLFLOW_SERVER_PID > /dev/null 2>&1
        sleep 2
        kill -KILL $MLFLOW_SERVER_PID > /dev/null 2>&1
    fi
    if [ ! -z "$FLASK_APP_PID" ]; then
        kill -TERM $FLASK_APP_PID > /dev/null 2>&1
        sleep 2
        kill -KILL $FLASK_APP_PID > /dev/null 2>&1
    fi
    echo "[run_all] Da dung cac tien trinh con."
    exit 0
}

# Đặt bẫy (trap) để gọi hàm cleanup khi nhận tín hiệu INT (Ctrl+C) hoặc TERM
trap cleanup SIGINT SIGTERM

# --- Bước 1: Khởi chạy MLflow Server trong nền ---
./run_mlflow_server.sh &
MLFLOW_SERVER_PID=$! # Lấy Process ID của tiến trình vừa chạy nền

# Chờ một chút để MLflow server có thời gian khởi động hoàn toàn
sleep 10

# Kiểm tra xem server có chạy không (tùy chọn, kiểm tra cổng 5000)
# if ! nc -z 127.0.0.1 5000; then
#     echo "[run_all] LOI: Khong the ket noi den MLflow Server tren port 5000. Thoat."
#     cleanup # Dọn dẹp nếu server không chạy được
#     exit 1
# fi

# --- Bước 2: Chạy quá trình huấn luyện (chạy ở foreground) ---
echo "[run_all] Bat dau qua trinh huan luyen..."
./run_training.sh
TRAINING_EXIT_CODE=$? # Lấy mã thoát của tiến trình huấn luyện

if [ $TRAINING_EXIT_CODE -ne 0 ]; then
    echo "[run_all] LOI: Qua trinh huan luyen that bai (exit code: $TRAINING_EXIT_CODE)."
    cleanup # Dọn dẹp
    exit 1
fi
echo "[run_all] Qua trinh huan luyen hoan tat."

# --- Bước 3: Khởi chạy Flask App trong nền ---
./run_app.sh &
FLASK_APP_PID=$! # Lấy Process ID

wait $MLFLOW_SERVER_PID $FLASK_APP_PID