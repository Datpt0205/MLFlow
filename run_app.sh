#!/bin/bash

# --- Thêm đoạn code kill tiến trình cũ ---
FLASK_PORT=5555 # Định nghĩa port ở đây để dễ thay đổi

echo "[Setup] Kiem tra va dung tien trinh cu neu co tren port ${FLASK_PORT}..."

# Tìm PID của tiến trình đang listen trên port
# -t: Chỉ lấy PID
# -i:${FLASK_PORT}: Tìm theo port
# -sTCP:LISTEN: Chỉ tìm các kết nối TCP đang ở trạng thái LISTEN
PID_TO_KILL=$(lsof -t -i:${FLASK_PORT} -sTCP:LISTEN)

# Kiểm tra xem có tìm thấy PID không
if [ -n "$PID_TO_KILL" ]; then
  echo "[Setup] Tim thay tien trinh cu (PID: ${PID_TO_KILL}) dang chay tren port ${FLASK_PORT}. Dang dung..."
  # Kill tiến trình
  kill ${PID_TO_KILL}
  # Chờ một chút để đảm bảo tiến trình đã được kill
  sleep 2
  echo "[Setup] Da dung tien trinh cu."
else
  echo "[Setup] Khong tim thay tien trinh nao tren port ${FLASK_PORT}."
fi
# --- Kết thúc đoạn code kill ---

# --- Phần khởi động Flask app của bạn ---
echo "[Flask App] Dang chay ung dung Flask API bang Waitress tren port ${FLASK_PORT}..."
# Ví dụ lệnh chạy app (điều chỉnh nếu cần)
# Giả sử bạn chạy từ thư mục gốc mlflow/
# source .venv/bin/activate # Kích hoạt virtualenv nếu cần
python src/app.py
# --- Kết thúc phần khởi động ---