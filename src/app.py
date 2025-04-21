# src/app.py
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string # render_template nếu có file html riêng
import time
import os
import traceback

# Import cấu hình
from config import MLFLOW_TRACKING_URI, MLFLOW_MODEL_NAME, API_MODEL_STAGE, N_FEATURES

# --- Khởi tạo Flask app ---
app = Flask(__name__)

# --- Cấu hình và tải model ---
MODEL_NAME = MLFLOW_MODEL_NAME
STAGE = API_MODEL_STAGE # Stage được cấu hình để API load
model = None
model_uri_loaded = "Chua tai"
model_load_error = None

def load_production_model():
    """Tải model từ MLflow Registry dựa trên STAGE đã cấu hình."""
    global model, model_uri_loaded, model_load_error
    model = None # Reset state
    model_load_error = None
    print(f"\n--- Dang tai model cho API ---")
    print(f"Ket noi toi MLflow: {MLFLOW_TRACKING_URI}")
    print(f"Model Name: {MODEL_NAME}")
    print(f"Stage can tai: {STAGE}")

    if not STAGE or STAGE.lower() == 'none':
        model_load_error = "API_MODEL_STAGE chua duoc cau hinh hop le (can Staging hoac Production)."
        print(f"LOI: {model_load_error}")
        return

    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    model_uri_loaded = model_uri # Cập nhật URI sẽ thử tải

    try:
        # Đặt tracking URI trước khi load
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print(f"Dang thu tai model tu URI: {model_uri}...")
        # Thêm độ trễ nhỏ phòng trường hợp registry chưa cập nhật kịp
        time.sleep(3)
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        print(f"Tai model '{MODEL_NAME}' stage '{STAGE}' thanh cong!")

    except Exception as e:
        print(f"!!! LOI khi tai model tu registry: {e}")
        print(traceback.format_exc())
        model_load_error = f"Khong the tai model '{MODEL_NAME}/{STAGE}'. Loi: {e}"
        # API vẫn chạy nhưng sẽ báo lỗi ở endpoint predict

# --- Tải model ngay khi ứng dụng khởi chạy ---
# Trong Flask, không có sự kiện startup chuẩn như FastAPI, nên tải model lần đầu
# khi có request đầu tiên hoặc thực hiện việc này trước khi chạy app.
# Cách đơn giản là tải trong một hàm riêng và gọi nó.
load_production_model()

# --- Định nghĩa Route ---

# Route kiểm tra sức khỏe
@app.route("/health", methods=['GET'])
def health_check():
    if model is None:
        # Thử tải lại model nếu lần trước thất bại
        print("Health check: Model chua co, thu tai lai...")
        load_production_model()

    status = "OK" if model is not None else "ERROR"
    return jsonify({
        "status": status,
        "model_name": MODEL_NAME,
        "model_stage_configured": STAGE,
        "model_uri_attempted": model_uri_loaded,
        "model_load_error": model_load_error
    })

# Route dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    global model # Đảm bảo dùng biến model toàn cục
    if model is None:
        # Kiểm tra lại nếu model chưa tải được
        if model_load_error:
            return jsonify({"error": f"Model khong san sang: {model_load_error}"}), 503
        else:
            # Thử tải lại lần nữa nếu chưa có lỗi cụ thể
             load_production_model()
             if model is None:
                  return jsonify({"error": f"Model khong san sang sau khi thu lai: {model_load_error}"}), 503

    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "Du lieu khong hop le. Can key 'features' la mot list."}), 400

        features = data['features']
        if not isinstance(features, list) or len(features) != N_FEATURES:
             return jsonify({"error": f"Key 'features' phai la list co dung {N_FEATURES} phan tu."}), 400

        # Chuyển đổi thành DataFrame
        input_df = pd.DataFrame([features])

        # Dự đoán
        predictions = model.predict(input_df)
        predictions_list = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions

        print(f"Input: {features}, Prediction: {predictions_list}") # Log lại
        return jsonify({
            "predictions": predictions_list,
            "model_served": model_uri_loaded # Trả về URI đã cấu hình để load
        })

    except Exception as e:
        print(f"Loi trong qua trinh predict: {e}")
        print(traceback.format_exc())
        return jsonify({"error": f"Loi server khi du doan: {str(e)}"}), 500

# Route trang chủ đơn giản (Tùy chọn)
@app.route('/', methods=['GET'])
def home():
    # Cung cấp một form HTML đơn giản để test
    form_html = f"""
    <!DOCTYPE html>
    <html>
    <head><title>MLflow Flask API</title></head>
    <body>
        <h1>Du doan voi Model '{MODEL_NAME}' (Stage: {STAGE})</h1>
        <p>Model URI da load: {model_uri_loaded}</p>
        <p>Trang thai model: {'OK' if model else 'ERROR'}</p>
        {f'<p style="color:red;">Loi load model: {model_load_error}</p>' if model_load_error else ''}
        <hr>
        <form id="predictForm">
            <label for="features">Nhap {N_FEATURES} features (cach nhau boi dau phay):</label><br>
            <input type="text" id="features" name="features" size="50" value="{','.join(['0.5']*N_FEATURES)}"><br><br>
            <button type="button" onclick="sendPrediction()">Du doan</button>
        </form>
        <hr>
        <h2>Ket qua:</h2>
        <pre id="result"></pre>

        <script>
            function sendPrediction() {{
                const featuresInput = document.getElementById('features').value;
                const featuresArray = featuresInput.split(',').map(Number);
                const resultDiv = document.getElementById('result');
                resultDiv.textContent = 'Dang gui yeu cau...';

                if (featuresArray.length !== {N_FEATURES} || featuresArray.some(isNaN)) {{
                    resultDiv.textContent = 'Loi: Vui long nhap dung {N_FEATURES} so cach nhau boi dau phay.';
                    return;
                }}

                fetch('/predict', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ features: featuresArray }})
                }})
                .then(response => response.json())
                .then(data => {{
                    resultDiv.textContent = JSON.stringify(data, null, 2);
                }})
                .catch(error => {{
                    resultDiv.textContent = 'Loi khi goi API: ' + error;
                }});
            }}
        </script>
    </body>
    </html>
    """
    return render_template_string(form_html)


# --- Chạy Flask app bằng Waitress cho production-like local ---
if __name__ == '__main__':
    from waitress import serve
    print(f"\nKhoi chay Flask server bang Waitress tren http://0.0.0.0:5001")
    print(f"API se su dung model: '{MODEL_NAME}' stage '{STAGE}'")
    if model is None:
         print(f"!!! CANH BAO: Model chua duoc load thanh cong. API co the khong hoat dong dung. !!!")
         print(f"!!! Loi chi tiet: {model_load_error} !!!")
    # Chạy server waitress thay vì app.run() của Flask
    serve(app, host='0.0.0.0', port=5001)