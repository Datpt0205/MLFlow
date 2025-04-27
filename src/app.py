# src/app.py
from flask import Flask, request, jsonify, render_template  # Thêm render_template
import traceback
import joblib
import os
import numpy as np

# import pandas as pd # Không cần pandas nữa nếu chỉ dùng numpy
from waitress import serve

import config  # Vẫn dùng config để lấy đường dẫn model

app = Flask(__name__)

# Biến toàn cục để giữ model
model = None
model_loaded_status = False  # Biến theo dõi trạng thái load model


def load_local_model():
    """Tải model từ file .pkl cục bộ"""
    global model, model_loaded_status
    output_dir = config.LOCAL_MODEL_OUTPUT_DIR
    if not config.MODEL_TYPE:
        print("!!! LOI: MODEL_TYPE trong config chua duoc dat.")
        model_loaded_status = False
        return False
    filename = f"best_{config.MODEL_TYPE.lower()}_model.pkl"
    model_path = os.path.join(output_dir, filename)

    print("\n--- Dang tai model truc tiep tu file cuc bo ---")
    print(f"Duong dan file: {model_path}")
    try:
        if not os.path.exists(model_path):
            print(f"!!! LOI: Khong tim thay file model tai: {model_path}")
            model_loaded_status = False
            return False
        model = joblib.load(model_path)
        if not hasattr(model, "predict"):
            print(
                f"!!! CANH BAO: Object duoc tai tu {model_path} khong co phuong thuc 'predict'."
            )
            model = None
            model_loaded_status = False
            return False

        print(f"*** Tai model tu '{model_path}' thanh cong! ***")
        model_loaded_status = True
        return True
    except Exception as e:
        print(f"!!! LOI khi tai model tu file '{model_path}': {e}")
        # print(traceback.format_exc()) # Bỏ comment nếu cần debug chi tiết
        model = None
        model_loaded_status = False
        return False


@app.route("/", methods=["GET"])
def home():
    """Hiển thị trang nhập liệu chính (Giữ nguyên)"""
    return render_template("index.html", prediction_result=None, error_message=None)


@app.route("/predict", methods=["POST"])
def predict():
    """Nhận dữ liệu JSON, dự đoán và trả về kết quả JSON"""
    if not model_loaded_status or model is None:
        # Trả về lỗi dạng JSON
        return jsonify({"error": "Model is not available"}), 503

    try:
        # **QUAN TRỌNG**: Kiểm tra xem JavaScript đang gửi gì
        # Nếu JS gửi { "features": [f1, f2, ...] } -> dùng request.get_json()['features']
        # Nếu JS gửi [f1, f2, ...] -> dùng request.get_json() trực tiếp
        data = request.get_json()  # Lấy dữ liệu JSON gửi lên

        # Kiểm tra xem data có phải là list không
        if not isinstance(data, list):
            # Nếu JS gửi dạng {"features": [...]}, thì lấy list đó ra
            if (
                isinstance(data, dict)
                and "features" in data
                and isinstance(data["features"], list)
            ):
                features_list = data["features"]
            else:
                return (
                    jsonify(
                        {
                            "error": "Invalid input format. Expecting a JSON list of features or {'features': [...]}."
                        }
                    ),
                    400,
                )
        else:
            features_list = data  # Nếu JS gửi thẳng list

        # Kiểm tra số lượng feature
        if len(features_list) != 10:  # Giả sử cần 10 features
            return (
                jsonify(
                    {
                        "error": f"Invalid input. Expected 10 features, got {len(features_list)}."
                    }
                ),
                400,
            )

        # Kiểm tra xem tất cả có phải là số không (có thể bỏ qua nếu tin tưởng input)
        if not all(isinstance(f, (int, float)) for f in features_list):
            return (
                jsonify({"error": "Invalid input. All features must be numeric."}),
                400,
            )

        # Chuyển thành numpy array 2D
        features_array = np.array([features_list])

        # Thực hiện dự đoán
        prediction = model.predict(features_array)
        prediction_value = prediction[0]  # Lấy kết quả

        # Trả về kết quả dự đoán dạng JSON
        # Ví dụ trả về: {"predictions": [0]} hoặc {"predictions": [1]}
        return jsonify(
            {"predictions": [prediction_value.item()]}
        )  # Dùng .item() để chuyển numpy type sang Python type nếu cần

    except Exception as e:
        print(f"Error during prediction: {e}")
        # print(traceback.format_exc()) # Bỏ comment nếu cần debug chi tiết
        # Trả về lỗi dạng JSON
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    # Tải model khi khởi động
    load_local_model()
    if not model_loaded_status:
        print(
            "!!! CANH BAO: Model chua duoc load thanh cong khi khoi dong app. UI co the khong hoat dong dung."
        )
    else:
        print("Model da san sang.")

    flask_port = 5555  # Đảm bảo port này không bị lỗi
    print(
        f"Khoi chay Flask server (co UI) bang Waitress tren http://0.0.0.0:{flask_port}"
    )
    print(
        f"Mo trinh duyet va truy cap: http://127.0.0.1:{flask_port}"
    )  # Hướng dẫn truy cập
    try:
        serve(app, host="0.0.0.0", port=flask_port)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"!!! LOI: Port {flask_port} da duoc su dung.")
        else:
            print(f"!!! LOI khi khoi chay Waitress: {e}")
    except Exception as e:
        print(f"!!! LOI khong xac dinh khi khoi chay Waitress: {e}")
