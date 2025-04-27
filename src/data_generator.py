# src/data_generator.py
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd  # Import pandas nếu cần xem dữ liệu

from src.config import (
    N_SAMPLES_ORIGINAL,
    N_FEATURES,
    N_CLASSES,
    RANDOM_STATE_DATA,
    TEST_SIZE,
    RANDOM_STATE_SPLIT,
    USE_AUGMENTATION,
    N_SAMPLES_AUGMENTED,
    ADD_NOISE,
    NOISE_LEVEL,
    USE_SCALER,
)


def get_data():
    """
    Tạo hoặc tải dữ liệu, áp dụng augmentation, noise, scaling nếu được cấu hình.
    Trả về: X_train, X_test, y_train, y_test
    """
    if USE_AUGMENTATION:
        print(f"--- Dang su dung Data Augmentation ---")
        print(f"Tao {N_SAMPLES_AUGMENTED} mau du lieu.")
        X, y = make_classification(
            n_samples=N_SAMPLES_AUGMENTED,
            n_features=N_FEATURES,
            n_informative=max(2, N_FEATURES // 2),
            n_redundant=max(0, N_FEATURES // 4),
            n_repeated=0,
            n_classes=N_CLASSES,
            n_clusters_per_class=2,
            random_state=RANDOM_STATE_DATA,
        )

        if ADD_NOISE:
            print(f"Them Gaussian noise voi muc: {NOISE_LEVEL}")
            noise = np.random.normal(0, NOISE_LEVEL, X.shape)
            X = X + noise
        else:
            print("Khong them noise.")

    else:
        print(f"--- Dang su dung Data goc ({N_SAMPLES_ORIGINAL} mau) ---")
        X, y = make_classification(
            n_samples=N_SAMPLES_ORIGINAL,
            n_features=N_FEATURES,
            n_informative=max(2, N_FEATURES // 2),
            n_redundant=max(0, N_FEATURES // 4),
            n_repeated=0,
            n_classes=N_CLASSES,
            n_clusters_per_class=2,
            random_state=RANDOM_STATE_DATA,
        )
        print("Khong them noise (chỉ áp dụng cho augmented data).")

    print(f"Chia du lieu: test_size={TEST_SIZE}, random_state={RANDOM_STATE_SPLIT}")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE_SPLIT,
        stratify=y,  # Giữ stratify để đảm bảo phân bố lớp trong train/test
    )

    if USE_SCALER:
        print("Ap dung StandardScaler cho du lieu.")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # Lưu scaler nếu cần deploy model sau này (ví dụ: log scaler như artifact trong MLflow)
        # Hoặc tích hợp scaler vào pipeline như trong train.py hiện tại
    else:
        print("Khong ap dung StandardScaler.")

    print(
        f"Kich thuoc tap huan luyen: {X_train.shape}, Kich thuoc tap kiem tra: {X_test.shape}"
    )
    return X_train, X_test, y_train, y_test


# if __name__ == '__main__':
#     # Test thử hàm get_data
#     X_train, X_test, y_train, y_test = get_data()
#     print("\nShapes sau khi chạy get_data:")
#     print("X_train:", X_train.shape)
#     print("X_test:", X_test.shape)
#     print("y_train:", y_train.shape)
#     print("y_test:", y_test.shape)
#     # print("\nDu lieu mau X_train (5 dong dau):")
#     # print(pd.DataFrame(X_train).head())
