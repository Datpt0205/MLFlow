# src/utils.py
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,  # Import thêm
)


def evaluate_model(y_true, y_pred):
    """
    Tính toán các metrics đánh giá và in classification_report.
    """
    try:
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

        # Tạo dictionary metrics để log vào MLflow
        metrics = {
            "accuracy": accuracy,
            "f1_score_weighted": f1,
            "precision_weighted": precision,
            "recall_weighted": recall,
        }

        # In classification_report để xem chi tiết từng lớp
        print("\n--- Classification Report ---")
        try:
            # target_names có thể được truyền vào nếu bạn có tên cụ thể cho các lớp
            report = classification_report(y_true, y_pred, zero_division=0)
            print(report)
        except Exception as e_report:
            print(f"Loi khi tao classification_report: {e_report}")
        print("-----------------------------\n")

        return metrics

    except Exception as e:
        print(f"Loi khi tinh toan metrics chinh: {e}")
        # In classification_report ngay cả khi có lỗi tính metrics khác
        try:
            print("\n--- Classification Report (co loi metrics chinh) ---")
            report = classification_report(y_true, y_pred, zero_division=0)
            print(report)
            print("----------------------------------------------------\n")
        except Exception as e_report_err:
            print(
                f"Loi khi tao classification_report trong luc xu ly loi: {e_report_err}"
            )

        return {
            "accuracy": 0.0,
            "f1_score_weighted": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
        }
