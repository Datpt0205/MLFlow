# src/utils.py
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_model(y_true, y_pred):
    try:
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics = {
            "accuracy": accuracy, "f1_score_weighted": f1,
            "precision_weighted": precision, "recall_weighted": recall
        }
        # print(f"Calculated Metrics: {metrics}") # Bỏ bớt print nếu muốn gọn
        return metrics
    except Exception as e:
        print(f"Loi khi tinh toan metrics: {e}")
        return {"accuracy": 0.0, "f1_score_weighted": 0.0, "precision_weighted": 0.0, "recall_weighted": 0.0}