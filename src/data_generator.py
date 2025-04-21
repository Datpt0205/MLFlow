# src/data_generator.py
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from config import N_SAMPLES, N_FEATURES, N_CLASSES, RANDOM_STATE_DATA, TEST_SIZE, RANDOM_STATE_SPLIT

def get_data():
    X, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES, n_informative=max(2, N_FEATURES // 2), n_redundant=max(0, N_FEATURES // 4), n_repeated=0, n_classes=N_CLASSES, n_clusters_per_class=2, random_state=RANDOM_STATE_DATA)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE_SPLIT, stratify=y)
    print(f"Kich thuoc tap huan luyen: {X_train.shape}, Kich thuoc tap kiem tra: {X_test.shape}")
    return X_train, X_test, y_train, y_test