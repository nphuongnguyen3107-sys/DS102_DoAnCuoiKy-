import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# Chuyển thư mục làm việc về thư mục chứa script để tránh lỗi đường dẫn tương đối
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Đảm bảo stdout/stderr sử dụng UTF-8 để tránh UnicodeEncodeError trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import warnings
warnings.filterwarnings('ignore')

import ml_pipeline

def main():
    model_results_path = "models/all_trained_models.joblib"
    if not os.path.exists(model_results_path):
        print(f"[Error] Không tìm thấy file: {model_results_path}")
        print("Vui lòng chạy 'python run_training.py' một lần trước để huấn luyện và lưu mô hình.")
        sys.exit(1)

    # 1. Tải dữ liệu
    print("1. Đang tải dữ liệu...")
    X_train, X_test, y_train, y_test = ml_pipeline.load_data(
        x_path='data/X_rf.csv', 
        y_path='data/y.csv'
    )

    # 2. Tải kết quả mô hình đã lưu
    print(f"2. Đang tải kết quả mô hình từ {model_results_path}...")
    results = joblib.load(model_results_path)

    # 3. Lấy thông tin mô hình được chọn (XGBoost) và ngưỡng tối ưu được tối ưu hóa động
    xgb_pipeline, _, threshold = results["xgb"]

    y_proba_test = xgb_pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= threshold).astype(int)

    print("\n=======================================================")
    print(" EVALUATION ON TEST SET (UNSEEN DATA) — XGBOOST")
    print("=======================================================")
    print(f"Test set size: {len(X_test)}")
    print(f"Threshold used: {threshold:.3f}\n")
    
    cls_report_str = classification_report(y_test, y_pred_test, target_names=["Susceptible", "Resistant"])
    print(cls_report_str)

    roc_auc = roc_auc_score(y_test, y_proba_test)
    ap_score = average_precision_score(y_test, y_proba_test)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC (Average Precision): {ap_score:.4f}")
    print("=======================================================")

    # 4. Sinh lại báo cáo bằng hàm dùng chung
    print("3. Đang tái tạo báo cáo huấn luyện từ dữ liệu đã lưu...")
    report_path = "reports/training_report.md"
    from ml_pipeline.reporting import generate_and_save_training_report
    generate_and_save_training_report(X_train, X_test, y_train, y_test, results, report_path)

if __name__ == "__main__":
    main()
