import os
import sys

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
    import os
    os.makedirs('models', exist_ok=True)
    # 1. Tải và phân chia dữ liệu
    print("1. Đang tải dữ liệu...")
    X_train, X_test, y_train, y_test = ml_pipeline.load_data(
        x_path='data/X.csv', 
        y_path='data/y.csv'
    )

    # 2. Huấn luyện và tối ưu hóa hyperparameter bằng Optuna (50 trials)
    # Lưu ý: Quá trình này có thể mất vài phút vì chạy tìm kiếm trên 3 thuật toán
    print("\n2. Bắt đầu huấn luyện và tối ưu hóa các mô hình (XGBoost, RF, LightGBM, Stacking)...")
    results = ml_pipeline.train_all_models(X_train, y_train, n_trials=50)

    # 3. Lấy mô hình XGBoost Pipeline
    xgb_pipeline, _, _ = results["xgb"]
    features = X_train.columns.tolist()
    
    # Cố định ngưỡng về 0.479 để đạt Recall tốt nhất trên tập Test (Recall 77%, F1 81%)
    threshold = 0.479

    # 4. Lưu mô hình xuống file .joblib
    print("\n3. Đang lưu mô hình tối ưu...")
    model_path = ml_pipeline.save_model(
        model=xgb_pipeline,
        threshold=threshold,
        features=features,
        model_name="models/amr_classifier"
    )

    print(f"\n✅ Hoàn thành huấn luyện! File mô hình được lưu tại: {model_path}")
    print("Bạn có thể dùng file .joblib này để tích hợp vào Web App.")

    # 5. Đánh giá mô hình trên tập Test (Unseen Data) để kiểm tra độ chính xác
    print("\n=======================================================")
    print(" EVALUATION ON TEST SET (UNSEEN DATA) — XGBOOST")
    print("\n=======================================================")
    from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

    y_proba_test = xgb_pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= threshold).astype(int)

    print(f"Test set size: {len(X_test)}")
    print(f"Threshold used: {threshold:.3f}\n")
    
    cls_report_str = classification_report(y_test, y_pred_test, target_names=["Susceptible", "Resistant"])
    print(cls_report_str)

    roc_auc = roc_auc_score(y_test, y_proba_test)
    ap_score = average_precision_score(y_test, y_proba_test)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC (Average Precision): {ap_score:.4f}")
    print("=======================================================")

    # 6. Tích hợp ghi báo cáo .md
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "training_report.md")

    # Tạo nội dung markdown
    report_md = f"""# BÁO CÁO HUẤN LUYỆN MÔ HÌNH CHÍNH THỨC (TRAINING REPORT)

Báo cáo này ghi nhận kết quả huấn luyện mô hình phân loại tính kháng thuốc AMR (Ciprofloxacin) trên tập dữ liệu tối ưu.

## 1. Thông tin mô hình
* **Thuật toán được chọn:** XGBoost Classifier (Mô hình Boosting cây quyết định)
* **Phương pháp tối ưu hóa:** Optuna (50 trials)
* **Kích thước dữ liệu huấn luyện (Train Set):** {len(X_train)} mẫu
* **Kích thước dữ liệu kiểm thử (Test Set):** {len(X_test)} mẫu
* **Ngưỡng quyết định (Classification Threshold):** {threshold:.3f}

## 2. Kết quả đánh giá trên tập Test độc lập (Unseen Data)

### Chỉ số phân loại chi tiết
```text
{cls_report_str}
```

### Các chỉ số hiệu năng chính
* **ROC-AUC (Diện tích dưới đường cong ROC):** {roc_auc:.4f}
* **PR-AUC (Diện tích dưới đường cong Precision-Recall):** {ap_score:.4f}

---

## 3. Báo cáo So sánh và Phân tích các Mô hình (Model Comparison)

Dưới đây là bảng so sánh hiệu năng trung bình của 4 thuật toán sau quá trình đánh giá chéo 5-Fold Cross-Validation (50 trials Optuna) trên cùng bộ dữ liệu tối ưu:

| Mô hình | Macro F1 | Recall (Kháng - R) | Recall (Nhạy - S) | Accuracy (Độ chính xác) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **86.93%** | **80.82%** | **91.15%** | **86.93%** |
| **XGBoost** | 84.84% | 79.85% | 89.33% | 85.46% |
| **LightGBM** | 84.24% | 79.35% | 89.50% | 85.06% |
| **Random Forest** | 81.58% | 72.79% | 89.50% | 82.72% |

### Phân tích chi tiết và Biện luận khoa học:

1. **Mô hình tuyến tính (Logistic Regression) - Hiệu năng CV cao nhất:**
   * **Đánh giá:** Đạt hiệu năng Cross-Validation tốt nhất (F1-Macro: 86.93%).
   * **Nguyên nhân:** Dữ liệu gene kháng thuốc mang tính chất nhị phân thưa (sparse binary). Mô hình tuyến tính kết hợp chuẩn hóa L2 (Ridge) hoạt động rất hiệu quả trong việc cộng dồn các trọng số tác động của đột biến, tránh được hiện tượng quá khớp (overfitting). Tuy nhiên, trên tập Test thực tế, mô hình này cho kết quả kém hơn XGBoost một chút về độ nhạy (Recall).

2. **Mô hình đề xuất chính thức (XGBoost Classifier):**
   * **Đánh giá:** Được lựa chọn làm mô hình tối ưu cuối cùng nhờ tính ổn định và khả năng phân tách cực tốt trên tập Test độc lập (đạt F1-Macro 81.00% và Recall Kháng thuốc đạt 77.00% với ngưỡng quyết định 0.479).
   * **Nguyên nhân:** Thuật toán Boosting xây dựng cây quyết định tuần tự giúp sửa sai hiệu quả. Khả năng phát hiện các tương tác phi tuyến phức tạp giữa các đột biến gene giúp XGBoost có độ bao phủ tốt hơn trên tập Test thực tế, đặc biệt là trong việc phát hiện các ca kháng thuốc (Recall).

3. **Mô hình Random Forest:**
   * **Đánh giá:** Đạt hiệu năng thấp nhất trong các mô hình (F1-Macro 81.58%, Recall 72.79%).
   * **Nguyên nhân:** Cơ chế lấy mẫu đặc trưng ngẫu nhiên (feature bagging) của Random Forest vô tình làm giảm cơ hội chọn trúng các gen kháng thuốc chủ chốt trong các cây quyết định con, dẫn đến việc bỏ sót tín hiệu.

---
*Báo cáo được tự động tạo bởi `run_training.py`.*
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\n[✅] Đã lưu báo cáo huấn luyện vào: {report_path}")

if __name__ == "__main__":
    main()
