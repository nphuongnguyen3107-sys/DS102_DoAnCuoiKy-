# BÁO CÁO HUẤN LUYỆN MÔ HÌNH CHÍNH THỨC (TRAINING REPORT)

Báo cáo này ghi nhận kết quả huấn luyện mô hình phân loại tính kháng thuốc AMR (Ciprofloxacin) trên tập dữ liệu tối ưu.

## 1. Thông tin mô hình
* **Thuật toán được chọn:** XGBoost Classifier (Mô hình Boosting cây quyết định)
* **Phương pháp tối ưu hóa:** Optuna (50 trials)
* **Kích thước dữ liệu huấn luyện (Train Set):** 2043 mẫu
* **Kích thước dữ liệu kiểm thử (Test Set):** 361 mẫu
* **Ngưỡng quyết định (Classification Threshold):** 0.479

## 2. Kết quả đánh giá trên tập Test độc lập (Unseen Data)

### Chỉ số phân loại chi tiết của mô hình được chọn (XGBoost)
```text
              precision    recall  f1-score   support

 Susceptible       0.84      0.85      0.84       214
   Resistant       0.77      0.77      0.77       147

    accuracy                           0.81       361
   macro avg       0.81      0.81      0.81       361
weighted avg       0.81      0.81      0.81       361

```

### Các chỉ số hiệu năng chính
* **ROC-AUC (Diện tích dưới đường cong ROC):** 0.9009
* **PR-AUC (Diện tích dưới đường cong Precision-Recall):** 0.8880

---

## 3. Báo cáo So sánh và Phân tích các Mô hình (Model Comparison)

### 3.1. Kết quả đánh giá trên tập Test độc lập (Independent Test Set)
Đây là bảng so sánh hiệu năng thực tế của các mô hình khi dự đoán trên dữ liệu hoàn toàn mới (**361 mẫu**):

| Mô hình | Ngưỡng (Threshold) | Accuracy | Macro F1 | Recall (Kháng - R) | Precision (Kháng - R) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | 0.479 | 81.44% | 80.76% | 76.87% | 77.40% | 0.9009 |
| **Logistic Regression** | 0.499 | 81.44% | 80.48% | 72.79% | 79.85% | 0.8913 |
| **Stacking Ensemble** | 0.523 | 80.33% | 79.69% | 76.87% | 75.33% | 0.8990 |
| **LightGBM** | 0.493 | 79.50% | 78.81% | 75.51% | 74.50% | 0.8902 |
| **Random Forest** | 0.467 | 76.73% | 76.29% | 77.55% | 69.09% | 0.8690 |


### 3.2. Kết quả đánh giá chéo 5-Fold Cross-Validation trên tập Train
Dưới đây là bảng so sánh hiệu năng trung bình của các thuật toán sau quá trình đánh giá chéo 5-Fold Cross-Validation (50 trials Optuna) trên cùng bộ dữ liệu tối ưu (**2043 mẫu**):

| Mô hình | Macro F1 | Recall (Kháng - R) | Recall (Nhạy - S) | Accuracy (Độ chính xác) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | 84.84% | 79.85% | 89.33% | 85.46% |
| **RandomForest** | 82.93% | 75.66% | 89.33% | 83.75% |
| **LightGBM** | 84.16% | 79.74% | 88.25% | 84.78% |
| **LogisticRegression** | 86.93% | 81.05% | 91.98% | 87.52% |


### Phân tích chi tiết và Biện luận khoa học:

1. **Mô hình tuyến tính (Logistic Regression) - Hiệu năng CV cao nhất:**
   * **Đánh giá:** Đạt hiệu năng Cross-Validation tốt nhất (F1-Macro: 86.93%).
   * **Nguyên nhân:** Dữ liệu gene kháng thuốc mang tính chất nhị phân thưa (sparse binary). Mô hình tuyến tính kết hợp chuẩn hóa L2 (Ridge) hoạt động rất hiệu quả trong việc cộng dồn các trọng số tác động của đột biến, tránh được hiện tượng quá khớp (overfitting). Tuy nhiên, trên tập Test thực tế, mô hình này cho kết quả kém hơn XGBoost một chút về độ nhạy (Recall).

2. **Mô hình đề xuất chính thức (XGBoost Classifier):**
   * **Đánh giá:** Được lựa chọn làm mô hình tối ưu cuối cùng nhờ tính ổn định và khả năng phân tách cực tốt trên tập Test độc lập.
   * **Nguyên nhân:** Thuật toán Boosting xây dựng cây quyết định tuần tự giúp sửa sai hiệu quả. Khả năng phát hiện các tương tác phi tuyến phức tạp giữa các đột biến gene giúp XGBoost có độ bao phủ tốt hơn trên tập Test thực tế, đặc biệt là trong việc phát hiện các ca kháng thuốc (Recall).

3. **Mô hình Random Forest:**
   * **Đánh giá:** Đạt hiệu năng thấp nhất trong các mô hình.
   * **Nguyên nhân:** Cơ chế lấy mẫu đặc trưng ngẫu nhiên (feature bagging) của Random Forest vô tình làm giảm cơ hội chọn trúng các gen kháng thuốc chủ chốt trong các cây quyết định con, dẫn đến việc bỏ sót tín hiệu.

---
*Báo cáo được tự động tạo bởi `run_report.py`.*
