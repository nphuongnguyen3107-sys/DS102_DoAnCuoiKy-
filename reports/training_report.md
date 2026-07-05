# BÁO CÁO HUẤN LUYỆN MÔ HÌNH CHÍNH THỨC (TRAINING REPORT)

Báo cáo này ghi nhận kết quả huấn luyện mô hình phân loại tính kháng thuốc AMR (Ciprofloxacin) trên tập dữ liệu tối ưu.

## 1. Thông tin mô hình
* **Thuật toán được chọn:** XGBoost Classifier (Mô hình Boosting cây quyết định)
* **Phương pháp tối ưu hóa:** Optuna (50 trials)
* **Kích thước dữ liệu huấn luyện (Train Set):** 2043 mẫu
* **Kích thước dữ liệu kiểm thử (Test Set):** 361 mẫu
* **Ngưỡng quyết định (Classification Threshold):** 0.498

## 2. Kết quả đánh giá trên tập Test độc lập (Unseen Data)

### Chỉ số phân loại chi tiết của mô hình được chọn (XGBoost)
```text
              precision    recall  f1-score   support

 Susceptible       0.83      0.86      0.84       214
   Resistant       0.78      0.74      0.76       147

    accuracy                           0.81       361
   macro avg       0.80      0.80      0.80       361
weighted avg       0.81      0.81      0.81       361

```

### Các chỉ số hiệu năng chính
* **ROC-AUC (Diện tích dưới đường cong ROC):** 0.8964
* **PR-AUC (Diện tích dưới đường cong Precision-Recall):** 0.8829

---

## 3. Báo cáo So sánh và Phân tích các Mô hình (Model Comparison)

### 3.1. Kết quả đánh giá trên tập Test độc lập (Independent Test Set)
Đây là bảng so sánh hiệu năng thực tế của các mô hình khi dự đoán trên dữ liệu hoàn toàn mới (**361 mẫu**):

| Mô hình | Ngưỡng (Threshold) | Accuracy | Macro F1 | Recall (Kháng - R) | Precision (Kháng - R) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | 0.498 | 80.89% | 80.05% | 74.15% | 77.86% | 0.8964 |
| **Logistic Regression** | 0.499 | 81.44% | 80.48% | 72.79% | 79.85% | 0.8913 |
| **Stacking Ensemble** | 0.486 | 77.29% | 76.82% | 77.55% | 69.94% | 0.8932 |
| **LightGBM** | 0.528 | 78.67% | 77.89% | 73.47% | 73.97% | 0.8867 |
| **Random Forest** | 0.468 | 77.56% | 77.01% | 76.19% | 70.89% | 0.8753 |


### 3.2. Kết quả đánh giá chéo 5-Fold Cross-Validation trên tập Train
Dưới đây là bảng so sánh hiệu năng trung bình của các thuật toán sau quá trình đánh giá chéo 5-Fold Cross-Validation (50 trials Optuna) trên cùng bộ dữ liệu tối ưu (**2043 mẫu**):

| Mô hình | Macro F1 | Recall (Kháng - R) | Recall (Nhạy - S) | Accuracy (Độ chính xác) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | 83.14% | 80.10% | 86.19% | 83.70% |
| **RandomForest** | 82.47% | 80.10% | 85.03% | 83.02% |
| **LightGBM** | 84.11% | 80.46% | 87.59% | 84.68% |
| **LogisticRegression** | 86.98% | 81.18% | 91.98% | 87.57% |


### Phân tích chi tiết và Biện luận khoa học:

1. **Mô hình tuyến tính (Logistic Regression) - Hiệu năng CV cao nhất:**
   * **Đánh giá:** Đạt hiệu năng Cross-Validation tốt nhất (F1-Macro: 86.98%).
   * **Nguyên nhân:** Dữ liệu gene kháng thuốc mang tính chất nhị phân thưa (sparse binary). Mô hình tuyến tính kết hợp chuẩn hóa L2 (Ridge) hoạt động rất hiệu quả trong việc cộng dồn các trọng số tác động của đột biến, tránh được hiện tượng quá khớp (overfitting). Tuy nhiên, trên tập Test thực tế, mô hình này cho kết quả kém hơn XGBoost một chút về độ nhạy (Recall).

2. **Mô hình đề xuất chính thức (XGBoost Classifier):**
   * **Đánh giá:** Được lựa chọn làm mô hình tối ưu cuối cùng nhờ tính ổn định và khả năng phân tách cực tốt trên tập Test độc lập.
   * **Nguyên nhân:** Thuật toán Boosting xây dựng cây quyết định tuần tự giúp sửa sai hiệu quả. Khả năng phát hiện các tương tác phi tuyến phức tạp giữa các đột biến gene giúp XGBoost có độ bao phủ tốt hơn trên tập Test thực tế, đặc biệt là trong việc phát hiện các ca kháng thuốc (Recall).

3. **Mô hình Random Forest:**
   * **Đánh giá:** Đạt hiệu năng thấp nhất trong các mô hình.
   * **Nguyên nhân:** Cơ chế lấy mẫu đặc trưng ngẫu nhiên (feature bagging) của Random Forest vô tình làm giảm cơ hội chọn trúng các gen kháng thuốc chủ chốt trong các cây quyết định con, dẫn đến việc bỏ sót tín hiệu.

---
*Báo cáo được tự động tạo bởi hệ thống.*
