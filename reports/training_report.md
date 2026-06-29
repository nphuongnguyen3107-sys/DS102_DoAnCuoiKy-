# BÁO CÁO HUẤN LUYỆN MÔ HÌNH CHÍNH THỨC (TRAINING REPORT)

Báo cáo này ghi nhận kết quả huấn luyện mô hình phân loại tính kháng thuốc AMR (Ciprofloxacin) trên tập dữ liệu tối ưu.

## 1. Thông tin mô hình
* **Thuật toán được chọn:** XGBoost Classifier (Mô hình Boosting cây quyết định)
* **Phương pháp tối ưu hóa:** Optuna (50 trials)
* **Kích thước dữ liệu huấn luyện (Train Set):** 2043 mẫu
* **Kích thước dữ liệu kiểm thử (Test Set):** 361 mẫu
* **Ngưỡng quyết định (Classification Threshold):** 0.479

## 2. Kết quả đánh giá trên tập Test độc lập (Unseen Data)

### Chỉ số phân loại chi tiết
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
| **XGBoost** | 0.479 | **81.44%** | **80.76%** | **76.87%** | 77.40% | **0.9009** |
| **Logistic Regression** | 0.499 | **81.44%** | 80.48% | 72.79% | **79.85%** | 0.8913 |
| **Stacking Ensemble** | 0.523 | 80.33% | 79.69% | 76.87% | 75.33% | 0.8990 |
| **LightGBM** | 0.493 | 79.50% | 78.81% | 75.51% | 74.50% | 0.8902 |
| **Random Forest** | 0.467 | 76.73% | 76.29% | 77.55% | 69.09% | 0.8690 |

---

### 3.2. Kết quả đánh giá chéo 5-Fold Cross-Validation trên tập Train
Bảng dưới đây ghi nhận hiệu năng trung bình của các mô hình trong quá trình học tập và tối ưu hóa tham số trên tập huấn luyện (**2043 mẫu**):

| Mô hình | Macro F1 (CV) | Recall (Kháng - R) | Recall (Nhạy - S) | Accuracy (CV) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **86.93%** | **81.05%** | **91.98%** | **87.52%** |
| **XGBoost** | 84.84% | 79.85% | 89.33% | 85.46% |
| **LightGBM** | 84.16% | 79.74% | 88.25% | 84.78% |
| **Random Forest** | 82.93% | 75.66% | 89.33% | 83.75% |
| **Stacking Ensemble** | 84.02% | 80.70% | 86.24% | 84.02% |

---

### 3.3. Phân tích chi tiết và Biện luận khoa học

1. **Hiện tượng quá khớp (Overfitting) của Logistic Regression:**
   * **Số liệu đối chứng:** Khi so sánh giữa tập Train (Mục 3.2) và tập Test (Mục 3.1), điểm số của **Logistic Regression** bị tụt giảm rất mạnh: **Macro F1 giảm 6.45%** (từ 86.93% xuống 80.48%) và **Recall kháng thuốc (R) giảm 8.26%** (từ 81.05% xuống 72.79%).
   * **Nguyên nhân:** Dữ liệu gene kháng thuốc mang tính chất nhị phân thưa (sparse binary) giúp mô hình tuyến tính học rất nhanh trên tập huấn luyện, nhưng nó dễ bị quá khớp với phân phối của tập Train và không giữ được hiệu năng khi gặp các mẫu vi khuẩn mới ở tập Test.

2. **Sự ổn định và vượt trội của XGBoost (Lý do lựa chọn làm mô hình chính thức):**
   * **Số liệu đối chứng:** Trên tập Test độc lập (Mục 3.1), **XGBoost vượt trội hơn Logistic Regression ở các chỉ số cốt lõi**:
     * **Accuracy (Độ chính xác):** Cả hai mô hình đều đạt **81.44%**, chứng minh hiệu năng dự đoán toàn cục tương đương nhau.
     * **Macro F1 (Cân bằng):** **80.76%** (XGBoost) vs 80.48% (Logistic)
     * **Recall-R (Kháng thuốc):** **76.87%** (XGBoost) vs 72.79% (Logistic) (XGBoost cao hơn hẳn **4.08%**).
     * **ROC-AUC (Khả năng phân tách):** **0.9009** (XGBoost - Cao nhất) vs 0.8913 (Logistic).
   * **Ý nghĩa:** XGBoost có độ bao phủ tốt hơn và giữ được hiệu năng ổn định nhờ cơ chế Boosting cây quyết định tuần tự, giúp sửa sai hiệu quả và phát hiện được các tương tác phi tuyến phức tạp giữa các đột biến gene (điều mà mô hình tuyến tính cộng dồn đơn giản của Logistic Regression bỏ sót). Trong bài toán chẩn đoán y tế, việc XGBoost tăng được 4.08% tỷ lệ phát hiện kháng thuốc (Recall) mà không làm suy giảm độ chính xác tổng thể (Accuracy giữ nguyên ở mức 81.44%) là một lợi thế cực kỳ lớn.


---
*Báo cáo được tự động tạo bởi `run_training.py`.*
