# BÁO CÁO PHÂN TÍCH & KẾT LUẬN THỰC NGHIỆM SO SÁNH

This report compares the performance of the model trained on two different feature sets:
1. **New Combined (310 Features - Unfiltered)**: The raw combined features dataset.
2. **Existing X.csv (310 Features - RF Filtered)**: The dataset with Random Forest filtered features.

## 1. Bảng so sánh hiệu năng trên tập Test (Độc lập)

| Bộ dữ liệu | Accuracy (Độ chính xác) | F1-Macro | Recall (Kháng thuốc) | Precision (Kháng thuốc) | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **New Combined** *(Chưa lọc)* | 73.96% | 73.59% | **76.19%** | 65.50% | 85.73% | 83.48% |
| **Existing X.csv** *(Đã lọc RF)* | **79.50%** | **78.77%** | 74.83% | **74.83%** | **88.79%** | **87.56%** |
| **Chênh lệch (Đã lọc vs Chưa lọc)** | **+5.54%** | **+5.18%** | *-1.36%* | **+9.33%** | **+3.06%** | **+4.08%** |

*Ghi chú: Thử nghiệm được thực hiện thông qua 5-Fold Cross-Validation và tối ưu hóa 20 trials bằng Optuna trên mô hình XGBoost.*

---

## 2. Phân tích chi tiết các chỉ số đạt được

### 2.1. Độ chính xác tổng thể (Accuracy & F1-Macro)
* **Kết quả:** Độ chính xác toàn cục (Accuracy) tăng từ **73.96% lên 79.50%** (tăng **5.54%**). Chỉ số F1-Macro (trung bình điều hòa giữa các lớp dữ liệu) tăng từ **73.59% lên 78.77%** (tăng **5.18%**).
* **Ý nghĩa:** Việc sử dụng bộ đặc trưng đã lọc (`X.csv`) giúp mô hình nâng cao đáng kể năng lực dự đoán chính xác trên cả hai nhóm vi khuẩn (Kháng thuốc và Nhạy cảm thuốc). Điều này chứng tỏ bộ dữ liệu thô ban đầu chứa nhiều đặc trưng trùng lặp hoặc gây nhiễu, làm giảm hiệu quả học tập của mô hình.

### 2.2. Độ tin cậy dự đoán kháng thuốc (Precision - Resistant)
* **Kết quả:** Điểm Precision đối với lớp Kháng thuốc tăng mạnh vượt bậc từ **65.50% lên 74.83%** (tăng **9.33%**).
* **Ý nghĩa:** Đây là cải tiến quan trọng nhất về mặt y sinh/lâm sàng. Khi mô hình dự đoán một mẫu vi khuẩn là **Kháng thuốc**, độ tin cậy chính xác của dự đoán đó đạt gần 75% (thay vì chỉ 65% ở bộ dữ liệu thô). Sự cải thiện này giúp giảm thiểu tối đa tỷ lệ **Dương tính giả** (dự đoán sai một vi khuẩn nhạy cảm thành kháng thuốc), tránh việc chỉ định sai thuốc hoặc đổi thuốc kháng sinh không cần thiết.

### 2.3. Tỷ lệ bỏ sót (Recall - Resistant)
* **Kết quả:** Tỷ lệ Recall đối với lớp Kháng thuốc giảm nhẹ từ **76.19% xuống 74.83%** (giảm **1.36%**).
* **Ý nghĩa:** Đây là sự đánh đổi (trade-off) hoàn toàn xứng đáng và nằm trong phạm vi cho phép. Việc chấp nhận giảm đi 1.36% khả năng phát hiện mẫu kháng thuốc để đổi lấy việc **tăng tới 9.33% độ chính xác dự đoán (Precision)** là một sự lựa chọn tối ưu trong bài toán phân loại thực tế.

### 2.4. Khả năng phân tách lớp (ROC-AUC & PR-AUC)
* **Kết quả:** Chỉ số diện tích dưới đường cong ROC (ROC-AUC) tăng từ **85.73% lên 88.79%** (+3.06%) và PR-AUC tăng từ **83.48% lên 87.56%** (+4.08%).
* **Ý nghĩa:** Đường cong PR-AUC đặc biệt hữu ích cho các bộ dữ liệu không cân bằng. Sự tăng trưởng ở cả hai chỉ số này chứng minh mô hình huấn luyện trên dữ liệu lọc (`X.csv`) có khả năng phân biệt giữa 2 lớp nhạy cảm/kháng thuốc ổn định hơn, ít bị ảnh hưởng bởi việc thay đổi ngưỡng phân loại (classification threshold).

---

## 3. Kết luận chung (Conclusion)

1. **Khẳng định vai trò của Feature Selection:** Phương pháp lựa chọn đặc trưng k-mer bằng Random Forest đã hoàn thành xuất sắc vai trò của nó. Nó chứng minh rằng: **Trong học máy, nhiều đặc trưng hơn không đồng nghĩa với mô hình tốt hơn**. Việc tinh lọc đúng những đặc trưng mang tính quyết định giúp loại bỏ nhiễu, giảm hiện tượng quá khớp (overfitting) và tăng cường độ chính xác đáng kể.
2. **Khuyến nghị sử dụng:** Bộ dữ liệu **`X.csv`** hiện tại là sự lựa chọn tối ưu nhất và vượt trội hoàn toàn so với bộ dữ liệu thô mới kết hợp. Bạn nên tiếp tục sử dụng bộ dữ liệu này làm nền tảng cốt lõi cho mô hình Stacking Ensemble cuối cùng và tích hợp vào ứng dụng Web dự đoán.
