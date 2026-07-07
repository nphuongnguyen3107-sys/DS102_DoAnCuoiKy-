# Báo cáo Thử nghiệm Độ nhạy Đặc trưng k-mer (K-mer Feature Sensitivity Report)

Báo cáo này trình bày kết quả thử nghiệm sinh tin học và học máy nhằm đánh giá sự ảnh hưởng của số lượng đặc trưng k-mer đến hiệu năng dự đoán tính kháng thuốc **Ciprofloxacin** ở vi khuẩn *E. coli*.

---

## 1. Lý do sử dụng đặc trưng k-mer nền (Background k-mers)
Trong nghiên cứu kháng kháng sinh (AMR), bên cạnh các gen kháng thuốc trực tiếp (nhu nhóm gen mã hóa bơm thải thuốc hoặc beta-lactamase), kiểu hình kháng thuốc còn bị ảnh hưởng mạnh bởi:
- Các đột biến điểm trên gen đích (như vùng QRDR của gen *gyrA* và *parC*).
- Cấu trúc nền tảng của hệ gen/hệ protein vi khuẩn (phản ánh đặc điểm dòng tiến hóa, cấu trúc màng tế bào, tính thấm, v.v.).

**k-mer** (các đoạn axit amin/nucleotide độ dài $k$ cố định) được đếm từ chuỗi protein đóng vai trò là các chỉ dấu sinh học (biomarkers) đại diện cho thông tin nền này. Việc kết hợp **210 gen kháng thuốc AMR** (đặc trưng nhị phân) và **các đặc trưng k-mer** (đặc trưng liên tục) giúp mô hình học máy có cái nhìn toàn diện hơn.

---

## 2. Thiết kế Thử nghiệm
Chúng tôi tiến hành huấn luyện mô hình **XGBoost** kết hợp phương pháp chọn lọc đặc trưng RFE (Recursive Feature Elimination) trên 3 kịch bản số lượng k-mer nền khác nhau (sau khi đã xếp hạng độ quan trọng bằng Random Forest):
1. **Bộ dữ liệu 50 k-mer:** 210 gen AMR + 50 k-mer quan trọng nhất (Tổng cộng 260 đặc trưng).
2. **Bộ dữ liệu 100 k-mer:** 210 gen AMR + 100 k-mer quan trọng nhất (Tổng cộng 310 đặc trưng).
3. **Bộ dữ liệu 200 k-mer:** 210 gen AMR + 200 k-mer quan trọng nhất (Tổng cộng 410 đặc trưng).

---

## 3. Kết quả Hiệu năng Mô hình
Dưới đây là bảng so sánh hiệu năng (đánh giá bằng phân tách tập Test độc lập):

| Kịch bản | Số lượng k-mer | Accuracy (Độ chính xác) | Macro F1-score | ROC-AUC | Ghi chú |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Kịch bản 1** | 50 k-mers | 79.22% | 78.10% | 88.50% | Hiệu năng tốt nhưng bỏ sót một số tín hiệu dòng tiến hóa ẩn. |
| **Kịch bản 2** | **100 k-mers** | **81.44%** | **80.76%** | **90.09%** | **Tối ưu nhất: Đạt độ chính xác cao nhất, cân bằng giữa chi phí tính toán và hiệu năng sinh học.** |
| **Kịch bản 3** | 200 k-mers | 81.16% | 80.45% | 89.95% | Hiệu năng đi ngang hoặc giảm nhẹ do xuất hiện nhiễu thông tin (Overfitting) và tăng chi phí tính toán. |

---

## 4. Kết luận dành cho Hội đồng / Giảng viên
- **Ý nghĩa thực tiễn:** Việc chọn **100 k-mer** (Kịch bản 2) giúp mô hình XGBoost đạt hiệu năng dự đoán tối ưu nhất.
- **Tiết kiệm chi phí sinh học:** Kết hợp với thuật toán giảm chiều đặc trưng RFE ở bước sau giúp rút gọn tiếp từ 310 đặc trưng xuống chỉ còn **84 đặc trưng gen/k-mer cốt lõi** để đưa vào mô hình dự đoán. Điều này giúp giảm thiểu tối đa chi phí giải trình tự gen và xét nghiệm thực tế tại bệnh viện mà không làm suy giảm độ chính xác của chẩn đoán kháng thuốc.
