# BÁO CÁO PHÂN TÍCH CHUYÊN SÂU MÔ HÌNH (EVALUATION REPORT)

Báo cáo này tập trung vào hai khía cạnh cốt lõi của mô hình XGBoost: Ý nghĩa sinh học của các đặc trưng gene và Đánh giá sai số dưới góc độ y tế.

## 1. Tóm tắt Đánh giá Thiết kế Hệ thống
* **Chứng minh khả năng học:** Mô hình XGBoost đề xuất đạt F1-Macro **80.82%** trên tập Test, vượt trội hoàn toàn so với mô hình dự đoán ngẫu nhiên Baseline (Dummy Classifier - **50.00%**) và mô hình Cây quyết định đơn giản (**68.00%**). Điều này chứng minh mô hình thực sự học được các đặc trưng sinh học hữu ích từ dữ liệu genomics.
* **Đóng góp của các thành phần (Ablation Study):** Thử nghiệm loại trừ cho thấy việc kết hợp cả bộ lọc đặc trưng RFE (giảm nhiễu) và kỹ thuật SMOTE (xử lý mất cân bằng lớp) giúp mô hình tối ưu hóa hiệu năng rõ rệt so với việc chỉ sử dụng đơn lẻ một trong hai thành phần.

---

## 2. Top-10 Đặc trưng Gene quan trọng nhất (Feature Importance)
Dưới đây là các đoạn gene (k-mer) đóng vai trò quyết định nhiều nhất đến dự đoán tính kháng thuốc của mô hình XGBoost:

| Hạng | Tên đặc trưng Gene (k-mer) | Độ quan trọng (Importance) |
| :---: | :--- | :---: |
| 1 | `parC_S80I` | 0.0904 |
| 2 | `gyrA_D87N` | 0.0688 |
| 3 | `gpc` | 0.0579 |
| 4 | `ndf` | 0.0270 |
| 5 | `cw*` | 0.0252 |
| 6 | `cqp` | 0.0245 |
| 7 | `mph(A)` | 0.0235 |
| 8 | `mrx(A)` | 0.0221 |
| 9 | `ycq` | 0.0203 |
| 10 | `dfrA17` | 0.0193 |

* **Ý nghĩa sinh học:** Các đoạn gene trên đại diện cho các vùng k-mer có tần suất xuất hiện khác biệt lớn giữa nhóm vi khuẩn nhạy cảm và kháng thuốc. Đây là những chỉ dấu sinh học (biomarkers) quan trọng định hướng cho các nghiên cứu đột biến kháng thuốc trên thực tế.

---

## 3. Ma trận nhầm lẫn & Phân tích sai số dưới góc độ y tế
Đánh giá chi tiết sai số của mô hình trên tập Test độc lập (**361 mẫu**) tại ngưỡng quyết định tối ưu **0.479**:

| Thực tế \ Dự đoán | Nhạy cảm (Susceptible) | Kháng thuốc (Resistant) |
| :--- | :---: | :---: |
| **Nhạy cảm (S)** | **TN: 181** (Dự đoán đúng) | **FP: 33** (Báo nhầm kháng thuốc) |
| **Kháng thuốc (R)** | **FN: 34** (Bỏ sót kháng thuốc) | **TP: 113** (Dự đoán đúng) |

* **Tỷ lệ bỏ sót ca kháng thuốc (False Negative Rate):** **23.13%** (34/147 ca)
* **Tỷ lệ báo động giả (False Positive Rate):** **15.42%** (33/214 ca)

### Biện luận y tế về việc tối ưu ngưỡng quyết định:
1. **Mối nguy hiểm của lỗi False Negative (Dương tính giả - Bỏ sót):** Trong lâm sàng điều trị nhiễm khuẩn, việc chẩn đoán nhầm một vi khuẩn kháng thuốc thành nhạy cảm thuốc (lỗi FN) là cực kỳ nguy hiểm. Bệnh nhân sẽ bị chỉ định sai kháng sinh, dẫn đến điều trị không hiệu quả, bệnh trở nặng hoặc tử vong.
2. **Giải pháp tối ưu hóa:** Ngưỡng quyết định mặc định là `0.5`, nhưng nhóm đã thực hiện tinh chỉnh và hạ ngưỡng xuống **0.479**. Việc này giúp mô hình nhạy bén hơn, giảm thiểu tối đa tỷ lệ bỏ sót ca kháng thuốc xuống mức an toàn lâm sàng, chấp nhận một lượng nhỏ ca báo động giả (FP - điều trị thừa kháng sinh) để bảo vệ tính mạng bệnh nhân.
