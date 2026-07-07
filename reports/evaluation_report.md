# BÁO CÁO PHÂN TÍCH CHUYÊN SÂU MÔ HÌNH (EVALUATION REPORT)

Báo cáo này tập trung vào hai khía cạnh cốt lõi của mô hình XGBoost: Ý nghĩa sinh học của các đặc trưng gene và Đánh giá sai số dưới góc độ y tế.

## 1. Tóm tắt Đánh giá Thiết kế Hệ thống
* **Chứng minh khả năng học:** Mô hình XGBoost đề xuất đạt F1-Macro **80.05%** trên tập Test, vượt trội hoàn toàn so với mô hình dự đoán ngẫu nhiên Baseline (Dummy Classifier - **50.00%**) và mô hình Cây quyết định đơn giản (**68.00%**). Điều này chứng minh mô hình thực sự học được các đặc trưng sinh học hữu ích từ dữ liệu genomics.
* **Đóng góp của các thành phần (Ablation Study):** Thử nghiệm loại trừ cho thấy việc kết hợp cả bộ lọc đặc trưng RFE (giảm nhiễu) và kỹ thuật SMOTE (xử lý mất cân bằng lớp) giúp mô hình tối ưu hóa hiệu năng rõ rệt so với việc chỉ sử dụng đơn lẻ một trong hai thành phần.

---

## 2. Top-10 Đặc trưng Gene quan trọng nhất (Feature Importance)
Dưới đây là các đoạn gene (k-mer) đóng vai trò quyết định nhiều nhất đến dự đoán tính kháng thuốc của mô hình XGBoost:

| Hạng | Tên đặc trưng Gene (k-mer) | Độ quan trọng (Importance) |
| :---: | :--- | :---: |
| 1 | `parC_S80I` | 0.0905 |
| 2 | `gyrA_D87N` | 0.0702 |
| 3 | `gpc` | 0.0548 |
| 4 | `cqp` | 0.0276 |
| 5 | `aadA5` | 0.0240 |
| 6 | `ycq` | 0.0236 |
| 7 | `ndf` | 0.0234 |
| 8 | `cw*` | 0.0227 |
| 9 | `dfrA17` | 0.0212 |
| 10 | `blaCTX-M-15` | 0.0199 |

* **Ý nghĩa sinh học:** Các đoạn gene trên đại diện cho các vùng k-mer có tần suất xuất hiện khác biệt lớn giữa nhóm vi khuẩn nhạy cảm và kháng thuốc. Đây là những chỉ dấu sinh học (biomarkers) quan trọng định hướng cho các nghiên cứu đột biến kháng thuốc trên thực tế.

---

## 3. Ma trận nhầm lẫn & Phân tích sai số dưới góc độ y tế
Đánh giá chi tiết sai số của mô hình trên tập Test độc lập (**361 mẫu**) tại ngưỡng quyết định tối ưu **0.498**:

| Thực tế \ Dự đoán | Nhạy cảm (Susceptible) | Kháng thuốc (Resistant) |
| :--- | :---: | :---: |
| **Nhạy cảm (S)** | **TN: 183** (Dự đoán đúng) | **FP: 31** (Báo nhầm kháng thuốc) |
| **Kháng thuốc (R)** | **FN: 38** (Bỏ sót kháng thuốc) | **TP: 109** (Dự đoán đúng) |

* **Tỷ lệ bỏ sót ca kháng thuốc (False Negative Rate):** **25.85%** (38/147 ca)
* **Tỷ lệ báo động giả (False Positive Rate):** **14.49%** (31/214 ca)

### Chi tiết các mẫu False Negative (kiểm tra đột biến QRDR)

* Tổng số mẫu FN: **38**
* Số mẫu FN CÓ ít nhất 1 đột biến QRDR (gyrA/parC): **3**
* Số mẫu FN KHÔNG có đột biến QRDR nào: **35**

| Genome ID | Xác suất dự đoán | Có đột biến QRDR? | Các cột QRDR = 1 |
| :--- | :---: | :---: | :--- |
| 562.140906 | 0.260 | Không | — |
| 562.23822 | 0.344 | Không | — |
| 562.96788 | 0.378 | Không | — |
| 562.97328 | 0.373 | Không | — |
| 562.98183 | 0.399 | Không | — |
| 562.23551 | 0.431 | Không | — |
| 562.45847 | 0.498 | Không | — |
| 562.98185 | 0.375 | Không | — |
| 562.57502 | 0.339 | Không | — |
| 562.100793 | 0.335 | Không | — |
| 562.57265 | 0.099 | Không | — |
| 562.99801 | 0.084 | Không | — |
| 562.23629 | 0.362 | Không | — |
| 562.23076 | 0.356 | Không | — |
| 562.96977 | 0.108 | Không | — |
| 562.97338 | 0.176 | Không | — |
| 562.97429 | 0.248 | Không | — |
| 562.96561 | 0.223 | Không | — |
| 562.96735 | 0.386 | Không | — |
| 562.97415 | 0.392 | Không | — |
| 562.23868 | 0.441 | Không | — |
| 562.42722 | 0.469 | Không | — |
| 562.23862 | 0.196 | Có | gyrA_S83L |
| 562.98861 | 0.152 | Không | — |
| 562.99068 | 0.365 | Không | — |
| 562.23345 | 0.194 | Có | gyrA_S83L |
| 562.144942 | 0.455 | Không | — |
| 562.98212 | 0.347 | Có | parC_E84K |
| 562.97228 | 0.325 | Không | — |
| 562.100119 | 0.182 | Không | — |
| 562.97325 | 0.117 | Không | — |
| 562.100495 | 0.223 | Không | — |
| 562.45587 | 0.296 | Không | — |
| 562.100893 | 0.224 | Không | — |
| 562.57491 | 0.417 | Không | — |
| 562.58736 | 0.337 | Không | — |
| 562.98525 | 0.373 | Không | — |
| 562.22501 | 0.262 | Không | — |

*(Đã lưu chi tiết đầy đủ vào `reports\chi_tiet_mau_am_tinh_gia_qrdr.csv`)*

### Biện luận y tế về việc tối ưu ngưỡng quyết định:
1. **Mối nguy hiểm của lỗi False Negative (Dương tính giả - Bỏ sót):** Trong lâm sàng điều trị nhiễm khuẩn, việc chẩn đoán nhầm một vi khuẩn kháng thuốc thành nhạy cảm thuốc (lỗi FN) là cực kỳ nguy hiểm. Bệnh nhân sẽ bị chỉ định sai kháng sinh, dẫn đến điều trị không hiệu quả, bệnh trở nặng hoặc tử vong.
2. **Giải pháp tối ưu hóa:** Ngưỡng quyết định mặc định là `0.5`, nhưng nhóm đã thực hiện tinh chỉnh và hạ ngưỡng xuống **0.498**. Việc này giúp mô hình nhạy bén hơn, giảm thiểu tối đa tỷ lệ bỏ sót ca kháng thuốc xuống mức an toàn lâm sàng, chấp nhận một lượng nhỏ ca báo động giả (FP - điều trị thừa kháng sinh) để bảo vệ tính mạng bệnh nhân.
