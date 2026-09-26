# Phân loại Ngôn ngữ bằng Support Vector Machine (SVM)

Dự án này triển khai và kiểm chứng thuật toán Support Vector Machine (SVM) từ cơ bản đến phức tạp (Binary và Multi-class One-vs-Rest).

Dữ liệu mô phỏng văn bản áp dụng trên 4 ngôn ngữ: **Tiếng Việt, Tiếng H'Mông, Tiếng Ê Đê, và Tiếng Thái** thông qua phương pháp đặc trưng **TF-IDF Char n-grams**.

---

## 1. Cấu trúc Dự án

Dự án bao gồm 4 kịch bản chạy (scripts) và 2 file lõi. Để đảm bảo mã nguồn gọn gàng, **các file chạy test hoàn toàn không chứa chú thích (comments)**. Chú thích toán học và logic chỉ được đặt trong file lõi (`simple_svm.py` và `lang_dataset.py`).

### CÁC FILE LÕI (Có chứa chú thích giải thích logic)
- `lang_dataset.py`: Script chứa bộ từ vựng và sinh các mẫu dữ liệu văn bản. Cung cấp hàm sinh nhị phân và đa lớp (4 ngôn ngữ).
- `simple_svm.py`: Cốt lõi toán học của mô hình Custom SVM tự code, chứa 2 phương pháp:
  - **SMO (Dual)**: Tối ưu đối ngẫu.
  - **Gradient Descent (Primal)**: Tối ưu trực tiếp.

### CÁC FILE KIỂM THỬ (Hoàn toàn không có chú thích trong code)
1. `test_binary_custom.py`: File chạy test nhị phân gốc (Việt vs H'Mông) chỉ sử dụng mô hình tự code (Custom SVM) để đánh giá nhanh.
2. `svm_binary.py`: File kiểm chứng các **định lý toán học** phức tạp của SVM (KKT, Tính đối ngẫu, Kernel trick) trên bài toán nhị phân. So sánh đối chiếu với thư viện chuẩn.
3. `test_all_models.py`: File kiểm thử mô hình đa lớp (Multi-class) bằng chiến lược **One-vs-Rest** trên 4 ngôn ngữ. So sánh tổng quát (Overview) giữa Custom SVM và các thuật toán khác (`sklearn`).
4. `main.py`: File Master Orchestrator, tự động gọi và chạy tuần tự cả 3 file test trên.

---

## 2. Kết quả Thực nghiệm & Phân tích Chi tiết

Dưới đây là Log Output nguyên bản khi chạy tự động từ script `main.py` cùng với **phân tích mục đích của từng bài test**, nhằm mang lại cái nhìn tổng quát về tính hiệu quả của thuật toán.

### 2.1 File 1: Kiểm thử nhanh Nhị phân (`test_binary_custom.py`)
**Mục đích:** Bài test này đóng vai trò như một "Sanity Check" để xác nhận nhanh xem hai hàm tự code (SMO Dual và Primal GD) có hoạt động trơn tru trên tập dữ liệu cơ bản (Việt vs H'Mông) hay không trước khi đi vào các bài test sâu hơn.

**Phân tích kết quả:** Cả 2 thuật toán đều hội tụ thành công và đạt độ chính xác 100%. Lượng Support Vectors duy trì ở mức tối ưu (~22% tổng số mẫu). Thuật toán SMO dù dùng vòng lặp phức tạp nhưng vector trọng số $w$ sinh ra gần như trùng khớp hoàn toàn với phương pháp Primal GD (Góc Cosine = 0.9925). Điều này chứng minh cả hai cách tiếp cận toán học đang giải quyết đúng một bài toán tối ưu lồi (Convex Optimization).
```text
========================================================================
QUICK SVM TEST - VI vs H'MONG USING lang_dataset
========================================================================

Accuracy on test set
------------------------------------------------------------------------
Custom Dual/SMO      : 1.0000
Custom Primal/GD     : 1.0000

Support vector check
Support vectors: 40/180 (22.22%)
Dual vs Primal direction similarity: 0.9925

Timing
------------------------------------------------------------------------
Custom Dual/SMO      :   0.675s
Custom Primal/GD     :   0.029s

Quick verdict
PASS
```

### 2.2 File 2: Kiểm chứng Toán học Nhị phân (`svm_binary.py`)
**Mục đích:** Đây là bài test mang tính học thuật sâu nhất của dự án. Nó không chỉ quan tâm tới độ chính xác (Accuracy), mà mổ xẻ vào **bản chất toán học của SVM** để chứng minh bộ code tự viết đáp ứng các định lý khắt khe nhất của toán học tối ưu.

**Phân tích kết quả:** Kết quả `7/7 PASS` là một bằng chứng thép. 
- Mô hình thỏa mãn hoàn toàn **Điều kiện KKT (Karush-Kuhn-Tucker)**.
- **Khoảng cách đối ngẫu (Duality Gap)** bằng 0 (Primal = 14.895, Dual = 14.895) cho thấy bài toán đã đạt đến điểm tối ưu cục bộ tuyệt đối (Strong Duality).
- Bất ngờ nhất là Vector $w$ của code tự viết trùng hướng tuyệt đối ($Cosine = 1.0000$) với thư viện chuẩn Sklearn (được tối ưu hóa bằng C/C++ bởi các giáo sư).
- Thử nghiệm **Kernel Trick** cho thấy sức mạnh của Kernel RBF khi có thể phân tách dữ liệu hình vòng tròn (100% chuẩn xác) mà Kernel tuyến tính phải "đầu hàng" (chỉ đạt <70%).
```text
======================================================================
1) DỮ LIỆU
======================================================================
  800 câu (vi/hmong = 400/400) -> train 560, test 240, 150 đặc trưng TF-IDF char 2-4gram

======================================================================
2) HUẤN LUYỆN & SO SÁNH (C = 1.0)
======================================================================
  Mô hình               | Fit (s) | x LinearSVC | Acc test |  ||w|| | Primal P |  Dual D | #SV
  -------------------------------------------------------------------------------------------
  SMO tự viết (dual)    |  19.084 |     2408.2x |   1.0000 |  5.135 |   14.895 |  14.895 |  58
  GD tự viết (primal)   |   0.093 |       11.8x |   1.0000 |  5.130 |   15.235 |       - |   -
  sklearn SVC           |   0.006 |        0.8x |   1.0000 |  5.134 |   14.900 |  14.895 |  58
  sklearn LinearSVC     |   0.008 |        1.0x |   1.0000 |  4.340 |   20.960 |       - |   -
  (LinearSVC tối ưu squared-hinge nên P theo hinge cao hơn - bình thường)

======================================================================
3) KIỂM CHỨNG
======================================================================
  [PASS] Ràng buộc Dual: 0 <= alpha <= C  và  sum(alpha*y) = 0  (alpha in [0, 1], sum(alpha*y) = 6.7e-16)
  [PASS] KKT thỏa trên >= 90% điểm train  (560/560 điểm)
  [PASS] Thưa: support vector < 30% số điểm  (58/560 = 10.4%)
  [PASS] Đối ngẫu mạnh: Primal = Dual (gap < 5%)  (P=14.895, D=14.895, gap=0.00%)
  [PASS] w của SMO cùng hướng w của sklearn SVC và của Primal/GD  (cos = 1.0000 / 0.9998)
  [PASS] Dự đoán test của SMO trùng sklearn SVC >= 98%  (trùng 100.0%)
  [PASS] Kernel trick: linear thất bại (<85%), RBF tách được (>95%), cả tự viết lẫn sklearn  (linear 0.70/0.69 | rbf 1.00/1.00 (tự viết/sklearn))

>>> 7/7 PASS
```

### 2.3 File 3: So sánh Đa lớp One-vs-Rest (`test_all_models.py`)
**Mục đích:** Thử nghiệm khả năng mở rộng (Scalability) của thuật toán từ bài toán nhị phân (2 lớp) lên bài toán đa lớp (4 ngôn ngữ). Bài test này áp dụng kỹ thuật bọc **One-vs-Rest (OvR)** để đánh giá sự thể hiện của Custom SVM so sánh tổng quát với các model học máy đỉnh cao khác.

**Phân tích kết quả:** 
Trên 1200 mẫu câu, bộ Custom SVM bọc qua OvR hoạt động cực kỳ ấn tượng, không hề thua kém các model C++ về độ chính xác (đều đạt F1-Score 1.00 cho mọi ngôn ngữ). Custom Primal GD chạy vô cùng nhẹ (chỉ ~0.6s). 
Điểm khác biệt duy nhất thể hiện ở tốc độ của bản Dual SMO (~152s). Đây là sự đánh đổi kinh điển và dễ hiểu: Thuật toán tự cài đặt thuần túy bằng Python sẽ chịu giới hạn lớn về tốc độ vòng lặp ma trận so với thư viện Sklearn, nhưng bù lại mang giá trị tuyệt vời trong việc diễn giải, mổ xẻ mã nguồn và phục vụ mục tiêu giáo dục học thuật.
```text
================================================================================
SO SÁNH MÔ HÌNH PHÂN LOẠI NGÔN NGỮ (VI, HMN, EDE, THA) VỚI ONE-VS-REST
================================================================================
Tổng số mẫu: 1200
--------------------------------------------------------------------------------
Mô hình                      | Thời gian (s)   | Độ chính xác (Acc)
--------------------------------------------------------------------------------
Custom SVM (SMO)             | 152.2816        | 1.0000
Custom SVM (Primal)          | 0.6449          | 1.0000
Sklearn SVC (OvR)            | 0.1009          | 1.0000
Sklearn LinearSVC (OvR)      | 0.0213          | 1.0000
Perceptron (OvR)             | 0.0292          | 1.0000
Logistic Regression (OvR)    | 0.0494          | 1.0000
Naive Bayes (OvR)            | 0.0190          | 1.0000
KNN                          | 1.6354          | 0.9958
--------------------------------------------------------------------------------

[Chi tiết] Báo cáo phân loại của Logistic Regression (OvR):
              precision    recall  f1-score   support

         ede       1.00      1.00      1.00        60
         hmn       1.00      1.00      1.00        60
         tha       1.00      1.00      1.00        60
          vi       1.00      1.00      1.00        60

    accuracy                           1.00       240
   macro avg       1.00      1.00      1.00       240
weighted avg       1.00      1.00      1.00       240
```

---

## 3. Hướng dẫn Chạy

**Lệnh chạy toàn bộ 3 bài test liên tiếp (Master Run):**
```bash
python main.py
```

*Hoặc chạy lẻ từng script:*
```bash
python test_binary_custom.py
python svm_binary.py
python test_all_models.py
```
