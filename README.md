# SVM cho nhận diện tiếng Việt vs tiếng H'Mông

Project này triển khai và kiểm chứng SVM nhị phân trên dữ liệu mô phỏng bằng tiếng Việt và tiếng H'Mông. Mục tiêu chính là minh họa cách SVM hoạt động trong bài toán phân loại văn bản, đồng thời kiểm chứng các khái niệm toán học nền tảng như margin, support vector, KKT, duality và kernel trick.

## 1. Tóm tắt bài toán

Bài toán ở đây là: cho một câu văn bản, xác định xem đó là tiếng Việt (+1) hay tiếng H'Mông (-1).

Dữ liệu được sinh ngẫu nhiên từ các từ vựng tiếng Việt và tiếng H'Mông, sau đó được biến đổi thành vector theo đặc trưng ký tự (char n-gram). Đây là cách phù hợp để mô hình hóa văn bản bằng SVM mà không cần xử lý quá sâu ngữ pháp.

## 2. Ý tưởng thuật toán

### 2.1 SVM tuyến tính

SVM tối ưu hóa:

- margin lớn nhất
- đồng thời giảm số lỗi phân loại

Với bài toán nhị phân, mô hình tìm w và b sao cho:

- y_i (w^T x_i + b) >= 1 nếu mẫu đúng lớp dương
- y_i (w^T x_i + b) <= -1 nếu mẫu đúng lớp âm

Trong project này, hai hướng giải được minh họa:

- Dual/SMO: giải bài toán đối ngẫu bằng SMO
- Primal/GD: giải trực tiếp trên primal bằng gradient descent với hinge loss

### 2.2 Kernel trick

Để chứng minh khả năng của SVM trên dữ liệu không tuyến tính, project còn cho chạy ví dụ hình học vòng tròn lồng nhau với:

- linear kernel: không phân tách tốt
- RBF kernel: phân tách tốt

Đây là minh chứng trực quan cho kernel trick: ánh xạ dữ liệu sang không gian có chiều cao hơn, nơi các lớp trở nên tuyến tính tách được.

## 3. Cấu trúc project

Project gồm các file chính sau:

- `lang_dataset.py`
  - Tạo từ vựng tiếng Việt và tiếng H'Mông
  - Sinh câu mô phỏng
  - Tạo dataset nhị phân
  - Có thể export CSV nếu cần

- `simple_svm.py`
  - Chứa các class và hàm toán học quan trọng của SVM
  - Bao gồm:
    - `linear_kernel()`
    - `polynomial_kernel()`
    - `rbf_kernel()`
    - `SimpleSVM_SMO`: triển khai SVM Dual qua SMO
    - `LinearSVM_Primal`: triển khai SVM Primal qua gradient descent

- `svm_binary.py`
  - File kiểm tra toán học và thực nghiệm đầy đủ cho SVM
  - Tạo dữ liệu, huấn luyện các mô hình, kiểm tra độ chính xác, KKT, support vector và kernel trick

- `main.py`
  - File test nhanh, dùng dataset thật từ `lang_dataset.py`
  - Chỉ dùng `simple_svm.py` và `lang_dataset.py`, không dùng sklearn cho mô hình
  - In ra accuracy, support vector, cosine similarity giữa w dual/primal và thời gian chạy

## 4. File map ý nghĩa

### `lang_dataset.py`

File này là bộ sinh dữ liệu. Nó không phải mô hình học, mà là nơi tạo ra các câu văn bản giả lập.

Nó chứa:

- `VI_WORDS`: từ vựng tiếng Việt
- `HMONG_WORDS`: từ vựng tiếng H'Mông
- `make_vi_sentence()`
- `make_hmong_sentence()`
- `build_language_dataset()`
- hàm trích xuất từ điển từ JSON nếu có sẵn

Mục đích: tạo dataset để SVM học nhận diện ngôn ngữ dựa trên đặc trưng ký tự (char n-gram).

### `simple_svm.py`

File này là “hạt nhân thuật toán”.

Nó chứa:

- giá trị kernel
- quy trình huấn luyện SMO
- quy trình gradient descent cho primal
- các hàm dự đoán và tính toán vector w, b

Đây là phần cốt lõi nếu bạn muốn hiểu thuật toán SVM từ đầu.

### `svm_binary.py`

File này là phiên bản kiểm tra đầy đủ hơn. Nó đánh giá:

- độ chính xác trên tập test
- điều kiện KKT
- sparsity của support vector
- định lý đối ngẫu mạnh
- kernel trick
- hiệu năng so sánh giữa các phương pháp

### `main.py`

File này là file test nhanh, sử dụng dataset thật từ `lang_dataset.py` và chỉ chạy các mô hình tự viết trong `simple_svm.py`.

Nó thực hiện:

- sinh dataset tiếng Việt/H'Mông
- chia train/test
- vector hóa văn bản theo char n-gram bằng `TfidfVectorizer`
- chạy 2 mô hình custom: Dual/SMO và Primal/GD
- in accuracy, support vector, độ tương đồng hướng `w`, và thời gian chạy

## 5. Cách chạy project

### Yêu cầu

- Python 3.10+
- NumPy
- pandas
- scikit-learn

Nếu chưa cài môi trường thì dùng:

```powershell
python -m pip install numpy pandas scikit-learn
```

### Cách chạy trong project này

Từ thư mục gốc của project:

```powershell
cd b:\CodeRun\Py\support_vector_machine
```

Chạy test nhanh:

```powershell
python main.py
```

Chạy phiên bản chi tiết hơn:

```powershell
python svm_binary.py
```

### Lưu ý trên Windows

Vì project in tiếng Việt, nếu terminal không bật UTF-8 thì có thể bị lỗi:

```text
UnicodeEncodeError: 'charmap' codec can't encode character ...
```

Cách khắc phục:

```powershell
chcp 65001
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python main.py
```

Hoặc chạy bằng venv sẵn có:

```powershell
cd b:\CodeRun\Py\support_vector_machine
.\.venv\Scripts\python.exe main.py
```

## 6. Thông số test hiện tại

Project hiện có 2 loại test chính:

### 6.1 Test nhanh (`main.py`)

`main.py` hiện chạy với các thông số cố định sau:

```python
np.random.seed(42)

dataset = build_language_dataset(n_per_class=120)
texts = dataset["text"].tolist()
labels = np.where(dataset["language"] == "vi", 1.0, -1.0)

texts_train, texts_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels,
)

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    max_features=150,
    sublinear_tf=True,
)

svm_dual = SimpleSVM_SMO(C=1.0, kernel="linear", max_passes=8)
svm_primal = LinearSVM_Primal(C=1.0, lr=0.01, n_iters=1500)
```

Các thông số này được thiết kế để:

- chạy nhanh hơn file test toán học đầy đủ
- vẫn đủ mạnh để đánh giá khả năng phân loại đúng trên dữ liệu thật
- định lượng được và có thể lặp lại

### 6.2 Test chi tiết hơn (`svm_binary.py`)

`svm_binary.py` vẫn là phiên bản chi tiết và kỹ lưỡng hơn, dùng dữ liệu lớn hơn và đánh giá nhiều hơn như:

- KKT
- support vector sparsity
- duality check
- kernel trick với dữ liệu hình học vòng tròn
- bảng tổng kết thời gian giữa các phương pháp

## 7. Kết quả chạy mẫu

### 7.1 Kết quả chạy `main.py`

Khi chạy `main.py`, kết quả này là dữ liệu thực tế trên môi trường hiện tại:

- Dataset: 240 samples (180 train / 60 test)
- Feature shape: (180, 150)
- Accuracy trên test: 1.0000 cho cả 2 mô hình custom
- Support vectors: 39/180 (21.67%)
- Dual vs Primal direction similarity: 0.9940
- Quick verdict: PASS

### 7.2 Kết quả chạy `svm_binary.py`

Trên môi trường hiện tại, `svm_binary.py` cho ra dạng kết quả sau:

- manual code - Dual/SMO: 5.3342s
- manual code - Primal/GD: 0.0790s
- scikit-learn - SVC(linear): 0.0070s
- scikit-learn - LinearSVC: 0.0040s
- Accuracy trên test: 1.0000 cho hầu hết mô hình
- KKT PASS
- Support vector nhỏ hơn 30%
- Dual và Primal có cùng hướng gần như tuyệt đối
- RBF kernel phân tách dữ liệu vòng tròn tốt

## 8. Ghi chú về tính lặp lại kết quả

Kết quả của 2 file có thể giống nhau hoặc chênh lệch nhẹ tùy phiên bản Python, NumPy và scikit-learn, máy chạy, và môi trường hệ thống.

- Thời gian chạy thường thay đổi rõ hơn so với accuracy
- Accuracy và logic kiểm tra thường ổn định nếu seed và cấu hình đã cố định
- Vì vậy, khi làm báo cáo hoặc demo, nên ghi rõ đây là kết quả chạy lần đầu trên môi trường hiện tại, không phải kết quả tuyệt đối cố định trên mọi máy

## 9. Bài test và kiểm định

Project có 2 dạng “test” chính:

### 9.1 Test logic/định lý toán học

Được chạy trong `svm_binary.py`.

Nó kiểm tra:

- `KKT` đối với alpha
- `support vector` nhỏ hơn 30% số mẫu train
- `w` từ primal và dual cùng hướng
- `linear kernel` kém trên dữ liệu phi tuyến
- `RBF kernel` tốt trên dữ liệu phi tuyến

### 9.2 Test thực nghiệm với dữ liệu ngôn ngữ

Được chạy trong `main.py`.

Nó kiểm tra:

- độ chính xác của 2 mô hình custom trên dữ liệu tiếng Việt/H'Mông
- thời gian chạy
- support vector và độ tương đồng hướng `w`
- khả năng phân loại thực tế với `lang_dataset.py`

## 10. Điểm cần lưu ý khi đọc code

- `SimpleSVM_SMO` là phiên bản học thuật, không tối ưu như libsvm
- `max_passes` và `tol` ảnh hưởng lớn đến độ hội tụ
- `TF-IDF char n-gram` là cách đánh giá văn bản phù hợp cho bài toán này
- `LinearSVC` trong scikit-learn sử dụng tối ưu hóa khác so với SVC/SMO, nên có thể khác một chút về nghiệm nhưng vẫn đạt cùng ý nghĩa

## 11. Nếu muốn mở rộng

Bạn có thể:

- tăng `n_per_class` trong `lang_dataset.py` hoặc `build_language_dataset()`
- thay đổi `max_features` trong `TfidfVectorizer`
- thử `poly` hoặc `rbf` kernel với `SimpleSVM_SMO`
- thay tên từ vựng hoặc thêm dữ liệu thực tế cho tiếng H'Mông
- chuyển dataset sang CSV hoặc JSON để đánh giá ngoài thực tế

## 12. Tóm tắt nhanh

Project này là một ví dụ đầy đủ để hiểu SVM từ nền tảng:

- dữ liệu văn bản
- trích chọn đặc trưng
- tối ưu hóa SVM
- kiểm chứng toán học
- so sánh với thư viện chuẩn

Đây là project rất tốt để học vì nó kết hợp cả lý thuyết và thực nghiệm.

## 8. Kết quả chạy mẫu và ghi chú thực tế

### 8.1 Kết quả chạy `main.py` (test nhanh)

Khi chạy `main.py` trên môi trường hiện tại, kết quả gần như sau:

- Dataset: 240 samples (180 train / 60 test)
- Feature shape: (180, 150)
- Accuracy trên test: 1.0000 cho tất cả các mô hình
- Support vectors: 39/180 (21.67%)
- Dual vs Primal direction similarity: ~0.9940
- Quick verdict: PASS

### 8.2 Kết quả chạy `svm_binary.py` (so sánh đầy đủ)

Khi chạy `svm_binary.py`, dữ liệu lớn hơn và kiểm tra nhiều hơn, kết quả thực tế có dạng như sau:

- manual code - Dual/SMO: 5.3342s
- manual code - Primal/GD: 0.0790s
- scikit-learn - SVC(linear): 0.0070s
- scikit-learn - LinearSVC: 0.0040s
- Accuracy trên test: 1.0000 cho hầu hết mô hình
- KKT PASS
- Support vector nhỏ hơn 30%
- Dual và Primal có cùng hướng gần như tuyệt đối
- RBF kernel phân tách dữ liệu vòng tròn tốt

### 8.3 Vì sao thực tế nên dùng sklearn thay vì code thuần thuật toán?

Mặc dù `SimpleSVM_SMO` và `LinearSVM_Primal` rất tốt để học, thì trong thực tế nên dùng scikit-learn vì các lý do sau:

- Cài đặt libsvm/liblinear đã được tối ưu về hiệu năng và ổn định hơn rất nhiều
- `SVC` và `LinearSVC` xử lý tốt hơn với tập dữ liệu lớn và nhiều đặc trưng
- Có kiểm soát tốt hơn về regularization, kernel, class weight, tol và max_iter
- Giảm rủi ro lỗi số học và tối ưu hóa cục bộ như trong phiên bản tự viết

Nói ngắn gọn: custom code phục vụ mục tiêu học thuật và minh họa toán học; sklearn phục vụ mục tiêu production và tốc độ thực thi.

### 8.4 Ghi chú về tính lặp lại của kết quả

Kết quả của 2 file này có thể giống nhau hoặc chênh lệch nhẹ tùy phiên bản Python, NumPy và scikit-learn, máy chạy, và môi trường hệ thống.

- Thời gian chạy thường thay đổi rõ hơn so với accuracy
- Accuracy và logic kiểm tra thường ổn định nếu seed và cấu hình đã cố định
- Vì vậy, khi làm báo cáo hoặc demo, nên ghi rõ đây là kết quả chạy lần đầu trên môi trường hiện tại, không phải kết quả tuyệt đối cố định trên mọi máy

## 9. Bài test và kiểm định

Project có 2 dạng “test” chính:

### 9.1 Test logic/định lý toán học

Được chạy trong `svm_binary.py`.

Nó kiểm tra:

- `KKT` đối với alpha
- `support vector` nhỏ hơn 30% số mẫu train
- `w` từ primal và dual cùng hướng
- `linear kernel` kém trên dữ liệu phi tuyến
- `RBF kernel` tốt trên dữ liệu phi tuyến

### 9.2 Test thực nghiệm với dữ liệu ngôn ngữ

Được chạy trong `main.py`.

Nó kiểm tra:

- độ chính xác của 4 mô hình trên dữ liệu tiếng Việt/H'Mông
- thời gian chạy
- support vector và độ tương đồng hướng `w`
- khả năng phân loại thực tế với `lang_dataset.py`

## 10. Điểm cần lưu ý khi đọc code

- `SimpleSVM_SMO` là phiên bản học thuật, không tối ưu như libsvm
- `max_passes` và `tol` ảnh hưởng lớn đến độ hội tụ
- `TF-IDF char n-gram` là cách đánh giá văn bản phù hợp cho bài toán này
- `LinearSVC` trong scikit-learn sử dụng tối ưu hóa khác so với SVC/SMO, nên có thể khác một chút về nghiệm nhưng vẫn đạt cùng ý nghĩa

## 11. Nếu muốn mở rộng

Bạn có thể:

- tăng `N_PER_CLASS` trong `svm_binary.py`
- thay đổi `max_features` trong `TfidfVectorizer`
- thử `poly` hoặc `rbf` kernel với `SimpleSVM_SMO`
- thay tên từ vựng hoặc thêm dữ liệu thực tế cho tiếng H'Mông
- chuyển dataset sang CSV hoặc JSON để đánh giá ngoài thực tế

## 12. Tóm tắt nhanh

Project này là một ví dụ đầy đủ để hiểu SVM từ nền tảng:

- dữ liệu văn bản
- trích chọn đặc trưng
- tối ưu hóa SVM
- kiểm chứng toán học
- so sánh với thư viện chuẩn

Đây là project rất tốt để học vì nó kết hợp cả lý thuyết và thực nghiệm.
