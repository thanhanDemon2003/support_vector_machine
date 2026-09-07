"""
SVM NHỊ PHÂN: TIẾNG VIỆT vs TIẾNG H'MÔNG - TRỌNG TÂM TOÁN HỌC

Bài toán: phân loại 1 câu là TIẾNG VIỆT (+1) hay TIẾNG H'MÔNG (-1).
Đây là bài toán con thực tế có logic "nếu confidence thấp -> mặc định về tiếng Việt" - tức bài toán nhị
phân "có phải tiếng Việt hay không" chính là lõi quan trọng nhất
trong hệ thống nhận diện ngôn ngữ của bạn.
"""

import time
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC

from simple_svm import SimpleSVM_SMO, LinearSVM_Primal
from lang_dataset import make_vi_sentence, make_hmong_sentence

np.random.seed(42)


# ================================================================
# MINI TEST FRAMEWORK
# ================================================================
_results = []

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    _results.append((name, status))
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return condition

def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ================================================================
# 1) TẠO DỮ LIỆU: tiếng Việt (+1) vs H'Mông (-1), LƯỢNG LỚN
# ================================================================
section("BƯỚC 1 - TẠO DỮ LIỆU (vi vs hmong, mô phỏng)")

N_PER_CLASS = 400
# 400 câu/lớp = 800 câu tổng - đã kiểm tra chạy hội tụ đầy đủ trong
# ~22s. Có thể tăng lên nếu bạn muốn nhiều dữ liệu hơn, nhưng thời
# gian chạy SMO manual code sẽ tăng nhanh (gần bậc 2 theo n, đã phân
# tích ở phần "độ phức tạp" trước đó).

vi_texts = [make_vi_sentence() for _ in range(N_PER_CLASS)]
hmn_texts = [make_hmong_sentence() for _ in range(N_PER_CLASS)]
texts = vi_texts + hmn_texts
labels = np.array([1] * N_PER_CLASS + [-1] * N_PER_CLASS, dtype=float)

print(f"Tổng số câu: {len(texts)} ({N_PER_CLASS} tiếng Việt + {N_PER_CLASS} H'Mông)")
print(f"Ví dụ câu tiếng Việt : {vi_texts[0]}")
print(f"Ví dụ câu H'Mông     : {hmn_texts[0]}")

texts_train, texts_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)

# --- TF-IDF ký tự (char n-gram) - GIỐNG kiến trúc trainer.py gốc ---
# max_features giảm từ 50000 (bản gốc) xuống 150 vì SimpleSVM_SMO tự
# viết cần ma trận kernel O(n^2) và KHÔNG hỗ trợ ma trận thưa (sparse)
# như scikit-learn - đây là giới hạn thực tế cần nhớ khi đọc kết quả.
tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4),
                          max_features=150, sublinear_tf=True)
X_train = tfidf.fit_transform(texts_train).toarray()
X_test = tfidf.transform(texts_test).toarray()

print(f"\nTrain: {X_train.shape[0]} mẫu | Test: {X_test.shape[0]} mẫu | "
      f"Đặc trưng: {X_train.shape[1]} chiều")


# ================================================================
# 2) HUẤN LUYỆN 4 MÔ HÌNH: 2 manual code + 2 scikit-learn
# ================================================================
section("BƯỚC 2 - HUẤN LUYỆN (Dual manual code / Primal manual code / scikit-learn)")

# --- 2.1 Dual (SMO) manual code ---
t0 = time.time()
svm_dual = SimpleSVM_SMO(C=1.0, kernel="linear", max_passes=10)
svm_dual.fit(X_train, y_train)
t_dual = time.time() - t0
print(f"manual code - Dual/SMO       : fit_time={t_dual:8.4f}s  "
      f"| SV={len(svm_dual.support_)}/{len(y_train)}")

# --- 2.2 Primal (Gradient Descent trên hinge loss) manual code ---
t0 = time.time()
svm_primal = LinearSVM_Primal(C=1.0, lr=0.001, n_iters=3000)
svm_primal.fit(X_train, y_train)
t_primal = time.time() - t0
print(f"manual code - Primal/GD      : fit_time={t_primal:8.4f}s")

# --- 2.3 scikit-learn SVC (libsvm, kernel linear) - tương đương lý
#     thuyết CHÍNH XÁC NHẤT với Dual/SMO manual code (cùng giải 1 bài
#     toán QP hinge-loss, không phải squared-hinge như LinearSVC) ---
t0 = time.time()
sk_svc = SVC(C=1.0, kernel="linear")
sk_svc.fit(X_train, y_train)
t_svc = time.time() - t0
print(f"scikit-learn - SVC(linear): fit_time={t_svc:8.4f}s  "
      f"| SV={len(sk_svc.support_)}/{len(y_train)}")

# --- 2.4 scikit-learn LinearSVC (liblinear) - ĐÚNG thuật toán mà
#     build_language_pipeline() trong trainer.py gốc của bạn dùng ---
t0 = time.time()
sk_linsvc = LinearSVC(max_iter=2000, random_state=42, class_weight="balanced")
sk_linsvc.fit(X_train, y_train)
t_linsvc = time.time() - t0
print(f"scikit-learn - LinearSVC  : fit_time={t_linsvc:8.4f}s")


# ================================================================
# 3) BỘ TEST LOGIC TOÁN HỌC (áp dụng TRỰC TIẾP trên dữ liệu vi/hmong)
# ================================================================
section("BƯỚC 3 - KIỂM CHỨNG CÁC TÍNH CHẤT TOÁN HỌC")

print("\n--- Test 3.1: Độ chính xác (mọi mô hình nên gần bằng nhau,"
      " vì cùng giải 1 bài toán tối ưu lồi) ---")
for name, model, is_custom in [
    ("manual code - Dual/SMO", svm_dual, True),
    ("manual code - Primal/GD", svm_primal, True),
    ("scikit-learn - SVC(linear)", sk_svc, False),
    ("scikit-learn - LinearSVC", sk_linsvc, False),
]:
    pred = model.predict(X_test)
    acc = np.mean(pred == y_test)
    print(f"  {name:28s}: test accuracy = {acc:.4f}")

print("\n--- Test 3.2: Điều kiện KKT (complementary slackness) ---")
# Với mỗi điểm huấn luyện, alpha_i phải rơi vào đúng 1 trong 3 trường
# hợp suy ra từ complementary slackness (xem giải thích chi tiết
# trong svm_math_annotated.py, phần docstring đầu file)
eps = 0.15
n_train = len(y_train)
violations = 0
for i in range(n_train):
    fi = svm_dual._decision_raw(i)
    margin_i = y_train[i] * fi
    a = svm_dual.alpha[i]
    if a < 1e-6:
        ok = margin_i >= 1 - eps
    elif a > svm_dual.C - 1e-6:
        ok = margin_i <= 1 + eps
    else:
        ok = abs(margin_i - 1) <= eps
    if not ok:
        violations += 1
check("KKT complementary slackness thỏa mãn trên >=90% điểm train",
      violations <= n_train * 0.10,
      f"{n_train - violations}/{n_train} điểm thỏa KKT")
check("Mọi alpha_i nằm trong [0, C]",
      np.all(svm_dual.alpha >= -1e-9) and np.all(svm_dual.alpha <= svm_dual.C + 1e-9))

print("\n--- Test 3.3: Tính sparsity (support vector << n) ---")
n_sv = len(svm_dual.support_)
check("Số support vector nhỏ hơn 30% tổng số điểm train",
      n_sv < n_train * 0.3, f"{n_sv}/{n_train} điểm là support vector")

print("\n--- Test 3.4: Định lý đối ngẫu mạnh (Strong Duality) ---")
w_dual = svm_dual.get_w()
w_primal = svm_primal.w
w_svc = sk_svc.coef_.flatten()
w_linsvc = sk_linsvc.coef_.flatten()

cos_dp = cosine_sim(w_dual, w_primal)
cos_ds = cosine_sim(w_dual, w_svc)
cos_dl = cosine_sim(w_dual, w_linsvc)

check("w(Dual manual code) và w(Primal manual code) gần như CÙNG HƯỚNG",
      cos_dp > 0.95, f"cosine similarity = {cos_dp:.4f}")
check("w(Dual manual code) và w(SVC scikit-learn, cùng bài toán QP hinge-loss)"
      " gần như CÙNG HƯỚNG",
      cos_ds > 0.95, f"cosine similarity = {cos_ds:.4f}")
print(f"  (Tham khảo, không bắt buộc PASS) cos(Dual, LinearSVC) = {cos_dl:.4f}"
      f" - LinearSVC dùng squared-hinge loss mặc định nên nghiệm có thể"
      f" khác 1 chút so với hinge loss thuần của SVC/Dual manual code")


# ================================================================
# 4) KERNEL TRICK: minh họa TÁCH BIỆT bằng dữ liệu hình học đơn giản
# ================================================================
section("BƯỚC 4 - KERNEL TRICK (minh họa bằng dữ liệu hình học,"
        " tách biệt khỏi bài toán ngôn ngữ)")

n2 = 80
theta = np.random.uniform(0, 2 * np.pi, n2 // 2)
r_in = np.random.normal(1.0, 0.1, n2 // 2)
r_out = np.random.normal(3.0, 0.1, n2 // 2)
X_in = np.stack([r_in * np.cos(theta), r_in * np.sin(theta)], axis=1)
X_out = np.stack([r_out * np.cos(theta), r_out * np.sin(theta)], axis=1)
X_circ = np.vstack([X_in, X_out])
y_circ = np.hstack([np.ones(n2 // 2), -np.ones(n2 // 2)])

svm_lin_fail = SimpleSVM_SMO(C=1.0, kernel="linear", max_passes=20)
svm_lin_fail.fit(X_circ, y_circ)
acc_lin = np.mean(svm_lin_fail.predict(X_circ) == y_circ)

svm_rbf_ok = SimpleSVM_SMO(C=1.0, kernel="rbf", gamma=0.5, max_passes=20)
svm_rbf_ok.fit(X_circ, y_circ)
acc_rbf = np.mean(svm_rbf_ok.predict(X_circ) == y_circ)

check("Linear kernel KHÔNG phân tách được dữ liệu vòng tròn (acc < 85%)",
      acc_lin < 0.85, f"accuracy linear = {acc_lin:.4f}")
check("RBF kernel phân tách được dữ liệu vòng tròn (acc > 95%)",
      acc_rbf > 0.95, f"accuracy RBF = {acc_rbf:.4f}")


# ================================================================
# 5) BẢNG TỔNG KẾT TỐC ĐỘ
# ================================================================
section("BƯỚC 5 - TỔNG KẾT TỐC ĐỘ")

print(f"\n{'Phương pháp':30s} | {'Thời gian (s)':>14s} | {'Chậm hơn LinearSVC':>20s}")
print("-" * 70)
for name, t in [
    ("manual code - Dual/SMO", t_dual),
    ("manual code - Primal/GD", t_primal),
    ("scikit-learn - SVC(linear)", t_svc),
    ("scikit-learn - LinearSVC", t_linsvc),
]:
    ratio = t / t_linsvc
    print(f"{name:30s} | {t:14.4f} | {ratio:19.1f}x")


# ================================================================
# 6) TỔNG KẾT TEST
# ================================================================
section("TỔNG KẾT")
n_pass = sum(1 for _, s in _results if s == "PASS")
n_total = len(_results)
print(f"\nBộ test logic: {n_pass}/{n_total} PASS")
for name, status in _results:
    if status == "FAIL":
        print(f"  !! CẦN XEM LẠI: {name}")