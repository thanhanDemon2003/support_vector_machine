"""
DEMO SVM NHỊ PHÂN: TIẾNG VIỆT (+1) vs TIẾNG H'MÔNG (-1)   (bản log gọn)

Chứng minh bản chất SVM bằng số liệu, đối chiếu code tự viết với scikit-learn:
  - Ràng buộc Dual + điều kiện KKT + support vector thưa
  - Đối ngẫu mạnh: giá trị Primal = giá trị Dual
  - Nghiệm của code tự viết = nghiệm của thư viện
  - Kernel trick: linear thất bại, RBF tách được
Cần: simple_svm.py và lang_dataset.py
"""
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC

from simple_svm import SimpleSVM_SMO, LinearSVM_Primal
from lang_dataset import make_vi_sentence, make_hmong_sentence

np.random.seed(42)
_results = []


def header(t):
    print(f"\n{'=' * 70}\n{t}\n{'=' * 70}")


def check(name, ok, detail=""):
    _results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  ({detail})")


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def primal_obj(w, b, X, y, C):   # P = 1/2||w||^2 + C*sum hinge
    return 0.5 * w @ w + C * np.maximum(0, 1 - y * (X @ w + b)).sum()


def dual_obj(alpha, w):          # D = sum(alpha) - 1/2||w||^2
    return alpha.sum() - 0.5 * w @ w


# ---------------------------------------------------------------- 1. DỮ LIỆU
header("1) DỮ LIỆU")
N = 400
texts = [make_vi_sentence() for _ in range(N)] + [make_hmong_sentence() for _ in range(N)]
labels = np.array([1] * N + [-1] * N, dtype=float)
tr, te, y_train, y_test = train_test_split(texts, labels, test_size=0.3,
                                           random_state=42, stratify=labels)
# 150 chiều vì SMO tự viết dùng ma trận kernel n x n, không hỗ trợ sparse
tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=150, sublinear_tf=True)
X_train, X_test = tfidf.fit_transform(tr).toarray(), tfidf.transform(te).toarray()
n = len(y_train)
print(f"  {len(texts)} câu (vi/hmong = {N}/{N}) -> train {n}, test {len(y_test)}, "
      f"{X_train.shape[1]} đặc trưng TF-IDF char 2-4gram")

# ---------------------------------------------------------------- 2. SO SÁNH
header("2) HUẤN LUYỆN & SO SÁNH (C = 1.0)")
C = 1.0
models, times = {}, {}
for name, m in [
    ("SMO tự viết (dual)", SimpleSVM_SMO(C=C, kernel="linear", max_passes=10)),
    ("GD tự viết (primal)", LinearSVM_Primal(C=C, lr=0.001, n_iters=3000)),
    ("sklearn SVC", SVC(C=C, kernel="linear")),
    ("sklearn LinearSVC", LinearSVC(C=C, max_iter=2000, random_state=42, class_weight="balanced")),
]:
    t0 = time.perf_counter()
    m.fit(X_train, y_train)
    models[name], times[name] = m, time.perf_counter() - t0

svm, gd, svc, lin = models.values()
w_svm, b_svm = svm.get_w(), svm.b
W = {"SMO tự viết (dual)": (w_svm, b_svm),
     "GD tự viết (primal)": (gd.w, gd.b),
     "sklearn SVC": (svc.coef_.ravel(), float(svc.intercept_[0])),
     "sklearn LinearSVC": (lin.coef_.ravel(), float(lin.intercept_[0]))}
alpha_svc = np.abs(svc.dual_coef_).ravel()
D_val = {"SMO tự viết (dual)": dual_obj(svm.alpha, w_svm),
         "sklearn SVC": dual_obj(alpha_svc, W["sklearn SVC"][0])}
n_sv = {"SMO tự viết (dual)": len(svm.support_), "sklearn SVC": len(svc.support_)}

t_ref = times["sklearn LinearSVC"]   # mốc so sánh tốc độ
print(f"  {'Mô hình':22s}|{'Fit (s)':>8s} |{'x LinearSVC':>12s} |{'Acc test':>9s} |{'||w||':>7s} |{'Primal P':>9s} |{'Dual D':>8s} |{'#SV':>4s}")
print("  " + "-" * 91)
for name, m in models.items():
    w, b = W[name]
    d = f"{D_val[name]:8.3f}" if name in D_val else f"{'-':>8s}"
    sv = f"{n_sv[name]:4d}" if name in n_sv else f"{'-':>4s}"
    print(f"  {name:22s}|{times[name]:8.3f} |{times[name] / t_ref:11.1f}x |{np.mean(m.predict(X_test) == y_test):9.4f} |"
          f"{np.linalg.norm(w):7.3f} |{primal_obj(w, b, X_train, y_train, C):9.3f} |{d} |{sv}")
print("  (LinearSVC tối ưu squared-hinge nên P theo hinge cao hơn - bình thường)")

# ---------------------------------------------------------------- 3. KIỂM CHỨNG
header("3) KIỂM CHỨNG")
alpha, y = svm.alpha, y_train

check("Ràng buộc Dual: 0 <= alpha <= C  và  sum(alpha*y) = 0",
      alpha.min() >= -1e-9 and alpha.max() <= C + 1e-9 and abs(np.sum(alpha * y)) < 1e-6,
      f"alpha in [{max(alpha.min(), 0):.2g}, {alpha.max():.2g}], sum(alpha*y) = {np.sum(alpha * y):.1e}")

eps, viol = 0.15, 0     # KKT: alpha=0 -> y*f>=1 ; 0<alpha<C -> y*f=1 ; alpha=C -> y*f<=1
for i in range(n):
    m = y[i] * svm._decision_raw(i)
    ok = (m >= 1 - eps) if alpha[i] < 1e-6 else (m <= 1 + eps) if alpha[i] > C - 1e-6 else abs(m - 1) <= eps
    viol += not ok
check("KKT thỏa trên >= 90% điểm train", viol <= 0.1 * n, f"{n - viol}/{n} điểm")

check("Thưa: support vector < 30% số điểm", len(svm.support_) < 0.3 * n,
      f"{len(svm.support_)}/{n} = {len(svm.support_) / n:.1%}")

P, D = primal_obj(w_svm, b_svm, X_train, y_train, C), dual_obj(alpha, w_svm)
check("Đối ngẫu mạnh: Primal = Dual (gap < 5%)", (P - D) / abs(P) < 0.05,
      f"P={P:.3f}, D={D:.3f}, gap={(P - D) / abs(P):.2%}")

c_sv, c_gd = cosine(w_svm, W["sklearn SVC"][0]), cosine(w_svm, gd.w)
check("w của SMO cùng hướng w của sklearn SVC và của Primal/GD", c_sv > 0.95 and c_gd > 0.95,
      f"cos = {c_sv:.4f} / {c_gd:.4f}")

agree = np.mean(svm.predict(X_test) == svc.predict(X_test))
check("Dự đoán test của SMO trùng sklearn SVC >= 98%", agree >= 0.98, f"trùng {agree:.1%}")

# Kernel trick: 2 vòng tròn đồng tâm (không tách tuyến tính được)
th = np.random.uniform(0, 2 * np.pi, 40)
ri, ro = np.random.normal(1, .1, 40), np.random.normal(3, .1, 40)
Xc = np.vstack([np.c_[ri * np.cos(th), ri * np.sin(th)], np.c_[ro * np.cos(th), ro * np.sin(th)]])
yc = np.hstack([np.ones(40), -np.ones(40)])
acc = {}
for k, m in [("linear", SimpleSVM_SMO(kernel="linear", max_passes=20)),
             ("rbf", SimpleSVM_SMO(kernel="rbf", gamma=0.5, max_passes=20)),
             ("sk-linear", SVC(kernel="linear")), ("sk-rbf", SVC(kernel="rbf", gamma=0.5))]:
    acc[k] = np.mean(m.fit(Xc, yc).predict(Xc) == yc)
check("Kernel trick: linear thất bại (<85%), RBF tách được (>95%), cả tự viết lẫn sklearn",
      max(acc["linear"], acc["sk-linear"]) < 0.85 and min(acc["rbf"], acc["sk-rbf"]) > 0.95,
      f"linear {acc['linear']:.2f}/{acc['sk-linear']:.2f} | rbf {acc['rbf']:.2f}/{acc['sk-rbf']:.2f} (tự viết/sklearn)")

print(f"\n>>> {sum(_results)}/{len(_results)} PASS")