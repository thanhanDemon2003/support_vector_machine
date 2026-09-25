
import numpy as np
# numpy: chỉ dùng để biểu diễn vector/ma trận (x_i là vector trong R^d,
# X là ma trận n×d).

# ================================================================
# KERNEL FUNCTIONS
# Cơ sở toán học: MERCER'S THEOREM
#   Nếu K(x,y) là một hàm đối xứng, nửa xác định dương (positive
#   semi-definite) thì tồn tại một ánh xạ phi(.) sao cho
#       K(x,y) = <phi(x), phi(y)>
#   => Ta có thể làm việc với K(x,y) mà KHÔNG cần biết phi(x) tường
#      minh. Đây là nền tảng của "kernel trick".
# ================================================================

def linear_kernel(X1, X2):
    """
    K(x,y) = x^T y   (phi(x) = x, tức không biến đổi gì)
    """
    return X1 @ X2.T
    # X1 @ X2.T : nhân ma trận (n1×d) với (d×n2) -> kết quả (n1×n2)
    # Đây chính là tính đồng thời TẤT CẢ các cặp <x_i, x_j> trong 1 dòng,
    # tương đương n1*n2 phép tích vô hướng, nhưng viết dạng ma trận
    # cho gọn (vẫn là phép toán tuyến tính thuần túy, không "tối ưu hoá").


def polynomial_kernel(X1, X2, degree=3, coef0=1.0):
    """
    K(x,y) = (x^T y + coef0)^degree
    Đây là kernel ứng với phi(x) chứa các đơn thức bậc <= degree
    của các thành phần của x (ví dụ degree=2, d=2:
    phi(x)=(x1^2, x2^2, sqrt(2)x1x2, sqrt(2)x1, sqrt(2)x2, 1))
    """
    return (X1 @ X2.T + coef0) ** degree
    # X1 @ X2.T          : ma trận tích vô hướng <x_i,x_j> (giống linear kernel)
    # + coef0            : cộng hằng số (tương ứng thêm bậc thấp hơn vào phi(x))
    # ** degree           : lũy thừa từng phần tử ma trận (elementwise power)


def rbf_kernel(X1, X2, gamma=0.5):
    """
    K(x,y) = exp(-gamma * ||x-y||^2)
    Khai triển Taylor của exp(.) cho thấy phi(x) tương ứng có VÔ HẠN
    chiều (chứa mọi bậc đa thức của x). gamma điều khiển độ "hẹp"
    của vùng ảnh hưởng quanh mỗi điểm dữ liệu.
    """
    sq_dists = (
        np.sum(X1 ** 2, axis=1)[:, None]      # ||x_i||^2 cho từng dòng, shape (n1,1)
        + np.sum(X2 ** 2, axis=1)[None, :]    # ||x_j||^2 cho từng dòng, shape (1,n2)
        - 2 * X1 @ X2.T                        # -2 x_i^T x_j
    )
    # Đẳng thức đại số:  ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i^T x_j
    # (khai triển bình phương chuẩn Euclid — không phải "mẹo lập trình",
    #  đây là hằng đẳng thức (a-b)^2 = a^2+b^2-2ab áp dụng cho vector)
    return np.exp(-gamma * sq_dists)
    # Áp dụng hàm mũ elementwise lên toàn bộ ma trận khoảng cách bình phương


# ================================================================
# CLASS 1: SimpleSVM_SMO
# Giải bài toán DUAL:
#     max_alpha  sum_i alpha_i - (1/2) sum_i sum_j alpha_i alpha_j y_i y_j K(x_i,x_j)
#     s.t.       sum_i alpha_i y_i = 0,   0 <= alpha_i <= C
#
# Cơ sở toán học của việc CHUYỂN từ Primal sang Dual:
#   ĐỊNH LÝ ĐỐI NGẪU MẠNH (Strong Duality) cho bài toán lồi (convex):
#   Nếu bài toán primal là lồi (ở đây đúng vì (1/2)||w||^2 là lồi,
#   các ràng buộc là tuyến tính) và thỏa điều kiện Slater (tồn tại
#   điểm khả thi nghiêm ngặt), thì:
#       giá trị tối ưu của Primal = giá trị tối ưu của Dual
#   => Giải Dual cho nghiệm CHÍNH XÁC như giải Primal, không phải
#      xấp xỉ. Đây là lý do ta được phép "chuyển sân" sang Dual.
# ================================================================
class SimpleSVM_SMO:

    def __init__(self, C=1.0, kernel="linear", tol=1e-3,
                 max_passes=10, **kernel_params):
        self.C = C
        # C: hằng số điều chỉnh trade-off margin-rộng vs lỗi-ít
        # (hệ số nhân với sum xi_i trong hàm mục tiêu Primal gốc)
        self.tol = tol
        # tol: ngưỡng sai số cho phép khi kiểm tra điều kiện KKT
        # (vì số thực dấu phẩy động không bao giờ = 0 tuyệt đối)
        self.max_passes = max_passes
        # Số vòng lặp liên tiếp KHÔNG có alpha nào thay đổi trước khi
        # coi là đã hội tụ (điều kiện dừng thực nghiệm, không phải
        # công thức toán, chỉ là tiêu chí dừng vòng lặp)
        self.kernel_params = kernel_params

        if kernel == "linear":
            self.kernel_fn = lambda A, B: linear_kernel(A, B)
        elif kernel == "poly":
            self.kernel_fn = lambda A, B: polynomial_kernel(A, B, **kernel_params)
        elif kernel == "rbf":
            self.kernel_fn = lambda A, B: rbf_kernel(A, B, **kernel_params)
        else:
            raise ValueError("kernel phải là 'linear', 'poly', hoặc 'rbf'")
        # self.kernel_fn lưu lại HÀM K(.,.) sẽ dùng xuyên suốt —
        # đây chính là chỗ "kernel trick" được cắm vào: toàn bộ thuật
        # toán bên dưới chỉ gọi self.kernel_fn(A,B), không quan tâm
        # công thức cụ thể là gì.

    # ------------------------------------------------------------
    def fit(self, X, y):
        n = X.shape[0]
        # n = số điểm dữ liệu -> cũng là số biến alpha_i cần tìm trong Dual

        self.X = X
        self.y = y.astype(float)
        # Lưu lại để dùng trong decision_function / get_w sau này.
        # ép kiểu float vì y ban đầu có thể là {-1,1} dạng int, mà
        # công thức toán cần phép nhân/chia số thực.

        self.K = self.kernel_fn(X, X)
        # Tính SẴN toàn bộ ma trận K[i,j] = K(x_i, x_j) MỘT LẦN.
        # Đây chính là ma trận Q trong công thức Dual:
        #     Q_ij = y_i y_j K(x_i,x_j)
        # (ở đây mới chỉ tính phần K, phần y_i*y_j sẽ nhân vào lúc dùng)

        self.alpha = np.zeros(n)
        # KHỞI TẠO alpha = 0 cho mọi điểm.
        # Điểm khởi tạo alpha=0 luôn KHẢ THI (feasible) vì:
        #   sum(0 * y_i) = 0                (thỏa ràng buộc đẳng thức)
        #   0 <= 0 <= C                      (thỏa ràng buộc hộp)
        # SMO chỉ di chuyển giữa các điểm khả thi, nên điểm bắt đầu
        # BẮT BUỘC phải khả thi — đây là lý do chọn alpha=0.

        self.b = 0.0
        # Khởi tạo b=0, sẽ được tính lại chính xác ở cuối bằng
        # _recompute_b() dựa trên điều kiện KKT.

        passes = 0
        while passes < self.max_passes:
            # Vòng lặp ngoài: lặp cho đến khi trải qua max_passes lần
            # QUÉT TOÀN BỘ dữ liệu mà KHÔNG có cập nhật nào xảy ra
            # (dấu hiệu đã đạt điểm dừng / hội tụ tới nghiệm KKT).

            num_changed = 0
            for i in range(n):
                # Quét từng điểm i, kiểm tra xem alpha_i có đang VI PHẠM
                # điều kiện KKT hay không.

                Ei = self._decision_raw(i) - self.y[i]
                # E_i = f(x_i) - y_i
                # f(x_i) là giá trị dự đoán hiện tại (trước khi ép dấu),
                # y_i là nhãn thật. E_i đo "sai số" — nếu E_i lớn nghĩa
                # là điểm i đang bị dự đoán rất sai so với margin mong muốn.

                # ------ KIỂM TRA VI PHẠM KKT ------
                # Nhắc lại điều kiện KKT đầy đủ (suy từ complementary
                # slackness alpha_i[y_i f(x_i) - 1 + xi_i] = 0 và
                # mu_i*xi_i=0, C-alpha_i-mu_i=0):
                #
                #   alpha_i = 0        <=>  y_i f(x_i) >= 1
                #   0 < alpha_i < C     <=>  y_i f(x_i) = 1
                #   alpha_i = C         <=>  y_i f(x_i) <= 1
                #
                # Viết lại theo E_i = f(x_i)-y_i (chú ý y_i^2=1 vì y_i=+-1):
                #   y_i * E_i = y_i f(x_i) - y_i^2 = y_i f(x_i) - 1
                # nên:
                #   y_i*E_i < 0  và alpha_i < C   => đang VI PHẠM
                #      (lẽ ra phải có alpha_i=C nhưng chưa đạt, vì
                #       y_i f(x_i) < 1 nghĩa là điểm trong margin/sai)
                #   y_i*E_i > 0  và alpha_i > 0   => đang VI PHẠM
                #      (lẽ ra phải có alpha_i=0 nhưng chưa đạt, vì
                #       y_i f(x_i) > 1 nghĩa là điểm đã an toàn ngoài margin)
                cond1 = (self.y[i] * Ei < -self.tol) and (self.alpha[i] < self.C)
                cond2 = (self.y[i] * Ei > self.tol) and (self.alpha[i] > 0)
                if not (cond1 or cond2):
                    continue
                    # Điểm i đã thỏa KKT (trong sai số tol) -> bỏ qua,
                    # không cần cập nhật gì cho điểm này nữa.

                j = np.random.randint(n - 1)
                if j >= i:
                    j += 1
                # Chọn ngẫu nhiên một điểm j khác i.
                # (Bản gốc Platt 1998 chọn j theo |E_i - E_j| lớn nhất để
                #  bước nhảy hiệu quả hơn — ở đây đơn giản hóa bằng random
                #  để code ngắn, dễ đọc, vẫn hội tụ đúng về mặt toán học
                #  vì mọi cặp (i,j) hợp lệ đều làm tăng hàm mục tiêu Dual
                #  hoặc giữ nguyên, không bao giờ làm giảm.)

                Ej = self._decision_raw(j) - self.y[j]
                # Tính E_j tương tự E_i, cần để đưa vào công thức cập nhật.

                changed = self._take_step(i, j, Ei, Ej)
                # Gọi hàm giải bài toán con 2 biến (i,j) — xem chi tiết
                # công thức toán trong _take_step bên dưới.

                num_changed += changed
                # Đếm số cặp thực sự có thay đổi giá trị (để biết có
                # nên reset bộ đếm passes hay không)

            passes = passes + 1 if num_changed == 0 else 0
            # Nếu 1 vòng quét toàn bộ mà KHÔNG có cặp nào thay đổi
            # (num_changed==0) -> tăng passes (gần hội tụ hơn).
            # Ngược lại, còn cập nhật -> reset passes về 0 (chưa xong).

        self._recompute_b()
        # Sau khi alpha đã hội tụ (thỏa KKT với sai số tol), tính lại
        # b chính xác một lần cuối bằng công thức KKT (xem hàm bên dưới).

        self.support_ = np.where(self.alpha > 1e-6)[0]
        # Support vectors = các điểm có alpha_i > 0 (theo định nghĩa
        # suy ra từ complementary slackness ở mục 4 trong docstring gốc)
        return self

    # ------------------------------------------------------------
    def _decision_raw(self, idx):
        """f(x_idx) = sum_k alpha_k y_k K(x_k, x_idx) + b"""
        return np.sum(self.alpha * self.y * self.K[:, idx]) + self.b
        # self.alpha * self.y          : elementwise, vector (alpha_k y_k) cho mọi k
        # self.K[:, idx]                : cột thứ idx của ma trận kernel
        #                                 = [K(x_0,x_idx), K(x_1,x_idx), ..., K(x_n,x_idx)]
        # (alpha_k y_k) * K(x_k,x_idx)  : elementwise nhân từng cặp
        # np.sum(...)                   : cộng lại -> đúng công thức
        #                                 sum_k alpha_k y_k K(x_k, x_idx)
        # + self.b                      : cộng bias
        # => Đây CHÍNH XÁC là công thức quyết định f(x) suy ra từ
        #    w = sum alpha_k y_k x_k  (thay <x_k, x> bằng K(x_k,x))

    # ------------------------------------------------------------
    def _take_step(self, i, j, Ei, Ej):
        """
        Giải giải tích (closed-form) bài toán con chỉ 2 biến alpha_i, alpha_j
        """
        if i == j:
            return 0
            # Không thể tối ưu 1 biến với chính nó theo ràng buộc đẳng thức
            # (cần ít nhất 2 biến khác nhau để "di chuyển" mà vẫn giữ
            #  sum alpha_i y_i = const)

        alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
        yi, yj = self.y[i], self.y[j]
        # Lưu giá trị CŨ lại vì công thức cập nhật cần dùng chúng
        # (không được ghi đè self.alpha[i] trước khi tính xong alpha_j)

        # ================================================================
        # BƯỚC 1: Xác định miền [L, H] hợp lệ cho alpha_j_new
        #
        # Xuất phát từ ràng buộc ĐẲNG THỨC của Dual:
        #     alpha_i * y_i + alpha_j * y_j = const = alpha_i_old*yi + alpha_j_old*yj
        # Đây là phương trình một ĐƯỜNG THẲNG trong mặt phẳng (alpha_i, alpha_j).
        # Kết hợp với ràng buộc HỘP  0 <= alpha_i, alpha_j <= C,
        # đoạn thẳng đó bị "cắt" thành một đoạn hữu hạn [L, H] cho alpha_j.
        # ================================================================
        if yi != yj:
            # Trường hợp yi != yj: đường thẳng có dạng alpha_i - alpha_j = const
            # => alpha_j = alpha_i - const, khi alpha_i tăng thì alpha_j cũng tăng
            L = max(0, alpha_j_old - alpha_i_old)
            H = min(self.C, self.C + alpha_j_old - alpha_i_old)
            # Suy ra bằng cách xét 2 đầu mút alpha_i in [0,C]:
            #   alpha_i=0   => alpha_j = alpha_j_old - alpha_i_old (là 1 cận)
            #   alpha_i=C   => alpha_j = C + alpha_j_old - alpha_i_old (cận kia)
            # rồi lấy max với 0 (do alpha_j>=0) và min với C (do alpha_j<=C)
        else:
            # Trường hợp yi == yj: đường thẳng có dạng alpha_i + alpha_j = const
            L = max(0, alpha_i_old + alpha_j_old - self.C)
            H = min(self.C, alpha_i_old + alpha_j_old)
            # Tương tự, xét alpha_i=0 và alpha_i=C để suy ra 2 cận cho alpha_j
        if L == H:
            return 0
            # Nếu miền suy biến thành 1 điểm -> không có gì để tối ưu,
            # bỏ qua cặp (i,j) này.

        # ================================================================
        # BƯỚC 2: Tính eta — đạo hàm bậc 2 (độ cong) của hàm mục tiêu Dual
        # theo alpha_j dọc theo đường thẳng ràng buộc.
        #
        # Xuất phát từ việc thế alpha_i = (const - alpha_j y_j)/y_i vào
        # hàm mục tiêu Dual W(alpha) = sum alpha_i - 1/2 sum sum alpha_i
        # alpha_j y_i y_j K(x_i,x_j), rồi lấy đạo hàm bậc 2 theo alpha_j,
        # ta được:
        #     eta = 2K(x_i,x_j) - K(x_i,x_i) - K(x_j,x_j)
        # ================================================================
        eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
        if eta >= 0:
            return 0
            # Về mặt toán học, eta = -||phi(x_i) - phi(x_j)||^2 <= 0 luôn
            # đúng (vì K nửa xác định dương theo Mercer). eta=0 xảy ra khi
            # phi(x_i)=phi(x_j) (2 điểm trùng nhau trong không gian đặc
            # trưng) — trường hợp thoái hóa, bản SMO đầy đủ xử lý riêng
            # bằng cách đánh giá trực tiếp 2 đầu mút L,H; ở đây đơn giản
            # hóa bằng cách bỏ qua.

        # ================================================================
        # BƯỚC 3: Công thức cập nhật alpha_j (đặt đạo hàm bậc 1 = 0)
        #
        # Đạo hàm bậc 1 của W(alpha) theo alpha_j (dọc đường ràng buộc)
        # dẫn tới nghiệm KHÔNG RÀNG BUỘC:
        #     alpha_j_new_unc = alpha_j_old - y_j*(Ei - Ej)/eta
        # (đây là 1 bước Newton: nghiệm = điểm cũ - đạo hàm bậc1/đạo hàm bậc2)
        # ================================================================
        alpha_j_new = alpha_j_old - (yj * (Ei - Ej)) / eta

        alpha_j_new = min(max(alpha_j_new, L), H)
        # BƯỚC 4: chiếu (clip) nghiệm không ràng buộc về đoạn [L,H] hợp lệ
        # (vì bài toán gốc CÓ ràng buộc hộp, nghiệm không ràng buộc có thể
        #  nằm ngoài, phải cắt lại theo nguyên lý "projection")

        if abs(alpha_j_new - alpha_j_old) < 1e-7:
            return 0
            # Nếu thay đổi quá nhỏ (dưới sai số máy) -> coi như không đổi,
            # tránh vòng lặp vô ích / dao động số học

        alpha_i_new = alpha_i_old + yi * yj * (alpha_j_old - alpha_j_new)
        # BƯỚC 5: suy alpha_i_new từ ràng buộc đẳng thức
        #     alpha_i*yi + alpha_j*yj = alpha_i_old*yi + alpha_j_old*yj
        # => alpha_i_new = alpha_i_old + yi*yj*(alpha_j_old - alpha_j_new)
        # (nhân 2 vế với yi và dùng yi^2=1 để rút gọn)

        self.alpha[i], self.alpha[j] = alpha_i_new, alpha_j_new
        # Cập nhật đồng thời cả 2 giá trị vào mảng alpha chính thức
        return 1
        # Trả về 1 để fit() biết rằng vòng lặp NÀY có thay đổi thực sự

    # ------------------------------------------------------------
    def _recompute_b(self):
        """
        KKT: với mọi i có 0 < alpha_i < C  =>  y_i * f(x_i) = 1
        =>  b = y_i - sum_k alpha_k y_k K(x_k,x_i)
        """
        sv_on_margin = np.where((self.alpha > 1e-6) & (self.alpha < self.C - 1e-6))[0]
        # Tìm các index có 0 < alpha_i < C (support vector "nằm đúng
        # trên margin", theo phân loại 3 trường hợp KKT ở mục 4)

        if len(sv_on_margin) == 0:
            sv_on_margin = np.where(self.alpha > 1e-6)[0]
            # Trường hợp hiếm: không có điểm nào thỏa 0<alpha<C
            # (mọi support vector đều "bão hòa" ở alpha=C) -> dùng tạm
            # toàn bộ support vector để ước lượng b gần đúng.
        if len(sv_on_margin) == 0:
            self.b = 0.0
            return
            # Trường hợp cực đoan: không có support vector nào cả
            # (mọi alpha=0, ví dụ dữ liệu suy biến) -> gán b=0 mặc định.

        b_vals = []
        for i in sv_on_margin:
            s = np.sum(self.alpha * self.y * self.K[:, i])
            # s = sum_k alpha_k y_k K(x_k, x_i)  (giống _decision_raw
            # nhưng KHÔNG cộng b, vì đang muốn GIẢI ra b)
            b_vals.append(self.y[i] - s)
            # Từ y_i*(s + b) = 1  =>  b = y_i - s  (vì y_i^2=1 nên
            # y_i*(s+b)=1 <=> s+b=y_i <=> b=y_i-s)
        self.b = np.mean(b_vals)
        # Lấy TRUNG BÌNH trên nhiều support vector để giảm sai số làm
        # tròn số thực (mỗi điểm cho một ước lượng b hơi khác nhau do
        # sai số dấu phẩy động, trung bình giúp ổn định hơn)

    # ------------------------------------------------------------
    def decision_function(self, X_new):
        """f(x) = sum_i alpha_i y_i K(x_i, x) + b"""
        K_new = self.kernel_fn(self.X, X_new)
        # Tính K(x_train_i, x_new_j) cho MỌI cặp -> ma trận (n_train, n_new)
        return (self.alpha * self.y) @ K_new + self.b
        # (alpha*y) là vector (n_train,), K_new là ma trận (n_train,n_new)
        # => phép nhân vector-ma trận cho ra vector (n_new,) chính là
        #    f(x) cho từng điểm mới, đúng công thức quyết định.

    def predict(self, X_new):
        return np.sign(self.decision_function(X_new))
        # np.sign: trả về +1 nếu f(x)>0, -1 nếu f(x)<0
        # (đúng quy tắc phân loại nhị phân của SVM: dự đoán theo dấu
        #  của khoảng cách có hướng tới siêu phẳng)

    def get_w(self):
        """w = sum_i alpha_i y_i x_i  — CHỈ đúng khi kernel là linear"""
        return (self.alpha * self.y) @ self.X
        # (alpha*y) vector (n,), self.X ma trận (n,d)
        # => vector (d,) = sum_i alpha_i y_i x_i  (đúng công thức (*)
        #    suy ra từ dL/dw=0 trong phần Lagrangian)
        # Lưu ý: với kernel phi tuyến (rbf, poly), "w" không còn nằm
        # trong không gian gốc R^d nữa (nó nằm trong không gian đặc
        # trưng vô hạn/đa chiều của phi(x)), nên hàm này KHÔNG có ý
        # nghĩa với kernel != linear.


# ================================================================
# CLASS 2: LinearSVM_Primal
# Giải TRỰC TIẾP bài toán Primal bằng Gradient Descent, KHÔNG qua
# Lagrangian/Dual — dùng để đối chiếu, kiểm chứng 2 đường giải cho
# cùng 1 nghiệm (theo định lý đối ngẫu mạnh đã nêu ở trên).
# ================================================================
class LinearSVM_Primal:

    def __init__(self, C=1.0, lr=0.001, n_iters=1000):
        self.C = C          # trọng số hinge loss, giống C ở bài toán Primal gốc
        self.lr = lr        # learning rate (tốc độ học) của gradient descent
        self.n_iters = n_iters  # số vòng lặp cập nhật w, b

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        # Khởi tạo w = vector 0 trong R^d (điểm bắt đầu bất kỳ, vì hàm
        # mục tiêu Primal (1/2)||w||^2 + C*hinge_loss là LỒI (convex)
        # nên gradient descent hội tụ về nghiệm toàn cục bất kể điểm
        # khởi tạo, theo lý thuyết tối ưu lồi)
        self.b = 0.0
        y = y.astype(float)

        for _ in range(self.n_iters):
            margins = y * (X @ self.w + self.b)
            # margins[i] = y_i * (w^T x_i + b)
            # X @ self.w : nhân ma trận (n,d) với vector (d,) -> vector (n,)
            #              chính là w^T x_i cho MỌI i cùng lúc

            violate = margins < 1
            # Mask boolean: True tại các điểm VI PHẠM margin
            # (đúng điều kiện kích hoạt của subgradient hinge loss)

            grad_w = self.w - self.C * (y[violate] @ X[violate])
            # ĐẠO HÀM (subgradient) của hàm mục tiêu Primal theo w:
            #   d/dw [ (1/2)||w||^2 ] = w
            #   d/dw [ C * max(0, 1 - y_i(w^Tx_i+b)) ]
            #         = -C*y_i*x_i  nếu vi phạm (margin<1), else 0
            # y[violate] @ X[violate] : tính sum_{i vi phạm} y_i * x_i
            #   (y[violate] có shape (k,), X[violate] có shape (k,d),
            #    phép @ ở đây là tích vector-ma trận -> vector (d,))
            # => grad_w = w - C * sum_{i vi phạm} y_i x_i (đúng công thức)

            grad_b = -self.C * np.sum(y[violate])
            # Tương tự, d/db[hinge loss] = -C*y_i nếu vi phạm, else 0
            # => grad_b = -C * sum_{i vi phạm} y_i

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            # CÔNG THỨC GRADIENT DESCENT chuẩn:
            #     theta_new = theta_old - learning_rate * gradient
            # (đi ngược chiều gradient để giảm hàm mục tiêu — nguyên lý
            #  cơ bản nhất của tối ưu hóa lồi/trơn từng khúc)

        return self

    def decision_function(self, X):
        return X @ self.w + self.b
        # f(x) = w^T x + b, tính cho toàn bộ ma trận X cùng lúc

    def predict(self, X):
        return np.sign(self.decision_function(X))
        # Phân loại theo dấu, giống class trên