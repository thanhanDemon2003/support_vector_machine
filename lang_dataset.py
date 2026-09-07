"""
DATASET: TIẾNG VIỆT vs TIẾNG H'MÔNG (dùng từ vựng H'MÔNG THẬT)
==================================================================

dùng syllable ngẫu nhiên mô phỏng nữa - 78 từ H'Mông dưới
đây là từ THẬT (ib, ob, peb, koj, kuv, tsev kawm ntawv...), chỉ có
việc GHÉP chúng thành "câu" là ngẫu nhiên (không đúng ngữ pháp), vì
mục đích ở đây là tạo dữ liệu có ĐẶC TRƯNG KÝ TỰ (character n-gram)
thật để SVM học, không phải tạo câu có nghĩa.

CÁCH THÊM TỪ ĐIỂN MỚI:
    Nếu thêm file JSON khác cùng schema (chunks[].content dạng
    "Nhãn: Từ_Hmong, ..." hoặc "Nhãn là Từ_Hmong. ..."), gọi:
        new_words = extract_hmong_vocab_from_json("path/to/file.json")
    rồi thêm vào HMONG_WORDS bên dưới.
"""

import json
import re
import numpy as np
import pandas as pd

np.random.seed(42)


# ================================================================
# 1) TIẾNG VIỆT - từ vựng thật, phù hợp ngữ cảnh chatbot giáo dục
# ================================================================
VI_WORDS = [
    "xin", "chào", "bạn", "tôi", "là", "học", "sinh", "lớp", "mấy",
    "hôm", "nay", "thế", "nào", "cảm", "ơn", "rất", "nhiều", "giúp",
    "đỡ", "bài", "toán", "này", "khó", "quá", "giải", "thích", "cho",
    "em", "hiểu", "được", "không", "câu", "hỏi", "về", "phép", "tính",
    "cộng", "trừ", "nhân", "chia", "số", "học", "tiếng", "việt", "văn",
    "đọc", "viết", "chữ", "cái", "trường", "thầy", "cô", "giáo", "bạn",
    "ơi", "mình", "muốn", "biết", "thêm", "nữa", "đúng", "sai", "rồi",
]

def make_vi_sentence(min_len=3, max_len=8):
    n = np.random.randint(min_len, max_len + 1)
    return " ".join(np.random.choice(VI_WORDS, n))


# ================================================================
# 2) TIẾNG H'MÔNG - PARSER trích từ vựng THẬT từ file JSON từ điển
# ================================================================
_VN_META_MARKERS = ["Mông", "tiếng", "chủ đề", "Lưu ý", "đối chiếu",
                     "ngữ pháp", "danh từ"]
_NOISE_WORDS = {"hoặc", "uk"}


def _clean_term(t):
    """Bỏ phần trong ngoặc đơn (dạng thay thế), bỏ dấu nháy/dấu chấm/khoảng trắng thừa"""
    t = re.sub(r"\s*\([^)]*\)", "", t)
    t = t.strip().rstrip(".").strip()
    return t.strip("'\" ")


def _is_valid_hmong_term(t):
    """Lọc bỏ các đoạn văn bản tiếng Việt lẫn vào do parser bắt nhầm"""
    if not t or len(t) > 30:
        return False
    if any(marker in t for marker in _VN_META_MARKERS):
        return False
    if t.lower() in _NOISE_WORDS:
        return False
    return True


def extract_hmong_vocab_from_json(json_path):
    """
    Đọc 1 file JSON từ điển (schema: {"chunks": [{"content": "..."}]})
    và trích ra danh sách các TỪ H'MÔNG THẬT xuất hiện trong đó.

    Hỗ trợ 2 định dạng câu thường gặp trong nội dung (content):
        Định dạng A: "Nhãn tiếng Việt: Từ H'Mông, Nhãn khác: Từ khác, ..."
        Định dạng B: "Nhãn tiếng Việt là Từ H'Mông. Nhãn khác là Từ khác. ..."
    Cộng thêm: các từ đặt trong dấu nháy đơn 'xxx' (thường là từ chỉ loại).

    Trả về: list các cụm từ H'Mông (có thể nhiều từ, ví dụ "Tsev kawm ntawv")
    """
    with open(json_path, encoding="utf-8") as f:
        doc = json.load(f)

    terms = []
    for chunk in doc.get("chunks", []):
        content = chunk.get("content", "")

        # Định dạng A: tách theo dấu phẩy, lấy phần SAU dấu ':' cuối cùng
        for seg in content.split(", "):
            if ":" in seg:
                terms.append(seg.rsplit(":", 1)[1])

        # Định dạng B: tách theo dấu chấm, lấy phần SAU cụm ' là ' cuối cùng
        for seg in content.split(". "):
            if " là " in seg:
                terms.append(seg.rsplit(" là ", 1)[1])

        # Bonus: các từ trong dấu nháy đơn (ví dụ từ chỉ loại 'tus', 'lub')
        terms.extend(re.findall(r"'([^']+)'", content))

    cleaned = [_clean_term(t) for t in terms]
    valid = [t for t in cleaned if _is_valid_hmong_term(t)]
    return sorted(set(valid))


def _phrases_to_word_pool(phrases):
    """Tách các cụm từ nhiều-từ (vd 'Tsev kawm ntawv') thành các TỪ ĐƠN,
    để dùng làm pool sinh câu ngẫu nhiên (giống cách VI_WORDS hoạt động)."""
    tokens = set()
    for phrase in phrases:
        for tok in phrase.split():
            tokens.add(tok.lower())
    return sorted(tokens)


# --- Trích xuất TỰ ĐỘNG từ 2 file JSON gốc của dự án (nếu có sẵn) ---
_HMONG_JSON_PATHS = [
    "/mnt/user-data/uploads/hmong_dictionary.json",
    "/mnt/user-data/uploads/tu_vung_hmong.json",
]

_hmong_phrases = []
for _path in _HMONG_JSON_PATHS:
    try:
        _hmong_phrases.extend(extract_hmong_vocab_from_json(_path))
    except FileNotFoundError:
        pass  # file không có sẵn -> bỏ qua, dùng fallback bên dưới

if _hmong_phrases:
    HMONG_WORDS = _phrases_to_word_pool(_hmong_phrases)
else:
    # FALLBACK: nếu không tìm thấy file JSON (ví dụ chạy trên máy khác),
    # dùng lại đúng 78 từ đã trích xuất sẵn từ 2 file gốc, hard-code tại
    # đây để lang_dataset.py vẫn chạy được ĐỘC LẬP không cần file ngoài.
    HMONG_WORDS = [
        "ceg", "dab", "daim", "daj", "dawb", "dev", "dua", "dub", "faib",
        "fwb", "ib", "kawm", "koj", "kuv", "kwv", "lej", "liab", "loj",
        "lub", "me", "miv", "muab", "muag", "ncauj", "neeg", "nees",
        "niam", "noog", "npaug", "npua", "ntau", "ntawv", "ntsej",
        "ntsuab", "ntswg", "ntxhais", "ntxhuav", "ntxiv", "nyeem", "nyob",
        "nyuj", "ob", "pab", "peb", "phau", "plab", "plaub", "pog",
        "qaib", "qhov", "rab", "rho", "sau", "siab", "sib", "tau",
        "taubhau", "tes", "tij", "tsaug", "tsev", "tsi", "tsib", "tsis",
        "tub", "tus", "txhais", "txiv", "ua", "vog", "xav", "xiav",
        "xib", "xim", "yawg", "yog", "zaj", "zoo",
    ]


def make_hmong_sentence(min_len=3, max_len=8):
    n = np.random.randint(min_len, max_len + 1)
    return " ".join(np.random.choice(HMONG_WORDS, n))


# ================================================================
# TẠO DATASET NHỊ PHÂN - đúng schema text/language như trainer.py
# ================================================================
def build_language_dataset(n_per_class=400):
    """
    Trả về DataFrame với 2 cột: 'text', 'language' ('vi' hoặc 'hmn'),
    đã xáo trộn (shuffle) ngẫu nhiên.
    """
    rows = []
    for _ in range(n_per_class):
        rows.append({"text": make_vi_sentence(), "language": "vi"})
    for _ in range(n_per_class):
        rows.append({"text": make_hmong_sentence(), "language": "hmn"})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print(f"Số từ H'Mông trong pool: {len(HMONG_WORDS)}")
    print(f"Nguồn: {'file JSON thật' if _hmong_phrases else 'fallback hard-code'}")
    df = build_language_dataset(n_per_class=400)
    print(df.head(10))
    print("\nPhân bố lớp:")
    print(df["language"].value_counts())
    df.to_csv("/home/claude/language_dataset_vi_hmong.csv", index=False, encoding="utf-8")
    print("\nĐã lưu: language_dataset_vi_hmong.csv")