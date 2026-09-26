import numpy as np
import pandas as pd

np.random.seed(42)

# ================================================================
# 1) TIẾNG VIỆT
# ================================================================
VI_WORDS = [
    "xin", "chào", "bạn", "tôi", "là", "học", "sinh", "lớp", "mấy",
    "hôm", "nay", "thế", "nào", "cảm", "ơn", "rất", "nhiều", "giúp",
    "để", "bài", "toán", "này", "khó", "quá", "giải", "thích", "cho",
    "em", "hiểu", "được", "không", "câu", "hỏi", "về", "phép", "tính",
    "cộng", "trừ", "nhân", "chia", "số", "học", "tiếng", "việt", "văn",
    "đọc", "viết", "chữ", "cái", "trường", "thầy", "cô", "giáo", "bạn",
    "ơi", "mình", "muốn", "biết", "thêm", "nữa", "đúng", "sai", "rồi",
]

def make_vi_sentence(min_len=3, max_len=8):
    n = np.random.randint(min_len, max_len + 1)
    return " ".join(np.random.choice(VI_WORDS, n))

# ================================================================
# 2) TIẾNG H'MÔNG
# ================================================================
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
# 3) TIẾNG Ê ĐÊ
# ================================================================
EDE_WORDS = [
    "Klei", "bhian", "Bi", "mni", "Uă", "kâo", "Ơh", "Amâo", "Lač",
    "Mgi", "Pap", "Ih", "Sang", "hră", "M'nŭm", "Mnuih", "hrăm",
    "Nai", "pơtô", "Blŭk", "M'jăng", "Toán", "Yŭan", "Jăk", "Mlang",
    "Prông", "Đĭêt", "Sa", "Dua", "Tlâo", "Pă", "Êma", "Khăp",
    "amâo", "kơ", "diih", "jăk", "êdi", "ñu", "djiê", "rê", "đŏk"
]

def make_ede_sentence(min_len=3, max_len=8):
    n = np.random.randint(min_len, max_len + 1)
    return " ".join(np.random.choice(EDE_WORDS, n))

# ================================================================
# 4) TIẾNG THÁI
# ================================================================
THAI_WORDS = [
    "สวัสดี", "ขอบคุณ", "ใช่", "ไม่", "ลาก่อน", "พูด", "เข้าใจ", 
    "ภาษาไทย", "โรงเรียน", "ครู", "นักเรียน", "หนังสือ", "ดี", 
    "ใหญ่", "เล็ก", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "รัก",
    "กิน", "ข้าว", "น้ำ", "ไป", "มา", "นอน", "ดู", "ทำ", "อะไร"
]

def make_thai_sentence(min_len=3, max_len=8):
    n = np.random.randint(min_len, max_len + 1)
    return " ".join(np.random.choice(THAI_WORDS, n))

# ================================================================
# TẠO DATASET NHIỀU LỚP
# ================================================================
def build_language_dataset_4_langs(n_per_class=400):
    rows = []
    for _ in range(n_per_class):
        rows.append({"text": make_vi_sentence(), "language": "vi"})
        rows.append({"text": make_hmong_sentence(), "language": "hmn"})
        rows.append({"text": make_ede_sentence(), "language": "ede"})
        rows.append({"text": make_thai_sentence(), "language": "tha"})
        
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df

def build_language_dataset(n_per_class=400):
    rows = []
    for _ in range(n_per_class):
        rows.append({"text": make_vi_sentence(), "language": "vi"})
    for _ in range(n_per_class):
        rows.append({"text": make_hmong_sentence(), "language": "hmn"})
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df
