# face_db_vn.py
import sqlite3
import json
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

DB_PATH = "nhan_su.db"


def tao_db(db_path: str = DB_PATH):
    """Tạo DB nếu chưa có."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS nhan_su (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ho_ten TEXT NOT NULL,
            ma_nv TEXT,
            bo_phan TEXT,
            ngay_sinh TEXT,
            embed BLOB NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def _np_to_blob(vec: np.ndarray) -> bytes:
    vec = vec.astype(np.float32)
    return vec.tobytes()


def _blob_to_np(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def them_nhan_su(
    ho_ten: str,
    ma_nv: str,
    bo_phan: str,
    ngay_sinh: str,
    embed: np.ndarray,
    db_path: str = DB_PATH,
) -> int:
    """Thêm 1 nhân sự vào DB. Trả về id."""
    tao_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO nhan_su (ho_ten, ma_nv, bo_phan, ngay_sinh, embed) VALUES (?,?,?,?,?)",
        (ho_ten, ma_nv, bo_phan, ngay_sinh, _np_to_blob(embed)),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return int(new_id)


def tai_tat_ca(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Load toàn bộ nhân sự + embedding từ DB."""
    tao_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, ho_ten, ma_nv, bo_phan, ngay_sinh, embed FROM nhan_su")
    rows = cur.fetchall()
    conn.close()

    ds = []
    for (id_, ho_ten, ma_nv, bo_phan, ngay_sinh, emb_blob) in rows:
        ds.append(
            {
                "id": int(id_),
                "ho_ten": ho_ten,
                "ma_nv": ma_nv or "",
                "bo_phan": bo_phan or "",
                "ngay_sinh": ngay_sinh or "",
                "embed": _blob_to_np(emb_blob),
            }
        )
    return ds


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity (a,b)."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


def so_khop(
    embed: np.ndarray,
    ds_nhan_su: List[Dict[str, Any]],
    nguong_sim: float = 0.45,
) -> Tuple[Optional[Dict[str, Any]], float]:
    """
    Tìm người giống nhất theo cosine similarity.
    Trả về (nhan_su_dict hoặc None, sim_tốt_nhất)
    """
    if embed is None or len(ds_nhan_su) == 0:
        return None, 0.0

    best_person = None
    best_sim = -1.0

    for p in ds_nhan_su:
        sim = cosine_sim(embed, p["embed"])
        if sim > best_sim:
            best_sim = sim
            best_person = p

    if best_person is not None and best_sim >= nguong_sim:
        return best_person, float(best_sim)

    return None, float(best_sim)
