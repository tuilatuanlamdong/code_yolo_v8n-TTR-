# demo_face_vn.py
import os
import time
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

import cv2
import numpy as np
from ultralytics import YOLO

from sort_tracker import Sort
import face_db_vn

# InsightFace
from insightface.app import FaceAnalysis

# ===== CẤU HÌNH =====
MODEL_PATH = "yolov8n.pt"
CAM_INDEX = 0

# giảm lag:
IMGSZ = 300
CONF = 0.5
PERSON_ID = 0

# ngưỡng face similarity
NGUONG_SIM = 0.45

# mỗi track bao lâu mới nhận diện lại (giây) để đỡ lag
NHAN_DIEN_MOI = 5.0

# chạy YOLO mỗi N frame để nhẹ (3 = nhẹ hơn)
YOLO_EVERY_N = 3

# min diện tích bbox (lọc rác)
MIN_AREA = 12000


def chon_track_lon_nhat(tracks):
    """Chọn track có bbox lớn nhất (x2-x1)*(y2-y1)."""
    if tracks is None or len(tracks) == 0:
        return None
    best = None
    best_area = -1
    for x1, y1, x2, y2, tid in tracks:
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best = (int(x1), int(y1), int(x2), int(y2), int(tid))
    return best


def cat_roi_an_toan(frame, x1, y1, x2, y2, pad=10):
    h, w = frame.shape[:2]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad); y2 = min(h - 1, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def main():
    # DB
    face_db_vn.tao_db()
    ds_nhan_su = face_db_vn.tai_tat_ca()
    print(f"[DB] Đã tải {len(ds_nhan_su)} nhân sự")

    # Face model
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=-1, det_size=(320, 320))  # ctx_id=-1 = CPU

    # YOLO + SORT
    model = YOLO(MODEL_PATH)
    tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.35)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Không mở được webcam. Thử CAM_INDEX=1/2")
        return

    # giảm độ phân giải để mượt
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 620)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 460)

    # FPS
    t0 = time.time()
    frames = 0
    fps = 0.0

    # cache nhận diện theo track_id
    last_recog = {}   # tid -> time
    tid_to_person = {}  # tid -> dict person / None
    tid_to_sim = {}     # tid -> sim

    frame_id = 0
    dets_cache = np.empty((0, 5), dtype=np.float32)

    print("\nPHÍM ĐIỀU KHIỂN:")
    print("  R: Đăng ký người mới (lấy mặt trong bbox track lớn nhất)")
    print("  ESC: thoát\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # lật gương
        frame = cv2.flip(frame, 1)

        frame_id += 1

        # --- YOLO (chạy mỗi N frame để giảm lag) ---
        if frame_id % YOLO_EVERY_N == 0:
            res = model.predict(frame, imgsz=IMGSZ, conf=CONF, device=0, verbose=False)[0]
            dets = []
            for box in res.boxes:
                cls = int(box.cls[0])
                if cls != PERSON_ID:
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                score = float(box.conf[0])

                area = (x2 - x1) * (y2 - y1)
                if area < MIN_AREA:
                    continue

                dets.append([x1, y1, x2, y2, score])

            dets_cache = np.array(dets, dtype=np.float32) if len(dets) else np.empty((0, 5), dtype=np.float32)

        # SORT update (dù có det hay không)
        tracks = tracker.update(dets_cache)  # [x1,y1,x2,y2,tid]

        # --- nhận diện khuôn mặt theo track ---
        now = time.time()
        for x1, y1, x2, y2, tid in tracks:
            x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)

            # chỉ nhận diện lại mỗi NHAN_DIEN_MOI giây
            if (tid not in last_recog) or (now - last_recog[tid] > NHAN_DIEN_MOI):
                roi = cat_roi_an_toan(frame, x1, y1, x2, y2, pad=10)
                person = None
                sim = 0.0

                if roi is not None and roi.size > 0:
                    faces = face_app.get(roi)
                    if len(faces) > 0:
                        # chọn mặt lớn nhất trong roi
                        faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
                        emb = faces[0].normed_embedding.astype(np.float32)

                        person, sim = face_db_vn.so_khop(emb, ds_nhan_su, nguong_sim=NGUONG_SIM)

                tid_to_person[tid] = person
                tid_to_sim[tid] = sim
                last_recog[tid] = now

        # --- vẽ bbox + info ---
        for x1, y1, x2, y2, tid in tracks:
            x1, y1, x2, y2, tid = int(x1), int(y1), int(x2), int(y2), int(tid)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            person = tid_to_person.get(tid, None)
            sim = tid_to_sim.get(tid, 0.0)

            if person is None:
                text1 = f"ID {tid} | Chua biet | sim={sim:.2f}"
                cv2.putText(frame, text1, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            else:
                text1 = f"ID {tid} | {person['ho_ten']} | sim={sim:.2f}"
                text2 = f"Ma: {person['ma_nv']} | BP: {person['bo_phan']} | NS: {person['ngay_sinh']}"
                cv2.putText(frame, text1, (x1, max(20, y1 - 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.putText(frame, text2, (x1, max(40, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)

        # --- FPS ---
        frames += 1
        if time.time() - t0 >= 1.0:
            fps = frames / (time.time() - t0)
            t0 = time.time()
            frames = 0

        cv2.putText(frame, f"FPS: {fps:.1f} | tracks: {len(tracks)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow("NHAN DIEN KHUON MAT ", frame)

        key = cv2.waitKey(1) & 0xFF

        # ESC
        if key == 27:
            break

        # R: đăng ký
        if key in (ord('r'), ord('R')):
            best = chon_track_lon_nhat(tracks)
            if best is None:
                print("[DK] Không có người trong khung để đăng ký.")
                continue

            x1, y1, x2, y2, tid = best
            roi = cat_roi_an_toan(frame, x1, y1, x2, y2, pad=10)
            if roi is None:
                print("[DK] ROI lỗi.")
                continue

            faces = face_app.get(roi)
            if len(faces) == 0:
                print("[DK] Không thấy khuôn mặt rõ. Hãy nhìn thẳng và lại gần hơn.")
                continue

            faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
            emb = faces[0].normed_embedding.astype(np.float32)

            print("\n=== ĐĂNG KÝ NHÂN SỰ MỚI ===")
            ho_ten = input("Họ và tên: ").strip()
            ma_nv = input("Mã nhân viên: ").strip()
            bo_phan = input("Bộ phận / Tầng làm việc: ").strip()
            ngay_sinh = input("Ngày sinh (VD 2001-12-31): ").strip()

            new_id = face_db_vn.them_nhan_su(ho_ten, ma_nv, bo_phan, ngay_sinh, emb)
            print(f"[DK] Đã lưu vào DB, id={new_id}")

            # reload DB để nhận diện ngay
            ds_nhan_su = face_db_vn.tai_tat_ca()
            print(f"[DB] Reload: {len(ds_nhan_su)} nhân sự\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
