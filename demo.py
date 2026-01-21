import time
import cv2
from ultralytics import YOLO

def main():
    # ====== CONFIG ======
    MODEL_PATH = "yolov8n.pt"
    CAM_INDEX = 0

    # tăng conf để bớt nhầm, giảm iou để NMS "gắt" hơn (bớt bbox trùng)
    CONF = 0.50
    IOU = 0.50
    IMGSZ = 416

    # camera
    CAM_W, CAM_H = 640, 480
    REQUEST_FPS = 30

    # COCO: person = 0
    PERSON_ID = 0

    # ====================
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Khong mo duoc webcam. Thu CAM_INDEX = 1/2 hoac kiem tra quyen camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, REQUEST_FPS)

    # đo FPS theo cửa sổ 1 giây
    t0 = time.time()
    frames = 0
    infer_sum = 0.0
    fps_real = 0.0
    infer_ms = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # lật gương
        frame = cv2.flip(frame, 1)

        # ===== YOLO predict (CÁCH 3: chỉnh IOU) =====
        t_in0 = time.time()
        results = model.predict(
            frame,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            verbose=False
        )
        t_in1 = time.time()

        infer_sum += (t_in1 - t_in0)
        frames += 1

        r = results[0]

        # ===== đếm person =====
        if r.boxes is not None and len(r.boxes) > 0:
            person_count = int((r.boxes.cls == PERSON_ID).sum().item())
        else:
            person_count = 0

        # ===== vẽ bbox =====
        annotated = r.plot()

        # ===== cập nhật FPS mỗi 1 giây =====
        dt = time.time() - t0
        if dt >= 1.0:
            fps_real = frames / dt
            infer_ms = (infer_sum / max(frames, 1)) * 1000
            print(f"YOLO FPS(real): {fps_real:.1f} | infer avg: {infer_ms:.1f} ms | persons: {person_count}")

            t0 = time.time()
            frames = 0
            infer_sum = 0.0

        # ===== overlay text =====
        cv2.putText(annotated, f"Persons: {person_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
        cv2.putText(annotated, f"FPS: {fps_real:.1f}  infer: {infer_ms:.1f}ms", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(annotated, f"conf={CONF} iou={IOU} imgsz={IMGSZ}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Count + FPS (q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
