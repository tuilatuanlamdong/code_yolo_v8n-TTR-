import cv2
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Khong mo duoc webcam (source 0). Thu 1,2... hoac kiem tra quyen camera.")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)

        results = model.predict(frame, imgsz=640, conf=0.35, verbose=False)
        r = results[0]

        # class id của "person" trong COCO = 0
        person_id = 0

        # Đếm số bbox có class == person_id
        if r.boxes is not None and len(r.boxes) > 0:
            cls = r.boxes.cls  # tensor
            person_count = int((cls == person_id).sum().item())
        else:
            person_count = 0

        annotated = r.plot()

        # Vẽ số người lên ảnh
        cv2.putText(
            annotated,
            f"Persons: {person_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2
        )

        cv2.imshow("camera", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
