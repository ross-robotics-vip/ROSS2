import cv2
from ultralytics import YOLO

model = YOLO('yolo26n.pt')

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        print("Failed to Connect to Camera")

    results = model(frame, classes=[0], conf=0.5)

    human_count = len(results[0].boxes)
    print(f'Humans detected: {human_count}')

    annotated_frame = results[0].plot()

    cv2.imshow("Vision", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
