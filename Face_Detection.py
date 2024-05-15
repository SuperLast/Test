import cv2 
import mediapipe as mp

# เริ่มต้นโมเดลการตรวจจับใบหน้า
face_detection = mp.solutions.face_detection.FaceDetection()

# เปิดกล้องและเริ่มสตรีมวิดีโอ
cap = cv2.VideoCapture(0)

while True:
    # รับเฟรมจากกล้อง
    ret, frame = cap.read()
    if not ret:
        break
    
    # แปลงเฟรมเป็นสี RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ตรวจจับใบหน้าในเฟรม
    faces = face_detection.process(frame)
    
    # วาดกรอบรอบใบหน้าที่ตรวจพบ
    if faces.detections:
        for face in faces:
            x, y, w, h = face.bounding_box
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

    # แสดงเฟรม
    cv2.imshow("Frame", frame)

    # ตรวจสอบการกดแป้น ESC เพื่อออก
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ปิดกล้อง
cap.release()
cv2.destroyAllWindows()