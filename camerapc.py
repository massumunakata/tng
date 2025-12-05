import cv2

# カメラデバイス番号（通常は0）
cam_id = 0
cap = cv2.VideoCapture(cam_id)

if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

# 1枚だけ撮影
ret, frame = cap.read()
if ret:
    filename = "capture.jpg"
    cv2.imwrite(filename, frame)
    print(f"撮影完了 → {filename}")

cap.release()
