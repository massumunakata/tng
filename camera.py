import cv2
import threading
import time

# カメラデバイス番号（USBカメラなら 0,1,2）
camera_ids = [0, 1, 2]

def capture_image(cam_id):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"カメラ {cam_id} を開けませんでした")
        return
    ret, frame = cap.read()
    if ret:
        filename = f"camera_{cam_id}.jpg"
        cv2.imwrite(filename, frame)
        print(f"カメラ {cam_id} → 撮影完了: {filename}")
    cap.release()

# ===== 同時撮影 =====
threads = []
start_time = time.time()

for cam_id in camera_ids:
    t = threading.Thread(target=capture_image, args=(cam_id,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("全カメラ撮影完了！処理時間:", time.time() - start_time, "秒")
