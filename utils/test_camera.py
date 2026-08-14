import cv2

RTSP_URL = "rtsp://admin:RVXCEM@192.168.1.100:554/h264/ch1/main/av_stream"

cap = cv2.VideoCapture(RTSP_URL)

print("摄像头已连接，按 Q 退出...")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # 显示画面
    cv2.imshow("萤石 C6c - 实时画面", frame)

    # 按 Q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("已退出")
