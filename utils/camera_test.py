"""快速诊断摄像头状态"""
import cv2
import time

for index in range(3):
    print(f"\n--- 测试摄像头索引 {index} ---")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # DSHOW 后端更稳定
    print(f"  isOpened: {cap.isOpened()}")
    if not cap.isOpened():
        cap.release()
        continue

    print(f"  分辨率: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"  后端: {cap.getBackendName()}")

    print("  尝试读取第1帧 (timeout=5s)...")
    t0 = time.time()
    success, frame = cap.read()
    elapsed = time.time() - t0
    print(f"  read() 耗时: {elapsed:.2f}s")
    print(f"  是否成功: {success}")
    if success:
        print(f"  帧尺寸: {frame.shape if frame is not None else 'None'}")
        print(f"  帧均值: {frame.mean():.1f} (0=全黑, >0=有画面)")
    else:
        print(f"  read() 返回失败！")

    cap.release()

print("\n完成。如果 read() 耗时 > 3s 或返回失败，说明驱动有问题。")
