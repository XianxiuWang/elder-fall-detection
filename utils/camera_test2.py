"""模拟 inference.py 完整流程，精确诊断卡在哪里"""
import os
import time
import cv2

print("Step 1: 导入 OpenCV, 检查 CAP_DSHOW 常量")
print(f"  cv2.__version__: {cv2.__version__}")
print(f"  cv2.CAP_DSHOW = {cv2.CAP_DSHOW}")

print("\nStep 2: 打开摄像头 (CAP_DSHOW + index=0)")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print(f"  cap.isOpened(): {cap.isOpened()}")
print(f"  实际后端: {cap.getBackendName()}")
print(f"  分辨率: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

# 设置分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nStep 3: 尝试读取帧（先读 5 帧热身）")
for i in range(5):
    t0 = time.time()
    success, frame = cap.read()
    elapsed = time.time() - t0
    if not success:
        print(f"  第{i+1}帧: read() 失败! 耗时 {elapsed:.2f}s")
    elif frame is None:
        print(f"  第{i+1}帧: frame is None! 耗时 {elapsed:.2f}s")
    else:
        print(f"  第{i+1}帧: OK, shape={frame.shape}, mean={frame.mean():.1f}, 耗时 {elapsed:.2f}s")

# 翻转为镜像（和推理脚本一样）
if success and frame is not None:
    frame = cv2.flip(frame, 1)
    print(f"  翻转后 shape={frame.shape}")

print("\nStep 4: 创建窗口并显示")
cv2.namedWindow("Diagnostic Test", cv2.WINDOW_NORMAL)
cv2.imshow("Diagnostic Test", frame if (success and frame is not None) else None)
print("  imshow() 已调用")
print("  等待按键 (5秒后自动关闭，或手动按键)...")

key = cv2.waitKey(5000) if (success and frame is not None) else 0
print(f"  waitKey 返回: {key}")

cv2.destroyAllWindows()
cap.release()
print("\nDiagnostic complete.")
