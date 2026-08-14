"""Minimal test: process 5 frames of Subject.1"""
import os, sys, traceback, numpy as np, cv2, time, mediapipe as mp

img_dir = r"F:\动作数据集\数据集（1）\Subject.1\Fall backwards"
pngs = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])[:5]

print(f"Will process {len(pngs)} frames from Subject.1/Fall backwards", flush=True)

pose = mp.solutions.pose.Pose(
    static_image_mode=True, model_complexity=1, smooth_landmarks=False,
    min_detection_confidence=0.3, min_tracking_confidence=0.3,
)

detected = 0
for png_name in pngs:
    path = os.path.join(img_dir, png_name)
    try:
        raw = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if w > 640:
            img = cv2.resize(img, (640, int(h * 640 / w)))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        results = pose.process(img_rgb)
        dt = time.time() - t0
        has = results.pose_landmarks is not None
        detected += 1 if has else 0
        print(f"  {png_name}: {'OK' if has else 'MISS'} ({dt*1000:.0f}ms)", flush=True)
    except Exception:
        print(f"  {png_name}: ERROR", flush=True)
        traceback.print_exc()

pose.close()
print(f"\nDone: {detected}/{len(pngs)} detected", flush=True)
