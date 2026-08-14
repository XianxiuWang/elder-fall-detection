"""Minimal debug: first 3 frames with valgrind-style tracing"""
import os, sys, numpy as np, cv2, mediapipe as mp

d = r"F:\动作数据集\数据集（1）\Subject.1\Fall backwards"
pngs = sorted([f for f in os.listdir(d) if f.endswith('.png')])[:3]

pose = mp.solutions.pose.Pose(
    static_image_mode=True, model_complexity=1, smooth_landmarks=False,
    min_detection_confidence=0.3, min_tracking_confidence=0.3,
)

for pn in pngs:
    fp = os.path.join(d, pn)
    print(f"Reading {fp}", flush=True)
    raw = np.fromfile(fp, dtype=np.uint8)
    print(f"  raw: {raw.shape[0]} bytes", flush=True)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    print(f"  img: {img.shape if img is not None else None}", flush=True)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"  rgb: {rgb.shape}", flush=True)
    print(f"  calling pose.process...", flush=True)
    sys.stdout.flush()
    res = pose.process(rgb)
    print(f"  result: {'OK' if res.pose_landmarks else 'MISS'}", flush=True)

pose.close()
print("DONE", flush=True)
