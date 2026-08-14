"""No stderr tricks, just process Subject.1"""
import os, sys, numpy as np, cv2, time, mediapipe as mp

d = r"F:\动作数据集\数据集（1）\Subject.1\Fall backwards"
files = sorted([f for f in os.listdir(d) if f.endswith('.png')])[:150]

print(f"Will process {len(files)} frames", flush=True)
print("Init Pose...", flush=True)
t0 = time.time()
pose = mp.solutions.pose.Pose(
    static_image_mode=True, model_complexity=1, smooth_landmarks=False,
    min_detection_confidence=0.3, min_tracking_confidence=0.3,
)
print(f"Pose ready ({time.time()-t0:.1f}s), processing...", flush=True)

kps = []
t0 = time.time()
for i, fn in enumerate(files):
    fp = os.path.join(d, fn)
    raw = np.fromfile(fp, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None: continue
    h, w = img.shape[:2]
    if w > 640:
        img = cv2.resize(img, (640, int(h*640/w)))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if res.pose_landmarks:
        lm = np.zeros((33, 4), dtype=np.float32)
        for j, l in enumerate(res.pose_landmarks.landmark):
            lm[j] = [l.x, l.y, l.z, l.visibility]
        kps.append(lm)
    if (i+1) % 50 == 0:
        dt = time.time() - t0
        print(f"  {i+1}/{len(files)} | {i/dt:.1f}fps | det={len(kps)}", flush=True)

pose.close()
dt = time.time() - t0
print(f"Done: {len(kps)}/{len(files)} det in {dt:.0f}s ({len(kps)/dt:.1f}fps)", flush=True)
