"""Ultra-simple: process Subject.1 Fall backwards only"""
import os, sys, numpy as np, cv2, mediapipe as mp

# Redirect stderr to file during init
stderr_file = r"E:\老人跌倒\data\subject_features\_stderr.txt"
os.makedirs(os.path.dirname(stderr_file), exist_ok=True)

sys.stderr = open(stderr_file, 'w')
pose = mp.solutions.pose.Pose(
    static_image_mode=True, model_complexity=1, smooth_landmarks=False,
    min_detection_confidence=0.3, min_tracking_confidence=0.3,
)
sys.stderr.close()
sys.stderr = sys.__stderr__

print("Pose ready", flush=True)

d = r"F:\动作数据集\数据集（1）\Subject.1\Fall backwards"
files = sorted([f for f in os.listdir(d) if f.endswith('.png')])
print(f"Processing {len(files)} files from Fall backwards", flush=True)

kps = []
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
    if i % 50 == 0:
        print(f"  {i}/{len(files)} det={len(kps)}", flush=True)

pose.close()

if kps:
    out = r"E:\老人跌倒\data\subject_features\TEST_fallback.npz"
    kp = np.array(kps, dtype=np.float32)
    labels = np.zeros(len(kps), dtype=np.int32)
    np.savez_compressed(out, keypoints=kp, labels=labels, category=np.int64(0))
    print(f"Saved {len(kps)} frames to {out}", flush=True)
else:
    print("No detections!", flush=True)
