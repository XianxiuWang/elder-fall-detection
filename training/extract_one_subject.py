"""Process single subject: MediaPipe tracking mode + lite model"""
import os, sys, time
import numpy as np
import cv2
import mediapipe as mp

SUBJECT_DIR = r"F:\动作数据集\数据集（1）"
OUTPUT_DIR = r"E:\老人跌倒\data\subject_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ACTION_MAP = {"Fall backwards": 0, "Fall forward": 0, "Fall left": 0,
              "Fall right": 0, "Fall sitting": 0, "Sit down": 1, "Walk": 3}
MAX_WIDTH = 640
SEGMENT = 100

sid = int(sys.argv[1])
sp = os.path.join(SUBJECT_DIR, f"Subject.{sid}")
print(f"Subject.{sid}", flush=True)

all_imgs = []
for an in sorted(os.listdir(sp)):
    ap = os.path.join(sp, an)
    if not os.path.isdir(ap) or an not in ACTION_MAP: continue
    cat = ACTION_MAP[an]
    for pn in sorted([f for f in os.listdir(ap) if f.endswith('.png')]):
        all_imgs.append((os.path.join(ap, pn), cat, an))
n = len(all_imgs)
print(f"  {n} frames", flush=True)
if n == 0:
    print(f"RESULT: sid={sid} det=0 miss=0 fail=0 total=0", flush=True)
    sys.exit(0)

# Use tracking mode + lite model - should be faster and more stable
pose = mp.solutions.pose.Pose(
    static_image_mode=False,  # Use tracking between frames
    model_complexity=0,       # Lite model
    smooth_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
)

kpb, lbb = [], []; cur_cat = None
sd = sm = sf = 0; seg = 0
t1 = time.time()

def save_buffer(cat_int):
    global seg, kpb, lbb
    if not kpb: return 0
    kp = np.array(kpb, dtype=np.float32)
    lb = np.array(lbb, dtype=np.int32)
    cn = {0:"Fall",1:"SitDown",3:"Walking"}.get(cat_int,"X")
    fn = f"SUBJ_S{sid:02d}_{cn}_seg{seg:03d}.npz"
    np.savez_compressed(os.path.join(OUTPUT_DIR, fn),
                      keypoints=kp, labels=lb, category=np.int64(cat_int))
    saved = len(kpb)
    kpb.clear(); lbb.clear()
    seg += 1
    return saved

for i, (ip, cat, an) in enumerate(all_imgs):
    if i > 0 and i % 100 == 0:
        dt = time.time() - t1
        print(f"  {i}/{n} | {i/dt:.1f}fps | det={sd}", flush=True)
    try:
        raw = np.fromfile(ip, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None: sf += 1; continue
        if img.shape[1] > MAX_WIDTH:
            h, w = img.shape[:2]
            img = cv2.resize(img, (MAX_WIDTH, int(h*MAX_WIDTH/w)))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        
        if res.pose_landmarks:
            lm = np.zeros((33, 4), dtype=np.float32)
            for j, l in enumerate(res.pose_landmarks.landmark):
                lm[j] = [l.x, l.y, l.z, l.visibility]
            
            if cur_cat is not None and (cat != cur_cat or len(kpb) >= SEGMENT):
                save_buffer(cur_cat)
            
            kpb.append(lm); lbb.append(cat); cur_cat = cat; sd += 1
        else:
            sm += 1
    except Exception as e:
        sf += 1
        if sf <= 3: print(f"  ERR {i}: {e}", flush=True)

save_buffer(cur_cat)
pose.close()

dt = time.time() - t1
print(f"  DONE {dt:.0f}s | saved {seg} segs | det={sd}/{n} ({sd/n*100:.0f}%) | miss={sm} fail={sf}", flush=True)
print(f"RESULT: sid={sid} det={sd} miss={sm} fail={sf} total={n}", flush=True)
