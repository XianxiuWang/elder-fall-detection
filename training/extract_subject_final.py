"""Process Subject.1~9 (all PNG frames → MediaPipe → .npz)"""
import os, sys, time, json
import numpy as np
import cv2
import mediapipe as mp

SUBJECT_DIR = r"F:\动作数据集\数据集（1）"
OUTPUT_DIR = r"E:\老人跌倒\data\subject_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ACTION_MAP = {"Fall backwards": 0, "Fall forward": 0, "Fall left": 0,
              "Fall right": 0, "Fall sitting": 0, "Sit down": 1, "Walk": 3}
MAX_WIDTH = 640
MIN_DETECTION_CONF = 0.3

# ── Init Pose once ──
print("Init Pose...", flush=True)
pose = mp.solutions.pose.Pose(
    static_image_mode=True, model_complexity=1, smooth_landmarks=False,
    min_detection_confidence=MIN_DETECTION_CONF,
    min_tracking_confidence=MIN_DETECTION_CONF,
)
print("Ready.", flush=True)

total_start = time.time()
seg_counter = 0
cat_counts = {}
grand = {"detected": 0, "missed": 0, "failed": 0, "total": 0}

for sid in range(1, 10):
    sp = os.path.join(SUBJECT_DIR, f"Subject.{sid}")
    if not os.path.isdir(sp):
        print(f"S{sid}: SKIP (not found)", flush=True)
        continue
    
    # Gather
    all_imgs = []
    for an in sorted(os.listdir(sp)):
        ap = os.path.join(sp, an)
        if not os.path.isdir(ap) or an not in ACTION_MAP:
            continue
        cat = ACTION_MAP[an]
        for pn in sorted([f for f in os.listdir(ap) if f.endswith('.png')]):
            all_imgs.append((os.path.join(ap, pn), cat, an))
    
    n = len(all_imgs)
    if n == 0:
        print(f"S{sid}: SKIP (no images)", flush=True)
        continue
    
    print(f"Subject.{sid}: {n} frames...", flush=True)
    kpb, lbb = [], []
    cur_cat = None
    sd = sm = sf = 0
    t1 = time.time()
    
    for i, (ip, cat, an) in enumerate(all_imgs):
        if i > 0 and i % 100 == 0:
            dt = time.time() - t1
            print(f"  {i}/{n} | {i/dt:.1f}fps | det={sd}/{i}", flush=True)
        
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
                
                if cur_cat is not None and cat != cur_cat and kpb:
                    kp = np.array(kpb, dtype=np.float32)
                    lb = np.array(lbb, dtype=np.int32)
                    cn = {0:"Fall",1:"SitDown",3:"Walking"}.get(cur_cat,"X")
                    fn = f"SUBJ_S{sid:02d}_{cn}_seg{seg_counter:03d}.npz"
                    np.savez_compressed(os.path.join(OUTPUT_DIR, fn),
                                      keypoints=kp, labels=lb,
                                      category=np.int64(cur_cat))
                    cat_counts[cur_cat] = cat_counts.get(cur_cat, 0) + len(kpb)
                    seg_counter += 1
                    kpb, lbb = [], []
                
                kpb.append(lm)
                lbb.append(cat)
                cur_cat = cat
                sd += 1
            else:
                sm += 1
        except Exception as e:
            sf += 1
            if sf <= 3:
                print(f"  ERR {i}: {e}", flush=True)
    
    # Flush
    if kpb and cur_cat is not None:
        kp = np.array(kpb, dtype=np.float32)
        lb = np.array(lbb, dtype=np.int32)
        cn = {0:"Fall",1:"SitDown",3:"Walking"}.get(cur_cat,"X")
        fn = f"SUBJ_S{sid:02d}_{cn}_seg{seg_counter:03d}.npz"
        np.savez_compressed(os.path.join(OUTPUT_DIR, fn),
                          keypoints=kp, labels=lb, category=np.int64(cur_cat))
        cat_counts[cur_cat] = cat_counts.get(cur_cat, 0) + len(kpb)
        seg_counter += 1
    
    grand["detected"] += sd
    grand["missed"] += sm
    grand["failed"] += sf
    grand["total"] += n
    dt = time.time() - t1
    print(f"  DONE {dt:.0f}s | det={sd}/{n} ({sd/n*100:.1f}%) | miss={sm} fail={sf}", flush=True)

pose.close()

total_t = time.time() - total_start
print(f"\n{'='*60}", flush=True)
print(f"ALL DONE: {total_t:.0f}s / {total_t/60:.1f}min", flush=True)
print(f"Stats: {grand}", flush=True)
cn = {0:"Fall",1:"SitDown",3:"Walking",5:"Standing"}
for cat, n in sorted(cat_counts.items()):
    print(f"  {cat} ({cn[cat]}): {n} frames", flush=True)

report = {"stats": grand, "categories": {str(k):v for k,v in cat_counts.items()},
          "elapsed_s": total_t, "segments": seg_counter}
with open(os.path.join(OUTPUT_DIR, "report.json"), 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"Report: {os.path.join(OUTPUT_DIR, 'report.json')}", flush=True)
