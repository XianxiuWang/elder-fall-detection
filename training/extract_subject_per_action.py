"""Process Subject.1~9, save per-action with small segments"""
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
SEGMENT_SIZE = 100  # Save every 100 frames (prevents huge buffers)

pose = mp.solutions.pose.Pose(
    static_image_mode=True, model_complexity=1, smooth_landmarks=False,
    min_detection_confidence=0.3, min_tracking_confidence=0.3)

seg_counter = 0
cat_counts = {}
grand = {"detected": 0, "missed": 0, "failed": 0, "total": 0}
total_start = time.time()

for sid in range(1, 10):
    sp = os.path.join(SUBJECT_DIR, f"Subject.{sid}")
    if not os.path.isdir(sp): continue
    
    for an in sorted(os.listdir(sp)):
        ap = os.path.join(sp, an)
        if not os.path.isdir(ap) or an not in ACTION_MAP: continue
        cat = ACTION_MAP[an]
        pngs = sorted([f for f in os.listdir(ap) if f.endswith('.png')])
        if not pngs: continue
        
        t1 = time.time()
        kpb, lbb = [], []
        sd = sm = sf = 0
        
        for i, pn in enumerate(pngs):
            fp = os.path.join(ap, pn)
            try:
                raw = np.fromfile(fp, dtype=np.uint8)
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
                    kpb.append(lm); lbb.append(cat); sd += 1
                else:
                    sm += 1
            except Exception as e:
                sf += 1
                if sf <= 2:
                    print(f"  ERR S{sid}/{an}/{pn}: {e}", flush=True)
            
            # Save every SEGMENT_SIZE frames
            if len(kpb) >= SEGMENT_SIZE:
                kp = np.array(kpb, dtype=np.float32)
                lb = np.array(lbb, dtype=np.int32)
                cn = {0:"Fall",1:"SitDown",3:"Walking"}.get(cat,"X")
                fn = f"SUBJ_S{sid:02d}_{cn}_seg{seg_counter:03d}.npz"
                np.savez_compressed(os.path.join(OUTPUT_DIR, fn),
                                  keypoints=kp, labels=lb, category=np.int64(cat))
                cat_counts[cat] = cat_counts.get(cat, 0) + len(kpb)
                seg_counter += 1
                kpb, lbb = [], []
        
        # Save remaining
        if kpb:
            kp = np.array(kpb, dtype=np.float32)
            lb = np.array(lbb, dtype=np.int32)
            cn = {0:"Fall",1:"SitDown",3:"Walking"}.get(cat,"X")
            fn = f"SUBJ_S{sid:02d}_{cn}_seg{seg_counter:03d}.npz"
            np.savez_compressed(os.path.join(OUTPUT_DIR, fn),
                              keypoints=kp, labels=lb, category=np.int64(cat))
            cat_counts[cat] = cat_counts.get(cat, 0) + len(kpb)
            seg_counter += 1
        
        grand["detected"] += sd
        grand["missed"] += sm
        grand["failed"] += sf
        grand["total"] += len(pngs)
        dt = time.time() - t1
        print(f"S{sid} {an:20s} | {sd}/{len(pngs)} det ({sd/len(pngs)*100:.0f}%) | miss={sm} | {dt:.0f}s", flush=True)

pose.close()
total_t = time.time() - total_start
print(f"\nALL DONE: {total_t:.0f}s", flush=True)
print(f"Stats: {grand}", flush=True)
cn = {0:"Fall",1:"SitDown",3:"Walking"}
for cat, n in sorted(cat_counts.items()):
    print(f"  {cat} ({cn.get(cat,'?')}): {n} frames", flush=True)
with open(os.path.join(OUTPUT_DIR, "report.json"), 'w') as f:
    json.dump({"stats": grand, "categories": {str(k):v for k,v in cat_counts.items()}}, f, indent=2, ensure_ascii=False)
