#!/usr/bin/env python3
"""Process all subjects, write progress to log file"""
import os, sys, time, json, traceback
import numpy as np
import cv2
import mediapipe as mp

SUBJECT_DIR = r"F:\动作数据集\数据集（1）"
OUTPUT_DIR = r"E:\老人跌倒\data\subject_features"
LOG_PATH = r"E:\老人跌倒\data\subject_features\extract.log"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ACTION_MAP = {"Fall backwards": 0, "Fall forward": 0, "Fall left": 0,
              "Fall right": 0, "Fall sitting": 0, "Sit down": 1, "Walk": 3}
SUBJECTS = list(range(1, 10))
MAX_WIDTH = 640

logf = open(LOG_PATH, 'w', encoding='utf-8')

def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    logf.write(line + '\n')
    logf.flush()

log("=" * 60)
log("Subject.1~9 MediaPipe Extraction")
log("=" * 60)
log(f"Subjects: {SUBJECTS}")
log(f"Actions: {list(ACTION_MAP.keys())}")
log(f"Output: {OUTPUT_DIR}")

log("Init MediaPipe Pose...")
t0 = time.time()
pose = mp.solutions.pose.Pose(
    static_image_mode=True, model_complexity=1, smooth_landmarks=False,
    min_detection_confidence=0.3, min_tracking_confidence=0.3,
)
log(f"Ready ({time.time()-t0:.1f}s)")

grand = {"detected": 0, "missed": 0, "failed": 0, "total": 0}
cat_counts = {}
seg_counter = 0

try:
    for sid in SUBJECTS:
        sp = os.path.join(SUBJECT_DIR, f"Subject.{sid}")
        if not os.path.isdir(sp):
            log(f"Subject.{sid}: NOT FOUND, skip")
            continue
        
        # Gather images
        all_imgs = []
        for aname in sorted(os.listdir(sp)):
            ap = os.path.join(sp, aname)
            if not os.path.isdir(ap) or aname not in ACTION_MAP:
                continue
            cat = ACTION_MAP[aname]
            for pn in sorted([f for f in os.listdir(ap) if f.endswith('.png')]):
                all_imgs.append((os.path.join(ap, pn), cat, aname))
        
        n = len(all_imgs)
        if n == 0:
            log(f"Subject.{sid}: no images, skip")
            continue
        
        log(f"Subject.{sid}: {n} frames, starting...")
        kpb, lbb = [], []
        cur_cat = None
        sd, sm, sf = 0, 0, 0
        t1 = time.time()
        
        for i, (imgp, cat, aname) in enumerate(all_imgs):
            if i > 0 and i % 100 == 0:
                et = time.time() - t1
                log(f"  {i}/{n} | {i/et:.1f}fps | det={sd}/{i}")
            
            try:
                raw = np.fromfile(imgp, dtype=np.uint8)
                img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if img is None:
                    sf += 1; continue
                h, w = img.shape[:2]
                if w > MAX_WIDTH:
                    img = cv2.resize(img, (MAX_WIDTH, int(h*MAX_WIDTH/w)))
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                
                if res.pose_landmarks:
                    lm = np.zeros((33, 4), dtype=np.float32)
                    for j, lm_j in enumerate(res.pose_landmarks.landmark):
                        lm[j] = [lm_j.x, lm_j.y, lm_j.z, lm_j.visibility]
                    
                    if cur_cat is not None and cat != cur_cat and kpb:
                        kp = np.array(kpb, dtype=np.float32)
                        lbs = np.array(lbb, dtype=np.int32)
                        cn = {0:"Fall",1:"SitDown",3:"Walking"}.get(cur_cat,"X")
                        fn = f"SUBJ_S{sid:02d}_{cn}_seg{seg_counter:03d}.npz"
                        np.savez_compressed(os.path.join(OUTPUT_DIR, fn),
                                          keypoints=kp, labels=lbs, category=np.int64(cur_cat))
                        log(f"    -> {fn} ({len(kpb)} fr)")
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
                    log(f"  ERR frame {i}: {e}")
        
        # Flush remaining
        if kpb and cur_cat is not None:
            kp = np.array(kpb, dtype=np.float32)
            lbs = np.array(lbb, dtype=np.int32)
            cn = {0:"Fall",1:"SitDown",3:"Walking"}.get(cur_cat,"X")
            fn = f"SUBJ_S{sid:02d}_{cn}_seg{seg_counter:03d}.npz"
            np.savez_compressed(os.path.join(OUTPUT_DIR, fn),
                              keypoints=kp, labels=lbs, category=np.int64(cur_cat))
            log(f"    -> {fn} ({len(kpb)} fr)")
            cat_counts[cur_cat] = cat_counts.get(cur_cat, 0) + len(kpb)
            seg_counter += 1
        
        grand["detected"] += sd
        grand["missed"] += sm
        grand["failed"] += sf
        grand["total"] += n
        log(f"  DONE {time.time()-t1:.0f}s | det={sd} miss={sm} fail={sf}/{n}")

finally:
    pose.close()

total_t = time.time() - t0
log(f"\n{'='*60}")
log(f"ALL DONE ({total_t:.0f}s / {total_t/60:.1f}min)")
log(f"{'='*60}")
log(f"Total: {grand}")
log(f"Categories: {cat_counts}")

report = {"stats": grand, "categories": {str(k):v for k,v in cat_counts.items()},
          "elapsed": total_t, "segments": seg_counter}
with open(os.path.join(OUTPUT_DIR, "report.json"), 'w') as f:
    json.dump(report, f, indent=2)

logf.close()
print(f"\nLog saved to {LOG_PATH}")
