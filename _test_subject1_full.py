"""Process full Subject.1 with robust error handling"""
import os, sys, traceback, numpy as np, cv2, time, mediapipe as mp

SUBJECT_DIR = r"F:\动作数据集\数据集（1）"
OUTPUT_DIR = r"E:\老人跌倒\data\subject_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ACTION_MAP = {
    "Fall backwards": 0, "Fall forward": 0, "Fall left": 0,
    "Fall right": 0, "Fall sitting": 0, "Sit down": 1, "Walk": 3,
}

sp = os.path.join(SUBJECT_DIR, "Subject.1")
all_images = []
for action_name in sorted(os.listdir(sp)):
    ap = os.path.join(sp, action_name)
    if not os.path.isdir(ap) or action_name not in ACTION_MAP:
        continue
    category = ACTION_MAP[action_name]
    for png_name in sorted([f for f in os.listdir(ap) if f.endswith('.png')]):
        all_images.append((os.path.join(ap, png_name), category, action_name))

print(f"Subject.1: {len(all_images)} frames", flush=True)

pose = mp.solutions.pose.Pose(
    static_image_mode=True, model_complexity=1, smooth_landmarks=False,
    min_detection_confidence=0.3, min_tracking_confidence=0.3,
)

kp_buf, lbl_buf = [], []
detected, missed, failed = 0, 0, 0
t0 = time.time()
current_cat = None
seg_idx = 0

for i, (img_path, category, action_name) in enumerate(all_images):
    if i % 100 == 0 and i > 0:
        elapsed = time.time() - t0
        fps = i / elapsed if elapsed > 0 else 0
        det_rate = detected / max(i, 1) * 100
        eta = (len(all_images) - i) / max(fps, 0.01)
        print(f"  {i}/{len(all_images)} | {fps:.1f}fps | 检出:{det_rate:.0f}% | ETA:{eta:.0f}s", flush=True)
    
    try:
        if current_cat is not None and category != current_cat and kp_buf:
            kp = np.array(kp_buf, dtype=np.float32)
            labels_arr = np.array(lbl_buf, dtype=np.int32)
            fname = f"TEST_S01_{action_name}_seg{seg_idx:02d}.npz"
            np.savez_compressed(os.path.join(OUTPUT_DIR, fname),
                              keypoints=kp, labels=labels_arr,
                              category=np.int64(current_cat))
            print(f"    -> {fname} ({len(kp_buf)} 帧)", flush=True)
            kp_buf, lbl_buf = [], []
            seg_idx += 1
        
        raw = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            failed += 1; continue
        
        h, w = img.shape[:2]
        if w > 640:
            img = cv2.resize(img, (640, int(h * 640 / w)))
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        
        if results.pose_landmarks:
            lm = np.zeros((33, 4), dtype=np.float32)
            for j, landmark in enumerate(results.pose_landmarks.landmark):
                lm[j] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
            kp_buf.append(lm)
            lbl_buf.append(category)
            detected += 1
            current_cat = category
        else:
            missed += 1
    except Exception as e:
        failed += 1
        if failed <= 3:
            print(f"  [ERR:{i}] {os.path.basename(img_path)}: {e}", flush=True)

if kp_buf:
    kp = np.array(kp_buf, dtype=np.float32)
    labels_arr = np.array(lbl_buf, dtype=np.int32)
    fname = f"TEST_S01_{action_name}_seg{seg_idx:02d}.npz"
    np.savez_compressed(os.path.join(OUTPUT_DIR, fname),
                      keypoints=kp, labels=labels_arr, category=np.int64(current_cat))
    print(f"    -> {fname} ({len(kp_buf)} 帧)", flush=True)

pose.close()
elapsed = time.time() - t0
print(f"\nDone: {detected}/{len(all_images)} ({detected/len(all_images)*100:.1f}%) in {elapsed:.0f}s", flush=True)
print(f"Missed: {missed}, Failed: {failed}", flush=True)
