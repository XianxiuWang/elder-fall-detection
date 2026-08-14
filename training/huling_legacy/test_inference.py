"""Headless inference test - captures webcam frames and logs predictions."""
import time, joblib, os, sys
import cv2
import mediapipe as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import FeatureExtractor, landmarks_from_mediapipe

MODEL_PATH = r"D:\Users\wangxianxiu\.openclaw\workspace\huling_model\models\pose_classifier.joblib"

# Load model
print("[1] Loading model...", flush=True)
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]
state_names = bundle["state_names"]
print(f"    Model: RF, {state_names}", flush=True)
print(f"    Metrics: acc={bundle['metrics']['accuracy']:.2%}, cv_f1={bundle['metrics']['cv_f1_mean']:.4f}", flush=True)

# Init
print("[2] Starting MediaPipe...", flush=True)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
extractor = FeatureExtractor(use_motion=False)

# Open camera
print("[3] Opening camera...", flush=True)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("ERROR: Camera not available!", flush=True)
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("    Camera ready", flush=True)

# Warmup - discard first few frames
for _ in range(10):
    cap.read()

# Run inference on 100 frames
print("[4] Running inference on 100 frames...", flush=True)
results = []
fps_times = []

for i in range(100):
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_results = pose.process(frame_rgb)

    if mp_results.pose_landmarks:
        landmarks = landmarks_from_mediapipe(mp_results.pose_landmarks)
        fv = extractor.extract(landmarks)
        X = scaler.transform(fv.values.reshape(1, -1))
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        confidence = proba[pred]
        results.append((pred, confidence))
    else:
        results.append((-1, 0.0))  # no detection

    fps_times.append(time.time() - t0)

cap.release()
pose.close()

# Report
valid = [r for r in results if r[0] >= 0]
print(f"\n===== RESULTS =====", flush=True)
print(f"Total frames: {len(results)}", flush=True)
print(f"Detected: {len(valid)}/{len(results)} ({len(valid)/len(results)*100:.0f}%)", flush=True)

from collections import Counter
pred_counts = Counter(r[0] for r in valid)
print(f"\nPredictions:", flush=True)
for cls_id, cnt in pred_counts.most_common():
    name = state_names[cls_id]
    avg_conf = sum(r[1] for r in valid if r[0] == cls_id) / cnt
    print(f"  {name:12s}: {cnt:3d} frames, avg confidence={avg_conf:.3f}", flush=True)

avg_fps = 1.0 / np.mean(fps_times) if fps_times else 0
print(f"\nPerformance: {avg_fps:.1f} FPS (avg {np.mean(fps_times)*1000:.0f}ms/frame)", flush=True)
print(f"\n[OK] Inference pipeline works! Use 'python inference.py' for live GUI.", flush=True)
