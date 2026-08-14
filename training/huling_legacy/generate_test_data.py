"""
Generate test data + Python reference outputs for C code verification.

Output: deploy/test_data.h — C header with landmark data, feature reference,
         and prediction reference, structured for test_deploy.c to compare against.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import joblib
from feature_extractor import FeatureExtractor, Landmark3D, landmarks_from_array
from config import MODEL_DIR, LANDMARK

# --- Load model ---
bundle = joblib.load(os.path.join(MODEL_DIR, "pose_classifier.joblib"))
model = bundle["model"]
scaler = bundle["scaler"]

print(f"Model: {type(model).__name__}")
print(f"  n_features_in: {model.n_features_in_}")
print(f"  classes_: {model.classes_}")
print(f"  scaler.mean_ shape: {scaler.mean_.shape}")
print(f"  scaler.scale_ shape: {scaler.scale_.shape}")

# ================================================================
# Create test poses
# ================================================================

def make_standing():
    """Simulated standing pose (33 keypoints)"""
    lm = []
    for i in range(33):
        name = list(LANDMARK.keys())[i]
        visibility = 0.95
        if "shoulder" in name:
            y, z = 0.35, 0.0
            x = 0.42 if "left" in name else 0.58 if "right" in name else 0.50
        elif "hip" in name:
            y, z = 0.55, 0.0
            x = 0.44 if "left" in name else 0.56 if "right" in name else 0.50
        elif "knee" in name:
            y, z = 0.75, 0.0
            x = 0.45 if "left" in name else 0.55 if "right" in name else 0.50
        elif "ankle" in name or "heel" in name or "foot" in name:
            y, z = 0.92, 0.0
            x = 0.45 if "left" in name else 0.55 if "right" in name else 0.50
        elif "elbow" in name:
            y, z = 0.45, 0.0
            x = 0.35 if "left" in name else 0.65 if "right" in name else 0.50
        elif "wrist" in name:
            y, z = 0.55, 0.0
            x = 0.30 if "left" in name else 0.70 if "right" in name else 0.50
        elif "nose" in name:
            y, x, z = 0.18, 0.50, 0.0
        elif "eye" in name:
            y, z = 0.16, 0.0
            x = 0.47 if "left" in name else 0.53 if "right" in name else 0.50
        elif "ear" in name:
            y, z = 0.17, 0.0
            x = 0.42 if "left" in name else 0.58 if "right" in name else 0.50
        elif "pinky" in name:
            y, z = 0.55, 0.0
            x = 0.28 if "left" in name else 0.72 if "right" in name else 0.50
        elif "index" in name:
            y, z = 0.53, 0.0
            x = 0.29 if "left" in name else 0.71 if "right" in name else 0.50
        elif "thumb" in name:
            y, z = 0.52, 0.0
            x = 0.31 if "left" in name else 0.69 if "right" in name else 0.50
        else:
            y, x, z = 0.20, 0.50, 0.0
        lm.append(Landmark3D(x=x, y=y, z=z, visibility=visibility))
    return lm


def make_sitting():
    """Simulated sitting pose"""
    lm = make_standing()
    # Adjust hip, knee positions for sitting
    for i in range(33):
        name = list(LANDMARK.keys())[i]
        if "hip" in name:
            lm[i].y = 0.50
        elif "knee" in name:
            lm[i].y = 0.65
            lm[i].visibility = 0.7  # partially occluded
        elif "ankle" in name or "heel" in name or "foot" in name:
            lm[i].y = 0.80
    return lm


def make_lying():
    """Simulated lying pose (horizontal body)"""
    lm = []
    for i in range(33):
        name = list(LANDMARK.keys())[i]
        visibility = 0.90
        if "shoulder" in name:
            y, z = 0.48, 0.0
            x = 0.30 if "left" in name else 0.60
        elif "hip" in name:
            y, z = 0.52, 0.0
            x = 0.32 if "left" in name else 0.62
        elif "knee" in name:
            y, z = 0.58, 0.0
            x = 0.33 if "left" in name else 0.61
        elif "ankle" in name or "heel" in name or "foot" in name:
            y, z = 0.62, 0.0
            x = 0.33 if "left" in name else 0.61
        elif "elbow" in name:
            y, z = 0.50, 0.0
            x = 0.25 if "left" in name else 0.65
        elif "wrist" in name:
            y, z = 0.55, 0.0
            x = 0.22 if "left" in name else 0.68
        elif "nose" in name:
            y, x, z = 0.46, 0.45, 0.0
        elif "eye" in name:
            y, z = 0.44, 0.0
            x = 0.43 if "left" in name else 0.47
        elif "ear" in name:
            y, z = 0.45, 0.0
            x = 0.40 if "left" in name else 0.50
        elif "pinky" in name:
            y, z = 0.55, 0.0
            x = 0.20 if "left" in name else 0.70
        elif "index" in name:
            y, z = 0.53, 0.0
            x = 0.21 if "left" in name else 0.69
        elif "thumb" in name:
            y, z = 0.52, 0.0
            x = 0.23 if "left" in name else 0.67
        else:
            y, x, z = 0.50, 0.45, 0.0
        lm.append(Landmark3D(x=x, y=y, z=z, visibility=visibility))
    return lm


def make_fall():
    """Simulated fall pose (on ground, limbs splayed)"""
    lm = make_lying()
    # Make arms more spread (typical fall posture)
    for i in range(33):
        name = list(LANDMARK.keys())[i]
        if "elbow" in name:
            lm[i].x = 0.18 if "left" in name else 0.82
            lm[i].y = 0.52
        elif "wrist" in name:
            lm[i].x = 0.12 if "left" in name else 0.88
            lm[i].y = 0.54
        elif "pinky" in name:
            lm[i].x = 0.10 if "left" in name else 0.90
            lm[i].y = 0.56
        elif "index" in name:
            lm[i].x = 0.11 if "left" in name else 0.89
            lm[i].y = 0.54
        elif "thumb" in name:
            lm[i].x = 0.13 if "left" in name else 0.87
            lm[i].y = 0.53
    # Make legs asymmetric (one straight, one bent)
    lm[LANDMARK["left_knee"]].y = 0.57
    lm[LANDMARK["left_ankle"]].y = 0.63
    lm[LANDMARK["left_foot_index"]].y = 0.64
    lm[LANDMARK["left_heel"]].y = 0.63
    return lm


def make_empty():
    """Empty frame (no person detected - all visibility=0)"""
    lm = []
    for i in range(33):
        lm.append(Landmark3D(x=0.0, y=0.0, z=0.0, visibility=0.0))
    return lm


# ================================================================
# Compute Python reference for each pose
# ================================================================

extractor = FeatureExtractor(use_motion=False)  # single-frame, no motion features

test_cases = []

for pose_name, landmarks_func in [
    ("standing", make_standing),
    ("sitting",  make_sitting),
    ("lying",    make_lying),
    ("fall",     make_fall),
    ("empty",    make_empty),
]:
    lm = landmarks_func()
    # Feature extraction
    fv = extractor.extract(lm)
    features = fv.values.astype(np.float64)  # 98-dim
    
    # Standardize
    features_norm = scaler.transform(features.reshape(1, -1))[0]
    
    # Predict
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features_norm.reshape(1, -1))[0]
        known_classes = getattr(model, 'classes_', list(range(len(proba))))
        pred_class = int(known_classes[np.argmax(proba)])
        scores = np.zeros(6)
        for cls_idx, p in zip(known_classes, proba):
            scores[int(cls_idx)] = float(p)
    else:
        pred_class = int(model.predict(features_norm.reshape(1, -1))[0])
        scores = np.zeros(6)
        scores[pred_class] = 1.0
    
    # Break features into modules
    torso_feat = fv.torso_features.tolist()
    joint_feat = fv.joint_features.tolist()
    angle_feat = fv.angle_features.tolist()
    struct_feat = fv.structure_features.tolist()
    motion_feat = fv.motion_features.tolist()
    sensor_feat = fv.sensor_features.tolist()
    
    test_cases.append({
        "name": pose_name,
        "landmarks": lm,
        "features_raw": features.tolist(),
        "features_norm": features_norm.tolist(),
        "pred_class": pred_class,
        "scores": scores.tolist(),
        "torso": torso_feat,
        "joints": joint_feat,
        "angles": angle_feat,
        "structure": struct_feat,
        "motion": motion_feat,
        "sensor": sensor_feat,
    })

# Also add a second standing frame for motion test
lm2 = make_standing()
# Slightly shift positions
for i, nm in enumerate(list(LANDMARK.keys())):
    if "shoulder" in nm or "hip" in nm or "knee" in nm:
        lm2[i].x += 0.02
        lm2[i].y += 0.01
    elif "elbow" in nm or "wrist" in nm:
        lm2[i].x -= 0.03
        lm2[i].y += 0.02

extractor_motion = FeatureExtractor(use_motion=True, smooth_window=0)
# First frame (init, no prev)
fv1 = extractor_motion.extract_with_motion(test_cases[0]["landmarks"])
# Second frame (has prev, produces motion features)
fv2 = extractor_motion.extract_with_motion(lm2)
features_motion = fv2.values.astype(np.float64)
features_motion_norm = scaler.transform(features_motion.reshape(1, -1))[0]
if hasattr(model, "predict_proba"):
    proba_m = model.predict_proba(features_motion_norm.reshape(1, -1))[0]
    known_classes = getattr(model, 'classes_', list(range(len(proba_m))))
    pred_class_m = int(known_classes[np.argmax(proba_m)])
    scores_m = np.zeros(6)
    for cls_idx, p in zip(known_classes, proba_m):
        scores_m[int(cls_idx)] = float(p)
else:
    pred_class_m = int(model.predict(features_motion_norm.reshape(1, -1))[0])
    scores_m = np.zeros(6)
    scores_m[pred_class_m] = 1.0

test_cases.append({
    "name": "standing_motion",
    "landmarks": lm2,
    "prev_landmarks": test_cases[0]["landmarks"],
    "features_raw": features_motion.tolist(),
    "features_norm": features_motion_norm.tolist(),
    "pred_class": pred_class_m,
    "scores": scores_m.tolist(),
    "motion_features": fv2.motion_features.tolist(),
})

# ================================================================
# Write C test data header
# ================================================================

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deploy")
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "test_data.h"), "w", encoding="utf-8") as f:
    f.write("// Auto-generated test data for C code verification\n")
    f.write("// Generated by generate_test_data.py\n")
    f.write(f"#define N_TEST_CASES {len(test_cases)}\n")
    f.write("#define N_FEATURES 98\n")
    f.write("#define N_KEYPOINTS 33\n")
    f.write("#define N_CLASSES 6\n\n")
    
    f.write("// Test case names\n")
    f.write("static const char *test_names[N_TEST_CASES] = {\n")
    for tc in test_cases:
        f.write(f'    "{tc["name"]}",\n')
    f.write("};\n\n")
    
    # Landmark data (33 keypoints per test case)
    f.write("// Landmark data: test_landmarks[case][keypoint] = {x, y, z, visibility}\n")
    f.write("static const float test_landmarks[N_TEST_CASES][N_KEYPOINTS][4] = {\n")
    for tc in test_cases:
        f.write(f"    {{ // {tc['name']}\n")
        for lm in tc["landmarks"]:
            f.write(f"        {{{lm.x:.10f}f, {lm.y:.10f}f, {lm.z:.10f}f, {lm.visibility:.10f}f}},\n")
        f.write("    },\n")
    f.write("};\n\n")
    
    # Previous landmarks for motion test (same format, only used for motion test cases)
    f.write("// Previous landmarks for motion test cases (0=no prev)\n")
    f.write("static const int test_has_prev[N_TEST_CASES] = {\n")
    for tc in test_cases:
        has = 1 if "prev_landmarks" in tc else 0
        f.write(f"    {has},\n")
    f.write("};\n\n")
    
    f.write("static const float test_prev_landmarks[N_TEST_CASES][N_KEYPOINTS][4] = {\n")
    for tc in test_cases:
        if "prev_landmarks" in tc:
            f.write(f"    {{ // {tc['name']} prev\n")
            for lm in tc["prev_landmarks"]:
                f.write(f"        {{{lm.x:.10f}f, {lm.y:.10f}f, {lm.z:.10f}f, {lm.visibility:.10f}f}},\n")
            f.write("    },\n")
        else:
            f.write(f"    {{ // {tc['name']} (no prev)\n")
            f.write("        {0},\n" * 33)
            f.write("    },\n")
    f.write("};\n\n")
    
    # Reference feature vectors (raw, unstandardized — from C feature extraction)
    f.write("// Python reference: raw features (before standardization)\n")
    f.write("static const double py_features_raw[N_TEST_CASES][N_FEATURES] = {\n")
    for tc in test_cases:
        f.write(f"    {{ // {tc['name']}\n")
        for i in range(0, 98, 8):
            vals = ", ".join(f"{tc['features_raw'][j]:.15e}" for j in range(i, min(i+8, 98)))
            f.write(f"        {vals},\n")
        f.write("    },\n")
    f.write("};\n\n")
    
    # Reference: standardized features
    f.write("// Python reference: standardized features\n")
    f.write("static const double py_features_norm[N_TEST_CASES][N_FEATURES] = {\n")
    for tc in test_cases:
        f.write(f"    {{ // {tc['name']}\n")
        for i in range(0, 98, 8):
            vals = ", ".join(f"{tc['features_norm'][j]:.15e}" for j in range(i, min(i+8, 98)))
            f.write(f"        {vals},\n")
        f.write("    },\n")
    f.write("};\n\n")
    
    # Reference: prediction results
    f.write("// Python reference: predicted class (0-5)\n")
    f.write("static const int py_pred_class[N_TEST_CASES] = {\n")
    for tc in test_cases:
        f.write(f"    {tc['pred_class']},  // {tc['name']}\n")
    f.write("};\n\n")
    
    f.write("// Python reference: prediction scores (per class)\n")
    f.write("static const double py_scores[N_TEST_CASES][N_CLASSES] = {\n")
    for tc in test_cases:
        f.write(f"    {{ // {tc['name']}\n")
        vals = ", ".join(f"{tc['scores'][j]:.15e}" for j in range(6))
        f.write(f"        {vals},\n")
        f.write("    },\n")
    f.write("};\n\n")
    
    # Per-module reference for detailed debug (standard test cases only, skip motion variant)
    f.write("// Python reference: per-module features\n")
    n_standard = 5  # standing, sitting, lying, fall, empty
    for mod_key in ["torso", "joints", "angles", "structure", "motion", "sensor"]:
        dims = [len(tc[mod_key]) for tc in test_cases[:n_standard]]
        max_dim = max(dims)
        f.write(f"#define PY_{mod_key.upper()}_DIM {max_dim}\n")
        f.write(f"static const double py_{mod_key}[{n_standard}][PY_{mod_key.upper()}_DIM] = {{\n")
        for tc in test_cases[:n_standard]:
            vals = ", ".join(f"{tc[mod_key][j]:.15e}" for j in range(len(tc[mod_key])))
            padding = ", 0.0" * (max_dim - len(tc[mod_key]))
            f.write(f"    {{ // {tc['name']}\n        {vals}{padding}\n    }},\n")
        f.write("};\n\n")

print(f"\n[OK] test_data.h written to {out_dir}")
print(f"  Test cases: {len(test_cases)}")
for tc in test_cases:
    print(f"    {tc['name']}: pred={tc['pred_class']}, scores={[f'{s:.4f}' for s in tc['scores']]}")

# ================================================================
# Verify scaler params match
# ================================================================
print("\n=== Scaler Parameter Verification ===")
for i in range(98):
    print(f"  [{i:2d}] mean={scaler.mean_[i]:.10f}  scale={scaler.scale_[i]:.10f}")

print("\nDone!")
