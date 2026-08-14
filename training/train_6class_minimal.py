"""
train_6class_minimal.py — 极简版训练
无超参搜索，直接 XGBoost 默认参数
"""
import os, sys, pickle, json, time, warnings
from collections import Counter
import numpy as np

warnings.filterwarnings("ignore")

MERGE_DIR = r"E:\老人跌倒\data\custom_6class_merged"
MODEL_OUT = r"E:\老人跌倒\models\fall_classifier_6class.pkl"
CLASS_NAMES = ["Fall", "SitDown", "StandUp", "Walking", "WakeUp", "Standing"]

sys.path.insert(0, r"E:\老人跌倒\training")
from train_fall_classifier import FeatureExtractor

class V3FeatureExtractor(FeatureExtractor):
    def extract_window(self, window):
        base_vec, _ = super().extract_window(window)
        torso_y = window[:, 11, 1]; n = len(torso_y)
        extra = [
            torso_y[:n//3].mean(), torso_y[n//3:2*n//3].mean(), torso_y[2*n//3:].mean(),
            np.polyfit(np.arange(n), torso_y, 1)[0],
            np.polyfit(np.arange(n), window[:, 0, 1], 1)[0],
        ]
        speeds = np.linalg.norm(np.diff(window[:,:,:2], axis=0), axis=2)
        extra.append(speeds.std(axis=0).mean())
        for idx in [0, 11, 23]:
            extra.append(window[-1, idx, 1] - window[0, idx, 1])
        return np.concatenate([base_vec, np.array(extra)]), []

def augment_standup(kpts, n_augment=2):
    versions = [kpts.copy()]
    f = kpts.copy(); f[:,:,0] = 1.0 - f[:,:,0]; versions.append(f)
    return versions[:n_augment+1]

print("=" * 60)
print("  6-Class Fall Classifier (MINIMAL)")
print("=" * 60)

# --- Load ---
extractor = V3FeatureExtractor(window_size=30)
print("\n[1] Loading data...")
X_all, y_all = [], []
files = sorted([f for f in os.listdir(MERGE_DIR) if f.endswith('.npz')])
for i, fn in enumerate(files):
    path = os.path.join(MERGE_DIR, fn)
    try:
        d = np.load(path, allow_pickle=True)
        kp = d['keypoints']; cat = int(d['category'])
        if kp.shape[0] < 30: continue
        vers = [kp]
        if cat == 2: vers = augment_standup(kp, 2)
        for v in vers:
            k3 = v[:,:,:3]; w = v.shape[0]
            idxs = list(range(0, w-29, 5))
            if len(idxs) > 40:
                idxs = sorted(np.random.choice(idxs, 40, replace=False).tolist())
            for s in idxs:
                vec, _ = extractor.extract_window(k3[int(s):int(s)+30])
                X_all.append(vec); y_all.append(cat)
    except: pass
    if (i+1) % 50 == 0: print(f"  {i+1}/{len(files)} ({len(X_all)} samples)")

X = np.nan_to_num(np.array(X_all, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
y = np.array(y_all, dtype=np.int32)
print(f"\n  Loaded: {X.shape[0]} samples, {X.shape[1]} features")
for ci, cnt in sorted(Counter(y).items()):
    print(f"  {CLASS_NAMES[ci]:12s}: {cnt:5d}")

# --- Split ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train); X_val_s = scaler.transform(X_val)

# --- SMOTE ---
print("\n[2] SMOTE...")
from imblearn.over_sampling import SMOTE
before = Counter(y_train)
print(f"  Before: {dict(sorted(before.items()))}")
target = max(before.values())
k = max(1, min(3, min(before.values())-1))
sm = SMOTE(sampling_strategy={c: max(before[c], min(target, 1500)) for c in range(6)},
           random_state=42, k_neighbors=k)
X_train_s, y_train = sm.fit_resample(X_train_s, y_train)
after = Counter(y_train)
print(f"  After: {dict(sorted(after.items()))} ({len(y_train)} samples)")

# --- Train XGBoost ---
print("\n[3] Training XGBoost...")
import xgboost as xgb
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    objective='multi:softprob', num_class=6,
    random_state=42, verbosity=0,
)
t0 = time.time()
model.fit(X_train_s, y_train)
tt = time.time() - t0
print(f"  Done: {tt:.1f}s")

from sklearn.metrics import (accuracy_score, f1_score, classification_report, confusion_matrix)
y_prob = model.predict_proba(X_val_s)
y_pred = np.argmax(y_prob, axis=1)
acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred, average='weighted')
print(f"\n  Accuracy: {acc:.4f} | Weighted F1: {f1:.4f}")

cm = confusion_matrix(y_val, y_pred)
print(f"\n  Confusion Matrix:")
hdr = f"  {'':>10s}" + "".join(f"{n:>8s}" for n in CLASS_NAMES)
print(hdr)
for i, n in enumerate(CLASS_NAMES):
    print(f"  {n:>10s}" + "".join(f"{cm[i][j]:8d}" for j in range(6)))

# --- Thresholds ---
thresholds = {}
for ci in range(6):
    pr = y_prob[:, ci]; tb = (y_val==ci).astype(int)
    bt, bs = 1.0/6, 0
    for thr in np.linspace(0.05,0.95,91):
        pb = (pr>=thr).astype(int)
        tp = ((pb==1)&(tb==1)).sum(); fn = ((pb==0)&(tb==1)).sum(); fp = ((pb==1)&(tb==0)).sum()
        rec = tp/max(tp+fn,1); prec = tp/max(tp+fp,1)
        sc = 2*rec*prec/max(rec+prec,1e-10)
        if ci==0: sc += 0.15*rec
        if sc>bs: bs=sc; bt=thr
    thresholds[CLASS_NAMES[ci]] = float(bt)

print(f"\n  Per-class (with optimized thresholds):")
for i, n in enumerate(CLASS_NAMES):
    t = cm[i].sum(); tp = cm[i,i]
    ws = [f"{CLASS_NAMES[j]}:{cm[i][j]}" for j in range(6) if j!=i and cm[i][j]>0]
    tag = f"  -> {','.join(ws)}" if ws else "  OK"
    print(f"    {n:10s}: {tp:3d}/{t:3d} ({100*tp/max(t,1):5.1f}%) thr={thresholds[n]:.2f}{tag}")

print(f"\n  Report:")
print(classification_report(y_val, y_pred, target_names=CLASS_NAMES, zero_division=0))

# --- CV ---
from sklearn.model_selection import StratifiedKFold, cross_val_score
aX = np.vstack([X_train_s, X_val_s]); ay = np.concatenate([y_train, y_val])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cva = cross_val_score(model, aX, ay, cv=cv, scoring='accuracy', n_jobs=-1)
cvf = cross_val_score(model, aX, ay, cv=cv, scoring='f1_weighted', n_jobs=-1)
print(f"\n[4] CV accuracy: {cva.mean():.4f} +/- {cva.std():.4f}")
print(f"    CV f1:       {cvf.mean():.4f} +/- {cvf.std():.4f}")

# --- Save ---
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
bundle = {
    "model": model, "scaler": scaler, "classes": CLASS_NAMES,
    "feature_dim": X.shape[1],
    "config": {"window_size": 30, "stride": 5, "thresholds": thresholds, "smooth_alpha": 0.3},
    "metrics": {"accuracy": acc, "f1": f1, "cv_accuracy_mean": float(cva.mean()), "cv_f1_mean": float(cvf.mean())},
}
with open(MODEL_OUT, "wb") as f: pickle.dump(bundle, f)

report = {"classes": CLASS_NAMES, "accuracy": acc, "f1": f1, "thresholds": thresholds,
          "per_class": {CLASS_NAMES[i]: float(cm[i,i]/max(cm[i].sum(),1)) for i in range(6)},
          "cv_accuracy": float(cva.mean()), "cv_f1": float(cvf.mean())}
with open(MODEL_OUT.replace(".pkl","_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"  MODEL: {MODEL_OUT}  |  {acc:.2%}  |  F1={f1:.4f}")
print(f"{'='*60}")
