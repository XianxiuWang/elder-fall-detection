"""Quick train script for URFD data - no emoji, minimal output."""
import csv, os, time, joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV_PATH = r"D:\Users\wangxianxiu\.openclaw\workspace\huling_model\data\urfd_features_20260507_205529.csv"
MODEL_DIR = r"D:\Users\wangxianxiu\.openclaw\workspace\huling_model\models"
os.makedirs(MODEL_DIR, exist_ok=True)

STATE_NAMES = ["walking", "sitting", "lying", "long_sit", "abnormal", "fall"]

# ===== Load =====
print("[1/5] Loading data...", flush=True)
rows = []
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for r in reader:
        rows.append(r)

has_split = (header[-1] == 'split')
label_idx = -3 if has_split else -2
feat_count = len(header) + label_idx  # number of feature columns

X = np.array([[float(v) for v in r[:feat_count]] for r in rows], dtype=np.float32)
y = np.array([int(r[label_idx]) for r in rows], dtype=np.int32)
print(f"  Samples: {len(X)}, Features: {X.shape[1]}", flush=True)

from collections import Counter
for cls_id, cnt in sorted(Counter(y).items()):
    print(f"  Class {cls_id} ({STATE_NAMES[cls_id]}): {cnt}", flush=True)

# ===== Split =====
if has_split:
    train_idx = [i for i, r in enumerate(rows) if r[-1] == 'train']
    test_idx = [i for i, r in enumerate(rows) if r[-1] == 'test']
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"  Split: train={len(train_idx)}, test={len(test_idx)}", flush=True)
else:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"  Split: train={len(X_train)}, test={len(X_test)}", flush=True)

# ===== Scale =====
print("[2/5] Scaling...", flush=True)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ===== Train =====
print("[3/5] Training RandomForest...", flush=True)
t0 = time.time()
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_s, y_train)
print(f"  Done in {time.time()-t0:.1f}s", flush=True)

# ===== Evaluate =====
print("[4/5] Evaluating...", flush=True)
y_pred = model.predict(X_test_s)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\n  Accuracy : {acc:.4f} ({acc*100:.1f}%)")
print(f"  Precision: {prec:.4f}")
print(f"  Recall   : {rec:.4f}")
print(f"  F1-score : {f1:.4f}", flush=True)

present = sorted(set(y_test) | set(y_pred))
print("\n" + classification_report(
    y_test, y_pred,
    labels=present,
    target_names=[STATE_NAMES[i] for i in present],
    zero_division=0,
), flush=True)

# ===== CV =====
print("[5/5] Cross-validation...", flush=True)
cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='f1_weighted')
print(f"  CV F1: mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}", flush=True)

# ===== Confusion Matrix =====
cm = confusion_matrix(y_test, y_pred, labels=present)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[STATE_NAMES[i] for i in present],
)
disp.plot(cmap='Blues', values_format='d')
plt.tight_layout()
cm_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150)
print(f"  Confusion matrix: {cm_path}", flush=True)
plt.close()

# ===== Save =====
model_path = os.path.join(MODEL_DIR, 'pose_classifier.joblib')
joblib.dump({
    'model': model,
    'scaler': scaler,
    'metrics': {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'cv_f1_mean': cv_scores.mean(), 'cv_f1_std': cv_scores.std(),
    },
    'state_names': STATE_NAMES,
    'feature_count': X.shape[1],
    'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
}, model_path)
print(f"\nModel saved: {model_path}", flush=True)
print("===== TRAINING COMPLETE =====", flush=True)
