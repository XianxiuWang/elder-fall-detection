"""Merge URFD + main_data, retrain model. Minimal version with flush."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['PYTHONUNBUFFERED'] = '1'

import pandas as pd
import numpy as np
from config import DATA_DIR, MODEL_DIR, STATE_NAMES, STATE_LABELS
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

BASE = DATA_DIR

print("="*60, flush=True)
print("Loading data...", flush=True)

urfd = pd.read_csv(os.path.join(BASE, 'urfd_features_20260507_205529.csv'))
main_data = pd.read_csv(os.path.join(BASE, 'main_data_features_20260509_162752.csv'))

print(f"URFD: {urfd.shape[0]} records", flush=True)
for i, name in enumerate(STATE_NAMES):
    cnt = int((urfd['label'] == i).sum())
    if cnt > 0:
        print(f"  {name}: {cnt}", flush=True)

print(f"Main: {main_data.shape[0]} records", flush=True)
for i, name in enumerate(STATE_NAMES):
    cnt = int((main_data['label'] == i).sum())
    if cnt > 0:
        print(f"  {name}: {cnt}", flush=True)

# Unify column names
urfd_feat_cols = [c for c in urfd.columns if c not in ('label', 'label_name', 'split')]
main_feat_cols = [c for c in main_data.columns if c not in ('label', 'label_name', 'split')]
assert len(urfd_feat_cols) == len(main_feat_cols) == 98
rename_map = dict(zip(main_feat_cols, urfd_feat_cols))
main_data = main_data.rename(columns=rename_map)

# Merge
merged = pd.concat([urfd, main_data], ignore_index=True)
print(f"\nMerged: {merged.shape[0]} records", flush=True)
for i, name in enumerate(STATE_NAMES):
    cnt = int((merged['label'] == i).sum())
    if cnt > 0:
        print(f"  {name}: {cnt} ({cnt/len(merged)*100:.1f}%)", flush=True)

# Split
train_df = merged[merged['split'].isin(['train'])]
test_df = merged[merged['split'].isin(['test', 'valid'])]

X_train = train_df[urfd_feat_cols].values.astype(np.float32)
y_train = train_df['label'].values.astype(np.int32)
X_test = test_df[urfd_feat_cols].values.astype(np.float32)
y_test = test_df['label'].values.astype(np.int32)

print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}", flush=True)

# Normalize
print("Normalizing...", flush=True)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train
print("Training RandomForest (n=300, depth=20)...", flush=True)
t0 = time.time()

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=1,
)
model.fit(X_train_s, y_train)

elapsed = time.time() - t0
print(f"Done in {elapsed:.1f}s", flush=True)

# Evaluate
y_pred = model.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}", flush=True)
print("\n" + classification_report(y_test, y_pred, labels=range(len(STATE_NAMES)), target_names=STATE_NAMES, zero_division=0), flush=True)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=list(range(len(STATE_NAMES))))
print("Confusion Matrix (rows=true, cols=pred):", flush=True)
header = f"{'':>12}" + "".join(f"{n:>8}" for n in STATE_NAMES)
print(header, flush=True)
for i, name in enumerate(STATE_NAMES):
    if (y_test == i).sum() == 0:
        continue
    row = f"{name:>12}" + "".join(f"{cm[i][j]:8d}" for j in range(len(STATE_NAMES)))
    print(row, flush=True)

# Sitting→fall misclassification
si = STATE_LABELS['sitting']
fi = STATE_LABELS['fall']
sit_test_cnt = int((y_test == si).sum())
sit_as_fall = int(((y_test == si) & (y_pred == fi)).sum())
if sit_test_cnt > 0:
    print(f"\nSitting misclassified as fall: {sit_as_fall}/{sit_test_cnt} ({sit_as_fall/sit_test_cnt*100:.1f}%)", flush=True)

# Save
print("\nSaving model...", flush=True)
bundle = {
    "model": model,
    "scaler": scaler,
    "metrics": {
        "model_type": "RandomForest",
        "accuracy": float(acc),
        "feature_dim": X_train.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_time": elapsed,
        "class_names": STATE_NAMES,
    }
}

save_path = os.path.join(MODEL_DIR, "pose_classifier.joblib")
if os.path.exists(save_path):
    backup = os.path.join(MODEL_DIR, "pose_classifier_backup_20260509.joblib")
    os.rename(save_path, backup)
    print(f"Backed up old model to: {backup}", flush=True)

joblib.dump(bundle, save_path)
print(f"Model saved to: {save_path}", flush=True)

print("\n" + "="*60, flush=True)
print("TRAINING COMPLETE", flush=True)
print(f"Accuracy: {acc:.4f}", flush=True)
print("="*60, flush=True)
