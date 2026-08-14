#!/usr/bin/env python3
"""
train_6class_v4_final.py — 最终打磨版
========================================
v3 51维特征 + SMOTE平衡 + 阈值调优 + 推理级时序平滑
"""
import os, sys, pickle, json, time, warnings
from collections import Counter
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_fall_classifier import FeatureExtractor, TrainConfig

DATA_DIR = r"E:\老人跌倒\data\custom_6class"
MODEL_OUT = r"E:\老人跌倒\models\fall_classifier_6class.pkl"
CLASS_NAMES = ["Fall", "SitDown", "StandUp", "Walking", "WakeUp", "Standing"]

import __main__
__main__.TrainConfig = TrainConfig


class V3FeatureExtractor(FeatureExtractor):
    """v3: 42 base + 9 temporal = 51 features"""
    def extract_window(self, window):
        base_vec, base_names = super().extract_window(window)
        torso_y = window[:, 11, 1]; n = len(torso_y)
        extra = [
            torso_y[:n//3].mean(), torso_y[n//3:2*n//3].mean(), torso_y[2*n//3:].mean(),
            np.polyfit(np.arange(n), torso_y, 1)[0],
            np.polyfit(np.arange(n), window[:, 0, 1], 1)[0],
        ]
        speeds = np.linalg.norm(np.diff(window[:,:,:2], axis=0), axis=2)
        extra.append(speeds.std(axis=0).mean())
        for idx in [0,11,23]:
            extra.append(window[-1,idx,1] - window[0,idx,1])
        names = base_names + ['torso_y_early','torso_y_mid','torso_y_late',
                              'torso_y_slope','head_y_slope','joint_speed_std',
                              'head_y_delta','shoulder_y_delta','hip_y_delta']
        return np.concatenate([base_vec, np.array(extra)]), names


def augment_standup(kpts, n_augment=3):
    versions = [kpts.copy()]
    f = kpts.copy(); f[:,:,0] = 1.0 - f[:,:,0]; versions.append(f)
    if kpts.shape[0] > 15:
        idx = np.sort(np.random.choice(kpts.shape[0], size=kpts.shape[0], replace=True))
        versions.append(kpts[idx].copy())
    n = kpts.copy(); n[:,:,:2] += np.random.normal(0,0.005,n[:,:,:2].shape).astype(np.float32)
    versions.append(n)
    return versions[:n_augment+1]


def load_all_data(data_dir, extractor, window_size=30, stride=5, augment=True):
    X_all, y_all = [], []
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    print(f"Found {len(files)} .npz files")
    su_f, su_v = 0, 0
    for fn in files:
        path = os.path.join(data_dir, fn)
        try:
            d = np.load(path, allow_pickle=True); kp = d['keypoints']; cat = int(d['category'])
            nf = kp.shape[0]
            if nf < window_size: continue
            vers = [kp]
            if augment and cat == 2:
                vers = augment_standup(kp, 3); su_f += 1; su_v += len(vers)
            for v in vers:
                k3 = v[:,:,:3]
                for s in range(0, v.shape[0]-window_size+1, stride):
                    vec, _ = extractor.extract_window(k3[s:s+window_size])
                    X_all.append(vec); y_all.append(cat)
        except: pass
    if augment and su_f > 0:
        print(f"  StandUp: {su_f} files -> {su_v} versions (augmented)")
    X = np.nan_to_num(np.array(X_all, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y_all, dtype=np.int32)
    print(f"\nLoaded: {X.shape[0]} samples, {X.shape[1]} features")
    for ci, cnt in sorted(Counter(y).items()):
        print(f"  {CLASS_NAMES[ci]:12s}: {cnt:6d} ({100*cnt/len(y):5.1f}%)")
    return X, y


def main():
    print("="*60)
    print("  V4 FINAL: v3 features + SMOTE + Thresholds")
    print("="*60)
    
    extractor = V3FeatureExtractor(window_size=30)
    
    print("\n[1/5] Loading data...")
    X, y = load_all_data(DATA_DIR, extractor, augment=True)
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train); X_val_s = scaler.transform(X_val)
    print(f"\n  Train: {len(X_train_s)} | Val: {len(X_val_s)} | Features: {X.shape[1]}")
    
    # SMOTE
    print("\n[2/5] SMOTE balancing...")
    smote_done = False
    try:
        from imblearn.over_sampling import SMOTE
        before = Counter(y_train)
        ratio = max(before.values())/min(before.values())
        print(f"  Before: {dict(sorted(before.items()))}  (ratio {ratio:.1f}:1)")
        k = min(3, min(Counter(y_train).values())-1)
        sm = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=max(k,1))
        X_train_s, y_train = sm.fit_resample(X_train_s, y_train)
        after = Counter(y_train)
        print(f"  After:  {dict(sorted(after.items()))}  ({len(y_train)} samples)")
        smote_done = True
    except Exception as e:
        print(f"  [SKIP] {e}")
    
    # Hyperparameter search
    print("\n[3/5] Hyperparameter search (XGBoost, 50 iters)...")
    import xgboost as xgb
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    
    param_dist = {
        'n_estimators': randint(150,500), 'max_depth': randint(4,14),
        'learning_rate': uniform(0.01,0.1), 'subsample': uniform(0.6,0.4),
        'colsample_bytree': uniform(0.6,0.4),
    }
    base = xgb.XGBClassifier(objective='multi:softprob', num_class=6,
                             reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)
    t0 = time.time()
    search = RandomizedSearchCV(base, param_dist, n_iter=50, cv=3,
                                scoring='accuracy', n_jobs=-1, random_state=42, verbose=1)
    search.fit(X_train_s, y_train)
    st = time.time()-t0
    print(f"\n  Search: {st:.1f}s | Best CV: {search.best_score_:.4f}")
    print(f"  Params: {search.best_params_}")
    
    # Train
    print("\n[4/5] Training + thresholds...")
    model = xgb.XGBClassifier(**search.best_params_, objective='multi:softprob',
                              num_class=6, reg_alpha=0.1, reg_lambda=1.0,
                              random_state=42, verbosity=0)
    t0 = time.time(); model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
    tt = time.time()-t0
    print(f"  Train: {tt:.1f}s")
    
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                  classification_report, confusion_matrix)
    y_prob = model.predict_proba(X_val_s); y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_val, y_pred); f1 = f1_score(y_val, y_pred, average='weighted')
    print(f"\n  Accuracy: {acc:.4f} | F1: {f1:.4f}")
    
    cm = confusion_matrix(y_val, y_pred)
    print(f"\n  Confusion Matrix:")
    hdr = f"  {'':>10s}" + "".join(f"{n:>8s}" for n in CLASS_NAMES)
    print(hdr)
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {n:>10s}" + "".join(f"{cm[i][j]:8d}" for j in range(6)))
    
    # Threshold optimization
    print(f"\n  Threshold optimization:")
    thresholds = {}
    for ci in range(6):
        pr = y_prob[:,ci]; tb = (y_val==ci).astype(int)
        bt, bs = 1.0/6, 0
        for thr in np.linspace(0.05,0.95,91):
            pb = (pr>=thr).astype(int)
            tp=((pb==1)&(tb==1)).sum(); fn=((pb==0)&(tb==1)).sum()
            fp=((pb==1)&(tb==0)).sum()
            rec=tp/max(tp+fn,1); prec=tp/max(tp+fp,1)
            sc=2*rec*prec/max(rec+prec,1e-10)
            if ci==0: sc+=0.15*rec
            if sc>bs: bs=sc; bt=thr
        thresholds[CLASS_NAMES[ci]]=float(bt)
        br = ((y_pred==ci)&(y_val==ci)).sum()/max((y_val==ci).sum(),1)
        ar = ((pr>=bt)&(y_val==ci)).sum()/max((y_val==ci).sum(),1)
        print(f"    {CLASS_NAMES[ci]:10s}: thr={bt:.2f}  recall: {br:.2%} -> {ar:.2%}")
    
    # Per-class
    print(f"\n  Per-class:")
    for i, n in enumerate(CLASS_NAMES):
        t = cm[i].sum(); tp = cm[i,i]
        ws = [f"{CLASS_NAMES[j]}:{cm[i][j]}" for j in range(6) if j!=i and cm[i][j]>0]
        tag = f"  -> {', '.join(ws)}" if ws else "  OK"
        print(f"    {n:10s}: {tp:3d}/{t:3d} ({100*tp/max(t,1):5.1f}%){tag}")
    
    print(f"\n  Report:")
    print(classification_report(y_val, y_pred, target_names=CLASS_NAMES, zero_division=0))
    
    # CV
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aX = np.vstack([X_train_s, X_val_s]); ay = np.concatenate([y_train, y_val])
    cva = cross_val_score(model, aX, ay, cv=cv, scoring='accuracy', n_jobs=-1)
    cvf = cross_val_score(model, aX, ay, cv=cv, scoring='f1_weighted', n_jobs=-1)
    print(f"\n[5/5] CV acc: {cva.mean():.4f} +/- {cva.std():.4f}")
    print(f"      CV f1:  {cvf.mean():.4f} +/- {cvf.std():.4f}")
    
    # Save
    bp = {k: (int(v) if isinstance(v,np.integer) else float(v) if isinstance(v,np.floating) else v)
          for k,v in search.best_params_.items()}
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    bundle = {
        "model": model, "scaler": scaler, "classes": CLASS_NAMES, "feature_dim": X.shape[1],
        "config": {"window_size":30, "stride":5, "best_params":bp,
                   "augmentation":True, "smote":smote_done, "thresholds":thresholds, "smooth_alpha":0.3},
        "metrics": {"accuracy":acc, "f1":f1, "train_time_s":tt+st,
                    "n_samples":len(X), "n_classes":6, "cv_accuracy_mean":float(cva.mean())},
    }
    with open(MODEL_OUT, "wb") as f: pickle.dump(bundle, f)
    
    report = {
        "classes": CLASS_NAMES, "accuracy":acc, "f1":f1, "n_features":X.shape[1],
        "thresholds":thresholds, "smote":smote_done,
        "per_class": {CLASS_NAMES[i]: float(cm[i,i]/max(cm[i].sum(),1))
                      for i in range(min(6,cm.shape[0]))},
        "cv_accuracy":float(cva.mean()), "cv_f1":float(cvf.mean()),
    }
    with open(MODEL_OUT.replace(".pkl","_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"  FINAL MODEL SAVED!")
    print(f"  {acc:.2%} acc | {f1:.4f} f1 | CV {cva.mean():.4f}")
    print(f"  SMOTE: {smote_done} | Features: {X.shape[1]}")
    print(f"  Model: {MODEL_OUT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
