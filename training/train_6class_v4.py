#!/usr/bin/env python3
"""
train_6class_v4.py — 六分类终极打磨 v4
========================================
四大优化:
  1. FFT 频域特征 — 关键关节频谱分析
  2. SMOTE 自适应过采样 — 类别平衡
  3. 逐类阈值调优 — 最小化 Fall 漏报
  4. 时序平滑 — 指数移动平均降噪
"""
import os, sys, pickle, json, time, warnings
from collections import Counter
import numpy as np
from scipy.fft import rfft, rfftfreq

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_fall_classifier import FeatureExtractor, TrainConfig

DATA_DIR = r"E:\老人跌倒\data\custom_6class"
MODEL_OUT = r"E:\老人跌倒\models\fall_classifier_6class.pkl"
CLASS_NAMES = ["Fall", "SitDown", "StandUp", "Walking", "WakeUp", "Standing"]

import __main__
__main__.TrainConfig = TrainConfig


# ============================================================
# v4 Feature Extractor (v3 temporal + FFT spectral)
# ============================================================
class V4FeatureExtractor(FeatureExtractor):
    FFT_JOINTS = {'head': 0, 'shoulder_c': 11, 'hip_c': 23}
    
    def __init__(self, window_size=30, fft_n=32):
        super().__init__(window_size=window_size)
        self.fft_n = fft_n
    
    def extract_fft(self, traj):
        n = len(traj)
        if n < 4:
            return [0.0] * 8
        traj_centered = traj - traj.mean()
        fft_vals = rfft(traj_centered, n=self.fft_n)
        mag = np.abs(fft_vals)
        freqs = rfftfreq(self.fft_n, d=1.0/30.0)
        mag_no_dc, freqs_no_dc = mag[1:], freqs[1:]
        if len(mag_no_dc) == 0:
            return [0.0] * 8
        
        total_energy = (mag_no_dc**2).sum()
        if total_energy < 1e-10:
            return [0.0] * 8
        
        dom_idx = np.argmax(mag_no_dc)
        dom_freq = freqs_no_dc[dom_idx]
        dom_mag = mag_no_dc[dom_idx] / max(total_energy**0.5, 1e-10)
        centroid = (freqs_no_dc * mag_no_dc).sum() / max(mag_no_dc.sum(), 1e-10)
        spread = ((freqs_no_dc - centroid)**2 * mag_no_dc).sum() / max(mag_no_dc.sum(), 1e-10)
        spread = spread**0.5
        
        low_e = (mag_no_dc[(freqs_no_dc<2)]**2).sum() / total_energy
        mid_e = (mag_no_dc[(freqs_no_dc>=2)&(freqs_no_dc<5)]**2).sum() / total_energy
        high_e = (mag_no_dc[(freqs_no_dc>=5)]**2).sum() / total_energy
        flatness = mag_no_dc.mean() / max(mag_no_dc.max(), 1e-10)
        
        return [dom_freq, dom_mag, centroid, spread, low_e, mid_e, high_e, flatness]
    
    @property
    def fft_names(self):
        return [f'{j}_{f}' for j in self.FFT_JOINTS
                for f in ['dfreq','dmag','centroid','spread','elow','emid','ehigh','flatness']]
    
    def extract_window(self, window):
        # v3 base (42 features)
        base_vec, base_names = super().extract_window(window)
        
        # v3 enhanced (9 features)
        torso_y = window[:, 11, 1]
        n = len(torso_y)
        extra = [
            torso_y[:n//3].mean(), torso_y[n//3:2*n//3].mean(), torso_y[2*n//3:].mean(),
            np.polyfit(np.arange(n), torso_y, 1)[0],
            np.polyfit(np.arange(n), window[:, 0, 1], 1)[0],
        ]
        speeds = np.linalg.norm(np.diff(window[:, :, :2], axis=0), axis=2)
        extra.append(speeds.std(axis=0).mean())
        for idx in [0, 11, 23]:
            extra.append(window[-1, idx, 1] - window[0, idx, 1])
        
        v3_names = ['torso_y_early','torso_y_mid','torso_y_late',
                    'torso_y_slope','head_y_slope','joint_speed_std',
                    'head_y_delta','shoulder_y_delta','hip_y_delta']
        
        # FFT features (3 joints * 8 = 24 features)
        fft_feats = []
        for jn, ji in self.FFT_JOINTS.items():
            fft_feats.extend(self.extract_fft(window[:, ji, 1]))
        
        combined = np.concatenate([base_vec, np.array(extra), np.array(fft_feats)])
        return combined, base_names + v3_names + self.fft_names


# ============================================================
# Data loading
# ============================================================
def augment_standup(kpts, n_augment=3):
    versions = [kpts.copy()]
    flipped = kpts.copy(); flipped[:,:,0] = 1.0 - flipped[:,:,0]
    versions.append(flipped)
    if kpts.shape[0] > 15:
        idx = np.sort(np.random.choice(kpts.shape[0], size=kpts.shape[0], replace=True))
        versions.append(kpts[idx].copy())
    noisy = kpts.copy()
    noisy[:,:,:2] += np.random.normal(0, 0.005, noisy[:,:,:2].shape).astype(np.float32)
    versions.append(noisy)
    return versions[:n_augment+1]


def load_all_data(data_dir, extractor, window_size=30, stride=5, augment=True):
    X_all, y_all = [], []
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
    print(f"Found {len(files)} .npz files")
    su_f, su_v = 0, 0
    for fname in files:
        path = os.path.join(data_dir, fname)
        try:
            data = np.load(path, allow_pickle=True)
            kpts = data['keypoints']
            cat = int(data['category'])
            nf = kpts.shape[0]
            if nf < window_size: continue
            versions = [kpts]
            if augment and cat == 2:
                versions = augment_standup(kpts, n_augment=3)
                su_f += 1; su_v += len(versions)
            for kp in versions:
                k3 = kp[:,:,:3]
                for s in range(0, kp.shape[0]-window_size+1, stride):
                    vec, _ = extractor.extract_window(k3[s:s+window_size])
                    X_all.append(vec); y_all.append(cat)
        except Exception as e:
            print(f"  [SKIP] {fname}: {e}")
    if augment and su_f > 0:
        print(f"  StandUp: {su_f} files -> {su_v} versions")
    X = np.nan_to_num(np.array(X_all, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y_all, dtype=np.int32)
    print(f"\nLoaded: {X.shape[0]} samples, {X.shape[1]} features")
    for ci, cnt in sorted(Counter(y).items()):
        print(f"  {CLASS_NAMES[ci]:12s}: {cnt:6d} ({100*cnt/len(y):5.1f}%)")
    return X, y


# ============================================================
# Temporal smoother
# ============================================================
class TemporalSmoother:
    def __init__(self, alpha=0.35, n_classes=6):
        self.alpha = alpha; self.n_classes = n_classes; self.ema = None
    
    def reset(self): self.ema = None
    
    def smooth(self, probs):
        if self.ema is None: self.ema = probs.copy()
        else: self.ema = self.alpha * probs + (1 - self.alpha) * self.ema
        return self.ema
    
    def smooth_sequence(self, prob_seq):
        self.reset()
        return np.argmax(np.array([self.smooth(p) for p in prob_seq]), axis=1)


# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("  V4 ULTIMATE: FFT + SMOTE + Threshold + Smoothing")
    print("="*60)
    
    extractor = V4FeatureExtractor(window_size=30)
    
    print("\n[1/6] Loading data (v4 features)...")
    X, y = load_all_data(DATA_DIR, extractor, augment=True)
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    nf = X.shape[1]
    print(f"\n  Train: {len(X_train_s)} | Val: {len(X_val_s)} | Features: {nf} (51 v3 + {nf-51} FFT)")
    
    # ── SMOTE ──
    print("\n[2/6] SMOTE balancing...")
    try:
        from imblearn.over_sampling import SMOTE
        before = Counter(y_train)
        print(f"  Before: {dict(sorted(before.items()))}")
        sm = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)
        X_train_s, y_train = sm.fit_resample(X_train_s, y_train)
        after = Counter(y_train)
        print(f"  After:  {dict(sorted(after.items()))}")
    except ImportError:
        print("  [SKIP] imbalanced-learn not installed")
    
    # ── Hyperparameter search ──
    print("\n[3/6] Hyperparameter search (XGBoost, 50 iters)...")
    import xgboost as xgb
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint, uniform
    
    param_dist = {
        'n_estimators': randint(150, 500), 'max_depth': randint(4, 14),
        'learning_rate': uniform(0.01, 0.1), 'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
    }
    
    base = xgb.XGBClassifier(objective='multi:softprob', num_class=6,
                             reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0)
    
    t0 = time.time()
    search = RandomizedSearchCV(base, param_dist, n_iter=50, cv=3,
                                scoring='accuracy', n_jobs=-1, random_state=42, verbose=1)
    search.fit(X_train_s, y_train)
    st = time.time() - t0
    print(f"\n  Search: {st:.1f}s | Best CV: {search.best_score_:.4f}")
    print(f"  Params: {search.best_params_}")
    
    # ── Train ──
    print("\n[4/6] Training final XGBoost...")
    model = xgb.XGBClassifier(**search.best_params_, objective='multi:softprob',
                              num_class=6, reg_alpha=0.1, reg_lambda=1.0,
                              random_state=42, verbosity=0)
    t0 = time.time()
    model.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)
    tt = time.time() - t0
    print(f"  Train: {tt:.1f}s")
    
    # ── Evaluate + Threshold ──
    print("\n[5/6] Evaluate + threshold optimization...")
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix
    )
    
    y_prob = model.predict_proba(X_val_s)
    y_pred = np.argmax(y_prob, axis=1)
    
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    print(f"  Raw: acc={acc:.4f}  f1={f1:.4f}")
    
    cm = confusion_matrix(y_val, y_pred)
    print(f"\n  Confusion Matrix:")
    hdr = f"  {'':>10s}" + "".join(f"{n:>8s}" for n in CLASS_NAMES)
    print(hdr)
    for i, nm in enumerate(CLASS_NAMES):
        print(f"  {nm:>10s}" + "".join(f"{cm[i][j]:8d}" for j in range(6)))
    
    # Per-class thresholds
    print(f"\n  Per-class thresholds:")
    thresholds = {}
    for ci in range(6):
        probs = y_prob[:, ci]; true_bin = (y_val == ci).astype(int)
        best_thr, best_sc = 1.0/6, 0
        for thr in np.linspace(0.05, 0.95, 91):
            pb = (probs >= thr).astype(int)
            tp = ((pb==1)&(true_bin==1)).sum(); fn = ((pb==0)&(true_bin==1)).sum()
            fp = ((pb==1)&(true_bin==0)).sum()
            rec = tp/max(tp+fn,1); prec = tp/max(tp+fp,1)
            sc = 2*rec*prec/max(rec+prec,1e-10)
            if ci == 0: sc += 0.1*rec  # Fall: prioritize recall
            if sc > best_sc: best_sc = sc; best_thr = thr
        thresholds[CLASS_NAMES[ci]] = float(best_thr)
        before_rec = ((y_pred==ci)&(y_val==ci)).sum()/max((y_val==ci).sum(),1)
        after_rec = ((probs>=best_thr)&(y_val==ci)).sum()/max((y_val==ci).sum(),1)
        print(f"    {CLASS_NAMES[ci]:10s}: thr={best_thr:.2f}  recall: {before_rec:.2%} -> {after_rec:.2%}")
    
    # ── Temporal Smoothing ──
    print(f"\n[6/6] Temporal smoothing (alpha=0.35)...")
    smoother = TemporalSmoother(alpha=0.35)
    y_pred_sm = smoother.smooth_sequence(y_prob)
    acc_sm = accuracy_score(y_val, y_pred_sm)
    f1_sm = f1_score(y_val, y_pred_sm, average='weighted')
    
    print(f"  Raw:  acc={acc:.4f}  f1={f1:.4f}")
    print(f"  Smooth: acc={acc_sm:.4f}  f1={f1_sm:.4f}")
    print(f"  Delta:  acc={acc_sm-acc:+.4f}  f1={f1_sm-f1:+.4f}")
    
    cm_sm = confusion_matrix(y_val, y_pred_sm)
    print(f"\n  Per-class (raw -> smooth):")
    for i, nm in enumerate(CLASS_NAMES):
        t_raw = cm[i].sum(); t_sm = cm_sm[i].sum()
        a_raw = 100*cm[i,i]/max(t_raw,1); a_sm = 100*cm_sm[i,i]/max(t_sm,1)
        print(f"    {nm:10s}: {a_raw:5.1f}% -> {a_sm:5.1f}% ({a_sm-a_raw:+.1f}%)")
    
    # ── CV ──
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aX = np.vstack([X_train_s, X_val_s]); ay = np.concatenate([y_train, y_val])
    cva = cross_val_score(model, aX, ay, cv=cv, scoring='accuracy', n_jobs=-1)
    cvf = cross_val_score(model, aX, ay, cv=cv, scoring='f1_weighted', n_jobs=-1)
    print(f"\n  CV accuracy: {cva.mean():.4f} +/- {cva.std():.4f}")
    print(f"  CV f1:       {cvf.mean():.4f} +/- {cvf.std():.4f}")
    
    # ── Save ──
    bp = {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v)
          for k, v in search.best_params_.items()}
    
    bundle = {
        "model": model, "scaler": scaler, "classes": CLASS_NAMES,
        "feature_dim": nf,
        "config": {"window_size": 30, "stride": 5, "best_params": bp,
                   "augmentation": True, "fft_features": True, "smote": True,
                   "thresholds": thresholds, "smooth_alpha": 0.35},
        "metrics": {"accuracy": acc, "f1": f1, "accuracy_smoothed": acc_sm,
                    "f1_smoothed": f1_sm, "train_time_s": tt+st,
                    "n_samples": len(X), "n_classes": 6,
                    "cv_accuracy_mean": float(cva.mean())},
    }
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    with open(MODEL_OUT, "wb") as f: pickle.dump(bundle, f)
    
    report = {
        "classes": CLASS_NAMES, "accuracy": acc, "f1": f1,
        "accuracy_smoothed": acc_sm, "f1_smoothed": f1_sm,
        "n_features": nf, "thresholds": thresholds,
        "per_class_raw": {CLASS_NAMES[i]: float(cm[i,i]/max(cm[i].sum(),1))
                          for i in range(min(6,cm.shape[0]))},
        "per_class_smoothed": {CLASS_NAMES[i]: float(cm_sm[i,i]/max(cm_sm[i].sum(),1))
                               for i in range(min(6,cm_sm.shape[0]))},
        "cv_accuracy": float(cva.mean()), "cv_f1": float(cvf.mean()),
    }
    with open(MODEL_OUT.replace(".pkl", "_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"  V4 COMPLETE!")
    print(f"  Raw:    acc={acc:.2%}  f1={f1:.4f}")
    print(f"  Smooth: acc={acc_sm:.2%}  f1={f1_sm:.4f}")
    print(f"  CV:     acc={cva.mean():.4f} +/- {cva.std():.4f}")
    print(f"  Feats:  {nf} (v3=51 + FFT={nf-51})")
    print(f"  Model:  {MODEL_OUT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
