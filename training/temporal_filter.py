#!/usr/bin/env python3
"""
temporal_filter.py — 时序后处理过滤器 + 评估
==============================================
在验证集上对比: 原始模型 vs 时序过滤后

用法:
  python temporal_filter.py
"""
import os, sys, pickle, time
import numpy as np
from collections import Counter, deque

sys.path.insert(0, r"E:\老人跌倒\training")
from train_fall_classifier import FeatureExtractor

CLASS_NAMES = ["Fall", "SitDown", "StandUp", "Walking", "WakeUp", "Standing"]
FALL_ID, STANDING_ID = 0, 5
MODEL_PATH = r"E:\老人跌倒\models\fall_classifier_6class.pkl"


# ============================================================
# 51-feature extractor (original V5, NO tilt features)
# ============================================================
class Extractor51(FeatureExtractor):
    def __init__(self, window_size=30):
        super().__init__(window_size=window_size)
    
    def extract_window(self, window):
        base_vec, base_names = super().extract_window(window)
        extra = []
        extra_names = []
        
        torso_y = window[:, 11, 1]
        n = len(torso_y)
        extra.extend([torso_y[:n//3].mean(), torso_y[n//3:2*n//3].mean(), torso_y[2*n//3:].mean()])
        extra_names += ['torso_y_early', 'torso_y_mid', 'torso_y_late']
        
        x = np.arange(n)
        extra.append(np.polyfit(x, torso_y, 1)[0])
        extra_names.append('torso_y_slope')
        
        head_y = window[:, 0, 1]
        extra.append(np.polyfit(x, head_y, 1)[0])
        extra_names.append('head_y_slope')
        
        speeds = np.linalg.norm(np.diff(window[:, :, :2], axis=0), axis=2)
        extra.append(speeds.std(axis=0).mean())
        extra_names.append('joint_speed_std')
        
        for idx in [0, 11, 23]:
            extra.append(window[-1, idx, 1] - window[0, idx, 1])
        extra_names += ['head_y_delta', 'shoulder_y_delta', 'hip_y_delta']
        
        return np.concatenate([base_vec, np.array(extra)]), base_names + extra_names


# ============================================================
# Data loading (same as V5, but with augment=False)
# ============================================================
DATA_ROOTS = {
    'subject_features': r"E:\老人跌倒\data\subject_features",
    'custom_6class':    r"E:\老人跌倒\data\custom_6class",
    'custom_5class':    r"E:\老人跌倒\data\custom_5class",
    'urfd_features':    r"E:\老人跌倒\data\urfd_features",
    'le2i_features':    r"E:\老人跌倒\data\le2i_features",
    'upfall_features':  r"E:\老人跌倒\data\upfall_features",
}

def load_for_eval(extractor, window_size=30, stride=6):
    X_all, y_all = [], []
    for source, d in DATA_ROOTS.items():
        if not os.path.exists(d): continue
        for f in sorted(os.listdir(d)):
            if not f.endswith('.npz'): continue
            try:
                data = np.load(os.path.join(d, f), allow_pickle=True)
                kp_key = 'keypoints' if 'keypoints' in data else 'landmarks'
                if kp_key not in data: continue
                kpts = data[kp_key]
                if kpts.ndim != 3 or kpts.shape[0] < window_size: continue
                
                cat = None
                if 'category' in data:
                    cat = int(data['category'].item())
                elif source == 'urfd_features' and 'label' in data:
                    if int(data['label'].item()) == 1: cat = 0
                    else: continue
                elif source == 'le2i_features' and 'frame_fall_map' in data:
                    if np.any(data['frame_fall_map'] > 0): cat = 0
                    else: continue
                if cat is None or cat not in range(6): continue
                
                kpts_3d = kpts[:, :, :3].astype(np.float32)
                for start in range(0, kpts_3d.shape[0] - window_size + 1, stride):
                    w = kpts_3d[start:start + window_size]
                    vec, _ = extractor.extract_window(w)
                    X_all.append(vec)
                    y_all.append(cat)
            except: pass
    
    X = np.nan_to_num(np.array(X_all, dtype=np.float32), nan=0.0)
    y = np.array(y_all, dtype=np.int32)
    return X, y


# ============================================================
# Temporal Filter
# ============================================================
class TemporalFilter:
    def __init__(self, vote_window=7, fall_hold=15, min_duration=9,
                 fall_prob_thresh=0.35, standing_to_fall_confirm=3):
        self.vote_window = vote_window
        self.fall_hold = fall_hold
        self.min_duration = min_duration
        self.fall_prob_thresh = fall_prob_thresh
        self.st2fall_confirm = standing_to_fall_confirm
        self.prob_hist = deque(maxlen=vote_window * 2)
        self.pred_hist = deque(maxlen=vote_window * 2)
        self.fall_ctr = 0
        self.cur_label = -1
        self.label_dur = 0
        self.switch_pending = 0
    
    def update(self, probs):
        raw = int(np.argmax(probs))
        self.pred_hist.append(raw)
        self.prob_hist.append(probs.copy())
        
        vn = min(self.vote_window, len(self.pred_hist))
        recent = list(self.pred_hist)[-vn:]
        cnt = Counter(recent)
        top, top_n = cnt.most_common(1)[0]
        
        if self.cur_label == STANDING_ID and top == FALL_ID:
            if top_n < vn - self.st2fall_confirm + 1:
                top = STANDING_ID
        
        if top == FALL_ID:
            self.fall_ctr += 1
        else:
            self.fall_ctr = max(0, self.fall_ctr - 1)
        
        if 0 < self.fall_ctr <= self.fall_hold and probs[FALL_ID] >= self.fall_prob_thresh:
            final = FALL_ID
        else:
            final = top
        
        if final != self.cur_label and self.cur_label >= 0:
            self.switch_pending += 1
            if self.switch_pending < self.min_duration:
                final = self.cur_label
            else:
                self.cur_label = final
                self.label_dur = 0
                self.switch_pending = 0
        else:
            if self.cur_label != final:
                self.label_dur = 0
                self.cur_label = final if self.cur_label < 0 else self.cur_label
            else:
                self.label_dur += 1
            self.switch_pending = 0
        
        conf = float(np.mean([p[self.cur_label] for p in list(self.prob_hist)[-vn:]]))
        return self.cur_label, conf
    
    def reset(self):
        self.prob_hist.clear()
        self.pred_hist.clear()
        self.fall_ctr = 0
        self.cur_label = -1
        self.label_dur = 0
        self.switch_pending = 0
    
    def process_stream(self, prob_seq):
        self.reset()
        labs, confs = [], []
        for p in prob_seq:
            l, c = self.update(p)
            labs.append(l)
            confs.append(c)
        return np.array(labs), np.array(confs)


# ============================================================
# Main
# ============================================================
def main():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    
    print("=" * 60)
    print("  Temporal Filter Evaluation")
    print("=" * 60)
    
    # Load model
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model, scaler = bundle["model"], bundle["scaler"]
    print(f"  Model: {MODEL_PATH}")
    print(f"  Feature dim: {bundle['feature_dim']}")
    
    # Load data
    print("\n[1/3] Loading data (no augmentation)...")
    extractor = Extractor51(window_size=30)
    X, y = load_for_eval(extractor)
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val_s = scaler.transform(X_val)
    print(f"  Val set: {len(X_val)} samples, {X_val.shape[1]} features")
    
    for cls_id, cnt in sorted(Counter(y_val).items()):
        print(f"    {CLASS_NAMES[cls_id]:12s}: {cnt:5d}")
    
    # Predict
    print("\n[2/3] Predicting...")
    t0 = time.time()
    proba = model.predict_proba(X_val_s)
    y_raw = np.argmax(proba, axis=1)
    pred_time = time.time() - t0
    print(f"  Inference: {pred_time:.3f}s ({len(X_val)/pred_time:.0f} samples/s)")
    
    # Filter
    print("\n[3/3] Applying temporal filter...")
    
    configs = [
        ("No filter", None),
        ("vote=7, hold=15 (默认)", dict(vote_window=7, fall_hold=15)),
        ("vote=9, hold=20 (保守)", dict(vote_window=9, fall_hold=20)),
        ("vote=5, hold=10 (激进)", dict(vote_window=5, fall_hold=10)),
    ]
    
    results = []
    for name, cfg in configs:
        if cfg:
            filt = TemporalFilter(**cfg)
            y_filt, _ = filt.process_stream(proba)
        else:
            y_filt = y_raw
        
        acc = accuracy_score(y_val, y_filt)
        cm = confusion_matrix(y_val, y_filt)
        
        # Fall metrics
        fall_tp = cm[FALL_ID, FALL_ID]
        fall_fn_standing = cm[FALL_ID, STANDING_ID]
        fall_fn_total = sum(cm[FALL_ID, j] for j in range(6) if j != FALL_ID)
        
        # Standing FP → Fall
        standing_fp_fall = cm[STANDING_ID, FALL_ID]
        
        results.append({
            'name': name, 'acc': acc, 'cm': cm,
            'fall_tp': fall_tp, 'fall_fn': fall_fn_total,
            'fall_fn_standing': fall_fn_standing,
            'standing_fp': standing_fp_fall,
        })
    
    # Print comparison
    print(f"\n{'Config':<30s} {'Acc':>7s} {'Fall FN→Stand':>15s} {'Fall FN':>10s} {'Stand FP→Fall':>15s}")
    print("-" * 77)
    for r in results:
        print(f"{r['name']:<30s} {r['acc']:7.4f} {r['fall_fn_standing']:15d} {r['fall_fn']:10d} {r['standing_fp']:15d}")
    
    best = results[1]  # Default config
    print(f"\n  Full report (filtered, default):")
    print(classification_report(y_val, best['cm'].argmax(axis=1) if best['name'] == 'No filter' else None, 
                                target_names=CLASS_NAMES, zero_division=0))
    
    # Show confusion matrix for best filtered
    cm = results[1]['cm']
    print(f"\n  Confusion Matrix (filtered):")
    hdr = f"  {'':>12s}" + "".join(f"{n:>8s}" for n in CLASS_NAMES)
    print(hdr)
    for i, name in enumerate(CLASS_NAMES):
        row = f"  {name:>12s}" + "".join(f"{cm[i][j]:8d}" for j in range(6))
        print(row)
    
    print(f"\n{'='*60}")
    print("  DONE")


if __name__ == "__main__":
    main()
