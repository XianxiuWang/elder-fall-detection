#!/usr/bin/env python3
"""验证 deploy_opi5/fall_inference.py 的提取器与 v8 训练提取器逐维对齐 + 模型可预测"""
import sys, os, glob, pickle
import numpy as np

sys.path.insert(0, r"E:\老人跌倒\deploy_opi5")
sys.path.insert(0, r"E:\老人跌倒\training")

from fall_inference import EnhancedFeatureExtractor as ProdExtractor
from train_6class_v8 import EnhancedFeatureExtractor as TrainExtractor

# 1. 找几个真实样本 (30+ 帧)
samples = []
for d in [r"E:\老人跌倒\data\custom_sitdown", r"E:\老人跌倒\data\subject_features",
          r"E:\老人跌倒\data\custom_standup_wakeup"]:
    for f in glob.glob(os.path.join(d, "*.npz"))[:5]:
        data = np.load(f, allow_pickle=True)
        for k in ['keypoints', 'landmarks']:
            if k in data and data[k].ndim == 3 and data[k].shape[0] >= 30:
                samples.append((f, k, data[k][:30, :, :3].astype(np.float32)))
                break
        if len(samples) >= 9:
            break
    if len(samples) >= 9:
        break

print(f"测试样本数: {len(samples)}")
prod = ProdExtractor(window_size=30)
train = TrainExtractor(window_size=30)

max_diff = 0.0
for fname, k, win in samples:
    pv = prod.extract_window(win)
    tv, _ = train.extract_window(win)
    d = np.abs(pv - tv).max()
    max_diff = max(max_diff, float(d))
    assert pv.shape == (51,), f"prod dim {pv.shape}"
    assert tv.shape == (51,), f"train dim {tv.shape}"
    print(f"  {os.path.basename(fname)}: prod={pv.shape} train={tv.shape} maxdiff={d:.2e}")

print(f"\n最大逐维差异: {max_diff:.2e} (应 < 1e-4)")
assert max_diff < 1e-4, "特征不一致！"

# 2. 用 v8 模型 + 生产提取器预测
model_path = r"E:\老人跌倒\deploy_opi5\models\fall_classifier_6class_v8.pkl"
bundle = pickle.load(open(model_path, "rb"))
model, scaler = bundle["model"], bundle["scaler"]
print(f"\n模型 feature_dim = {bundle['feature_dim']} | version = {bundle['config'].get('version')}")

classes = ["Fall", "SitDown", "StandUp", "Walking", "WakeUp", "Standing"]
for fname, k, win in samples[:6]:
    vec = prod.extract_window(win)
    vec_s = scaler.transform(vec.reshape(1, -1))
    probs = model.predict_proba(vec_s)[0]
    top = int(np.argmax(probs))
    print(f"  {os.path.basename(fname)}: pred={classes[top]} (p={probs[top]:.3f})")
    assert vec_s.shape == (1, 51)

print("\n✅ 生产提取器与 v8 训练提取器完全对齐，模型可正常推理")
