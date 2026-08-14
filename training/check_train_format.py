"""检查训练数据格式"""
import os, sys, numpy as np, glob

sys.stdout.reconfigure(encoding='utf-8')

data_dir = r"E:\老人跌倒\data\custom_6class"
if os.path.isdir(data_dir):
    npzs = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    print(f"Total .npz files: {len(npzs)}")
    for n in npzs[:5]:
        d = np.load(n, allow_pickle=True)
        print(f"\n{os.path.basename(n)}:")
        for k in d.keys():
            val = d[k]
            if isinstance(val, np.ndarray):
                print(f"  {k}: shape={val.shape}, dtype={val.dtype}")
            else:
                print(f"  {k}: {val} ({type(val).__name__})")
        d.close()

# 统计各类别
from collections import Counter
cats = Counter()
total_frames = 0
for n in npzs:
    d = np.load(n, allow_pickle=True)
    # Try different key names
    cat = None
    for key in ['category', 'cat', 'label', 'class']:
        if key in d:
            cat = int(d[key]) if isinstance(d[key], np.ndarray) else d[key]
            break
    # Also check if category is in array
    if cat is None and 'arr_0' in d:
        # try to find it
        pass
    
    if cat is not None:
        cats[cat] += 1
        # count frames
        for key in ['landmarks', 'keypoints', 'kpts', 'data']:
            if key in d:
                total_frames += d[key].shape[0]
                break
    d.close()

print(f"\nCategories: {dict(cats)}")
print(f"Total frames: {total_frames}")
