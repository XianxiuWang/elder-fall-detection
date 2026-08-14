"""增强稀有类别数据"""
import os, shutil, numpy as np

MERGED = r"E:\老人跌倒\data\custom_6class_merged"

def augment_kpts(kpts, n_aug=10):
    """Generate augmented versions of a keypoint sequence"""
    versions = []
    T = kpts.shape[0]
    
    for _ in range(n_aug):
        v = kpts.copy()
        
        # 1. 水平翻转 (50% chance)
        if np.random.random() > 0.5:
            v[:, :, 0] = 1.0 - v[:, :, 0]  # mirror x
        
        # 2. 时间重采样 (speed variation: 0.85~1.15x)
        scale = np.random.uniform(0.85, 1.15)
        if scale < 0.95 or scale > 1.05:
            new_T = max(10, int(T * scale))
            idx = np.linspace(0, T-1, new_T).astype(int)
            idx = np.clip(idx, 0, T-1)
            v = v[idx]
        
        # 3. 高斯噪声 (std=0.005)
        noise = np.random.normal(0, 0.005, v[:, :, :2].shape).astype(np.float32)
        v[:, :, :2] += noise
        
        # 4. 随机旋转 (小角度，±3度)
        angle = np.random.uniform(-3, 3) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        xy = v[:, :, :2].reshape(-1, 2)
        rotated = np.zeros_like(xy)
        rotated[:, 0] = xy[:, 0] * cos_a - xy[:, 1] * sin_a
        rotated[:, 1] = xy[:, 0] * sin_a + xy[:, 1] * cos_a
        v[:, :, :2] = rotated.reshape(v.shape[0], v.shape[1], 2)
        
        versions.append(v)
    
    return versions

# Find underrepresented classes
from collections import Counter
cats = Counter()
rare_files = {2: [], 4: [], 5: []}  # StandUp, WakeUp, Standing
for f in os.listdir(MERGED):
    if f.endswith('.npz'):
        data = np.load(os.path.join(MERGED, f), allow_pickle=True)
        cat = int(data['category'])
        cats[cat] += 1
        if cat in rare_files:
            rare_files[cat].append(f)

print("Before augmentation:")
names = ['Fall', 'SitDown', 'StandUp', 'Walking', 'WakeUp', 'Standing']
for c in sorted(cats.keys()):
    print(f"  C{c} {names[c]:10s}: {cats[c]:3d} files")

# Augment rare classes
target_min = 20  # minimum files per class
total_added = 0
for cat in [2, 4, 5]:
    current = cats[cat]
    needed = target_min - current
    if needed <= 0:
        continue
    
    n_per_file = max(1, needed // len(rare_files[cat])) + 1
    print(f"\n  Augmenting C{cat} ({names[cat]}): {current} -> target {target_min}")
    
    added = 0
    for f in rare_files[cat]:
        if added >= needed:
            break
        src = os.path.join(MERGED, f)
        data = np.load(src, allow_pickle=True)
        kpts = data['keypoints']
        
        aug_versions = augment_kpts(kpts, min(n_per_file, needed - added))
        for i, v in enumerate(aug_versions):
            dst = os.path.join(MERGED, f.replace('.npz', f'_aug{i:02d}.npz'))
            np.savez_compressed(dst, keypoints=v.astype(np.float32), 
                              category=np.array(cat, dtype=np.int64),
                              source=np.array('augmented'))
            added += 1
            total_added += 1
    
    cats[cat] += added
    print(f"    Added {added} augmented samples")

print(f"\nAfter augmentation: ({total_added} total added)")
for c in sorted(cats.keys()):
    print(f"  C{c} {names[c]:10s}: {cats[c]:3d} files")
