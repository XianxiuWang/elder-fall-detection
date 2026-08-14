"""
merge_and_train.py — 合并 subject_features + existing data → 训练六分类模型
=====================================================================
1. 从 subject_features 复制数据 (类别: Fall=0, SitDown=1, Walk=3)
2. 从 custom_6class 补充缺失类别 (StandUp=2, WakeUp=4, Standing=5)
3. 运行 train_6class_v4_final.py 训练
"""
import os, shutil, sys, numpy as np
from collections import Counter

SUBJECT_DIR = r"E:\老人跌倒\data\subject_features"
EXISTING_DIR = r"E:\老人跌倒\data\custom_6class"
MERGED_DIR = r"E:\老人跌倒\data\custom_6class_merged"

os.makedirs(MERGED_DIR, exist_ok=True)

# --- Step 1: Copy subject_features files ---
print("[1] Copying subject_features...")
subject_files = sorted([f for f in os.listdir(SUBJECT_DIR) if f.endswith('.npz')])
copied = 0
for f in subject_files:
    src = os.path.join(SUBJECT_DIR, f)
    # Convert filename to clean prefix
    dst = os.path.join(MERGED_DIR, f"SUBJ_{f}")  # prefix with SUBJ_
    shutil.copy2(src, dst)
    copied += 1
    if copied % 20 == 0:
        print(f"  {copied}/{len(subject_files)}")
print(f"  Copied {copied} subject files")

# --- Step 2: Supplement missing classes from existing data ---
print("\n[2] Checking coverage...")
cats_present = set()
for f in os.listdir(MERGED_DIR):
    if f.endswith('.npz'):
        data = np.load(os.path.join(MERGED_DIR, f), allow_pickle=True)
        cats_present.add(int(data['category']))
print(f"  Present categories: {sorted(cats_present)}")

MISSING_CLASSES = {2: "StandUp", 4: "WakeUp", 5: "Standing"}
supplemented = 0
for cat in MISSING_CLASSES:
    if cat not in cats_present:
        added = 0
        for f in sorted(os.listdir(EXISTING_DIR)):
            if not f.endswith('.npz'): 
                continue
            data = np.load(os.path.join(EXISTING_DIR, f), allow_pickle=True)
            if int(data['category']) == cat:
                dst = os.path.join(MERGED_DIR, f"EXIST_{f}")
                shutil.copy2(os.path.join(EXISTING_DIR, f), dst)
                added += 1
        if added > 0:
            print(f"  Added {added} files for C{cat} ({MISSING_CLASSES[cat]})")
            supplemented += added
        else:
            print(f"  [WARNING] No files found for C{cat} ({MISSING_CLASSES[cat]})")

# --- Step 3: Final stats ---
print("\n[3] Merged dataset stats:")
total_files = 0
cat_counter = Counter()
total_frames = 0
for f in os.listdir(MERGED_DIR):
    if f.endswith('.npz'):
        data = np.load(os.path.join(MERGED_DIR, f), allow_pickle=True)
        cat = int(data['category'])
        nf = data['keypoints'].shape[0]
        cat_counter[cat] += 1
        total_files += 1
        total_frames += nf

CLASS_NAMES = ["Fall", "SitDown", "StandUp", "Walking", "WakeUp", "Standing"]
print(f"  Total: {total_files} files, {total_frames} frames")
for ci in sorted(cat_counter.keys()):
    name = CLASS_NAMES[ci] if ci < len(CLASS_NAMES) else f"C{ci}"
    print(f"  C{ci} {name:10s}: {cat_counter[ci]:4d} files")

# --- Step 4: Update DATA_DIR and run training ---
print("\n[4] Running training...")
print("=" * 60)

# Monkey-patch the DATA_DIR in train_6class_v4_final
train_script_path = r"E:\老人跌倒\training\train_6class_v4_final.py"

sys.path.insert(0, os.path.dirname(train_script_path))

# Read the training script and modify DATA_DIR
with open(train_script_path, 'r', encoding='utf-8') as f:
    code = f.read()

# Backup original
with open(train_script_path + '.bak', 'w', encoding='utf-8') as f:
    f.write(code)

# Replace DATA_DIR
code = code.replace(
    'DATA_DIR = r"E:\\老人跌倒\\data\\custom_6class"',
    f'DATA_DIR = r"{MERGED_DIR}"'
)

with open(train_script_path, 'w', encoding='utf-8') as f:
    f.write(code)

print(f"  Patched DATA_DIR -> {MERGED_DIR}")
print(f"  Backup saved to {train_script_path}.bak")

# Run training
exec(open(train_script_path, encoding='utf-8').read())

# Restore original
shutil.copy2(train_script_path + '.bak', train_script_path)
os.remove(train_script_path + '.bak')
print("\n[DONE] Restored original training script.")
