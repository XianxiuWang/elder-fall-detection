"""更深入检查数据和推理结果"""
import os, sys, cv2, numpy as np, json, glob

sys.stdout.reconfigure(encoding='utf-8')

root = r"F:\动作数据集"

# 找到目录
dirs = {}
for entry in os.scandir(root):
    if entry.is_dir():
        dirs[entry.name] = entry.path

print("Directories found:")
for k, v in dirs.items():
    print(f"  {k} -> {v}")

batch1 = dirs[list(dirs.keys())[0]]
walk_dir_name = list(dirs.keys())[1]

# 看看行走文件夹里有什么
print(f"\n=== Contents of [{walk_dir_name}] ===")
walk_dir = dirs[walk_dir_name]
for entry in sorted(os.scandir(walk_dir), key=lambda e: e.name):
    if entry.is_file():
        size_mb = entry.stat().st_size / (1024*1024)
        print(f"  {entry.name} ({size_mb:.1f} MB)")
    else:
        print(f"  [{entry.name}]/")

# 看看PNG到底是什么图 - 保存一张到临时目录看
print(f"\n=== Checking PNG sample more ===")
subj1_walk = os.path.join(batch1, "Subject.1", "Walk")
pngs = sorted([f.path for f in os.scandir(subj1_walk) if f.name.endswith('.png')])

with open(pngs[0], 'rb') as f:
    img_data = f.read()
img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
print(f"  Shape: {img.shape}")
print(f"  Channel 0 (B): mean={img[:,:,0].mean():.1f}, std={img[:,:,0].std():.1f}")
print(f"  Channel 1 (G): mean={img[:,:,1].mean():.1f}, std={img[:,:,1].std():.1f}")
print(f"  Channel 2 (R): mean={img[:,:,2].mean():.1f}, std={img[:,:,2].std():.1f}")

# 保存一张看看
cv2.imwrite(r"E:\老人跌倒\training\sample_walk_frame.png", img)
print(f"  Saved sample to E:\\老人跌倒\\training\\sample_walk_frame.png")

# 找推理结果JSON
print(f"\n=== Looking for inference results ===")
results_dir = r"D:\Users\wangxianxiu\clawd"
if os.path.isdir(results_dir):
    jsons = glob.glob(os.path.join(results_dir, "results_*.json"))
    for j in jsons:
        print(f"  {os.path.basename(j)} ({os.path.getsize(j)} bytes)")

# 检查训练数据目录格式
print(f"\n=== Training data format ===")
data_dir = r"E:\老人跌倒\data\custom_6class"
if os.path.isdir(data_dir):
    npzs = glob.glob(os.path.join(data_dir, "*.npz"))
    for n in npzs[:3]:
        d = np.load(n)
        print(f"  {os.path.basename(n)}: keys={list(d.keys())}, "
              f"landmarks shape={d['landmarks'].shape}, cat={d.get('category', '?')}")
        d.close()
