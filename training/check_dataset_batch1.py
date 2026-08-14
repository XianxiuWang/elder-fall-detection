"""检查数据集第1批的图片格式"""
import os, cv2, glob

base = r"F:\动作数据集\数据集第1批"

# 找一个 Walk PNG 看看
walk_dir = os.path.join(base, "Subject.1", "Walk")
pngs = sorted(glob.glob(os.path.join(walk_dir, "*.png")))
print(f"Walk PNGs: {len(pngs)}")
print(f"First: {os.path.basename(pngs[0])}")
print(f"Last:  {os.path.basename(pngs[-1])}")

img = cv2.imread(pngs[0])
if img is not None:
    print(f"\nImage shape: {img.shape}")
    print(f"dtype: {img.dtype}")
    print(f"Mean: {img.mean():.1f}")
    print(f"Min: {img.min()}, Max: {img.max()}")
    # 判断是否是深度图
    if len(img.shape) == 2:
        print("Type: Grayscale (possible depth map)")
    elif img.shape[2] == 3:
        # 检查三个通道是否相同
        b_eq_g = (img[:,:,0] == img[:,:,1]).all()
        g_eq_r = (img[:,:,1] == img[:,:,2]).all()
        if b_eq_g and g_eq_r:
            print("Type: 3-channel grayscale (possible depth stored as RGB)")
        else:
            print("Type: RGB color image")
else:
    print("ERROR: Cannot read image")

# 统计每个 Subject 各类别的帧数
print("\n--- All Subjects ---")
for i in range(1, 10):
    subj_dir = os.path.join(base, f"Subject.{i}")
    if not os.path.isdir(subj_dir):
        continue
    cats = [d for d in os.listdir(subj_dir) if os.path.isdir(os.path.join(subj_dir, d))]
    cat_counts = {}
    for cat in cats:
        pngs_c = glob.glob(os.path.join(subj_dir, cat, "*.png"))
        cat_counts[cat] = len(pngs_c)
    total = sum(cat_counts.values())
    print(f"Subject.{i}: {total} frames — {' | '.join(f'{k}:{v}' for k,v in sorted(cat_counts.items()))}")
