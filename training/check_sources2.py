"""用 Python os.scandir 绕过路径编码问题"""
import os, cv2, numpy as np

root = r"F:\动作数据集"

# 找到数据集第1批
batch1 = None
for entry in os.scandir(root):
    if entry.is_dir() and "1" in entry.name:
        batch1 = entry.path
        break

print(f"数据集第1批: {batch1}")

# 进入 Subject.1/Walk
subj1 = os.path.join(batch1, "Subject.1", "Walk")
print(f"Walk dir exists: {os.path.isdir(subj1)}")

# 用 os.scandir 找文件
png_files = []
for entry in os.scandir(subj1):
    if entry.name.endswith('.png'):
        png_files.append(entry.path)
png_files.sort()
print(f"PNG count: {len(png_files)}")

if png_files:
    # 用 numpy fromfile 读
    with open(png_files[0], 'rb') as f:
        data = f.read()
    print(f"File size: {len(data)} bytes")
    
    # numpy imdecode
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        print(f"Shape: {img.shape}, dtype: {img.dtype}")
        print(f"Range: [{img.min()}, {img.max()}]")
        if len(img.shape) == 3:
            cm = [img[:,:,i].mean() for i in range(3)]
            print(f"Channel means: {cm}")
            if max([abs(cm[i]-cm[j]) for i,j in [(0,1),(0,2),(1,2)]]) < 2:
                print("→ Grayscale (likely depth map or IR)")
            else:
                print("→ RGB image")
    else:
        print("cv2.imdecode FAILED")

# 走路视频
other = os.path.join(root, "其他")
walk_vids = []
for entry in os.scandir(other):
    if "走路" in entry.name and entry.name.endswith('.mp4'):
        walk_vids.append(entry.path)
print(f"\n走路视频: {len(walk_vids)}")
for v in sorted(walk_vids):
    cap = cv2.VideoCapture(v)
    if cap.isOpened():
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  {os.path.basename(v)}: {frames}f, {fps:.1f}fps, {frames/fps:.0f}s")
    cap.release()
