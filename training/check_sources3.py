"""测试PNG能否跑MediaPipe + 找其他文件夹"""
import os, cv2, numpy as np

root = r"F:\动作数据集"

# 找到两个子目录
dirs = []
for entry in os.scandir(root):
    if entry.is_dir():
        dirs.append((entry.name, entry.path))

print("Root dirs:")
for name, path in dirs:
    print(f"  [{name}] -> {path}")
    # 数里面的文件
    try:
        items = list(os.scandir(path))
        print(f"    {len(items)} items, first 3: {[i.name for i in items[:3]]}")
    except:
        print(f"    (cannot list)")

# 测试第一个PNG能否跑MediaPipe
batch1_path = dirs[0][1]
walk_dir = os.path.join(batch1_path, "Subject.1", "Walk")
pngs = []
for entry in os.scandir(walk_dir):
    if entry.name.endswith('.png'):
        pngs.append(entry.path)
pngs.sort()

# 测试 MediaPipe
with open(pngs[0], 'rb') as f:
    data = f.read()
arr = np.frombuffer(data, np.uint8)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

try:
    import mediapipe as mp
    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if result.pose_landmarks:
        kpts = [(lm.x, lm.y, lm.visibility) for lm in result.pose_landmarks.landmark]
        visible = sum(1 for _, _, v in kpts if v > 0.5)
        print(f"\n✅ MediaPipe works! {visible}/33 keypoints visible")
        print(f"   Nose: ({kpts[0][0]:.3f}, {kpts[0][1]:.3f})")
    else:
        print(f"\n⚠️ No pose detected in sample frame")
    pose.close()
except Exception as e:
    print(f"\n❌ MediaPipe error: {e}")

# 统计所有Subject的各类别帧数
print(f"\n--- All 9 Subjects frame counts ---")
grand_total = 0
for i in range(1, 10):
    subj_dir = os.path.join(batch1_path, f"Subject.{i}")
    if not os.path.isdir(subj_dir):
        continue
    cats = {}
    for entry in os.scandir(subj_dir):
        if entry.is_dir():
            n = 0
            for f in os.scandir(entry.path):
                if f.name.endswith('.png'):
                    n += 1
            cats[entry.name] = n
    total = sum(cats.values())
    grand_total += total
    items = [f"{k}:{v}" for k,v in sorted(cats.items())]
    print(f"  Subject.{i}: {total} total — " + " | ".join(items))
print(f"  GRAND TOTAL: {grand_total}")
