"""检查数据源 - 无unicode特殊字符版本"""
import os, sys, cv2, numpy as np

# 强制UTF-8输出
sys.stdout.reconfigure(encoding='utf-8')

root = r"F:\动作数据集"

# 找到两个子目录
dirs = []
for entry in os.scandir(root):
    if entry.is_dir():
        dirs.append((entry.name, entry.path))

batch1_path = dirs[0][1]
other_path = dirs[1][1]

print(f"Batch1: {batch1_path}")
print(f"Other:  {other_path}")

# 统计所有Subject Walk帧数
print(f"\n=== Walking frames per subject ===")
total_walk = 0
for i in range(1, 10):
    walk_dir = os.path.join(batch1_path, f"Subject.{i}", "Walk")
    if os.path.isdir(walk_dir):
        n = 0
        for f in os.scandir(walk_dir):
            if f.name.endswith('.png'):
                n += 1
        total_walk += n
        print(f"  Subject.{i}/Walk: {n} frames")
print(f"  TOTAL WALK: {total_walk}")

# 测试PNG是否能跑MediaPipe
print(f"\n=== Testing MediaPipe on sample PNG ===")
walk_dir = os.path.join(batch1_path, "Subject.1", "Walk")
pngs = sorted([f.path for f in os.scandir(walk_dir) if f.name.endswith('.png')])
with open(pngs[0], 'rb') as f:
    data = f.read()
img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

import mediapipe as mp
pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = pose.process(rgb)
if result.pose_landmarks:
    visible = sum(1 for lm in result.pose_landmarks.landmark if lm.visibility > 0.5)
    print(f"  OK! {visible}/33 keypoints visible")
else:
    print(f"  No pose detected in sample")
pose.close()

# 统计其他文件夹的走路相关视频
print(f"\n=== Walking videos in Other ===")
walking_vids = []
for entry in os.scandir(other_path):
    name = entry.name
    if name.endswith('.mp4') and ('Walk' in name or 'walk' in name):
        cap = cv2.VideoCapture(entry.path)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  {name}: {frames}f, {fps:.1f}fps")
        walking_vids.append((entry.path, frames, fps))
        cap.release()

# 读取Walk推理结果中的假阳窗口
print(f"\n=== Walk hard negatives from inference results ===")
import json
result_path = r"D:\Users\wangxianxiu\clawd\results_100 Ways to Walk.json"
if os.path.exists(result_path):
    with open(result_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    fp_windows = [r for r in results['predictions'] if r['filt'] == 'Fall']
    print(f"  Total predictions: {len(results['predictions'])}")
    print(f"  False positive Fall windows: {len(fp_windows)}")
    print(f"  FP time ranges: {fp_windows[0]['time_s']:.1f}s - {fp_windows[-1]['time_s']:.1f}s")
else:
    print(f"  Result file not found: {result_path}")
