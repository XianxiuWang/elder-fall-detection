"""检查数据源可用性"""
import os, cv2, glob, numpy as np

root = r"F:\动作数据集"
batch1 = None
for d in os.listdir(root):
    if "1" in d and os.path.isdir(os.path.join(root, d)):
        batch1 = os.path.join(root, d)
        break

print(f"数据集第1批: {batch1}")

# 1. 检查 PNG 图片类型
walk_dir = os.path.join(batch1, "Subject.1", "Walk")
pngs = sorted(glob.glob(os.path.join(walk_dir, "*.png")))
if pngs:
    img = cv2.imread(pngs[0])
    print(f"\nPNG sample: {pngs[0]}")
    print(f"  Shape: {img.shape}, dtype: {img.dtype}")
    print(f"  Value range: [{img.min()}, {img.max()}]")
    if len(img.shape) == 3:
        print(f"  Channel means: B={img[:,:,0].mean():.1f} G={img[:,:,1].mean():.1f} R={img[:,:,2].mean():.1f}")
        # 检查是不是三通道都一样的假RGB
        diffs = [np.abs(img[:,:,i].astype(float) - img[:,:,j].astype(float)).max() 
                 for i,j in [(0,1),(0,2),(1,2)]]
        print(f"  Max channel diffs: {diffs}")
        if max(diffs) < 5:
            print("  ⚠ Appears to be grayscale in 3-channel (depth map)")
    
    # 试 MediaPipe
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_landmarks:
            print(f"  ✅ MediaPipe works! Detected pose.")
        else:
            print(f"  ❌ MediaPipe couldn't detect pose")
        pose.close()
    except Exception as e:
        print(f"  ❌ MediaPipe error: {e}")
    
    # 统计 9 Subject Walk 帧数
    print(f"\nWalk frames per subject:")
    total = 0
    for i in range(1, 10):
        sd = os.path.join(batch1, f"Subject.{i}", "Walk")
        if os.path.isdir(sd):
            n = len(glob.glob(os.path.join(sd, "*.png")))
            total += n
            print(f"  Subject.{i}: {n}")
    print(f"  TOTAL: {total}")

# 2. 其他文件夹里的走路视频
other = os.path.join(root, "其他")
if os.path.isdir(other):
    walk_vids = [f for f in os.listdir(other) if "走路" in f and f.endswith(".mp4")]
    print(f"\n走路视频 in 其他: {walk_vids}")
    for v in walk_vids:
        cap = cv2.VideoCapture(os.path.join(other, v))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  {v}: {frames} frames, {fps:.1f} fps, {frames/fps:.0f}s")
        cap.release()
