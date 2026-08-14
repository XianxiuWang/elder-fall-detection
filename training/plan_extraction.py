"""检查已处理的视频和尚未处理的视频"""
import os, sys, numpy as np, glob

sys.stdout.reconfigure(encoding='utf-8')

# 1. 查看训练数据中已有的 Walking 来源
data_dir = r"E:\老人跌倒\data\custom_6class"
npzs = glob.glob(os.path.join(data_dir, "*.npz"))
walk_sources = set()
for n in npzs:
    d = np.load(n, allow_pickle=True)
    if int(d['category']) == 3:  # Walking
        src = str(d['source'])
        walk_sources.add(src)
    d.close()

print(f"Existing Walking sources ({len(walk_sources)}):")
for s in sorted(walk_sources):
    print(f"  {s}")

# 2. 查看行走文件夹的视频
walk_dir = r"F:\动作数据集\行走"
videos = []
for entry in os.scandir(walk_dir):
    if entry.name.endswith('.mp4'):
        videos.append(entry.name)

print(f"\nWalking videos in folder ({len(videos)}):")

# 判断哪些已经处理过
processed = set()
unprocessed = []
for v in sorted(videos):
    base = os.path.splitext(v)[0]
    is_processed = any(base in s or s in base for s in walk_sources)
    if is_processed:
        processed.add(v)
    else:
        unprocessed.append(v)

print(f"  Already processed: {len(processed)}")
print(f"  NOT processed: {len(unprocessed)}")
if unprocessed:
    print(f"\n  Unprocessed videos (showing first 30):")
    for v in sorted(unprocessed)[:30]:
        # 获取视频时长
        full = os.path.join(walk_dir, v)
        import cv2
        cap = cv2.VideoCapture(full)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        dur = frames/fps if fps > 0 else 0
        print(f"    {v} ({frames}f, {dur:.0f}s)")
        cap.release()

# 3. 看看Walk难例推理结果JSON
print(f"\n=== Hard negative analysis ===")
import json
result_json = r"D:\Users\wangxianxiu\clawd\results_100 Ways to Walk100种走路方式.json"
with open(result_json, 'r', encoding='utf-8') as f:
    r = json.load(f)

fps = r['fps']
fp_windows = [p for p in r['predictions'] if p['filt'] == 'Fall']
print(f"  Video: {r['video']}, {r['duration_s']:.0f}s, {fps:.1f}fps")
print(f"  Total predictions: {len(r['predictions'])}")
print(f"  False positive Fall windows: {len(fp_windows)}")
print(f"  FP率: {len(fp_windows)/len(r['predictions'])*100:.1f}%")

# 找出FP连续段
events = []
for p in fp_windows:
    t = p['time_s']
    if not events or t - events[-1]['end'] > 0.5:
        events.append({'start': t, 'end': t, 'frames': [p['frame']]})
    else:
        events[-1]['end'] = t
        events[-1]['frames'].append(p['frame'])

print(f"\n  FP events ({len(events)}):")
for i, ev in enumerate(events):
    dur = ev['end'] - ev['start']
    avg_conf = np.mean([p['filt_conf'] for p in fp_windows 
                        if ev['start'] <= p['time_s'] <= ev['end']])
    print(f"    Event {i+1}: {ev['start']:.1f}s - {ev['end']:.1f}s ({dur:.1f}s), "
          f"{len(ev['frames'])} windows, avg conf={avg_conf:.3f}")
    # 每个事件的帧范围
    start_f = ev['frames'][0]
    end_f = ev['frames'][-1]
    print(f"      Frames: {start_f} - {end_f}, raw range: {start_f/fps:.1f}s - {end_f/fps:.1f}s")
