import os, numpy as np, time

d = r"E:\老人跌倒\data\subject_features"

for i in range(30):  # Max 15 min
    time.sleep(30)
    sc = {}
    tc = 0
    total_frames = 0
    for f in sorted(os.listdir(d)):
        if f.startswith("SUBJ_S") and f.endswith(".npz"):
            s = f.split("_")[1]
            sc[s] = sc.get(s, 0) + 1
            tc += 1
            try:
                data = np.load(os.path.join(d, f), allow_pickle=True)
                total_frames += data["keypoints"].shape[0]
            except:
                pass
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {tc} files, {total_frames}fr | subjects: {dict(sorted(sc.items()))}", flush=True)
    if len(sc) >= 9:
        print("All 9 subjects have files!", flush=True)
        break
