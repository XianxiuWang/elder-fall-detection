"""Generate final summary report for all subjects"""
import os, json, numpy as np
from collections import Counter

OUTPUT_DIR = r"E:\老人跌倒\data\subject_features"
files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("SUBJ_S") and f.endswith(".npz")])

# Parse subject data
subjects = {}
for f in files:
    sid = f.split("_")[1]  # S01, S02, ...
    fp = os.path.join(OUTPUT_DIR, f)
    data = np.load(fp, allow_pickle=True)
    n = data["keypoints"].shape[0]
    cat = int(data.get("category", -1))
    
    if sid not in subjects:
        subjects[sid] = {"files": 0, "frames_by_cat": {}, "total_frames": 0}
    subjects[sid]["files"] += 1
    subjects[sid]["frames_by_cat"][cat] = subjects[sid]["frames_by_cat"].get(cat, 0) + n
    subjects[sid]["total_frames"] += n

# Expected frame counts (from earlier data)
expected = {
    "S01": 1133, "S02": 1001, "S03": 1356, "S04": 1316,
    "S05": 1602, "S06": 1626, "S07": 1392, "S08": 1434, "S09": 1491,
}

cn = {0: "Fall", 1: "SitDown", 3: "Walking"}

print("=" * 70)
print("  Subject Feature Extraction Final Report")
print("=" * 70)
print(f"{'Subject':<10} {'Expected':>8} {'Detected':>8} {'Rate':>7} {'Fall':>6} {'SitDown':>8} {'Walk':>6}")
print("-" * 70)

total_exp = 0
total_det = 0
total_fall = 0
total_sit = 0
total_walk = 0

for sid in sorted(subjects.keys()):
    s = subjects[sid]
    exp = expected.get(sid, 0)
    det = s["total_frames"]
    rate = det / exp * 100 if exp > 0 else 0
    fall = s["frames_by_cat"].get(0, 0)
    sit = s["frames_by_cat"].get(1, 0)
    walk = s["frames_by_cat"].get(3, 0)
    
    sid_num = int(sid[1:])
    print(f"Subject.{sid_num:<4} {exp:>8} {det:>8} {rate:>6.1f}% {fall:>6} {sit:>8} {walk:>6}")
    
    total_exp += exp
    total_det += det
    total_fall += fall
    total_sit += sit
    total_walk += walk

print("-" * 70)
print(f"{'TOTAL':<10} {total_exp:>8} {total_det:>8} {total_det/total_exp*100:>6.1f}% {total_fall:>6} {total_sit:>8} {total_walk:>6}")
print("=" * 70)
print(f"Total .npz files: {len(files)}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Disk usage: {sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in files) // 1024} KB")

# Save report JSON
report = {
    "subjects": {sid: {"total_frames": v["total_frames"], "expected": expected.get(sid, 0),
                       "rate": f"{v['total_frames']/expected[sid]*100:.1f}%" if expected.get(sid) else "N/A",
                       "by_category": {cn.get(k, str(k)): v2 for k, v2 in v["frames_by_cat"].items()}}
                 for sid, v in subjects.items()},
    "total": {"detected": total_det, "expected": total_exp,
              "rate": f"{total_det/total_exp*100:.1f}%",
              "fall": total_fall, "sit_down": total_sit, "walking": total_walk},
    "files": len(files),
}

with open(os.path.join(OUTPUT_DIR, "report.json"), "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("\nReport saved to:", os.path.join(OUTPUT_DIR, "report.json"))
