"""Batch runner: launch one process per subject"""
import subprocess, sys, time, os

SCRIPT = r"E:\老人跌倒\training\extract_one_subject.py"
PYTHON = r"d:\Anaconda3\envs\fall\python.exe"

print("Batch extracting Subject.1~9", flush=True)
results = []
total_start = time.time()

for sid in range(1, 10):
    print(f"\n--- Subject.{sid} ---", flush=True)
    t0 = time.time()
    try:
        # Run as separate process with unbuffered output
        proc = subprocess.run(
            [PYTHON, "-u", SCRIPT, str(sid)],
            capture_output=True, text=True, timeout=900,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        output = proc.stdout + proc.stderr
        elapsed = time.time() - t0
        code = proc.returncode
        
        # Parse result
        det = miss = fail = total = 0
        for line in output.split('\n'):
            if 'RESULT:' in line:
                parts = line.split()
                for p in parts:
                    if p.startswith('det='): det = int(p.split('=')[1])
                    if p.startswith('miss='): miss = int(p.split('=')[1])
                    if p.startswith('fail='): fail = int(p.split('=')[1])
                    if p.startswith('total='): total = int(p.split('=')[1])
        
        status = "OK" if code == 0 else f"EXIT={code}"
        print(f"  {status} | {elapsed:.0f}s | det={det}/{total} ({det/total*100:.0f}%) | miss={miss} fail={fail}", flush=True)
        results.append({"subject": sid, "detected": det, "missed": miss, "failed": fail, "total": total, "time": elapsed, "status": status})
        
        # Print last few lines if failed
        if code != 0:
            lines = [l.strip() for l in output.split('\n') if l.strip()]
            for l in lines[-5:]:
                print(f"    {l[:120]}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (15min)", flush=True)
        results.append({"subject": sid, "status": "TIMEOUT"})
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        results.append({"subject": sid, "status": str(e)})

print(f"\n{'='*60}", flush=True)
print(f"ALL DONE: {time.time()-total_start:.0f}s", flush=True)
total_det = sum(r.get('detected', 0) for r in results)
total_all = sum(r.get('total', 0) for r in results)
print(f"Total: {total_det}/{total_all} ({total_det/total_all*100:.0f}%)" if total_all else "No data", flush=True)
for r in results:
    print(f"  S{r['subject']:02d}: {r['status']}", flush=True)
