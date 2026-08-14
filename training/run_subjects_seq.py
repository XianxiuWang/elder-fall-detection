"""Batch: run extract_one_subject.py for subjects 1-9 sequentially"""
import subprocess, os, time

PYTHON = r"d:\Anaconda3\envs\fall\python.exe"
SCRIPT = r"E:\老人跌倒\training\extract_one_subject.py"

total_start = time.time()
results = []

for sid in range(1, 10):
    print(f"\n{'='*50}", flush=True)
    print(f"Subject.{sid}", flush=True)
    print(f"{'='*50}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        [PYTHON, "-u", SCRIPT, str(sid)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )
    output_lines = []
    for line in proc.stdout:
        line = line.rstrip()
        print(f"  {line}", flush=True)
        output_lines.append(line)
    proc.wait()
    elapsed = time.time() - t0
    rc = proc.returncode
    print(f"  EXIT={rc} | {elapsed:.0f}s", flush=True)
    results.append((sid, rc, elapsed))

print(f"\n{'='*50}", flush=True)
print(f"ALL DONE: {time.time()-total_start:.0f}s", flush=True)
for sid, rc, elapsed in results:
    print(f"  Subject.{sid}: EXIT={rc} ({elapsed:.0f}s)", flush=True)
