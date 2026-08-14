import subprocess, os, sys, time

SCRIPT = r"E:\老人跌倒\training\extract_one_subject.py"
PYTHON = r"d:\Anaconda3\envs\fall\python.exe"

for sid in range(5, 10):
    print(f"Subject.{sid}...", flush=True)
    t0 = time.time()
    try:
        r = subprocess.run(
            [PYTHON, "-u", SCRIPT, str(sid)],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        dt = time.time() - t0
        ok = "RESULT:" in r.stdout
        print(f"  Status: {'OK' if ok else 'FAIL'}, exit={r.returncode}, {dt:.0f}s", flush=True)
        for line in r.stdout.split("\n"):
            if "RESULT:" in line or "DONE" in line:
                print(f"  {line.strip()}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT (10min)", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

print("All done", flush=True)
