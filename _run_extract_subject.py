"""
Run extract_subject_skeletons.py with proper error capture
"""
import sys, os, traceback

sys.path.insert(0, r"E:\老人跌倒")

try:
    from training.extract_subject_skeletons import main
    main()
except Exception as e:
    print(f"\nFATAL ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
