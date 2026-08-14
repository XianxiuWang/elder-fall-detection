#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析 git 暂存区文件构成，找出该排除的目录"""
import subprocess, os, sys
from collections import defaultdict

os.chdir(r"E:\老人跌倒")
out = subprocess.check_output(["git", "ls-files"], encoding="utf-8", errors="replace")
lines = [l.strip().strip('"') for l in out.splitlines()]

by_dir = defaultdict(lambda: [0, 0])  # dir -> [count, size]
for l in lines:
    parts = l.replace("\\", "/").split("/")
    d = "/".join(parts[:2]) if len(parts) > 1 else "(root)"
    try:
        sz = os.path.getsize(l)
    except OSError:
        sz = 0
    by_dir[d][0] += 1
    by_dir[d][1] += sz

print(f"暂存文件总数: {len(lines)}\n")
print(f"{'目录':40s} {'文件数':>8s} {'大小MB':>10s}")
for d, (cnt, sz) in sorted(by_dir.items(), key=lambda x: -x[1][1]):
    print(f"  {d:40s} {cnt:8d} {sz/1e6:10.2f}")
