#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess, os
from collections import Counter
os.chdir(r"E:\老人跌倒")
out = subprocess.check_output(["git", "ls-files"], encoding="utf-8", errors="replace")
lines = [l.strip().strip('"') for l in out.splitlines()]
ext = Counter(os.path.splitext(l)[1].lower() for l in lines)
print("文件数:", len(lines))
print("扩展名:", dict(ext))
big = [l for l in lines if os.path.getsize(l) > 1000000]
print(">1MB 文件数:", len(big))
for l in big:
    print("  ", round(os.path.getsize(l)/1e6, 2), "MB", l)
