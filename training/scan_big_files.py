#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描 E:\老人跌倒 大文件 (>1MB)，用于写 .gitignore"""
import os
ROOT = r"E:\老人跌倒"
SKIP = {".git", "__pycache__", "node_modules", ".venv", "venv"}

big = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP]
    for fn in filenames:
        p = os.path.join(dirpath, fn)
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        if sz > 1_000_000:
            rel = os.path.relpath(p, ROOT)
            big.append((sz, rel))

big.sort(reverse=True)
total = sum(s for s, _ in big)
print(f"大文件数: {len(big)}, 总大小: {total/1024/1024:.0f} MB\n")
for sz, rel in big:
    print(f"  {sz/1024/1024:8.2f} MB  {rel}")
