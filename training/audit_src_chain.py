#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""排查 src/ 与根目录主模型链路是否还在使用"""
import os, sys, re, pickle, time

ROOT = r"E:\老人跌倒"

def fmt_ts(p):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))

print("=" * 70)
print("1. 根目录顶层结构")
print("=" * 70)
for name in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, name)
    tag = "[DIR]" if os.path.isdir(p) else f"[{os.path.getsize(p)//1024}KB]"
    print(f"  {tag:8s} {name:30s} {fmt_ts(p)}")

print("\n" + "=" * 70)
print("2. src/ 目录内容")
print("=" * 70)
src = os.path.join(ROOT, "src")
if os.path.isdir(src):
    for name in sorted(os.listdir(src)):
        p = os.path.join(src, name)
        tag = "[DIR]" if os.path.isdir(p) else f"[{os.path.getsize(p)//1024}KB]"
        print(f"  {tag:8s} {name:34s} {fmt_ts(p)}")
else:
    print("  src/ 不存在")

print("\n" + "=" * 70)
print("3. models/ 目录内容")
print("=" * 70)
mdl = os.path.join(ROOT, "models")
if os.path.isdir(mdl):
    for name in sorted(os.listdir(mdl)):
        p = os.path.join(mdl, name)
        tag = "[DIR]" if os.path.isdir(p) else f"[{os.path.getsize(p)//1024}KB]"
        print(f"  {tag:8s} {name:38s} {fmt_ts(p)}")
else:
    print("  models/ 不存在")

print("\n" + "=" * 70)
print("4. 根目录主模型 fall_classifier_6class.pkl 的版本/维度")
print("=" * 70)
for mf in ["fall_classifier_6class.pkl", "fall_classifier_6class_v6.pkl",
           "fall_classifier_6class_v7.pkl", "fall_classifier_6class_v8.pkl"]:
    p = os.path.join(mdl, mf)
    if os.path.exists(p):
        try:
            b = pickle.load(open(p, "rb"))
            cfg = b.get("config", {})
            print(f"  {mf}: dim={b.get('feature_dim')} ver={cfg.get('version')} "
                  f"model={type(b.get('model')).__name__} acc={cfg.get('val_acc')}")
        except Exception as e:
            print(f"  {mf}: 读取失败 {e}")
    else:
        print(f"  {mf}: (不存在)")

print("\n" + "=" * 70)
print("5. 全项目搜索关键引用 (fall_classifier_6class / ml_6class_detector / e2e_fall_monitor / src)")
print("=" * 70)
refs = {}
patterns = [
    r"fall_classifier_6class\.pkl",
    r"fall_classifier_6class_v\d+\.pkl",
    r"ml_6class_detector",
    r"e2e_fall_monitor",
    r"from src\.|import src",
    r"src[/\\]",
]
exts = {".py", ".bat", ".ps1", ".sh", ".service", ".md", ".txt", ".json", ".yaml", ".yml", ".cfg", ".ini"}
for dirpath, dirnames, filenames in os.walk(ROOT):
    # 跳过 venv/缓存
    dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", "node_modules", ".venv", "venv"}]
    for fn in filenames:
        if os.path.splitext(fn)[1].lower() not in exts:
            continue
        p = os.path.join(dirpath, fn)
        try:
            txt = open(p, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for pat in patterns:
            if re.search(pat, txt):
                rel = os.path.relpath(p, ROOT)
                refs.setdefault(pat, []).append(rel)

for pat in patterns:
    hits = refs.get(pat, [])
    print(f"\n  [{pat}] 命中 {len(hits)} 处:")
    for h in sorted(set(hits)):
        print(f"      - {h}")

print("\n" + "=" * 70)
print("6. deploy_opi5/ 目录 (对比活跃度)")
print("=" * 70)
dp = os.path.join(ROOT, "deploy_opi5")
if os.path.isdir(dp):
    for name in sorted(os.listdir(dp)):
        p = os.path.join(dp, name)
        tag = "[DIR]" if os.path.isdir(p) else f"[{os.path.getsize(p)//1024}KB]"
        print(f"  {tag:8s} {name:34s} {fmt_ts(p)}")
