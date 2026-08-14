#!/usr/bin/env python3
"""
config_compare.py — 配置对比工具
===============================
对比本地测试配置和生产配置，高亮差异。

用法:
    python config_compare.py                          # 自动对比
    python config_compare.py local.json prod.json     # 指定文件
"""

import sys
import os
import json
import argparse
from typing import Any


def flatten_dict(d: dict, prefix: str = "") -> dict:
    """展平嵌套字典为 key.path → value"""
    result = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            result.update(flatten_dict(v, full_key))
        elif isinstance(v, list):
            result[full_key] = f"[{len(v)} items]" if len(v) > 5 else str(v)
        else:
            result[full_key] = v
    return result


def compare(local_path: str, prod_path: str):
    with open(local_path, 'r', encoding='utf-8') as f:
        local = json.load(f)
    with open(prod_path, 'r', encoding='utf-8') as f:
        prod = json.load(f)

    flat_local = flatten_dict(local)
    flat_prod = flatten_dict(prod)

    all_keys = sorted(set(flat_local.keys()) | set(flat_prod.keys()))

    only_local = []
    only_prod = []
    differences = []
    same = 0

    for key in all_keys:
        lv = flat_local.get(key, "<MISSING>")
        pv = flat_prod.get(key, "<MISSING>")
        if lv == "<MISSING>":
            only_prod.append((key, pv))
        elif pv == "<MISSING>":
            only_local.append((key, lv))
        elif lv != pv:
            differences.append((key, lv, pv))
        else:
            same += 1

    print(f"{'='*70}")
    print(f"  Config Comparison: {os.path.basename(local_path)} vs {os.path.basename(prod_path)}")
    print(f"{'='*70}")
    print(f"  Same: {same}  |  Diff: {len(differences)}  |  "
          f"Local-only: {len(only_local)}  |  Prod-only: {len(only_prod)}")

    if differences:
        print(f"\n  ── Differences ──")
        print(f"  {'Key':<45s} {'Local':>10s}  {'Production':>10s}")
        print(f"  {'-'*67}")
        for key, lv, pv in differences:
            key_short = key if len(key) < 45 else "..." + key[-42:]
            lv_str = str(lv)[:10] if len(str(lv)) <= 10 else str(lv)[:9] + "…"
            pv_str = str(pv)[:10] if len(str(pv)) <= 10 else str(pv)[:9] + "…"
            print(f"  {key_short:<45s} {lv_str:>10s}  {pv_str:>10s}")

    if only_local:
        print(f"\n  ── Local-only Keys ──")
        for key, val in only_local:
            print(f"    {key} = {val}")

    if only_prod:
        print(f"\n  ── Production-only Keys ──")
        for key, val in only_prod:
            print(f"    {key} = {val}")

    print(f"\n{'='*70}")


def create_production_config(output_path: str):
    """基于默认配置创建生产配置"""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.fall_config import FallConfig

    cfg = FallConfig()

    # Orange Pi 5 Pro 适配
    cfg.e2e_monitor.rtsp_url = "rtsp://admin:RVXCEM@192.168.1.100:554/h264/ch1/main/av_stream"
    cfg.e2e_monitor.window_name = "Fall Monitor - Orange Pi 5 Pro"
    cfg.e2e_monitor.use_local_camera_fallback = True
    cfg.e2e_monitor.camera_buffer_size = 4  # RTSP 需要更大缓冲

    # MediaPipe 用轻量模式（Orange Pi 算力有限）
    cfg.mediapipe.model_complexity = 1
    cfg.mediapipe.min_detection_confidence = 0.3
    cfg.mediapipe.min_tracking_confidence = 0.3

    # 过程检测参数调优（生产环境更敏感）
    cfg.process_fall.cooldown_seconds = 10.0
    cfg.process_fall.consecutive_suspicious = 4

    # 久坐提醒 (v3.0)
    cfg.e2e_monitor.sedentary_warn_min = 45.0
    cfg.e2e_monitor.sedentary_alert_min = 60.0

    cfg.description = f"Orange Pi 5 Pro production config"
    cfg.save(output_path)
    print(f"Production config saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Config comparison tool")
    parser.add_argument("local", nargs="?", default=None, help="Local config path")
    parser.add_argument("prod", nargs="?", default=None, help="Production config path")
    parser.add_argument("--create-prod", action="store_true",
                        help="Create production config from default")
    args = parser.parse_args()

    if args.create_prod:
        out = args.prod or r"E:\老人跌倒\config_production.json"
        create_production_config(out)
        return

    local = args.local or r"E:\老人跌倒\config_local_camera.json"
    prod = args.prod or r"E:\老人跌倒\config_production.json"

    if not os.path.exists(prod):
        print(f"Production config not found: {prod}")
        print("Run with --create-prod to create one")
        return

    compare(local, prod)


if __name__ == "__main__":
    main()
