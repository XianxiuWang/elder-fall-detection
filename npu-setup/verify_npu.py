#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_npu.py — Orange Pi 5 Pro 板子端运行：快速检测 NPU 是否可用
用法：python verify_npu.py
"""

import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: str) -> str:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip()


def main():
    print("=" * 50)
    print("🔍 Orange Pi 5 Pro NPU 环境检测")
    print("=" * 50)

    all_ok = True

    # 1. 芯片型号
    print("\n[1] 芯片检测")
    soc = run_cmd("cat /proc/device-tree/compatible")
    if "rk3588" in soc.lower():
        print(f"    ✅ 芯片: {soc}")
    else:
        print(f"    ⚠️  芯片: {soc} (期望 RK3588)")
        all_ok = False

    # 2. NPU 驱动
    print("\n[2] NPU 驱动")
    dmesg = run_cmd("dmesg | grep -i rknpu | tail -5")
    if dmesg:
        print(f"    ✅ 驱动已加载")
        for line in dmesg.split("\n")[:3]:
            print(f"       {line[:80]}")
    else:
        print("    ❌ 未检测到 NPU 驱动！请确认镜像是否包含 NPU 驱动。")
        all_ok = False

    # 3. NPU 设备节点
    print("\n[3] NPU 设备节点")
    devices = list(Path("/dev").glob("rknpu*"))
    if devices:
        for d in devices:
            print(f"    ✅ {d}")
    else:
        print("    ⚠️  未找到 /dev/rknpu*（可能不影响使用）")

    # 4. RKNN Lite2
    print("\n[4] RKNN Lite2 (Python)")
    try:
        from rknnlite.api import RKNNLite
        print(f"    ✅ rknn-toolkit-lite2 已安装")
    except ImportError:
        print("    ❌ 未安装 rknn-toolkit-lite2")
        print("       安装: pip install rknn_toolkit_lite2-*-cp310-*.whl")
        all_ok = False

    # 5. 关键依赖
    print("\n[5] 关键依赖")
    deps = ["numpy", "opencv-python", "onnxruntime"]
    for dep in deps:
        try:
            __import__(dep.replace("-", "_"))
            print(f"    ✅ {dep}")
        except ImportError:
            print(f"    ⚠️  {dep} (可选，推荐安装)")

    # 6. Python 版本
    print(f"\n[6] Python 版本: {sys.version}")

    # 7. 内存
    print("\n[7] 内存")
    mem = run_cmd("free -h | grep Mem")
    print(f"    {mem}")

    # ─── 总结 ───
    print("\n" + "=" * 50)
    if all_ok:
        print("🎉 环境就绪！可以运行：python infer.py --model xxx.rknn")
    else:
        print("⚠️  部分检查未通过，请按上述提示修复。")
    print("=" * 50)


if __name__ == "__main__":
    main()
