#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer.py — Orange Pi 5 Pro 板子端运行：加载 RKNN 模型，NPU 推理验证
用法：python infer.py --model path/to/model.rknn [--image path/to/test.jpg]

前置依赖：
  pip install rknn-toolkit-lite2  # 从 https://github.com/airockchip/rknn-toolkit2 获取 wheel
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

try:
    from rknnlite.api import RKNNLite
except ImportError:
    print("❌ 未安装 rknn-toolkit-lite2，请先执行：")
    print("   pip install rknn_toolkit_lite2-*-cp310-*.whl")
    sys.exit(1)


def check_npu():
    """检查 NPU 驱动是否正常加载"""
    try:
        load_path = "/sys/kernel/debug/rknpu/load"
        if Path(load_path).exists():
            load = Path(load_path).read_text().strip()
            print(f"🔍 NPU 负载: {load}")
            return True
        else:
            print("⚠️  无法读取 NPU 负载（可能需要 root 权限），尝试 dmesg 检查...")
            import subprocess
            result = subprocess.run(
                ["dmesg", "|", "grep", "-i", "rknpu"],
                shell=True, capture_output=True, text=True
            )
            if "RKNPU" in result.stdout or "rknpu" in result.stdout:
                print("       ✅ dmesg 检测到 RKNPU")
                return True
            else:
                print("       ❌ 未检测到 RKNPU 驱动！请确认系统镜像是否包含 NPU 驱动。")
                return False
    except Exception as e:
        print(f"⚠️  NPU 状态检测异常: {e}")
        return True  # 继续尝试推理，可能权限问题而非驱动问题


def infer_rknn(model_path: str, image_path: str = None, loop: int = 5):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print(f"📦 模型: {model_path.name} ({model_path.stat().st_size / 1024:.1f} KB)")

    # ─── Step 1: 加载模型 ───
    print(f"[1/3] 加载 RKNN 模型...")
    rknn = RKNNLite()
    ret = rknn.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f"加载模型失败，错误码: {ret}")
    print("       ✅ 模型加载完成")

    # ─── Step 2: 初始化 NPU ───
    print(f"[2/3] 初始化 RK3588S NPU (6 TOPS)...")
    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"NPU 初始化失败，错误码: {ret}")
    print("       ✅ NPU 就绪")

    # 查看模型输入信息
    input_details = rknn.query(what="input_tensors")
    print(f"       📐 输入: {input_details[0]['fmt']} | "
          f"形状 {input_details[0]['dims']} | "
          f"类型 {input_details[0]['type']}")

    # ─── Step 3: 推理基准测试 ───
    print(f"[3/3] 推理测试 ({loop} 轮 )...")

    # 构造输入数据（与实际模型输入尺寸匹配）
    if image_path:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"图片不存在: {image_path}")
        # 调整到模型输入尺寸
        h, w = input_details[0]['dims'][:2] if len(input_details[0]['dims']) == 4 else (224, 224)
        img = cv2.resize(img, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = img.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)  # NCHW
        print(f"       🖼️  使用图片: {image_path} → ({img.shape[2]}×{img.shape[0]})")
    else:
        # 随机数据（先用随机数据验证通路）
        dims = input_details[0]['dims']
        input_data = np.random.randn(*dims).astype(np.float32)
        print(f"       🎲 使用随机数据 (形状: {dims})")

    # 预热
    print("       🔥 预热中...")
    _ = rknn.inference(inputs=[input_data])

    # 正式基准测试
    latencies = []
    for i in range(loop):
        start = time.perf_counter()
        outputs = rknn.inference(inputs=[input_data])
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        print(f"       第 {i+1:2d} 轮: {elapsed_ms:8.2f} ms")

    # ─── 统计 ───
    avg_ms = np.mean(latencies)
    min_ms = np.min(latencies)
    max_ms = np.max(latencies)
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0

    print(f"\n{'─'*50}")
    print(f"📊 推理统计 ({loop} 轮):")
    print(f"   平均延迟:  {avg_ms:8.2f} ms")
    print(f"   最小/最大:  {min_ms:.2f} / {max_ms:.2f} ms")
    print(f"   吞吐量:    {fps:8.2f} FPS")
    print(f"   输出数量:  {len(outputs)} | 第一个输出形状: {outputs[0].shape}")
    print(f"{'─'*50}")

    # ─── 性能评估 ───
    if fps >= 30:
        print("🎉 NPU 性能优秀，满足实时推理要求！")
    elif fps >= 10:
        print("✅ NPU 工作正常，可用于项目开发。")
    else:
        print("⚠️  NPU 能跑但偏慢，检查是否未做量化或模型太大。")

    rknn.release()
    return avg_ms, fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RKNN NPU 推理验证工具 (Orange Pi 5 Pro)")
    parser.add_argument("--model", "-m", required=True, help="RKNN 模型路径")
    parser.add_argument("--image", "-i", default=None, help="测试图片路径（可选，不传则用随机数据）")
    parser.add_argument("--loop", "-n", type=int, default=10, help="推理轮数（默认 10）")
    args = parser.parse_args()

    # 检查 NPU
    check_npu()

    try:
        avg_ms, fps = infer_rknn(args.model, args.image, args.loop)
        print(f"\n✅ 全部测试通过！NPU 已就绪。")
    except Exception as e:
        print(f"\n❌ 推理失败: {e}")
        sys.exit(1)
