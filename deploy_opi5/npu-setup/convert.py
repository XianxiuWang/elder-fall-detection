#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert.py — PC 端运行：将 ONNX 模型转换为 RKNN 格式
用法：python convert.py --model path/to/model.onnx --output path/to/model.rknn

前置依赖：
  pip install rknn-toolkit2  # 从 https://github.com/airockchip/rknn-toolkit2 获取 wheel
"""

import argparse
import sys
from pathlib import Path

try:
    from rknn.api import RKNN
except ImportError:
    print("❌ 未安装 rknn-toolkit2，请先执行：")
    print("   git clone https://github.com/airockchip/rknn-toolkit2.git")
    print("   pip install rknn-toolkit2/packages/rknn_toolkit2-*-cp310-*.whl")
    sys.exit(1)


def convert_onnx_to_rknn(model_path: str, output_path: str, target: str = "rk3588"):
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN()

    # ─── Step 1: 加载 ONNX ───
    print(f"[1/4] 加载 ONNX 模型: {model_path}")
    ret = rknn.load_onnx(model=str(model_path))
    if ret != 0:
        raise RuntimeError(f"加载 ONNX 失败，错误码: {ret}")
    print("       ✅ 加载成功")

    # ─── Step 2: 配置 ───
    print(f"[2/4] 配置 RKNN (目标平台: {target})")
    ret = rknn.config(
        target_platform=target,
        optimization_level=3,          # 0=最快转换, 3=最优性能
        # 先不量化，确保能跑通；稳定后再开 INT8:
        # do_quantization=True,
        # dataset='dataset.txt',       # INT8 量化校准数据集
    )
    if ret != 0:
        raise RuntimeError(f"配置失败，错误码: {ret}")
    print("       ✅ 配置完成")

    # ─── Step 3: 构建 RKNN ───
    print(f"[3/4] 构建 RKNN 模型（可能需要几分钟）...")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise RuntimeError(f"构建失败，错误码: {ret}")
    print("       ✅ 构建完成")

    # ─── Step 4: 导出 ───
    print(f"[4/4] 导出 RKNN 模型: {output_path}")
    ret = rknn.export_rknn(str(output_path))
    if ret != 0:
        raise RuntimeError(f"导出失败，错误码: {ret}")

    # 统计信息
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"       ✅ 导出成功！文件大小: {size_mb:.2f} MB")

    rknn.release()
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX → RKNN 模型转换工具")
    parser.add_argument("--model", "-m", required=True, help="输入 ONNX 模型路径")
    parser.add_argument("--output", "-o", default=None, help="输出 RKNN 路径（默认同目录同名 .rknn）")
    parser.add_argument("--target", "-t", default="rk3588", help="目标平台 (默认: rk3588)")
    args = parser.parse_args()

    if args.output is None:
        args.output = str(Path(args.model).with_suffix(".rknn"))

    try:
        convert_onnx_to_rknn(args.model, args.output, args.target)
        print(f"\n🎉 转换完成: {args.output}")
        print(f"   下一步：将这个 .rknn 文件拷贝到 Orange Pi 5 Pro 上运行 infer.py")
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        sys.exit(1)
