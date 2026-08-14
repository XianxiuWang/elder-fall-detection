"""
护龄 — 模型导出工具
====================
将训练好的 PyTorch 模型导出为可部署格式：
  · ONNX         → 通用推理引擎（OpenCV DNN / ONNX Runtime / 树莓派）
  · TorchScript  → PyTorch 原生部署
  · PTH          → 原始 PyTorch 权重（备份）

用法：
    # 导出所有格式
    python export_model.py

    # 只导出 ONNX
    python export_model.py --format onnx

    # 指定模型和 checkpoint
    python export_model.py --model efficientnet_b0 --checkpoint models/best_model.pth

    # 导出 YOLO 检测模型（给边缘设备用）
    python export_model.py --export_detector

导出产物：
    models/
    ├── classifier.onnx          ← ONNX 格式（推荐部署用）
    ├── classifier.torchscript   ← TorchScript 格式
    └── yolo_detector.onnx       ← YOLO 检测器 ONNX
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import config


# ============================================================
# 分类模型导出
# ============================================================

def export_classifier_onnx(model: nn.Module, output_path: Path,
                           input_size: tuple = None, device: str = "cpu"):
    """
    导出分类模型为 ONNX 格式

    ONNX 优势：
    - OpenCV DNN 可以直接加载（无需 PyTorch 环境）
    - ONNX Runtime 推理快
    - 树莓派/香橙派/手机 都支持
    """
    if input_size is None:
        input_size = (1, 3, *config.CLASSIFIER_INPUT_SIZE)

    model.eval()
    model = model.to("cpu")

    # 创建 dummy input
    dummy_input = torch.randn(*input_size)

    print(f"  输入形状: {input_size}")
    print(f"  ONNX opset: {config.ONNX_OPSET}")

    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=config.ONNX_OPSET,
        do_constant_folding=True,
        input_names=[config.ONNX_INPUT_NAME],
        output_names=[config.ONNX_OUTPUT_NAME],
        dynamic_axes={
            config.ONNX_INPUT_NAME: {0: 'batch_size'},
            config.ONNX_OUTPUT_NAME: {0: 'batch_size'},
        },
    )

    # 验证
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"  ✅ 导出成功: {output_path} ({file_size:.1f} MB)")

    # 验证推理一致性
    print(f"  验证: ONNX vs PyTorch 输出一致性...")
    import onnxruntime as ort
    ort_session = ort.InferenceSession(str(output_path))

    with torch.no_grad():
        pytorch_out = model(dummy_input).numpy()

    onnx_out = ort_session.run(
        [config.ONNX_OUTPUT_NAME],
        {config.ONNX_INPUT_NAME: dummy_input.numpy()}
    )[0]

    diff = np.abs(pytorch_out - onnx_out).max()
    print(f"  最大差异: {diff:.6f} {'✅ 一致' if diff < 1e-4 else '⚠️ 有差异'}")

    return output_path


def export_classifier_torchscript(model: nn.Module, output_path: Path,
                                  device: str = "cpu"):
    """导出为 TorchScript 格式"""
    model.eval()
    model = model.to("cpu")

    # 使用 trace 方式
    dummy_input = torch.randn(1, 3, *config.CLASSIFIER_INPUT_SIZE)

    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(str(output_path))

    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"  ✅ TorchScript 导出: {output_path} ({file_size:.1f} MB)")

    return output_path


def export_classifier_pth(model: nn.Module, checkpoint_path: Path,
                          output_path: Path, model_name: str):
    """
    导出精简版 PyTorch 权重（仅包含推理需要的状态字典）
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    export_data = {
        'model_state_dict': checkpoint['model_state_dict'],
        'model_name': model_name,
        'num_classes': config.NUM_CLASSES,
        'state_names': {str(k): str(v) for k, v in config.STATE_NAMES.items()},
        'state_names_en': {str(k): str(v) for k, v in config.STATE_NAMES_EN.items()},
        'input_size': list(config.CLASSIFIER_INPUT_SIZE),
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225],
        'export_date': str(Path(checkpoint_path).stat().st_mtime),
    }

    torch.save(export_data, output_path)
    file_size = output_path.stat().st_size / 1024 / 1024
    print(f"  ✅ PTH 导出: {output_path} ({file_size:.1f} MB)")

    return output_path


# ============================================================
# YOLO 检测模型导出
# ============================================================

def export_detector_onnx(detector_model: str = None):
    """
    导出 YOLO 检测模型为 ONNX

    用 ultralytics 自带的 export 方法，支持多种格式：
    onnx, tflite, ncnn, openvino 等
    """
    if detector_model is None:
        detector_model = config.DETECTION_MODEL

    print(f"\n{'='*60}")
    print(f"  导出检测模型: {detector_model}")
    print(f"{'='*60}")

    from ultralytics import YOLO

    model = YOLO(detector_model)

    # 导出 ONNX
    onnx_path = model.export(format="onnx", opset=config.ONNX_OPSET, simplify=True)
    print(f"  ✅ YOLO ONNX: {onnx_path}")

    # 导出 NCNN（国产芯片常用格式，嘉楠K230也支持）
    try:
        ncnn_path = model.export(format="ncnn")
        print(f"  ✅ YOLO NCNN: {ncnn_path}")
    except Exception as e:
        print(f"  ⚠️ NCNN导出失败: {e}")

    return onnx_path


def export_detector_tflite(detector_model: str = None):
    """导出 YOLO 为 TFLite（给手机/ESP32-S3部署用）"""
    if detector_model is None:
        detector_model = config.DETECTION_MODEL

    from ultralytics import YOLO
    model = YOLO(detector_model)

    try:
        tflite_path = model.export(format="tflite", int8=True)
        print(f"  ✅ YOLO TFLite (INT8): {tflite_path}")
    except Exception as e:
        print(f"  ⚠️ TFLite导出失败: {e}")


# ============================================================
# 部署配置生成
# ============================================================

def generate_deploy_config(model_name: str, output_path: Path):
    """
    生成部署配置文件 —— 方便在树莓派/K230/手机上加载模型
    """
    deploy_config = {
        "project": "护龄 - 独居老人居家安全监护系统",
        "model_info": {
            "classifier": str(output_path / "classifier.onnx"),
            "detector": str(output_path / "yolo_detector.onnx"),
            "model_type": model_name,
            "num_classes": config.NUM_CLASSES,
            "input_size": list(config.CLASSIFIER_INPUT_SIZE),
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "inference": {
            "detection_confidence": config.DETECTION_CONF,
            "temporal_window": config.TEMPORAL_WINDOW,
            "target_fps": config.INFERENCE_FPS,
        },
        "states": {
            str(k): {"name": v, "name_en": config.STATE_NAMES_EN[k],
                      "alert": config.ALERT_LEVEL[k][0]}
            for k, v in config.STATE_NAMES.items()
        },
    }

    config_path = output_path / "deploy_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(deploy_config, f, ensure_ascii=False, indent=2)

    print(f"  ✅ 部署配置: {config_path}")
    return config_path


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="护龄 模型导出")
    parser.add_argument("--model", type=str, default=config.DEFAULT_CLASSIFIER,
                        help=f"分类模型架构 (默认: {config.DEFAULT_CLASSIFIER})")
    parser.add_argument("--checkpoint", type=str,
                        default=str(config.MODEL_DIR / "best_model.pth"),
                        help="训练好的模型 checkpoint")
    parser.add_argument("--format", type=str, default="all",
                        choices=["all", "onnx", "torchscript", "pth"],
                        help="导出格式")
    parser.add_argument("--export_detector", action="store_true",
                        help="同时导出 YOLO 检测模型")
    parser.add_argument("--output_dir", type=str,
                        default=str(config.MODEL_DIR),
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cpu",
                        help="导出设备（建议用cpu避免兼容性问题）")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查 checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint 不存在: {checkpoint_path}")
        print("请先运行: python train_classifier.py")
        return

    print(f"\n{'='*60}")
    print(f"  护龄 v3 — 模型导出")
    print(f"{'='*60}")
    print(f"  分类模型: {args.model}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  输出目录: {output_dir}")
    print()

    # 加载模型
    print("📦 加载分类模型...")
    from train_classifier import build_model

    model = build_model(args.model, config.NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  ✅ 模型加载成功 (验证精度: {checkpoint.get('val_acc', 0)*100:.1f}%)")

    # ── 导出 ──
    formats = [args.format] if args.format != "all" else ["onnx", "torchscript", "pth"]
    exported_files = []

    for fmt in formats:
        if fmt == "onnx":
            path = export_classifier_onnx(model, output_dir / "classifier.onnx")
            exported_files.append(path)
        elif fmt == "torchscript":
            path = export_classifier_torchscript(model, output_dir / "classifier.torchscript")
            exported_files.append(path)
        elif fmt == "pth":
            path = export_classifier_pth(
                model, checkpoint_path,
                output_dir / "classifier.pth", args.model
            )
            exported_files.append(path)

    # ── 导出 YOLO 检测模型 ──
    if args.export_detector:
        print(f"\n🔍 导出检测模型...")
        detector_onnx = export_detector_onnx(config.DETECTION_MODEL)
        exported_files.append(Path(detector_onnx))

    # ── 生成部署配置 ──
    generate_deploy_config(args.model, output_dir)

    # ── 总结 ──
    print(f"\n{'='*60}")
    print(f"  ✅ 导出完成! 文件列表:")
    print(f"{'='*60}")
    for f in sorted(exported_files):
        size_mb = f.stat().st_size / 1024 / 1024 if f.exists() else 0
        print(f"  📁 {f.name:30s} {size_mb:7.1f} MB")
    print(f"  📁 deploy_config.json         ← 部署配置")

    print(f"\n📋 下一步:")
    print(f"  · OpenCV DNN 加载: net = cv2.dnn.readNetFromONNX('classifier.onnx')")
    print(f"  · ONNX Runtime:    session = ort.InferenceSession('classifier.onnx')")
    print(f"  · 移动端:          adb push classifier.onnx /data/local/tmp/")


if __name__ == "__main__":
    main()
