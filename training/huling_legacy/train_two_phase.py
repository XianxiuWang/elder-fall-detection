"""
护龄 — 两阶段训练脚本
======================
一键跑通"开源数据预训练 → 自采数据微调"全流程。

Phase 1: 开源数据集 → 基础模型
    · 冻结 backbone（ImageNet 预训练权重），只训练新的分类头
    · 数据来源：URFD + MCFD 公开跌倒数据集
    · 输出：models/best_model_phase1.pth

Phase 2: 基础模型 + 自采数据 → 最终模型
    · 加载 Phase 1 权重，解冻全模型做微调
    · 数据来源：公开数据集 + 你自己的采集数据（合并训练）
    · 输出：models/best_model_phase2.pth

用法:
    # 完整两阶段训练（推荐首次使用）
    python train_two_phase.py

    # 指定模型
    python train_two_phase.py --model mobilenet_v3_small

    # 指定自己采集数据的目录
    python train_two_phase.py --custom_data_dir D:/my_huling_data

    # Phase 1 训练更多轮（开源数据量大时）
    python train_two_phase.py --phase1_epochs 30

    # 只跑 Phase 1（还没收集自己数据时）
    python train_two_phase.py --phase1_only

    # 只跑 Phase 2（已有 Phase 1 模型，新加了自采数据）
    python train_two_phase.py --phase2_only --phase1_model models/best_model_phase1.pth

    # 下载数据集但不训练（先确认数据能下载）
    python train_two_phase.py --download_only

输出产物:
    models/
    ├── best_model_phase1.pth          ← Phase 1 最佳模型（基础模型）
    ├── best_model_phase2.pth          ← Phase 2 最佳模型（最终模型，部署用这个）
    ├── confusion_matrix_phase1.png    ← Phase 1 混淆矩阵
    ├── confusion_matrix_phase2.png    ← Phase 2 混淆矩阵
    ├── training_curves_phase1.png     ← Phase 1 训练曲线
    ├── training_curves_phase2.png     ← Phase 2 训练曲线
    └── two_phase_report.json         ← 两阶段对比报告
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter

# 添加父目录到 path（确保能 import config 和 train_classifier）
sys.path.insert(0, str(Path(__file__).parent))
import config

# 从现有训练脚本导入核心函数
from train_classifier import (
    build_model,
    train_model,
    get_transforms,
    evaluate,
    EarlyStopping,
    plot_confusion_matrix,
    plot_training_curves,
)


# ============================================================
# 工具函数
# ============================================================

def run_script(script_name: str, args: list = None) -> bool:
    """
    运行同目录下的 Python 脚本，返回是否成功。

    Args:
        script_name: 脚本文件名（如 'download_datasets.py'）
        args: 命令行参数列表

    Returns:
        True 表示运行成功，False 表示失败
    """
    script_path = Path(__file__).parent / script_name
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    print(f"\n{'─'*60}")
    print(f"  🔧 执行: {' '.join(cmd)}")
    print(f"{'─'*60}")

    result = subprocess.run(cmd, cwd=str(script_path.parent))
    return result.returncode == 0


def check_datasets_ready() -> bool:
    """
    检查公开数据集是否已下载并预处理完毕。

    Returns:
        True 表示数据就绪
    """
    prepared_dir = config.DATASET_DIR / "prepared"
    train_dir = prepared_dir / "train"

    if not train_dir.exists():
        return False

    # 检查是否至少有一个类别的图片
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        return False

    total_images = sum(1 for d in class_dirs for _ in d.iterdir() if _.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    return total_images > 0


def ensure_public_datasets(download_only: bool = False):
    """
    确保公开数据集已下载并预处理。
    如果缺失则自动执行 download → prepare 流程。
    """
    # Step 1: 检查原始数据集是否已下载
    urfd_dir = config.DATASET_DIR / "urfd"
    has_urfd = urfd_dir.exists() and any(urfd_dir.iterdir())

    if not has_urfd:
        print("\n📥 未检测到 URFD 数据集，开始下载...")
        if not run_script("download_datasets.py", ["--dataset", "urfd"]):
            print("⚠️ URFD 下载失败。可能原因：网络问题或链接失效。")
            print("   可以手动下载后放到 datasets/urfd/ 目录下，然后重新运行。")
            print("   下载地址: http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html")
            if not check_datasets_ready():
                print("\n❌ 无可用数据集，无法继续训练。")
                sys.exit(1)

    # Step 2: 检查预处理后的数据
    if check_datasets_ready():
        print(f"\n✅ 数据集已就绪: {config.DATASET_DIR / 'prepared'}")
        return

    if download_only:
        print("\n📥 --download_only 模式：数据集已下载，跳过训练。")
        print(f"   下次运行 python train_two_phase.py 时将自动预处理并训练。")
        sys.exit(0)

    print("\n🔧 数据未预处理，开始预处理...")
    if not run_script("prepare_data.py"):
        print("❌ 数据预处理失败。")
        sys.exit(1)

    if not check_datasets_ready():
        print("❌ 预处理完成但未找到训练数据，请检查 prepare_data.py 的输出。")
        sys.exit(1)

    print("✅ 数据预处理完成。")


def check_custom_data(data_dir: Path) -> Optional[dict]:
    """
    检查自定义数据目录是否存在且有效。

    Args:
        data_dir: 自定义数据根目录（应包含 train/val/test 子目录）

    Returns:
        如果数据有效，返回 {'train': N, 'val': N, 'test': N} 统计；
        如果无效，返回 None
    """
    if not data_dir.exists():
        return None

    stats = {}
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            return None

        # 统计图片数量
        count = 0
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        for d in class_dirs:
            count += sum(1 for f in d.iterdir()
                        if f.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        stats[split] = count

    total = sum(stats.values())
    if total == 0:
        return None

    return stats


def load_prepared_dataset(data_dir: Path, split: str, is_train: bool):
    """
    加载 ImageFolder 格式的数据集。

    Args:
        data_dir: 数据根目录（包含 train/val/test 子目录）
        split: 'train' / 'val' / 'test'
        is_train: 是否使用训练增强

    Returns:
        ImageFolder 数据集实例
    """
    split_dir = data_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {split_dir}")

    return datasets.ImageFolder(
        split_dir,
        transform=get_transforms(is_train=is_train)
    )


def merge_datasets(public_dir: Path, custom_dir: Path, split: str, is_train: bool):
    """
    合并公开数据集和自定义数据集。

    使用 ConcatDataset 在内存中合并，不复制文件。

    Args:
        public_dir: 公开数据集根目录
        custom_dir: 自定义数据集根目录
        split: 'train' / 'val' / 'test'
        is_train: 是否使用训练增强

    Returns:
        ConcatDataset 实例，或单独的 ImageFolder（当自定义数据不存在时）
    """
    public_ds = load_prepared_dataset(public_dir, split, is_train)

    custom_split_dir = custom_dir / split
    if not custom_split_dir.exists():
        return public_ds

    custom_ds = datasets.ImageFolder(
        custom_split_dir,
        transform=get_transforms(is_train=is_train)
    )

    # 验证类别一致性
    if public_ds.classes != custom_ds.classes:
        print(f"  ⚠️  类别映射不一致！")
        print(f"     公开数据: {public_ds.classes}")
        print(f"     自定义数据: {custom_ds.classes}")
        print(f"     将使用公开数据的类别映射，自定义数据中的未知类别会被跳过。")
        # TODO: 更健壮的类别映射处理

    print(f"    公开数据: {len(public_ds)} 张, 自定义数据: {len(custom_ds)} 张")
    return ConcatDataset([public_ds, custom_ds])


# ============================================================
# 两阶段报告生成
# ============================================================

def generate_report(phase1_metrics: dict, phase2_metrics: Optional[dict],
                    config_dict: dict, output_path: Path):
    """
    生成两阶段训练对比报告（JSON + 文本摘要）。

    Args:
        phase1_metrics: Phase 1 测试集指标
        phase2_metrics: Phase 2 测试集指标（可能为 None）
        config_dict: 训练配置摘要
        output_path: 报告保存路径
    """
    report = {
        'generated_at': datetime.now().isoformat(),
        'model': config_dict['model_name'],
        'device': config_dict['device'],
        'phase1': {
            'data_source': '公开数据集 (URFD)',
            'strategy': '冻结 backbone，仅训练分类头',
            'epochs': config_dict['phase1_epochs'],
            'learning_rate': config_dict['phase1_lr'],
            'test_metrics': {
                'accuracy': float(phase1_metrics['accuracy']),
                'precision': float(phase1_metrics['precision']),
                'recall': float(phase1_metrics['recall']),
                'f1': float(phase1_metrics['f1']),
            } if phase1_metrics else None,
        },
    }

    if phase2_metrics:
        report['phase2'] = {
            'data_source': '公开数据集 + 自采数据',
            'strategy': '加载 Phase 1 权重，全模型微调',
            'epochs': config_dict['phase2_epochs'],
            'learning_rate': config_dict['phase2_lr'],
            'test_metrics': {
                'accuracy': float(phase2_metrics['accuracy']),
                'precision': float(phase2_metrics['precision']),
                'recall': float(phase2_metrics['recall']),
                'f1': float(phase2_metrics['f1']),
            },
            'improvement_vs_phase1': {
                'accuracy': float(phase2_metrics['accuracy'] - phase1_metrics['accuracy'])
                if phase1_metrics else None,
                'f1': float(phase2_metrics['f1'] - phase1_metrics['f1'])
                if phase1_metrics else None,
            },
        }

    # 保存 JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"  📊 两阶段训练报告")
    print(f"{'='*60}")
    print(f"  模型: {report['model']}")
    print(f"  设备: {report['device']}")
    print()

    if phase1_metrics:
        m1 = report['phase1']['test_metrics']
        print(f"  Phase 1 (开源数据, 冻结 backbone):")
        print(f"    Accuracy:  {m1['accuracy']*100:.2f}%")
        print(f"    F1-Score:  {m1['f1']*100:.2f}%")

    if phase2_metrics:
        m2 = report['phase2']['test_metrics']
        imp = report['phase2']['improvement_vs_phase1']
        print(f"\n  Phase 2 (加入自采数据, 全模型微调):")
        print(f"    Accuracy:  {m2['accuracy']*100:.2f}%")
        print(f"    F1-Score:  {m2['f1']*100:.2f}%")
        if imp['accuracy'] is not None:
            sign = '+' if imp['accuracy'] >= 0 else ''
            print(f"\n  提升幅度: Accuracy {sign}{imp['accuracy']*100:.1f}pp, "
                  f"F1 {sign}{imp['f1']*100:.1f}pp")

    print(f"\n  报告已保存: {output_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="护龄 两阶段训练 — 开源数据预训练 → 自采数据微调",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train_two_phase.py                              # 完整两阶段训练
  python train_two_phase.py --phase1_only                # 只跑 Phase 1
  python train_two_phase.py --model mobilenet_v3_small   # 指定轻量模型
  python train_two_phase.py --custom_data_dir D:/my_data # 指定自采数据目录
  python train_two_phase.py --download_only              # 只下载数据集
        """
    )

    # ── 模型配置 ──
    parser.add_argument('--model', type=str, default=config.DEFAULT_CLASSIFIER,
                        choices=config.CLASSIFIER_OPTIONS,
                        help=f'分类模型 (默认: {config.DEFAULT_CLASSIFIER})')

    # ── Phase 配置 ──
    parser.add_argument('--phase1_only', action='store_true',
                        help='仅运行 Phase 1（开源数据 + 冻结 backbone）')
    parser.add_argument('--phase2_only', action='store_true',
                        help='仅运行 Phase 2（需已有 Phase 1 模型）')
    parser.add_argument('--download_only', action='store_true',
                        help='仅下载公开数据集，不训练')

    # ── 训练参数 ──
    parser.add_argument('--phase1_epochs', type=int, default=20,
                        help='Phase 1 训练轮数 (默认: 20)')
    parser.add_argument('--phase2_epochs', type=int, default=30,
                        help='Phase 2 训练轮数 (默认: 30)')
    parser.add_argument('--phase1_lr', type=float, default=1e-3,
                        help='Phase 1 学习率 (默认: 1e-3, 训练分类头可以大一点)')
    parser.add_argument('--phase2_lr', type=float, default=1e-4,
                        help='Phase 2 学习率 (默认: 1e-4, 全模型微调用小学习率)')
    parser.add_argument('--batch', type=int, default=config.BATCH_SIZE,
                        help=f'批次大小 (默认: {config.BATCH_SIZE})')

    # ── 数据路径 ──
    parser.add_argument('--custom_data_dir', type=str,
                        default=str(config.DATA_DIR / 'custom'),
                        help='自采数据目录（ImageFolder 格式，含 train/val/test）')
    parser.add_argument('--phase1_model', type=str, default=None,
                        help='Phase 1 模型路径（--phase2_only 时使用）')

    args = parser.parse_args()

    # ================================================================
    # 环境检查
    # ================================================================

    print(f"\n{'#'*60}")
    print(f"#  护龄 — 两阶段训练")
    print(f"#  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    # 设备
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif config.DEVICE == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\n💻 设备: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB")

    # 创建必要目录
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # 确保公开数据集就绪
    # ================================================================
    ensure_public_datasets(download_only=args.download_only)

    prepared_dir = config.DATASET_DIR / "prepared"

    # ================================================================
    # 检查自定义数据
    # ================================================================
    custom_dir = Path(args.custom_data_dir)
    custom_stats = check_custom_data(custom_dir)

    if custom_stats:
        print(f"\n📂 自采数据: {custom_dir}")
        print(f"   train: {custom_stats['train']} 张")
        print(f"   val:   {custom_stats['val']} 张")
        print(f"   test:  {custom_stats['test']} 张")
    else:
        if args.phase2_only:
            print(f"\n⚠️  未找到自采数据目录: {custom_dir}")
            print(f"   Phase 2 需要自采数据。请先运行 collect_data.py 采集数据。")
            print(f"   或指定正确的目录: --custom_data_dir <路径>")
            sys.exit(1)
        else:
            print(f"\n💡 未找到自采数据目录 ({custom_dir})。")
            print(f"   Phase 2 将跳过。后续收集到数据后运行:")
            print(f"   python train_two_phase.py --phase2_only --phase1_model models/best_model_phase1.pth")

    # ================================================================
    # Phase 1: 开源数据 → 基础模型
    # ================================================================

    phase1_model_path = config.MODEL_DIR / "best_model_phase1.pth"
    phase1_metrics = None

    if not args.phase2_only:
        print(f"\n{'='*60}")
        print(f"  🧊 Phase 1: 开源数据集 → 基础模型")
        print(f"     策略: 冻结 backbone（保留 ImageNet 特征），仅训练分类头")
        print(f"     数据: URFD 公开跌倒数据集")
        print(f"     轮数: {args.phase1_epochs}")
        print(f"     学习率: {args.phase1_lr}")
        print(f"{'='*60}")

        # 加载数据
        phase1_train_ds = load_prepared_dataset(prepared_dir, 'train', is_train=True)
        phase1_val_ds = load_prepared_dataset(prepared_dir, 'val', is_train=False)
        phase1_test_ds = load_prepared_dataset(prepared_dir, 'test', is_train=False)

        print(f"\n  数据统计:")
        print(f"    训练集: {len(phase1_train_ds)} 张")
        print(f"    验证集: {len(phase1_val_ds)} 张")
        print(f"    测试集: {len(phase1_test_ds)} 张")
        print(f"    类别: {phase1_train_ds.classes}")

        if len(phase1_train_ds) == 0:
            print("\n❌ 训练集为空！请检查数据预处理是否成功。")
            sys.exit(1)

        # DataLoader
        phase1_train_loader = DataLoader(
            phase1_train_ds, batch_size=args.batch, shuffle=True,
            num_workers=config.NUM_WORKERS, pin_memory=(device.type == "cuda"),
            drop_last=True
        )
        phase1_val_loader = DataLoader(
            phase1_val_ds, batch_size=args.batch, shuffle=False,
            num_workers=config.NUM_WORKERS
        )
        phase1_test_loader = DataLoader(
            phase1_test_ds, batch_size=args.batch, shuffle=False,
            num_workers=config.NUM_WORKERS
        )

        # 构建模型（冻结 backbone）
        model_phase1 = build_model(
            args.model,
            num_classes=config.NUM_CLASSES,
            freeze_backbone=True
        )
        model_phase1 = model_phase1.to(device)

        # TensorBoard
        phase1_writer = SummaryWriter(
            log_dir=str(config.LOG_DIR / f"phase1_{args.model}_{datetime.now():%Y%m%d_%H%M%S}")
        )

        # 训练
        phase1_metrics = train_model(
            model_phase1,
            phase1_train_loader, phase1_val_loader, phase1_test_loader,
            epochs=args.phase1_epochs, lr=args.phase1_lr, device=device,
            model_name=f"{args.model}_phase1", writer=phase1_writer
        )
        phase1_writer.close()

        # 重命名 Phase 1 产物（避免被 Phase 2 覆盖）
        default_best = config.MODEL_DIR / "best_model.pth"
        if default_best.exists():
            shutil.move(str(default_best), str(phase1_model_path))
            print(f"  💾 Phase 1 模型已保存: {phase1_model_path}")

        # 重命名混淆矩阵和训练曲线
        for fname in ['confusion_matrix.png', 'training_curves.png']:
            src = config.MODEL_DIR / fname
            if src.exists():
                dst = config.MODEL_DIR / fname.replace('.png', '_phase1.png')
                shutil.move(str(src), str(dst))

        print(f"\n✅ Phase 1 完成!")
        print(f"   基础模型: {phase1_model_path}")
        print(f"   测试集 F1: {phase1_metrics['f1']*100:.2f}%")
    else:
        # --phase2_only: 使用指定的 Phase 1 模型
        phase1_model_path = Path(args.phase1_model) if args.phase1_model else phase1_model_path
        if not phase1_model_path.exists():
            print(f"\n❌ Phase 1 模型不存在: {phase1_model_path}")
            print(f"   请先运行 Phase 1: python train_two_phase.py --phase1_only")
            sys.exit(1)
        print(f"\n📦 使用已有 Phase 1 模型: {phase1_model_path}")

    # 如果仅需要 Phase 1，到此结束
    if args.phase1_only:
        print(f"\n{'='*60}")
        print(f"  🏁 --phase1_only 模式，训练结束。")
        print(f"  基础模型: {phase1_model_path}")
        print(f"  后续收集自采数据后，运行:")
        print(f"  python train_two_phase.py --phase2_only --phase1_model {phase1_model_path}")
        print(f"{'='*60}")
        return

    # ================================================================
    # Phase 2: 基础模型 + 自采数据 → 最终模型
    # ================================================================

    if not custom_stats:
        print(f"\n⚠️  跳过 Phase 2（无自采数据）。")
        print(f"   收集数据后运行:")
        print(f"   python train_two_phase.py --phase2_only --phase1_model {phase1_model_path}")
        return

    print(f"\n{'='*60}")
    print(f"  🔥 Phase 2: 基础模型 + 自采数据 → 最终模型")
    print(f"     策略: 加载 Phase 1 权重，解冻 backbone 全模型微调")
    print(f"     数据: URFD 公开数据集 + 自采数据（{custom_dir}）")
    print(f"     轮数: {args.phase2_epochs}")
    print(f"     学习率: {args.phase2_lr}")
    print(f"{'='*60}")

    # 合并数据集
    print(f"\n  📊 合并数据集:")
    phase2_train_ds = merge_datasets(prepared_dir, custom_dir, 'train', is_train=True)
    phase2_val_ds = merge_datasets(prepared_dir, custom_dir, 'val', is_train=False)
    phase2_test_ds = merge_datasets(prepared_dir, custom_dir, 'test', is_train=False)

    print(f"    合并后训练集: {len(phase2_train_ds)} 张")
    print(f"    合并后验证集: {len(phase2_val_ds)} 张")
    print(f"    合并后测试集: {len(phase2_test_ds)} 张")

    # DataLoader
    phase2_train_loader = DataLoader(
        phase2_train_ds, batch_size=args.batch, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    phase2_val_loader = DataLoader(
        phase2_val_ds, batch_size=args.batch, shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    phase2_test_loader = DataLoader(
        phase2_test_ds, batch_size=args.batch, shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    # 构建模型（不冻结 backbone，全模型微调）
    model_phase2 = build_model(
        args.model,
        num_classes=config.NUM_CLASSES,
        freeze_backbone=False
    )

    # 加载 Phase 1 权重（加载的是我们自定义的分类头 + 预训练 backbone）
    checkpoint = torch.load(phase1_model_path, map_location=device, weights_only=False)
    # Phase 1 保存的分类头结构和 Phase 2 相同（都是 freeze_backbone 参数不同），可以直接加载
    try:
        model_phase2.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✅ 已加载 Phase 1 权重: {phase1_model_path}")
        print(f"     Phase 1 最佳 val_acc: {checkpoint.get('val_acc', 'N/A')}")
    except RuntimeError as e:
        # 如果 state_dict 不完全匹配（例如分类头结构微调），尝试宽松加载
        print(f"  ⚠️  严格加载失败，尝试宽松加载...")
        model_state = model_phase2.state_dict()
        pretrained_state = checkpoint['model_state_dict']
        matched = 0
        for k, v in pretrained_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                matched += 1
        model_phase2.load_state_dict(model_state)
        print(f"  ✅ 宽松加载完成: {matched}/{len(model_state)} 层匹配")

    model_phase2 = model_phase2.to(device)

    # TensorBoard
    phase2_writer = SummaryWriter(
        log_dir=str(config.LOG_DIR / f"phase2_{args.model}_{datetime.now():%Y%m%d_%H%M%S}")
    )

    # 训练
    phase2_metrics = train_model(
        model_phase2,
        phase2_train_loader, phase2_val_loader, phase2_test_loader,
        epochs=args.phase2_epochs, lr=args.phase2_lr, device=device,
        model_name=f"{args.model}_phase2", writer=phase2_writer
    )
    phase2_writer.close()

    # 保存 Phase 2 模型
    phase2_model_path = config.MODEL_DIR / "best_model_phase2.pth"
    default_best = config.MODEL_DIR / "best_model.pth"
    if default_best.exists():
        shutil.move(str(default_best), str(phase2_model_path))
        print(f"  💾 Phase 2 模型已保存: {phase2_model_path}")

    # 重命名图表
    for fname in ['confusion_matrix.png', 'training_curves.png']:
        src = config.MODEL_DIR / fname
        if src.exists():
            dst = config.MODEL_DIR / fname.replace('.png', '_phase2.png')
            shutil.move(str(src), str(dst))

    # ================================================================
    # 生成对比报告
    # ================================================================

    report_path = config.MODEL_DIR / "two_phase_report.json"
    generate_report(
        phase1_metrics=phase1_metrics,
        phase2_metrics=phase2_metrics,
        config_dict={
            'model_name': args.model,
            'device': str(device),
            'phase1_epochs': args.phase1_epochs,
            'phase2_epochs': args.phase2_epochs,
            'phase1_lr': args.phase1_lr,
            'phase2_lr': args.phase2_lr,
        },
        output_path=report_path,
    )

    # ================================================================
    # 完成
    # ================================================================

    print(f"\n{'='*60}")
    print(f"  🎉 两阶段训练全部完成!")
    print(f"{'='*60}")
    print(f"")
    print(f"  📦 产物清单:")
    print(f"     Phase 1 基础模型: {phase1_model_path}")
    print(f"     Phase 2 最终模型: {phase2_model_path}")
    print(f"     对比报告:         {report_path}")
    print(f"     Phase 1 混淆矩阵: {config.MODEL_DIR / 'confusion_matrix_phase1.png'}")
    print(f"     Phase 2 混淆矩阵: {config.MODEL_DIR / 'confusion_matrix_phase2.png'}")
    print(f"     Phase 1 训练曲线: {config.MODEL_DIR / 'training_curves_phase1.png'}")
    print(f"     Phase 2 训练曲线: {config.MODEL_DIR / 'training_curves_phase2.png'}")
    print(f"")
    print(f"  🚀 下一步:")
    print(f"     1. 导出模型: python export_model.py")
    print(f"     2. 部署推理: python inference_demo.py")
    print(f"     3. 启动 Web 监控: python webapp/server.py")
    print(f"")


if __name__ == '__main__':
    main()
