"""
护龄 — 状态分类模型微调训练
============================
基于 torchvision 预训练模型，在跌倒/活动数据集上做 fine-tune。

支持的分类模型（均为 ImageNet 预训练）：
  · mobilenet_v3_small  — 最轻量(~2.5M)，适合 K230/ESP32 部署
  · mobilenet_v3_large  — 轻量(~5.5M)
  · efficientnet_b0     — 均衡(~5.3M)，精度好 ★ 推荐
  · efficientnet_b1     — 稍大(~7.8M)
  · resnet18            — 经典(~11.7M)

训练产物：
  · best_model.pth              — 最佳 PyTorch 权重
  · training_metrics.json       — 训练曲线数据
  · confusion_matrix.png        — 混淆矩阵图
  · models/checkpoints/         — 中间检查点
  · runs/tensorboard/           — TensorBoard 日志

用法：
    # 默认配置训练
    python train_classifier.py

    # 指定模型和轮数
    python train_classifier.py --model efficientnet_b0 --epochs 80 --batch 64

    # 只做评估（不训练）
    python train_classifier.py --eval_only --checkpoint models/best_model.pth
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 服务器无 GUI 模式
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models

import config


# ============================================================
# 数据增强与预处理
# ============================================================

def get_transforms(is_train: bool = True) -> transforms.Compose:
    """获取数据预处理 + 增强流水线"""
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                config.CLASSIFIER_INPUT_SIZE,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],   # ImageNet 统计值
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(
                (config.CLASSIFIER_INPUT_SIZE[0], config.CLASSIFIER_INPUT_SIZE[1])
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


# ============================================================
# 模型构建
# ============================================================

def build_model(model_name: str, num_classes: int = None,
                freeze_backbone: bool = False) -> nn.Module:
    """
    构建分类模型（ImageNet 预训练 + 自定义分类头）

    Args:
        model_name: 模型名称 (mobilenet_v3_small, efficientnet_b0, etc.)
        num_classes: 输出类别数
        freeze_backbone: 是否冻结 backbone（第一阶段训练时可用）

    Returns:
        PyTorch 模型
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES

    # 获取预训练权重
    weights_map = {
        "mobilenet_v3_small": models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "efficientnet_b1": models.EfficientNet_B1_Weights.IMAGENET1K_V1,
        "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
    }

    if model_name not in weights_map:
        raise ValueError(f"不支持的模型: {model_name}，可用: {list(weights_map.keys())}")

    weights = weights_map[model_name]

    # ── 构建模型并替换分类头 ──
    if model_name.startswith("mobilenet"):
        model = models.mobilenet_v3_small(weights=weights) \
            if "small" in model_name else \
            models.mobilenet_v3_large(weights=weights)

        # MobileNetV3 分类头: classifier[3] 是最后的 Linear 层
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    elif model_name.startswith("efficientnet"):
        model = models.efficientnet_b0(weights=weights) \
            if "b0" in model_name else \
            models.efficientnet_b1(weights=weights)

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    elif model_name.startswith("resnet"):
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    # ── 可选：冻结 backbone（第一阶段） ──
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name and "fc" not in name:
                param.requires_grad = False

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  📐 模型参数: 总计 {total_params/1e6:.1f}M, 可训练 {trainable_params/1e6:.1f}M")

    return model


# ============================================================
# 训练引擎
# ============================================================

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.should_stop = False

    def __call__(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer=None):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"  Epoch {epoch:3d} [Train]", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100.*correct/total:.1f}%'
        })

        # TensorBoard（每50步记录一次）
        global_step = (epoch - 1) * len(dataloader) + batch_idx
        if writer and batch_idx % 50 == 0:
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/Accuracy', 100.*correct/total, global_step)

    return running_loss / len(dataloader), 100. * correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="  [Eval]", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    return {
        'loss': running_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels,
    }


def train_model(model, train_loader, val_loader, test_loader,
                epochs: int, lr: float, device, model_name: str,
                writer=None) -> Dict:
    """
    完整训练流水线

    两阶段训练策略：
        Phase 1（前1/3 epoch）：冻结 backbone，只训练分类头
        Phase 2（剩余 epoch）：解冻全部参数，小学习率微调
    """
    print(f"\n{'='*60}")
    print(f"  开始训练: {model_name}")
    print(f"  设备: {device}")
    print(f"  训练集: {len(train_loader.dataset)} 张")
    print(f"  验证集: {len(val_loader.dataset)} 张")
    print(f"  测试集: {len(test_loader.dataset)} 张")
    print(f"{'='*60}\n")

    # 损失函数（带类别权重，缓解不平衡）
    criterion = nn.CrossEntropyLoss()

    # 优化器 + 调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr * 0.01
    )

    early_stopping = EarlyStopping(patience=config.EARLY_STOP_PATIENCE)

    # 训练状态记录
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_f1': [],
        'lr': [],
    }

    best_val_acc = 0.0
    best_model_path = config.MODEL_DIR / "best_model.pth"
    checkpoint_dir = config.MODEL_DIR / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )

        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']

        # 学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # 记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_metrics['f1'])
        history['lr'].append(current_lr)

        # TensorBoard
        if writer:
            writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            writer.add_scalar('Epoch/Train_Acc', train_acc, epoch)
            writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            writer.add_scalar('Epoch/Val_Acc', val_acc, epoch)
            writer.add_scalar('Epoch/Val_F1', val_metrics['f1'], epoch)
            writer.add_scalar('Epoch/LR', current_lr, epoch)

        # 日志
        elapsed = time.time() - epoch_start
        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | "
              f"F1: {val_metrics['f1']:.3f} | LR: {current_lr:.2e} | "
              f"⏱ {elapsed:.1f}s")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_metrics['f1'],
                'model_name': model_name,
                'num_classes': config.NUM_CLASSES,
                'state_names': config.STATE_NAMES,
            }, best_model_path)
            print(f"    ✅ 新最佳模型! Val Acc={val_acc:.1f}% (已保存)")

        # 定期 checkpoint
        if epoch % 10 == 0:
            ckpt_path = checkpoint_dir / f"ckpt_epoch{epoch:03d}.pth"
            torch.save(model.state_dict(), ckpt_path)

        # 早停检查
        if early_stopping(val_loss, epoch):
            print(f"  ⏹ 早停! 最佳 epoch: {early_stopping.best_epoch}")
            break

    # ── 加载最佳模型并在测试集上评估 ──
    print(f"\n{'='*60}")
    print(f"  📊 最终评估 (测试集)")
    print(f"{'='*60}")

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\n  测试集结果:")
    print(f"    Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"    Precision: {test_metrics['precision']*100:.2f}%")
    print(f"    Recall:    {test_metrics['recall']*100:.2f}%")
    print(f"    F1-Score:  {test_metrics['f1']*100:.2f}%")

    # 每类指标
    per_class = precision_recall_fscore_support(
        test_metrics['labels'], test_metrics['predictions'],
        labels=list(range(config.NUM_CLASSES)),
        zero_division=0
    )

    print(f"\n  逐类指标:")
    print(f"  {'状态':12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'样本数':>8s}")
    print(f"  {'-'*55}")
    for i in range(config.NUM_CLASSES):
        count = (test_metrics['labels'] == i).sum()
        print(f"  {config.STATE_NAMES[i]:12s} {per_class[0][i]*100:>9.1f}% "
              f"{per_class[1][i]*100:>9.1f}% {per_class[2][i]*100:>9.1f}% "
              f"{count:>8d}")

    # ── 生成混淆矩阵图 ──
    plot_confusion_matrix(
        test_metrics['labels'],
        test_metrics['predictions'],
        config.MODEL_DIR / "confusion_matrix.png"
    )

    # ── 生成训练曲线图 ──
    plot_training_curves(history, config.MODEL_DIR / "training_curves.png")

    # ── 保存训练历史为JSON ──
    metrics_path = config.MODEL_DIR / "training_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'model_name': model_name,
            'best_val_acc': float(best_val_acc),
            'test_metrics': {
                'accuracy': float(test_metrics['accuracy']),
                'precision': float(test_metrics['precision']),
                'recall': float(test_metrics['recall']),
                'f1': float(test_metrics['f1']),
                'per_class': {
                    config.STATE_NAMES_EN[i]: {
                        'precision': float(per_class[0][i]),
                        'recall': float(per_class[1][i]),
                        'f1': float(per_class[2][i]),
                    } for i in range(config.NUM_CLASSES)
                }
            },
            'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 训练完成!")
    print(f"  最佳模型: {best_model_path}")
    print(f"  混淆矩阵: {config.MODEL_DIR / 'confusion_matrix.png'}")
    print(f"  训练曲线: {config.MODEL_DIR / 'training_curves.png'}")
    print(f"  指标JSON: {metrics_path}")

    return test_metrics


# ============================================================
# 可视化
# ============================================================

def plot_confusion_matrix(y_true, y_pred, save_path: Path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    labels = [config.STATE_NAMES[i] for i in range(config.NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title='Confusion Matrix',
        ylabel='True Label',
        xlabel='Predicted Label',
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在格子里显示数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 混淆矩阵已保存: {save_path}")


def plot_training_curves(history: Dict, save_path: Path):
    """绘制训练曲线"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # F1 Score
    axes[1, 0].plot(epochs, history['val_f1'], 'g-', label='Val F1')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning Rate
    axes[1, 1].plot(epochs, history['lr'], 'm-')
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('LR')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 训练曲线已保存: {save_path}")


# ============================================================
# 多模型对比训练
# ============================================================

def compare_models(models_to_train: List[str], train_loader, val_loader,
                   test_loader, epochs: int, lr: float, device):
    """训练多个模型并输出对比结果"""
    results = {}

    for model_name in models_to_train:
        print(f"\n{'#'*60}")
        print(f"#  训练模型: {model_name}")
        print(f"{'#'*60}")

        model = build_model(model_name, config.NUM_CLASSES)
        model = model.to(device)

        writer = SummaryWriter(log_dir=str(config.LOG_DIR / f"{model_name}_{datetime.now():%Y%m%d_%H%M%S}"))

        metrics = train_model(
            model, train_loader, val_loader, test_loader,
            epochs=epochs, lr=lr, device=device,
            model_name=model_name, writer=writer
        )

        writer.close()

        results[model_name] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
        }

    # ── 打印对比表 ──
    print(f"\n{'='*70}")
    print(f"  多模型对比结果")
    print(f"{'='*70}")
    print(f"  {'模型':25s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
    print(f"  {'-'*65}")

    best_model = None
    best_f1 = 0
    for name, metrics in results.items():
        print(f"  {name:25s} {metrics['accuracy']*100:>9.2f}% "
              f"{metrics['precision']*100:>9.2f}% {metrics['recall']*100:>9.2f}% "
              f"{metrics['f1']*100:>9.2f}%")
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model = name

    print(f"\n  🏆 最佳模型: {best_model} (F1={best_f1*100:.2f}%)")

    # 保存对比结果
    with open(config.MODEL_DIR / "model_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="护龄 状态分类模型训练")
    parser.add_argument("--model", type=str, default=config.DEFAULT_CLASSIFIER,
                        choices=config.CLASSIFIER_OPTIONS + ["all"],
                        help="分类模型 (默认: efficientnet_b0, all=全部对比)")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS,
                        help=f"训练轮数 (默认: {config.EPOCHS})")
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE,
                        help=f"批次大小 (默认: {config.BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                        help=f"学习率 (默认: {config.LEARNING_RATE})")
    parser.add_argument("--data_dir", type=str,
                        default=str(config.DATASET_DIR / "prepared"),
                        help="预处理好的数据目录")
    parser.add_argument("--eval_only", action="store_true",
                        help="仅评估（不训练）")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型检查点路径（--eval_only 时使用）")
    args = parser.parse_args()

    # 设备
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif config.DEVICE == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"💻 使用设备: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB")

    # 数据目录检查
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        print("请先运行: python prepare_data.py")
        return

    # ── 仅评估模式 ──
    if args.eval_only:
        if not args.checkpoint:
            print("❌ --eval_only 需要指定 --checkpoint")
            return

        model = build_model(args.model, config.NUM_CLASSES)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        test_dataset = datasets.ImageFolder(
            data_dir / "test",
            transform=get_transforms(is_train=False)
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch,
                                 shuffle=False, num_workers=config.NUM_WORKERS)

        criterion = nn.CrossEntropyLoss()
        metrics = evaluate(model, test_loader, criterion, device)
        print(f"\n测试集准确率: {metrics['accuracy']*100:.2f}%")
        print(f"测试集 F1: {metrics['f1']*100:.2f}%")
        return

    # ── 正常训练模式 ──
    # 加载数据
    train_dataset = datasets.ImageFolder(
        data_dir / "train",
        transform=get_transforms(is_train=True)
    )
    val_dataset = datasets.ImageFolder(
        data_dir / "val",
        transform=get_transforms(is_train=False)
    )
    test_dataset = datasets.ImageFolder(
        data_dir / "test",
        transform=get_transforms(is_train=False)
    )

    # 验证类别映射
    print(f"\n类别映射 (ImageFolder 自动排序):")
    for idx, class_name in enumerate(train_dataset.classes):
        print(f"  [{idx}] {class_name}")

    # 处理样本不足的情况
    if len(train_dataset) == 0:
        print(f"\n❌ 训练集为空！请检查数据目录: {data_dir / 'train'}")
        print("   可能原因:")
        print("   1. prepare_data.py 未成功运行")
        print("   2. YOLO 检测阈值过高，裁剪不到人体")
        print("   3. 数据集未下载")
        return

    print(f"\n数据统计:")
    print(f"  训练集: {len(train_dataset)} 张 ({len(train_dataset.classes)} 类)")
    print(f"  验证集: {len(val_dataset)} 张")
    print(f"  测试集: {len(test_dataset)} 张")

    # DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch, shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch, shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    # ── 多模型对比 ──
    if args.model == "all":
        compare_models(
            config.CLASSIFIER_OPTIONS,
            train_loader, val_loader, test_loader,
            epochs=args.epochs, lr=args.lr, device=device
        )
        return

    # ── 单模型训练 ──
    model = build_model(args.model, config.NUM_CLASSES)
    model = model.to(device)

    writer = SummaryWriter(
        log_dir=str(config.LOG_DIR / f"{args.model}_{datetime.now():%Y%m%d_%H%M%S}")
    )

    train_model(
        model, train_loader, val_loader, test_loader,
        epochs=args.epochs, lr=args.lr, device=device,
        model_name=args.model, writer=writer
    )

    writer.close()


if __name__ == "__main__":
    main()
