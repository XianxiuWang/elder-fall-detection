#!/bin/bash
# install.sh — Orange Pi 5 Pro 跌倒检测系统安装脚本
# 用法: bash install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "============================================"
echo "  老人跌倒检测 — Orange Pi 5 Pro 安装"
echo "  ${SCRIPT_DIR}"
echo "============================================"

# ── 检查 Python ──
echo ""
echo "[1/5] 检查 Python..."
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v $cmd &>/dev/null; then
        PYVER=$($cmd --version 2>&1 | grep -oP '\d+\.\d+')
        MAJOR=$(echo $PYVER | cut -d. -f1)
        MINOR=$(echo $PYVER | cut -d. -f2)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON=$cmd
            echo "  ✓ $PYTHON (version $PYVER)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "  ✗ 需要 Python >= 3.10"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "  Armbian: 通常已预装 Python 3.11+"
    exit 1
fi

# ── 可选：创建虚拟环境 ──
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo ""
    echo "[2/5] 创建虚拟环境..."
    $PYTHON -m venv "$SCRIPT_DIR/venv"
    echo "  ✓ venv 创建完成"
else
    echo ""
    echo "[2/5] 虚拟环境已存在，跳过"
fi

source "$SCRIPT_DIR/venv/bin/activate"

# ── 升级 pip ──
echo ""
echo "[3/5] 升级 pip..."
pip install --upgrade pip setuptools wheel

# ── 安装依赖 ──
echo ""
echo "[4/5] 安装依赖 (ARM64)..."
echo "  这可能需要 5-15 分钟，请耐心等待..."

# 先装 numpy（其他依赖需要）
pip install "numpy>=2.0,<3.0"

# 装其他核心依赖
pip install "scikit-learn>=1.5,<2.0" "xgboost>=2.0,<4.0"

# OpenCV
pip install "opencv-python>=4.8"

# MediaPipe — ARM64 特殊处理
echo "  安装 MediaPipe (ARM64)..."
pip install "mediapipe>=0.10.14,<0.11" 2>/dev/null || {
    echo "  ⚠ 标准 mediapipe 安装失败，尝试从源码安装..."
    echo "  手动安装: pip install mediapipe"
}

echo ""
echo "[5/5] 验证安装..."
$PYTHON -c "
import numpy; print(f'  numpy: {numpy.__version__}')
import sklearn; print(f'  sklearn: {sklearn.__version__}')
import xgboost; print(f'  xgboost: {xgboost.__version__}')
import cv2; print(f'  opencv: {cv2.__version__}')
try:
    import mediapipe; print(f'  mediapipe: {mediapipe.__version__}')
except ImportError:
    print('  mediapipe: ⚠ 未安装（需要单独处理）')
" || true

# ── 检查模型文件 ──
echo ""
if [ -f "$SCRIPT_DIR/models/fall_classifier_6class.pkl" ]; then
    echo "  ✓ 模型文件已就绪"
else
    echo "  ⚠ 模型文件未找到: models/fall_classifier_6class.pkl"
    echo "  请将模型文件复制到 deploy_opi5/models/ 目录"
fi

echo ""
echo "============================================"
echo "  安装完成！"
echo ""
echo "  运行方式:"
echo "    source venv/bin/activate"
echo "    python3 fall_inference.py <video.mp4>"
echo "    python3 fall_inference.py --realtime"
echo "    python3 fall_inference.py --benchmark"
echo ""
echo "  如果 MediaPipe 安装失败，请手动尝试:"
echo "    pip install mediapipe==0.10.14"
echo "  或使用 RKNN 加速版 (需自行编译):"
echo "    https://github.com/rockchip-linux/rknn-toolkit2"
echo "============================================"
