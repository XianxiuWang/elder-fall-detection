"""
护龄 — 公开数据集下载工具
===========================
自动下载并解压常用的跌倒检测公开数据集：
  · UR Fall Detection Dataset  (70个视频，包含跌倒 + 日常活动)
  · 还支持 Le2i、UP-Fall 等备选数据集

用法：
    python download_datasets.py              # 下载 URFD
    python download_datasets.py --dataset all  # 下载全部可用数据集
    python download_datasets.py --dataset le2i  # 只下载 Le2i

下载后目录结构：
    datasets/
    ├── urfd/
    │   ├── falls/          ← 跌倒视频 (cam0 + cam1)
    │   └── adls/           ← 日常活动视频 (cam0 + cam1)
    └── le2i/
        └── ...
"""

import os
import sys
import zipfile
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed

import config


# ============================================================
# 数据集元信息注册表
# ============================================================

DATASET_REGISTRY = {
    "urfd": {
        "name": "UR Fall Detection Dataset",
        "description": "波兰热舒夫大学发布，70个视频（30跌倒+40日常活动）RGB+深度双通道",
        "homepage": "http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html",
        "files": {
            "cam0_falls": {
                "url": "http://fenix.univ.rzeszow.pl/~mkepski/ds/data/urfall-cam0-falls.zip",
                "filename": "urfall-cam0-falls.zip",
            },
            "cam0_adls": {
                "url": "http://fenix.univ.rzeszow.pl/~mkepski/ds/data/urfall-cam0-adls.zip",
                "filename": "urfall-cam0-adls.zip",
            },
        },
        "citation": "@article{kwolek2014fall, title={Fall detection using Kinect sensor...}",
        "license": "CC BY-NC-SA 4.0",
    },
    "multicam": {
        "name": "Multiple Cameras Fall Dataset (MCFD)",
        "description": "多视角跌倒检测数据集，GitHub可直接下载，国内友好",
        "homepage": "https://github.com/AdailtonCerqueira/fall-detection-dataset",
        "files": {
            "main": {
                "url": "https://github.com/AdailtonCerqueira/fall-detection-dataset/archive/refs/heads/main.zip",
                "filename": "mcfd.zip",
            },
        },
    },
}


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """
    下载单个文件，带进度条和重试机制
    返回 True/False
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 如果已存在且大小>0，跳过
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ⏭️  已存在，跳过: {dest.name} ({dest.stat().st_size/1024/1024:.1f} MB)")
        return True

    print(f"  ⬇️  下载 {desc or dest.name} ...")

    def _report(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size / total_size * 100, 100)
            downloaded = count * block_size / 1024 / 1024
            total_mb = total_size / 1024 / 1024
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r    [{bar}] {pct:5.1f}%  {downloaded:.1f}/{total_mb:.1f} MB", end="")
            sys.stdout.flush()

    max_retries = 3
    for attempt in range(max_retries):
        try:
            urlretrieve(url, str(dest), reporthook=_report)
            print()  # 换行
            print(f"  ✅ 下载完成: {dest.name} ({dest.stat().st_size/1024/1024:.1f} MB)")
            return True
        except (URLError, HTTPError, ConnectionError) as e:
            print()
            if attempt < max_retries - 1:
                print(f"  ⚠️  下载失败 (尝试 {attempt+1}/{max_retries}): {e}，3秒后重试...")
                import time
                time.sleep(3)
            else:
                print(f"  ❌ 下载失败: {e}")
    return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """解压 ZIP 文件"""
    try:
        print(f"  📦 解压 {zip_path.name} → {extract_to.name}/ ...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # 检查是否有顶层目录，避免解压出散乱文件
            members = zf.namelist()
            root_dirs = set(m.split('/')[0] for m in members if '/' in m)

            zf.extractall(extract_to)
        zip_path.unlink()  # 解压后删除原始zip
        print(f"  ✅ 解压完成")
        return True
    except (zipfile.BadZipFile, OSError) as e:
        print(f"  ❌ 解压失败: {e}")
        return False


def download_dataset(dataset_key: str) -> bool:
    """下载指定数据集"""
    if dataset_key not in DATASET_REGISTRY:
        print(f"❌ 未知数据集: {dataset_key}")
        print(f"可用: {list(DATASET_REGISTRY.keys())}")
        return False

    info = DATASET_REGISTRY[dataset_key]
    print(f"\n{'='*60}")
    print(f"  下载: {info['name']}")
    print(f"  说明: {info['description']}")
    print(f"{'='*60}")

    dest_dir = config.DATASET_DIR / dataset_key
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 并行下载
    total = len(info["files"])
    success = 0

    for key, meta in info["files"].items():
        url = meta["url"]
        filename = meta.get("filename", f"{key}.zip")
        dest = dest_dir / filename

        ok = download_file(url, dest, f"[{key}]")
        if ok and dest.suffix == '.zip':
            extract_zip(dest, dest_dir)
        if ok:
            success += 1

    print(f"\n  📊 完成: {success}/{total} 个文件下载成功")

    # 统计下载的内容
    video_count = 0
    for ext in ['*.avi', '*.mp4', '*.mov', '*.mkv', '*.wmv']:
        video_count += len(list(dest_dir.rglob(ext)))

    print(f"  📹 检出视频: {video_count} 个")
    print(f"  📍 路径: {dest_dir}")

    if hasattr(info, 'citation'):
        print(f"\n  📝 引用: \n{info.get('citation', '')}")

    return success > 0


def main():
    parser = argparse.ArgumentParser(description="护龄 公开数据集下载")
    parser.add_argument("--dataset", type=str, default="urfd",
                        choices=list(DATASET_REGISTRY.keys()) + ["all"],
                        help="要下载的数据集 (默认: urfd)")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用数据集")
    args = parser.parse_args()

    if args.list:
        print("可用数据集:")
        for key, info in DATASET_REGISTRY.items():
            print(f"  [{key}] {info['name']}")
            print(f"        {info['description']}")
            print()
        return

    if args.dataset == "all":
        for key in DATASET_REGISTRY:
            download_dataset(key)
    else:
        download_dataset(args.dataset)

    print(f"\n{'='*60}")
    print("✅ 下载完成！下一步：")
    print(f"   python prepare_data.py    # 预处理数据集（切帧+标注）")
    print(f"   python train_classifier.py # 训练分类模型")
    print(f"\n⚠️  如果下载链接失效（学术服务器常有这种情况），")
    print(f"   可以手动下载数据集，放到 {config.DATASET_DIR}/ 目录下即可")


if __name__ == "__main__":
    main()
