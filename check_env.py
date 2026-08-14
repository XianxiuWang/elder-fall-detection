"""检查 conda fall 环境"""
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"torch: {torch.__version__}")
except ImportError:
    print("torch: NOT INSTALLED")

try:
    import mediapipe
    print(f"mediapipe: {mediapipe.__version__}")
except ImportError:
    print("mediapipe: NOT INSTALLED")

try:
    import cv2
    print(f"opencv: {cv2.__version__}")
except ImportError:
    print("opencv: NOT INSTALLED")

try:
    from ultralytics import YOLO
    print("ultralytics: OK")
except ImportError:
    print("ultralytics: NOT INSTALLED")

try:
    import lightgbm
    print(f"lightgbm: {lightgbm.__version__}")
except ImportError:
    print("lightgbm: NOT INSTALLED")

try:
    import numpy as np
    print(f"numpy: {np.__version__}")
except ImportError:
    print("numpy: NOT INSTALLED")
