"""
护龄 —— 全局配置
"""
import os

# ============================================================
# 项目路径
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# 开源数据集路径（SisFall 等多目录支持）
# ============================================================
EXTERNAL_DATASET_DIRS = [
    r"E:\main_data",          # SisFall 主数据集
    r"E:\Three Classes",      # SisFall 三类数据集
]

# SisFall 活动码 → 状态映射（可在此处扩展/覆盖默认映射）
SISFALL_ACTIVITY_OVERRIDES = {
    # 格式: "SA01": "walking",  # 覆盖默认映射
    #       "SE99": "fall",     # 添加新映射
}

# ============================================================
# MediaPipe 33 个关键点索引
# ============================================================
# 参考: https://google.github.io/mediapipe/solutions/pose.html
LANDMARK = {
    "nose": 0,
    "left_eye_inner": 1,  "left_eye": 2,  "left_eye_outer": 3,
    "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
    "left_ear": 7,        "right_ear": 8,
    "mouth_left": 9,      "mouth_right": 10,
    "left_shoulder": 11,  "right_shoulder": 12,
    "left_elbow": 13,     "right_elbow": 14,
    "left_wrist": 15,     "right_wrist": 16,
    "left_pinky": 17,     "right_pinky": 18,
    "left_index": 19,     "right_index": 20,
    "left_thumb": 21,     "right_thumb": 22,
    "left_hip": 23,       "right_hip": 24,
    "left_knee": 25,      "right_knee": 26,
    "left_ankle": 27,     "right_ankle": 28,
    "left_heel": 29,      "right_heel": 30,
    "left_foot_index": 31,"right_foot_index": 32,
}

# 用于特征提取的关键骨骼线段
BONES = [
    # 躯干
    ("left_shoulder", "right_shoulder"),   # 肩宽
    ("left_shoulder", "left_hip"),         # 左躯干
    ("right_shoulder", "right_hip"),       # 右躯干
    ("left_hip", "right_hip"),             # 髋宽
    # 脊柱近似
    ("nose", "left_hip"),
    ("nose", "right_hip"),
    # 左臂
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    # 右臂
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # 左腿
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    # 右腿
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

# 6 类状态
STATE_NAMES = ["walking", "sitting", "lying", "long_sit", "abnormal", "fall"]
STATE_LABELS = {name: i for i, name in enumerate(STATE_NAMES)}
STATE_LABELS_REVERSE = {i: name for i, name in enumerate(STATE_NAMES)}

# 状态对应的快捷键（用于录制打标签）
STATE_HOTKEYS = {
    ord('1'): "walking",
    ord('2'): "sitting",
    ord('3'): "lying",
    ord('4'): "long_sit",
    ord('5'): "abnormal",
    ord('6'): "fall",
    ord('0'): "none",       # 无人/无效帧
}

# ============================================================
# 跌倒检测传感器阈值（MPU6050）
# ============================================================
FALL_THRESHOLD_ACCEL = 2.5      # 合加速度阈值 (G)
FALL_THRESHOLD_ANGLE = 60.0     # 角度变化阈值 (度)
FALL_WINDOW_MS = 500            # 撞击后监测窗口 (ms)

# ============================================================
# 生命体征异常阈值
# ============================================================
HR_MIN = 40                     # 最低心率 (bpm)
HR_MAX = 120                    # 最高心率 (bpm)
SPO2_MIN = 90                   # 最低血氧 (%)
TEMP_MAX = 37.5                 # 最高体温 (°C)
TEMP_MIN = 35.5                 # 最低体温 (°C)

# ============================================================
# 模型参数
# ============================================================
MODEL_TYPE = "random_forest"    # random_forest / xgboost / svm
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 12
TEST_SIZE = 0.2
RANDOM_SEED = 42

# ============================================================
# 视觉参数
# ============================================================
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
MODEL_COMPLEXITY = 1            # MediaPipe: 0=最快, 1=中等, 2=最准
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
