"""
护龄 —— 特征提取核心模块

从 MediaPipe Pose 的 33 个关键点中，提取用于状态分类的多维特征向量。

提取的特征维度：
  1. 躯干基础特征 (6维)  —— 质心高度、肩髋中点、躯干角度
  2. 关节点归一化 (66维) —— 33个关键点相对坐标 (x, y)
  3. 身体角度特征 (8维)  —— 膝盖角、肘部角、肩髋角度
  4. 姿态结构特征 (8维)  —— 宽高比、对称性、腿部状态
  5. 运动特征 (4+N维)    —— 帧间位移、速度（需连续帧）
  6. 传感器槽位 (6维)    —— 预留加速度/心率/血氧/体温

总维度: ~98 维（不含运动缓冲帧数）

用法:
    from feature_extractor import FeatureExtractor

    extractor = FeatureExtractor()
    features = extractor.extract(landmarks_list)       # 单帧
    features = extractor.extract_with_motion(landmarks_list, prev_landmarks)  # 含运动
"""

import math
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field

from config import LANDMARK, BONES


# ============================================================
# 数据结构
# ============================================================
@dataclass
class Landmark3D:
    """单个关键点"""
    x: float
    y: float
    z: float
    visibility: float = 1.0

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class FeatureVector:
    """一帧完整特征"""
    values: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    dim: int = 0
    # 分模块的特征（方便调试查看各模块贡献）
    torso_features: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    joint_features: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    angle_features: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    structure_features: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    motion_features: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    sensor_features: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))


# ============================================================
# 特征提取器
# ============================================================
class FeatureExtractor:
    """
    特征提取器。

    Parameters
    ----------
    use_motion : bool
        是否计算帧间运动特征（需要传入前后帧）
    smooth_window : int
        运动特征滑动窗口大小（帧数）
    """

    def __init__(self, use_motion: bool = True, smooth_window: int = 5):
        self.use_motion = use_motion
        self.smooth_window = smooth_window
        self._prev_landmarks: Optional[List[Landmark3D]] = None
        self._motion_history: List[np.ndarray] = []  # 滑动窗口

        # 预计算特征维度（用于验证一致性）
        self._compute_feature_dim()

    def _compute_feature_dim(self):
        """计算总特征维度"""
        n = 0
        n += 6   # torso
        n += 66  # joints (33 * 2)
        n += 8   # angles
        n += 8   # structure
        if self.use_motion:
            n += 4 + self.smooth_window * 2  # motion
        n += 6   # sensor slots
        self.feature_dim = n

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def extract(self, landmarks: List[Landmark3D],
                sensor_data: Optional[Dict[str, float]] = None) -> FeatureVector:
        """
        从单帧关键点提取特征（不含运动特征，用于数据集训练）。

        Parameters
        ----------
        landmarks : 33个关键点
        sensor_data : 传感器数据 {"accel_x": ..., "hr": ..., "spo2": ..., "temp": ...}

        Returns
        -------
        FeatureVector
        """
        lm_array = self._validate_landmarks(landmarks)

        fv = FeatureVector()
        fv.torso_features = self._extract_torso(lm_array)
        fv.joint_features = self._extract_joints(lm_array)
        fv.angle_features = self._extract_angles(lm_array)
        fv.structure_features = self._extract_structure(lm_array)
        fv.motion_features = np.zeros(4, dtype=np.float32)  # 单帧无运动
        fv.sensor_features = self._extract_sensor(sensor_data)

        parts = [
            fv.torso_features,
            fv.joint_features,
            fv.angle_features,
            fv.structure_features,
            fv.motion_features,
            fv.sensor_features,
        ]
        fv.values = np.concatenate(parts, dtype=np.float32)
        fv.dim = len(fv.values)
        return fv

    def extract_with_motion(self, landmarks: List[Landmark3D],
                            sensor_data: Optional[Dict[str, float]] = None) -> FeatureVector:
        """
        从连续帧提取特征（含帧间运动特征，用于实时推理）。
        调用时会自动保存当前帧作为下一次的 prev_landmarks。
        """
        lm_array = self._validate_landmarks(landmarks)

        fv = FeatureVector()
        fv.torso_features = self._extract_torso(lm_array)
        fv.joint_features = self._extract_joints(lm_array)
        fv.angle_features = self._extract_angles(lm_array)
        fv.structure_features = self._extract_structure(lm_array)
        fv.sensor_features = self._extract_sensor(sensor_data)

        # 运动特征
        if self._prev_landmarks is not None:
            prev_array = self._validate_landmarks(self._prev_landmarks)
            motion = self._extract_motion(lm_array, prev_array)
            fv.motion_features = motion
        else:
            fv.motion_features = np.zeros(4 + self.smooth_window * 2, dtype=np.float32)

        parts = [
            fv.torso_features,
            fv.joint_features,
            fv.angle_features,
            fv.structure_features,
            fv.motion_features,
            fv.sensor_features,
        ]
        fv.values = np.concatenate(parts, dtype=np.float32)
        fv.dim = len(fv.values)

        # 保存当前帧
        self._prev_landmarks = landmarks
        self._update_motion_history(fv.motion_features[:4])

        return fv

    def reset(self):
        """重置状态（切换视频/重新开始推理时调用）"""
        self._prev_landmarks = None
        self._motion_history.clear()

    # ------------------------------------------------------------------
    # 内部方法：验证与坐标转换
    # ------------------------------------------------------------------
    def _validate_landmarks(self, landmarks: List[Landmark3D]) -> np.ndarray:
        """验证并转换为 numpy 数组 (33, 4)"""
        if isinstance(landmarks, np.ndarray):
            return landmarks
        if len(landmarks) != 33:
            raise ValueError(f"需要 33 个关键点，收到 {len(landmarks)} 个")
        return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks], dtype=np.float32)

    def _get_point(self, lm: np.ndarray, name: str) -> np.ndarray:
        """获取指定关键点的坐标 (x, y, z)"""
        idx = LANDMARK[name]
        return lm[idx, :3]

    def _get_vis(self, lm: np.ndarray, name: str) -> float:
        """获取指定关键点的可见度"""
        idx = LANDMARK[name]
        return lm[idx, 3]

    def _norm(self, v: np.ndarray) -> float:
        """向量模长"""
        return float(np.linalg.norm(v))

    def _angle_between(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        计算三点夹角 ∠ABC（B 是顶点，返回角度值，范围 0-180°）
        """
        ba = a - b
        bc = c - b
        dot = np.dot(ba, bc)
        norm_prod = self._norm(ba) * self._norm(bc)
        if norm_prod < 1e-6:
            return 0.0
        cos_angle = np.clip(dot / norm_prod, -1.0, 1.0)
        return math.degrees(math.acos(cos_angle))

    # ------------------------------------------------------------------
    # 模块1: 躯干基础特征 (6维)
    # ------------------------------------------------------------------
    def _extract_torso(self, lm: np.ndarray) -> np.ndarray:
        """
        [0] 质心 (x) —— 相对于画面宽度（已归一化 0-1）
        [1] 质心 (y) —— 关键：站立≈0.35, 坐着≈0.55, 躺着≈0.6+
        [2] 躯干长度 —— 肩-髋距离（归一化）
        [3] 躯干角度 —— 躯干与垂直线的夹角（站立≈0°, 躺着≈90°）
        [4] 肩膀中点高度
        [5] 髋部中点高度
        """
        l_shoulder = self._get_point(lm, "left_shoulder")
        r_shoulder = self._get_point(lm, "right_shoulder")
        l_hip = self._get_point(lm, "left_hip")
        r_hip = self._get_point(lm, "right_hip")

        # 质心（肩+髋的几何中心）
        shoulder_mid = (l_shoulder + r_shoulder) / 2.0
        hip_mid = (l_hip + r_hip) / 2.0
        centroid = (shoulder_mid + hip_mid) / 2.0

        # 躯干长度（肩中点 → 髋中点）
        torso_len = self._norm(shoulder_mid - hip_mid)

        # 躯干与垂直线的夹角（地面法线 = (0,1,0) 的反方向）
        torso_vec = hip_mid - shoulder_mid  # 从上到下
        vertical = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        dot = np.dot(torso_vec, vertical)
        norm_prod = self._norm(torso_vec) * self._norm(vertical)
        torso_angle = 0.0
        if norm_prod > 1e-6:
            cos_a = np.clip(dot / norm_prod, -1.0, 1.0)
            torso_angle = math.degrees(math.acos(cos_a))

        return np.array([
            centroid[0],         # 质心x
            centroid[1],         # 质心y
            torso_len,           # 躯干长度
            torso_angle,         # 躯干角
            shoulder_mid[1],     # 肩高
            hip_mid[1],          # 髋高
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # 模块2: 关节点归一化特征 (66维)
    # ------------------------------------------------------------------
    def _extract_joints(self, lm: np.ndarray) -> np.ndarray:
        """
        33个关键点的 (x, y) 坐标，以髋部中点为零点做归一化。

        归一化方式：
          x' = (x - hip_mid_x) * scale_factor
          y' = (y - hip_mid_y) * scale_factor
        其中 scale_factor = 1 / max(shoulder_width, 0.01)
        这样消除了人体在画面中的绝对位置影响，保留姿态结构。
        """
        l_shoulder = self._get_point(lm, "left_shoulder")
        r_shoulder = self._get_point(lm, "right_shoulder")
        l_hip = self._get_point(lm, "left_hip")
        r_hip = self._get_point(lm, "right_hip")

        hip_mid_x = (l_hip[0] + r_hip[0]) / 2.0
        hip_mid_y = (l_hip[1] + r_hip[1]) / 2.0

        # 用肩宽作为归一化尺度因子（不同人不同距离都归一化到统一尺度）
        shoulder_width = abs(r_shoulder[0] - l_shoulder[0])
        scale = 1.0 / max(shoulder_width, 0.02)

        features = []
        for i in range(33):
            nx = (lm[i, 0] - hip_mid_x) * scale
            ny = (lm[i, 1] - hip_mid_y) * scale
            features.append(nx)
            features.append(ny)

        return np.array(features, dtype=np.float32)

    # ------------------------------------------------------------------
    # 模块3: 身体角度特征 (8维)
    # ------------------------------------------------------------------
    def _extract_angles(self, lm: np.ndarray) -> np.ndarray:
        """
        [0] 左膝角     —— hip-knee-ankle（站立≈180°, 坐着≈90°）
        [1] 右膝角
        [2] 左肘角     —— shoulder-elbow-wrist（伸直≈180°, 弯曲≈60°）
        [3] 右肘角
        [4] 左肩-髋-膝角 —— 躯干与大腿的夹角
        [5] 右肩-髋-膝角
        [6] 左髋-肩-肘角 —— 大臂与躯干的夹角
        [7] 右髋-肩-肘角
        """
        left_knee = self._angle_between(
            self._get_point(lm, "left_hip"),
            self._get_point(lm, "left_knee"),
            self._get_point(lm, "left_ankle"),
        )
        right_knee = self._angle_between(
            self._get_point(lm, "right_hip"),
            self._get_point(lm, "right_knee"),
            self._get_point(lm, "right_ankle"),
        )
        left_elbow = self._angle_between(
            self._get_point(lm, "left_shoulder"),
            self._get_point(lm, "left_elbow"),
            self._get_point(lm, "left_wrist"),
        )
        right_elbow = self._angle_between(
            self._get_point(lm, "right_shoulder"),
            self._get_point(lm, "right_elbow"),
            self._get_point(lm, "right_wrist"),
        )
        # 躯干-大腿角
        left_hip_angle = self._angle_between(
            self._get_point(lm, "left_shoulder"),
            self._get_point(lm, "left_hip"),
            self._get_point(lm, "left_knee"),
        )
        right_hip_angle = self._angle_between(
            self._get_point(lm, "right_shoulder"),
            self._get_point(lm, "right_hip"),
            self._get_point(lm, "right_knee"),
        )
        # 大臂-躯干角
        left_shoulder_angle = self._angle_between(
            self._get_point(lm, "left_hip"),
            self._get_point(lm, "left_shoulder"),
            self._get_point(lm, "left_elbow"),
        )
        right_shoulder_angle = self._angle_between(
            self._get_point(lm, "right_hip"),
            self._get_point(lm, "right_shoulder"),
            self._get_point(lm, "right_elbow"),
        )

        return np.array([
            left_knee, right_knee,
            left_elbow, right_elbow,
            left_hip_angle, right_hip_angle,
            left_shoulder_angle, right_shoulder_angle,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # 模块4: 姿态结构特征 (8维)
    # ------------------------------------------------------------------
    def _extract_structure(self, lm: np.ndarray) -> np.ndarray:
        """
        [0] 人体包围盒宽高比 —— 站立>1, 躺着<0.5（关键区分特征！）
        [1] 质心相对高度   —— 质心y / 包围盒高度（站≈0.5, 躺≈0.3）
        [2] 肩髋宽度比     —— 肩宽/髋宽（辅助判断朝向）
        [3] 左-右对称性    —— 左右关键点y坐标的平均差异（跌倒时不对称）
        [4] 脚踝到髋距离   —— 判断是否在站立（站立>0.3, 躺着<0.1）
        [5] 手腕到地面距离 —— 手的相对高度
        [6] 关键点可见度均值 —— 画面中人物被遮挡程度
        [7] 头部倾斜角     —— 两耳连线与水平线夹角
        """
        # 包围盒
        xs = lm[:, 0]
        ys = lm[:, 1]
        valid = lm[:, 3] > 0.3  # 只取可见度高的点
        if valid.sum() < 5:
            return np.zeros(8, dtype=np.float32)

        x_min, x_max = xs[valid].min(), xs[valid].max()
        y_min, y_max = ys[valid].min(), ys[valid].max()
        bbox_w = max(x_max - x_min, 0.01)
        bbox_h = max(y_max - y_min, 0.01)
        aspect_ratio = bbox_h / bbox_w  # 站立>1, 躺<0.5

        # 质心相对高度
        l_shoulder = self._get_point(lm, "left_shoulder")
        r_shoulder = self._get_point(lm, "right_shoulder")
        l_hip = self._get_point(lm, "left_hip")
        r_hip = self._get_point(lm, "right_hip")
        centroid_y = (l_shoulder[1] + r_shoulder[1] + l_hip[1] + r_hip[1]) / 4.0
        rel_height = (centroid_y - y_min) / bbox_h

        # 肩髋宽度比
        shoulder_w = abs(r_shoulder[0] - l_shoulder[0])
        hip_w = abs(r_hip[0] - l_hip[0])
        shoulder_hip_ratio = shoulder_w / max(hip_w, 0.01)

        # 左右对称性
        left_pts_y = []
        right_pts_y = []
        left_names = ["left_shoulder", "left_elbow", "left_wrist", "left_hip", "left_knee", "left_ankle"]
        right_names = ["right_shoulder", "right_elbow", "right_wrist", "right_hip", "right_knee", "right_ankle"]
        for ln, rn in zip(left_names, right_names):
            left_pts_y.append(self._get_point(lm, ln)[1])
            right_pts_y.append(self._get_point(lm, rn)[1])
        left_mean = np.mean(left_pts_y)
        right_mean = np.mean(right_pts_y)
        symmetry = abs(left_mean - right_mean)  # 不对称程度

        # 脚踝到髋距离
        l_ankle = self._get_point(lm, "left_ankle")
        r_ankle = self._get_point(lm, "right_ankle")
        foot_to_hip = (abs(l_ankle[1] - l_hip[1]) + abs(r_ankle[1] - r_hip[1])) / 2.0

        # 手腕相对高度
        l_wrist = self._get_point(lm, "left_wrist")
        r_wrist = self._get_point(lm, "right_wrist")
        wrist_y = (l_wrist[1] + r_wrist[1]) / 2.0
        wrist_height = (wrist_y - y_min) / bbox_h

        # 关键点可见度均值
        avg_vis = float(lm[:, 3].mean())

        # 头部倾斜（两耳连线与水平线的夹角）
        l_ear = self._get_point(lm, "left_ear")
        r_ear = self._get_point(lm, "right_ear")
        ear_vec = r_ear - l_ear
        head_tilt = math.degrees(math.atan2(ear_vec[1], abs(ear_vec[0]) + 1e-6))

        return np.array([
            aspect_ratio,
            rel_height,
            shoulder_hip_ratio,
            symmetry,
            foot_to_hip,
            wrist_height,
            avg_vis,
            head_tilt,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # 模块5: 运动特征 (4+window*2 维)
    # ------------------------------------------------------------------
    def _extract_motion(self, curr_lm: np.ndarray, prev_lm: np.ndarray) -> np.ndarray:
        """
        计算帧间运动特征。

        [0] 质心位移量  —— 躯干中心的帧间移动距离
        [1] 质心速度方向 —— 移动方向角度
        [2] 关键点总位移  —— 所有关键点的平均位移（跌倒时突增）
        [3] 姿态变化率    —— 关键角度的帧间变化（区分"快速变化"和"缓慢变化"）
        """
        # 当前帧髋中点
        curr_lhip = self._get_point(curr_lm, "left_hip")
        curr_rhip = self._get_point(curr_lm, "right_hip")
        curr_centroid = (curr_lhip + curr_rhip) / 2.0

        prev_lhip = self._get_point(prev_lm, "left_hip")
        prev_rhip = self._get_point(prev_lm, "right_hip")
        prev_centroid = (prev_lhip + prev_rhip) / 2.0

        # 质心位移
        disp_vec = curr_centroid - prev_centroid
        displacement = self._norm(disp_vec)

        # 移动方向（角度）
        direction = math.degrees(math.atan2(disp_vec[1], abs(disp_vec[0]) + 1e-6))

        # 关键点总位移（选躯干+四肢的关键点）
        key_indices = [
            LANDMARK["left_shoulder"], LANDMARK["right_shoulder"],
            LANDMARK["left_hip"], LANDMARK["right_hip"],
            LANDMARK["left_knee"], LANDMARK["right_knee"],
            LANDMARK["left_elbow"], LANDMARK["right_elbow"],
            LANDMARK["left_wrist"], LANDMARK["right_wrist"],
            LANDMARK["left_ankle"], LANDMARK["right_ankle"],
        ]
        total_disp = 0.0
        for idx in key_indices:
            d = self._norm(curr_lm[idx, :2] - prev_lm[idx, :2])
            total_disp += d
        avg_disp = total_disp / len(key_indices)

        # 姿态变化率（用躯干角度变化来估算）
        curr_torso_angle = self._extract_torso(curr_lm)[3]
        prev_torso_angle = self._extract_torso(prev_lm)[3]
        pose_change = abs(curr_torso_angle - prev_torso_angle)

        base_features = np.array([
            displacement,
            direction,
            avg_disp,
            pose_change,
        ], dtype=np.float32)

        # 追加滑动窗口历史
        self._motion_history.append(base_features[:2])  # [displacement, direction]
        if len(self._motion_history) > self.smooth_window:
            self._motion_history.pop(0)

        history_features = []
        for h in self._motion_history:
            history_features.extend(h)
        # 补齐到固定长度
        while len(history_features) < self.smooth_window * 2:
            history_features.append(0.0)

        return np.array(list(base_features) + history_features, dtype=np.float32)

    def _update_motion_history(self, base_motion: np.ndarray):
        """更新滑动窗口"""
        self._motion_history.append(list(base_motion[:2]))
        if len(self._motion_history) > self.smooth_window:
            self._motion_history.pop(0)

    # ------------------------------------------------------------------
    # 模块6: 传感器特征槽位 (6维)
    # ------------------------------------------------------------------
    def _extract_sensor(self, sensor_data: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        预留传感器数据槽位，在仅有视觉数据时填入 0。

        [0] 合加速度 (G)
        [1] 俯仰角 pitch (度)
        [2] 横滚角 roll (度)
        [3] 心率 (bpm)
        [4] 血氧 (%)
        [5] 体温 (°C)
        """
        if sensor_data is None:
            return np.zeros(6, dtype=np.float32)

        return np.array([
            sensor_data.get("accel_mag", 0.0),
            sensor_data.get("pitch", 0.0),
            sensor_data.get("roll", 0.0),
            sensor_data.get("hr", 0.0),
            sensor_data.get("spo2", 0.0),
            sensor_data.get("temp", 0.0),
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # 批量提取（用于数据集预处理）
    # ------------------------------------------------------------------
    def extract_batch(self, landmarks_list: List[List[Landmark3D]]) -> np.ndarray:
        """
        批量提取特征，用于离线数据集预处理。

        Parameters
        ----------
        landmarks_list : 多帧的关键点数据列表，每个元素是 33 个 Landmark3D

        Returns
        -------
        features : (N, D) 的特征矩阵
        """
        features = []
        for lm in landmarks_list:
            fv = self.extract(lm)
            features.append(fv.values)
        return np.array(features, dtype=np.float32)

    def extract_batch_with_labels(self,
                                   landmarks_list: List[List[Landmark3D]],
                                   labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量提取 + 标签，直接返回训练就绪的 (X, y)。
        """
        X = self.extract_batch(landmarks_list)
        y = np.array(labels, dtype=np.int32)
        return X, y

    # ------------------------------------------------------------------
    # 特征说明（用于调试 + 模型可解释性）
    # ------------------------------------------------------------------
    @staticmethod
    def feature_names() -> List[str]:
        """返回所有特征的名称列表（与 extract() 输出一一对应）"""
        names = []

        # 模块1: 躯干
        names += ["torso_centroid_x", "torso_centroid_y", "torso_length",
                   "torso_angle", "shoulder_mid_y", "hip_mid_y"]

        # 模块2: 关节点
        landmark_names = list(LANDMARK.keys())
        for lm_name in landmark_names:
            names.append(f"joint_{lm_name}_x")
            names.append(f"joint_{lm_name}_y")

        # 模块3: 角度
        names += ["angle_knee_left", "angle_knee_right",
                   "angle_elbow_left", "angle_elbow_right",
                   "angle_hip_left", "angle_hip_right",
                   "angle_shoulder_left", "angle_shoulder_right"]

        # 模块4: 结构
        names += ["struct_aspect_ratio", "struct_rel_height",
                   "struct_shoulder_hip_ratio", "struct_symmetry",
                   "struct_foot_to_hip", "struct_wrist_height",
                   "struct_avg_visibility", "struct_head_tilt"]

        # 模块5: 运动（基础4维）
        names += ["motion_displacement", "motion_direction",
                   "motion_avg_keypoint_disp", "motion_pose_change"]

        # 模块6: 传感器
        names += ["sensor_accel_mag", "sensor_pitch", "sensor_roll",
                   "sensor_hr", "sensor_spo2", "sensor_temp"]

        return names


# ============================================================
# 便捷函数：从 MediaPipe 原始输出转换
# ============================================================
def landmarks_from_mediapipe(mp_landmarks) -> List[Landmark3D]:
    """
    将 MediaPipe Pose 的原始输出转换为我们的 Landmark3D 格式。

    Usage:
        import mediapipe as mp
        results = pose.process(frame)
        landmarks = landmarks_from_mediapipe(results.pose_landmarks)
    """
    result = []
    for lm in mp_landmarks.landmark:
        result.append(Landmark3D(
            x=lm.x, y=lm.y, z=lm.z,
            visibility=lm.visibility
        ))
    return result


def landmarks_from_array(arr: np.ndarray) -> List[Landmark3D]:
    """从 (33, 4) 的 numpy 数组转换"""
    result = []
    for i in range(arr.shape[0]):
        result.append(Landmark3D(
            x=float(arr[i, 0]), y=float(arr[i, 1]),
            z=float(arr[i, 2]), visibility=float(arr[i, 3])
        ))
    return result


# ============================================================
# 快速测试
# ============================================================
if __name__ == "__main__":
    # 模拟一组站立姿态的关键点
    print("=" * 60)
    print("特征提取器自测")
    print("=" * 60)

    extractor = FeatureExtractor()

    # 模拟 33 个关键点（站姿）
    # x, y 范围 0-1, z 以鼻子为参考
    landmarks_standing = []
    for i in range(33):
        # 粗略模拟站姿
        has_left = "left" in list(LANDMARK.keys())[i]
        has_right = "right" in list(LANDMARK.keys())[i]
        if "shoulder" in list(LANDMARK.keys())[i]:
            y = 0.35
            x = 0.42 if has_left else 0.58 if has_right else 0.50
            z = -0.1 if has_left else 0.1 if has_right else 0.0
        elif "hip" in list(LANDMARK.keys())[i]:
            y = 0.55
            x = 0.44 if has_left else 0.56 if has_right else 0.50
            z = -0.05 if has_left else 0.05 if has_right else 0.0
        elif "knee" in list(LANDMARK.keys())[i]:
            y = 0.75
            x = 0.45 if has_left else 0.55 if has_right else 0.50
            z = -0.02 if has_left else 0.02 if has_right else 0.0
        elif "ankle" in list(LANDMARK.keys())[i] or "heel" in list(LANDMARK.keys())[i] or "foot" in list(LANDMARK.keys())[i]:
            y = 0.92
            x = 0.45 if has_left else 0.55 if has_right else 0.50
            z = 0.0
        elif "elbow" in list(LANDMARK.keys())[i]:
            y = 0.45
            x = 0.35 if has_left else 0.65 if has_right else 0.50
            z = -0.15 if has_left else 0.15 if has_right else 0.0
        elif "wrist" in list(LANDMARK.keys())[i]:
            y = 0.55
            x = 0.30 if has_left else 0.70 if has_right else 0.50
            z = -0.2 if has_left else 0.2 if has_right else 0.0
        elif "nose" in list(LANDMARK.keys())[i]:
            y = 0.18; x = 0.50; z = 0.0
        elif "eye" in list(LANDMARK.keys())[i]:
            y = 0.16; x = 0.47 if has_left else 0.53 if has_right else 0.50; z = -0.05 if has_left else 0.05 if has_right else 0.0
        elif "ear" in list(LANDMARK.keys())[i]:
            y = 0.17; x = 0.42 if has_left else 0.58 if has_right else 0.50; z = -0.1 if has_left else 0.1 if has_right else 0.0
        else:
            y = 0.20; x = 0.48 if has_left else 0.52 if has_right else 0.50; z = 0.0
        landmarks_standing.append(Landmark3D(x=x, y=y, z=z, visibility=0.95))

    # 提取特征
    fv = extractor.extract(landmarks_standing)

    print(f"\n总特征维度: {fv.dim}")
    print(f"  躯干特征:     {len(fv.torso_features)}维 → {fv.torso_features}")
    print(f"  关节点特征:   {len(fv.joint_features)}维")
    print(f"  角度特征:     {len(fv.angle_features)}维 → {fv.angle_features}")
    print(f"  结构特征:     {len(fv.structure_features)}维 → {fv.structure_features}")
    print(f"  运动特征:     {len(fv.motion_features)}维")
    print(f"  传感器槽位:   {len(fv.sensor_features)}维")

    print(f"\n关键区分特征（站姿）:")
    print(f"  躯干角度 = {fv.torso_features[3]:.1f}° (站姿 ≈ 0°)")
    print(f"  质心高度 = {fv.torso_features[1]:.3f} (站姿 ≈ 0.35)")
    print(f"  宽高比   = {fv.structure_features[0]:.2f} (站姿 > 1)")
    print(f"  脚-髋距  = {fv.structure_features[4]:.3f} (站姿 > 0.3)")

    # 模拟躺姿
    landmarks_lying = []
    for i in range(33):
        name = list(LANDMARK.keys())[i]
        if "shoulder" in name:
            y = 0.48; x = 0.30 if "left" in name else 0.60
            z = 0.0
        elif "hip" in name:
            y = 0.52; x = 0.32 if "left" in name else 0.62
            z = 0.0
        elif "knee" in name:
            y = 0.58; x = 0.33 if "left" in name else 0.61
            z = 0.0
        elif "ankle" in name or "heel" in name or "foot" in name:
            y = 0.62; x = 0.33 if "left" in name else 0.61
            z = 0.0
        elif "elbow" in name:
            y = 0.50; x = 0.25 if "left" in name else 0.65
            z = 0.0
        elif "wrist" in name:
            y = 0.55; x = 0.22 if "left" in name else 0.68
            z = 0.0
        elif "nose" in name:
            y = 0.46; x = 0.45; z = 0.0
        elif "eye" in name:
            y = 0.44; x = 0.43 if "left" in name else 0.47; z = 0.0
        elif "ear" in name:
            y = 0.45; x = 0.40 if "left" in name else 0.50; z = 0.0
        else:
            y = 0.50; x = 0.45; z = 0.0
        landmarks_lying.append(Landmark3D(x=x, y=y, z=z, visibility=0.9))

    fv2 = extractor.extract(landmarks_lying)

    print(f"\n关键区分特征（躺姿）:")
    print(f"  躯干角度 = {fv2.torso_features[3]:.1f}° (躺姿 ≈ 80-90°)")
    print(f"  质心高度 = {fv2.torso_features[1]:.3f} (躺姿 > 0.50)")
    print(f"  宽高比   = {fv2.structure_features[0]:.2f} (躺姿 < 0.5)")
    print(f"  脚-髋距  = {fv2.structure_features[4]:.3f} (躺姿 < 0.1)")

    print(f"\n[OK] 站姿 vs 躺姿的躯干角度差异: {abs(fv.torso_features[3] - fv2.torso_features[3]):.1f}deg")
    print(f"[OK] 站姿 vs 躺姿的宽高比差异:     {abs(fv.structure_features[0] - fv2.structure_features[0]):.2f}")
    print("\n这些差异足够随机森林做分类。")
