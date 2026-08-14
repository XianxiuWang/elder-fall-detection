# 护龄——第一步：Windows 环境搭建 + MediaPipe 调通

## 本教程目标

在你的 Windows 笔记本上完成以下操作：
1. ✅ 安装 Python + 必要库
2. ✅ 插上 USB 摄像头并验证能工作
3. ✅ 跑通 MediaPipe Pose，实时看到自己身体的骨架
4. ✅ 保存人体关键点坐标数据（为后面训练模型准备）

**预计用时：30分钟-1小时**

---

## 一、安装 Python

### 1.1 检查是否已有 Python

按 `Win + R`，输入 `cmd` 回车，在命令行里输入：

```cmd
python --version
```

如果显示类似 `Python 3.10.x` 或 `3.11.x` → 跳过安装，直接到第二步。

如果显示 "未找到命令" → 跟着下面装：

### 1.2 下载安装 Python

1. 打开浏览器，访问 https://www.python.org/downloads/
2. 下载 **Python 3.10.x**（推荐3.10，兼容性最好，不要装3.13）
3. 安装时**务必勾选** ☑️ **"Add Python to PATH"**
4. 一路点 Next 安装完成

### 1.3 验证安装

重新打开一个 CMD 窗口：

```cmd
python --version
pip --version
```

两条都能看到版本号 → 成功。

---

## 二、安装必要库

### 2.1 用 pip 一键安装

打开 CMD，运行：

```cmd
pip install opencv-python mediapipe numpy flask flask-socketio websocket-client
```

这会安装你项目要用到的一整套库：
- **opencv-python** — 摄像头、图像处理
- **mediapipe** — 人体姿态关键点提取（Google出品，你要用的核心库）
- **numpy** — 数值计算
- **flask** — Web服务器（后面做前端用）
- **flask-socketio** — 实时推流

**安装过程可能碰到的情况：**
- 下载慢 → 加个国内源：`pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple`
- mediapipe 报错 → 确保 Python 版本在 3.8-3.11 之间（3.12/3.13 可能不兼容）
- 部分依赖安装失败 → 多看报错信息，一般是缺 C++ 编译工具，在评论区截图给我

### 2.2 验证安装

```cmd
python -c "import cv2; import mediapipe; print('✅ 成功')"
```

输出 `✅ 成功` → 装好了。

---

## 三、插上摄像头

### 3.1 检查摄像头

笔记本自带的摄像头或者 USB 外接摄像头都可以。

如果是外接的，插上 USB 口。

### 3.2 测试摄像头是否工作

创建一个测试文件 `test_cam.py`：

```python
import cv2

cap = cv2.VideoCapture(0)  # 0 = 第一个摄像头

if not cap.isOpened():
    print("❌ 摄像头没打开！检查是否被占用或索引不对")
    exit()

print("✅ 摄像头打开成功！按 Q 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 显示画面
    cv2.imshow("Camera Test", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

运行：

```cmd
python test_cam.py
```

能看到摄像头画面 → 成功。

如果报错或者黑屏：
- 检查摄像头是不是被别的软件占用（关掉微信/腾讯会议/QQ）
- 如果是外接摄像头，试试把 `0` 改成 `1`
- 如果是笔记本自带摄像头，确认摄像头盖板是打开的

**这一步过了，恭喜你，最难的部分已经搞定了。**

---

## 四、跑通 MediaPipe Pose——看到你的骨架

这是最令人兴奋的一步。创建一个文件 `pose_demo.py`：

```python
import cv2
import mediapipe as mp

# 初始化 MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 打开摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ 摄像头没打开")
    exit()

print("✅ 摄像头已打开，开始姿势检测...")
print("按 Q 退出")

# 创建 Pose 对象
# static_image_mode=False → 视频模式，连续检测
# model_complexity=1 → 中等精度（可选0/1/2，0最快，2最准）
# min_detection_confidence=0.5 → 检测置信度阈值
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("❌ 读不到摄像头帧")
            break
        
        # 转成 RGB（MediaPipe 需要 RGB）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 跑推理
        results = pose.process(frame_rgb)
        
        # 如果检测到人体
        if results.pose_landmarks:
            # 在原图上画出骨架
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # 显示画面
        cv2.imshow("护龄 v2 - MediaPipe Pose", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

运行：

```cmd
python pose_demo.py
```

**如果一切正常，你会看到：**

![示意图]
摄像头画面里的你，身上有一条彩色的骨架线条，这就是 **MediaPipe Pose 提取的人体33个关键点**。

在摄像头前走动、坐下、举手、弯腰——骨架会跟着你的动作实时变化。

---

## 五、进阶：看看关键点数据到底是什么

MediaPipe 在你的身体上提取了 **33个关键点**，每个点有 (x, y, z, visibility) 四个值。让我们把它打印出来看看：

创建一个 `pose_data.py`：

```python
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

# 人体33个关键点的名称
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    smooth_landmarks=True
) as pose:
    
    print("=" * 60)
    print("按 Q 退出，每5帧打印一次关键点数据")
    print("=" * 60)
    
    frame_count = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            frame_count += 1
            
            # 每5帧打印一次
            if frame_count % 5 == 0:
                landmarks = results.pose_landmarks.landmark
                
                print(f"\n--- 第 {frame_count} 帧 ---")
                print(f"{'关键点':20s} {'x':8s} {'y':8s} {'z':8s} {'可视度':6s}")
                print("-" * 50)
                
                # 打印几个关键点给你看（肩、髋、膝、踝）
                key_points = ["left_shoulder", "right_shoulder", 
                              "left_hip", "right_hip",
                              "left_knee", "right_knee",
                              "left_ankle", "right_ankle"]
                
                for name in key_points:
                    idx = mp_pose.PoseLandmark[name.upper()].value
                    lm = landmarks[idx]
                    print(f"{name:20s} {lm.x:8.3f} {lm.y:8.3f} {lm.z:8.3f} {lm.visibility:6.2f}")
        
        # 显示画面
        cv2.imshow("护龄 - 关键点数据", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

运行：

```cmd
python pose_data.py
```

你会看到类似这样的输出：

```
--- 第 5 帧 ---
关键点                x        y        z        可视度
--------------------------------------------------
left_shoulder      0.421    0.312    0.002   0.99
right_shoulder     0.589    0.308   -0.001   0.99
left_hip           0.438    0.552    0.005   0.98
right_hip          0.572    0.548   -0.003   0.98
left_knee          0.445    0.748    0.012   0.95
right_knee         0.560    0.742   -0.008   0.95
left_ankle         0.452    0.902    0.018   0.90
right_ankle        0.553    0.895   -0.010   0.90
```

**x, y 是归一化坐标（0-1，相对于画面宽高）**
**z 是深度（相对于鼻子的深度）**
**visibility 是置信度（0-1，越大越确定）**

**现在你能看到你自己的实时骨架数据了。**

---

## 六、核心观察

**站起来走动时**：
- 肩膀高度（y值）在 0.3 左右
- 膝盖高度在 0.7 左右
- 脚踝在 0.9 左右

**坐下时**：
- 肩膀高度降到 0.4-0.5
- 膝盖降到 0.6 左右
- 躯干和地面夹角变小

**躺下时**：
- 所有点的 y 值变化范围缩小
- 肩膀和髋部的 x 范围变宽

**跌倒时（可以模拟）**：
- 关键点 y 值突然变化
- 质心高度猛降
- 肩髋连线方向异常

这就是你后面**训练模型**的依据——**不同状态下，这33个关键点的坐标分布是不一样的。**

---

## 七、常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `ImportError: No module named cv2` | OpenCV没装好 | 重跑 `pip install opencv-python` |
| `ImportError: No module named mediapipe` | MediaPipe没装好 | 检查Python版本 ≤3.11，重装 |
| 摄像头黑屏 | 被其他软件占用 | 关掉微信/腾讯会议/QQ的视频功能 |
| 画面卡，骨架跟不上 | 笔记本性能不够 | `model_complexity=0`（最快但精度稍低） |
| 骨架抖动 | 置信度低 | 提高 `min_detection_confidence` 到 0.7 |

---

## 八、完成！你的下一步

恭喜！你现在已经：
- [x] 装好了Python开发环境
- [x] 装好了所有要用到的库
- [x] 让摄像头工作了
- [x] 看到了MediaPipe的实时骨架
- [x] 看到了人体关键点的原始数据

**跑通了告诉我一声**，我接下来给你出第二步教程：
> **《第二步：用开源数据集训练你的第一个状态分类模型》**

你会学到：
1. 找一个现成的跌倒检测数据集
2. 提取关键点特征
3. 用随机森林训练6类状态分类器
4. 跑通模型推理，实时判断"站着""坐着""躺着""跌倒"
