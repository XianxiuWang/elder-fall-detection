/**
 * huling_deploy.h — 护龄模型部署统一头文件
 *
 * 包含特征提取 + RandomForest 推理的完整 C 接口。
 * 目标平台: Milk-V Duo S (SG2000, ARM64 or RISC-V)
 *
 * 特征维度: 98 (与训练模型一致)
 * 类别: 0=walking, 1=sitting, 2=lying, 3=long_sit, 4=abnormal, 5=fall
 */
#ifndef HULING_DEPLOY_H
#define HULING_DEPLOY_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* ================================================================
 * 常量
 * ================================================================ */

#define HULING_N_KEYPOINTS  33      /* MediaPipe Pose 关键点数 */
#define HULING_N_FEATURES   98      /* 特征维度 */
#define HULING_N_CLASSES    6       /* 分类数 */
#define HULING_SMOOTH_WINDOW 0      /* 运动特征滑动窗口（匹配 Python smooth_window=0，98-dim 模型） */

/* 状态名称 */
static const char *const HULING_CLASS_NAMES[HULING_N_CLASSES] = {
    "walking", "sitting", "lying", "long_sit", "abnormal", "fall"
};

/* ================================================================
 * 数据结构
 * ================================================================ */

/* 单个关键点 (x, y, z 归一化到 0-1, visibility 0-1) */
typedef struct {
    float x, y, z;
    float visibility;
} Keypoint3D;

/* 一帧 33 个关键点 */
typedef struct {
    Keypoint3D kp[HULING_N_KEYPOINTS];
} PoseLandmarks;

/* 特征向量 (98 维，按模块顺序排列) */
typedef struct {
    float f[HULING_N_FEATURES];
} FeatureVector;

/* 推理结果 */
typedef struct {
    int   class_id;                   /* 0-5 */
    const char *class_name;           /* "walking" 等 */
    float confidence;                 /* 分类概率 (0-1) */
    float scores[HULING_N_CLASSES];   /* 各类得分 */
} PosePrediction;

/* ================================================================
 * API
 * ================================================================ */

/**
 * 从 33 个关键点提取 98 维特征（单帧，无运动历史）
 * 用途：离线推理、兼容现有训练数据
 *
 * @param lm    [in]  33 个关键点
 * @param fv    [out] 98 维特征向量
 */
void huling_extract_features(const PoseLandmarks *lm, FeatureVector *fv);

/**
 * 从连续帧提取 98 维特征（含运动特征）
 * 自动维护帧间历史。
 *
 * @param lm    [in]  当前帧关键点
 * @param fv    [out] 98 维特征向量
 */
void huling_extract_features_online(const PoseLandmarks *lm, FeatureVector *fv);

/**
 * 重置在线特征提取器的运动历史
 * 切换视频源 / 重新开始时调用
 */
void huling_extract_reset(void);

/**
 * 分类预测（完整管线：特征提取 + 标准化 + RandomForest）
 *
 * @param lm        [in]  33 个关键点
 * @param result    [out] 预测结果
 */
void huling_predict(const PoseLandmarks *lm, PosePrediction *result);

/**
 * 直接对特征向量分类（跳过特征提取）
 *
 * @param fv        [in]  98 维特征向量
 * @param result    [out] 预测结果
 */
void huling_predict_from_features(const FeatureVector *fv, PosePrediction *result);

/**
 * 初始化（模型加载，TDL-SDK 初始化等 — 平台相关，在此为桩）
 */
void huling_init(void);

#ifdef __cplusplus
}
#endif

#endif /* HULING_DEPLOY_H */
