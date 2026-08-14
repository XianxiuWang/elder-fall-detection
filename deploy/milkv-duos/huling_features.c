/**
 * huling_features.c — 特征提取器 C 实现
 *
 * 精确复现 feature_extractor.py 的数学公式。
 *
 * 特征顺序 (98维):
 *   0-5:   torso (6维)   — 躯干质心、角度
 *   6-71:  joints (66维)  — 33关键点×2 归一化坐标
 *   72-79: angles (8维)   — 膝盖、肘部、髋、肩角度
 *   80-87: structure (8维) — 宽高比、对称性等
 *   88-91: motion (4维)   — 帧间位移（单帧为零）
 *   92-97: sensor (6维)   — 传感器槽位（视觉模式为零）
 */
#include "huling_deploy.h"
#include <math.h>
#include <string.h>

/* ================================================================
 * 数学工具
 * ================================================================ */

static inline float vec_norm(float x, float y, float z) {
    return sqrtf(x*x + y*y + z*z);
}

static inline float vec_norm2(float x, float y) {
    return sqrtf(x*x + y*y);
}

/** 三点夹角 ∠ABC, B 为顶点, 返回角度 (0-180°) */
static float angle_between(const Keypoint3D *a,
                           const Keypoint3D *b,
                           const Keypoint3D *c) {
    float bax = a->x - b->x, bay = a->y - b->y, baz = a->z - b->z;
    float bcx = c->x - b->x, bcy = c->y - b->y, bcz = c->z - b->z;
    float dot = bax*bcx + bay*bcy + baz*bcz;
    float na = vec_norm(bax, bay, baz);
    float nb = vec_norm(bcx, bcy, bcz);
    float prod = na * nb;
    if (prod < 1e-6f) return 0.0f;
    float cos_a = dot / prod;
    if (cos_a > 1.0f) cos_a = 1.0f;
    if (cos_a < -1.0f) cos_a = -1.0f;
    return acosf(cos_a) * 57.295779513f; /* rad to deg */
}

#define KP(lm, idx)  ((lm)->kp[idx])

/* ================================================================
 * 模块1: 躯干基础特征 (6维)
 * ================================================================ */
static void extract_torso(const PoseLandmarks *lm, float *out) {
    /*
     * [0] centroid_x      — 肩+髋的几何中心 x
     * [1] centroid_y      — 肩+髋的几何中心 y
     * [2] torso_length    — 肩中点→髋中点的欧氏距离
     * [3] torso_angle     — 躯干与垂直线(0,1,0)的夹角
     * [4] shoulder_mid_y  — 两肩中点的 y 坐标
     * [5] hip_mid_y       — 两髋中点的 y 坐标
     */
    const Keypoint3D *ls = &KP(lm, 11);  /* left_shoulder  */
    const Keypoint3D *rs = &KP(lm, 12);  /* right_shoulder */
    const Keypoint3D *lh = &KP(lm, 23);  /* left_hip       */
    const Keypoint3D *rh = &KP(lm, 24);  /* right_hip      */

    float sh_mid_x = (ls->x + rs->x) * 0.5f;
    float sh_mid_y = (ls->y + rs->y) * 0.5f;
    float sh_mid_z = (ls->z + rs->z) * 0.5f;

    float hip_mid_x = (lh->x + rh->x) * 0.5f;
    float hip_mid_y = (lh->y + rh->y) * 0.5f;
    float hip_mid_z = (lh->z + rh->z) * 0.5f;

    /* 质心 */
    float centroid_x = (sh_mid_x + hip_mid_x) * 0.5f;
    float centroid_y = (sh_mid_y + hip_mid_y) * 0.5f;

    /* 躯干长度 */
    float tx = hip_mid_x - sh_mid_x;
    float ty = hip_mid_y - sh_mid_y;
    float tz = hip_mid_z - sh_mid_z;
    float torso_len = vec_norm(tx, ty, tz);

    /* 躯干角 (与垂直线 (0,1,0) 的夹角, 向量方向: 肩→髋) */
    float dot = ty; /* tx*0 + ty*1 + tz*0 */
    float torso_angle = 0.0f;
    if (torso_len > 1e-6f) {
        float cos_a = dot / torso_len;
        if (cos_a > 1.0f) cos_a = 1.0f;
        if (cos_a < -1.0f) cos_a = -1.0f;
        torso_angle = acosf(cos_a) * 57.295779513f;
    }

    out[0] = centroid_x;
    out[1] = centroid_y;
    out[2] = torso_len;
    out[3] = torso_angle;
    out[4] = sh_mid_y;
    out[5] = hip_mid_y;
}

/* ================================================================
 * 模块2: 关节点归一化特征 (66维)
 * ================================================================ */
static void extract_joints(const PoseLandmarks *lm, float *out) {
    /*
     * 33 个关键点的 (x, y), 以髋部中点为零点, 肩宽为单位做归一化:
     *   x' = (x - hip_mid_x) / max(shoulder_width, 0.02)
     *   y' = (y - hip_mid_y) / max(shoulder_width, 0.02)
     */
    const Keypoint3D *ls = &KP(lm, 11);
    const Keypoint3D *rs = &KP(lm, 12);
    const Keypoint3D *lh = &KP(lm, 23);
    const Keypoint3D *rh = &KP(lm, 24);

    float hip_mid_x = (lh->x + rh->x) * 0.5f;
    float hip_mid_y = (lh->y + rh->y) * 0.5f;
    float shoulder_w = fabsf(rs->x - ls->x);
    float scale = 1.0f / (shoulder_w > 0.02f ? shoulder_w : 0.02f);

    for (int i = 0; i < 33; i++) {
        *out++ = (KP(lm, i).x - hip_mid_x) * scale;
        *out++ = (KP(lm, i).y - hip_mid_y) * scale;
    }
}

/* ================================================================
 * 模块3: 身体角度特征 (8维)
 * ================================================================ */
static void extract_angles(const PoseLandmarks *lm, float *out) {
    /*
     * [0] left_knee_angle     hip(23)→knee(25)→ankle(27)
     * [1] right_knee_angle    hip(24)→knee(26)→ankle(28)
     * [2] left_elbow_angle    shoulder(11)→elbow(13)→wrist(15)
     * [3] right_elbow_angle   shoulder(12)→elbow(14)→wrist(16)
     * [4] left_hip_angle      shoulder(11)→hip(23)→knee(25)
     * [5] right_hip_angle     shoulder(12)→hip(24)→knee(26)
     * [6] left_shoulder_angle hip(23)→shoulder(11)→elbow(13)
     * [7] right_shoulder_angle hip(24)→shoulder(12)→elbow(14)
     */
    out[0] = angle_between(&KP(lm, 23), &KP(lm, 25), &KP(lm, 27));
    out[1] = angle_between(&KP(lm, 24), &KP(lm, 26), &KP(lm, 28));
    out[2] = angle_between(&KP(lm, 11), &KP(lm, 13), &KP(lm, 15));
    out[3] = angle_between(&KP(lm, 12), &KP(lm, 14), &KP(lm, 16));
    out[4] = angle_between(&KP(lm, 11), &KP(lm, 23), &KP(lm, 25));
    out[5] = angle_between(&KP(lm, 12), &KP(lm, 24), &KP(lm, 26));
    out[6] = angle_between(&KP(lm, 23), &KP(lm, 11), &KP(lm, 13));
    out[7] = angle_between(&KP(lm, 24), &KP(lm, 12), &KP(lm, 14));
}

/* ================================================================
 * 模块4: 姿态结构特征 (8维)
 * ================================================================ */
static void extract_structure(const PoseLandmarks *lm, float *out) {
    /*
     * [0] aspect_ratio        — 包围盒 h/w (站>1, 躺<0.5)
     * [1] rel_height          — 质心相对高度 (centroid_y-y_min)/h
     * [2] shoulder_hip_ratio  — 肩宽 / 髋宽
     * [3] symmetry            — |左关键点y均值 - 右关键点y均值|
     * [4] foot_to_hip         — 脚踝到髋的平均y距离
     * [5] wrist_height        — 手腕的相对高度
     * [6] avg_visibility      — 所有关键点可见度均值
     * [7] head_tilt           — 两耳连线与水平线夹角 (度)
     */

    /* 包围盒 (仅可见度>0.3的点) */
    float x_min = 1e10f, x_max = -1e10f;
    float y_min = 1e10f, y_max = -1e10f;
    int valid_cnt = 0;
    for (int i = 0; i < 33; i++) {
        if (KP(lm, i).visibility > 0.3f) {
            if (KP(lm, i).x < x_min) x_min = KP(lm, i).x;
            if (KP(lm, i).x > x_max) x_max = KP(lm, i).x;
            if (KP(lm, i).y < y_min) y_min = KP(lm, i).y;
            if (KP(lm, i).y > y_max) y_max = KP(lm, i).y;
            valid_cnt++;
        }
    }
    if (valid_cnt < 5) {
        memset(out, 0, 8 * sizeof(float));
        return;
    }

    float bbox_w = x_max - x_min;
    float bbox_h = y_max - y_min;
    if (bbox_w < 0.01f) bbox_w = 0.01f;
    if (bbox_h < 0.01f) bbox_h = 0.01f;
    float aspect_ratio = bbox_h / bbox_w;

    /* 质心相对高度 */
    float sh_mid_y = (KP(lm, 11).y + KP(lm, 12).y) * 0.5f;
    float hip_mid_y = (KP(lm, 23).y + KP(lm, 24).y) * 0.5f;
    float centroid_y = (sh_mid_y + hip_mid_y) * 0.5f;
    float rel_height = (centroid_y - y_min) / bbox_h;

    /* 肩髋宽度比 */
    float shoulder_w = fabsf(KP(lm, 12).x - KP(lm, 11).x);
    float hip_w = fabsf(KP(lm, 24).x - KP(lm, 23).x);
    float sh_hip_ratio = shoulder_w / (hip_w > 0.01f ? hip_w : 0.01f);

    /* 左右对称性 */
    float left_y_sum = 0, right_y_sum = 0;
    int left_idx[] = {11, 13, 15, 23, 25, 27};
    int right_idx[] = {12, 14, 16, 24, 26, 28};
    for (int i = 0; i < 6; i++) {
        left_y_sum += KP(lm, left_idx[i]).y;
        right_y_sum += KP(lm, right_idx[i]).y;
    }
    float left_mean = left_y_sum / 6.0f;
    float right_mean = right_y_sum / 6.0f;
    float symmetry = fabsf(left_mean - right_mean);

    /* 脚踝到髋距离 */
    float foot_to_hip = (fabsf(KP(lm, 27).y - KP(lm, 23).y)
                       + fabsf(KP(lm, 28).y - KP(lm, 24).y)) * 0.5f;

    /* 手腕相对高度 */
    float wrist_y = (KP(lm, 15).y + KP(lm, 16).y) * 0.5f;
    float wrist_height = (wrist_y - y_min) / bbox_h;

    /* 关键点可见度均值 */
    float vis_sum = 0;
    for (int i = 0; i < 33; i++) vis_sum += KP(lm, i).visibility;
    float avg_vis = vis_sum / 33.0f;

    /* 头部倾斜角 (两耳连线 vs 水平线) */
    float ear_dx = KP(lm, 8).x - KP(lm, 7).x;  /* right_ear(8) - left_ear(7) */
    float ear_dy = KP(lm, 8).y - KP(lm, 7).y;
    float head_tilt = atan2f(ear_dy, fabsf(ear_dx) + 1e-6f) * 57.295779513f;

    out[0] = aspect_ratio;
    out[1] = rel_height;
    out[2] = sh_hip_ratio;
    out[3] = symmetry;
    out[4] = foot_to_hip;
    out[5] = wrist_height;
    out[6] = avg_vis;
    out[7] = head_tilt;
}

/* ================================================================
 * 模块5: 运动特征 (4+窗口, 在线提取器使用)
 * ================================================================ */
/* 滑动窗口历史 */
static float motion_history[HULING_SMOOTH_WINDOW][2]; /* [displacement, direction] */
static int motion_history_cnt = 0;
static PoseLandmarks prev_landmarks;
static int has_prev = 0;

static void extract_motion(const PoseLandmarks *curr, float *out) {
    /*
     * [0] displacement     — 髋部质心帧间位移
     * [1] direction        — 移动方向角 (度)
     * [2] avg_keypoint_disp— 12 个关键点的平均位移
     * [3] pose_change      — 躯干角度的帧间变化
     *
     * 后续 10 维 (2×5窗口): 滑动窗口位移历史
     */
    if (!has_prev) {
        memset(out, 0, (4 + HULING_SMOOTH_WINDOW * 2) * sizeof(float));
        return;
    }

    /* 当前髋中点 */
    float curr_hx = (KP(curr, 23).x + KP(curr, 24).x) * 0.5f;
    float curr_hy = (KP(curr, 23).y + KP(curr, 24).y) * 0.5f;
    /* 前一帧髋中点 */
    float prev_hx = (KP(&prev_landmarks, 23).x + KP(&prev_landmarks, 24).x) * 0.5f;
    float prev_hy = (KP(&prev_landmarks, 23).y + KP(&prev_landmarks, 24).y) * 0.5f;

    /* [0] 位移量 */
    float dx = curr_hx - prev_hx;
    float dy = curr_hy - prev_hy;
    float displacement = vec_norm2(dx, dy);

    /* [1] 方向角 */
    float direction = atan2f(dy, fabsf(dx) + 1e-6f) * 57.295779513f;

    /* [2] 关键点平均位移 */
    int key_idx[] = {11,12, 23,24, 25,26, 13,14, 15,16, 27,28};
    float total_disp = 0;
    for (int i = 0; i < 12; i++) {
        int k = key_idx[i];
        float d = vec_norm2(KP(curr, k).x - KP(&prev_landmarks, k).x,
                            KP(curr, k).y - KP(&prev_landmarks, k).y);
        total_disp += d;
    }
    float avg_keypoint_disp = total_disp / 12.0f;

    /* [3] 姿态变化率 — 躯干角帧间变化（3D，与 Python _extract_torso 一致） */
    /* 当前躯干角（完整3D） */
    float c_shx = (KP(curr, 11).x + KP(curr, 12).x) * 0.5f;
    float c_shy = (KP(curr, 11).y + KP(curr, 12).y) * 0.5f;
    float c_shz = (KP(curr, 11).z + KP(curr, 12).z) * 0.5f;
    float c_hipx = (KP(curr, 23).x + KP(curr, 24).x) * 0.5f;
    float c_hipy = (KP(curr, 23).y + KP(curr, 24).y) * 0.5f;
    float c_hipz = (KP(curr, 23).z + KP(curr, 24).z) * 0.5f;
    float c_tx = c_hipx - c_shx, c_ty = c_hipy - c_shy;
    float c_tz = c_hipz - c_shz;
    float c_len = vec_norm(c_tx, c_ty, c_tz);
    float c_angle = (c_len > 1e-6f)
        ? acosf(c_ty / c_len) * 57.295779513f : 0.0f;

    /* 前一帧躯干角（完整3D） */
    float p_shx = (KP(&prev_landmarks, 11).x + KP(&prev_landmarks, 12).x) * 0.5f;
    float p_shy = (KP(&prev_landmarks, 11).y + KP(&prev_landmarks, 12).y) * 0.5f;
    float p_shz = (KP(&prev_landmarks, 11).z + KP(&prev_landmarks, 12).z) * 0.5f;
    float p_hipx = (KP(&prev_landmarks, 23).x + KP(&prev_landmarks, 24).x) * 0.5f;
    float p_hipy = (KP(&prev_landmarks, 23).y + KP(&prev_landmarks, 24).y) * 0.5f;
    float p_hipz = (KP(&prev_landmarks, 23).z + KP(&prev_landmarks, 24).z) * 0.5f;
    float p_tx = p_hipx - p_shx, p_ty = p_hipy - p_shy;
    float p_tz = p_hipz - p_shz;
    float p_len = vec_norm(p_tx, p_ty, p_tz);
    float p_angle = (p_len > 1e-6f)
        ? acosf(p_ty / p_len) * 57.295779513f : 0.0f;

    float pose_change = fabsf(c_angle - p_angle);

    out[0] = displacement;
    out[1] = direction;
    out[2] = avg_keypoint_disp;
    out[3] = pose_change;

    /* 追加滑动窗口历史 */
    float *hist = out + 4;
    for (int i = 0; i < HULING_SMOOTH_WINDOW; i++) {
        *hist++ = motion_history[i][0]; /* displacement */
        *hist++ = motion_history[i][1]; /* direction     */
    }
}

/* ================================================================
 * 模块6: 传感器特征槽位 (6维)
 * ================================================================ */
static void extract_sensor(float *out) {
    /*
     * [0] accel_mag (G)
     * [1] pitch (度)
     * [2] roll (度)
     * [3] hr (bpm)
     * [4] spo2 (%)
     * [5] temp (°C)
     * 纯视觉模式下全为零
     */
    memset(out, 0, 6 * sizeof(float));
}

/* ================================================================
 * 公共接口实现
 * ================================================================ */

void huling_extract_features(const PoseLandmarks *lm, FeatureVector *fv) {
    float *f = fv->f;
    extract_torso(lm, f);       f += 6;   /* f0-f5   */
    extract_joints(lm, f);      f += 66;  /* f6-f71  */
    extract_angles(lm, f);      f += 8;   /* f72-f79 */
    extract_structure(lm, f);   f += 8;   /* f80-f87 */
    memset(f, 0, 4 * sizeof(float)); f += 4;  /* f88-f91: 单帧无运动 */
    extract_sensor(f);                   /* f92-f97 */
}

void huling_extract_features_online(const PoseLandmarks *lm, FeatureVector *fv) {
    float *f = fv->f;
    extract_torso(lm, f);       f += 6;
    extract_joints(lm, f);      f += 66;
    extract_angles(lm, f);      f += 8;
    extract_structure(lm, f);   f += 8;
    extract_motion(lm, f);      f += 4;  /* base 4 motion features (window=0 matches Python smooth_window=0) */
    extract_sensor(f);

    /* 保存当前帧为下一帧的 prev */
    memcpy(&prev_landmarks, lm, sizeof(PoseLandmarks));
    has_prev = 1;

    /* 更新滑动窗口 */
    /* f[88]=displacement, f[89]=direction */
    if (HULING_SMOOTH_WINDOW > 0) {
        if (motion_history_cnt < HULING_SMOOTH_WINDOW) {
            motion_history[motion_history_cnt][0] = fv->f[88];
            motion_history[motion_history_cnt][1] = fv->f[89];
            motion_history_cnt++;
        } else {
            /* 移动: 丢掉最旧, 追加最新 */
            memmove(motion_history, motion_history + 1,
                    (HULING_SMOOTH_WINDOW - 1) * sizeof(motion_history[0]));
            motion_history[HULING_SMOOTH_WINDOW - 1][0] = fv->f[88];
            motion_history[HULING_SMOOTH_WINDOW - 1][1] = fv->f[89];
        }
    }
}

void huling_extract_reset(void) {
    has_prev = 0;
    motion_history_cnt = 0;
    memset(motion_history, 0, sizeof(motion_history));
    memset(&prev_landmarks, 0, sizeof(prev_landmarks));
}

/* ================================================================
 * 模型推理桩 (实际通过 random_forest.c 实现)
 * ================================================================ */
/* 由 random_forest_wrapper.c 提供 */
extern int predict_pose_class(const float *features);
extern const char* class_name(int cls);

void huling_predict(const PoseLandmarks *lm, PosePrediction *result) {
    FeatureVector fv;
    huling_extract_features(lm, &fv);
    huling_predict_from_features(&fv, result);
}

void huling_predict_from_features(const FeatureVector *fv, PosePrediction *result) {
    result->class_id = predict_pose_class(fv->f);
    result->class_name = class_name(result->class_id);
    /* 注意: predict_pose_class 只返回 class_id, 不返回 scores。
     * 如需 scores, 调用 predict_pose(scores_array, features) 后做 softmax。
     * 此处简化为: 最高票类别置信度 = 1.0 */
    result->confidence = 1.0f;
    for (int i = 0; i < HULING_N_CLASSES; i++)
        result->scores[i] = (i == result->class_id) ? 1.0f : 0.0f;
}

void huling_init(void) {
    /* 平台相关初始化桩 */
    huling_extract_reset();
}
