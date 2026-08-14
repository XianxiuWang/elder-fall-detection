/**
 * test_deploy.c — PC-side C code verification
 *
 * Compares C feature extraction + RandomForest inference against Python reference.
 *
 * Build (MinGW):
 *   gcc -o test_deploy huling_features.c random_forest.c random_forest_wrapper.c test_deploy.c -lm -O0 -Wall
 *
 * Build (MSVC):
 *   cl /Fe:test_deploy.exe huling_features.c random_forest.c random_forest_wrapper.c test_deploy.c
 */
#include "huling_deploy.h"
#include "scaler_params.h"
#include "test_data.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ---- Forward declarations from random_forest_wrapper.c ---- */
extern int predict_pose_class(const float *features);
extern const char* class_name(int cls);

/* We redeclare predict_pose to get raw scores for comparison */
extern void predict_pose(double *scores, const double *features);

#define N_MODEL_CLASSES 3
extern int MODEL_CLASS_MAP[];  /* not used externally, we'll hardcode the map */

static const int g_model_class_map[3] = {0, 1, 5};

/* ================================================================
 * Helper: print a floating-point diff analysis
 * ================================================================ */
static int compare_float(const char *label, float c_val, double py_val,
                         int idx, double *max_diff, int *max_idx,
                         const char **max_label) {
    double diff = fabs((double)c_val - py_val);
    if (diff > *max_diff) {
        *max_diff = diff;
        *max_idx = idx;
        *max_label = label;
    }
    /* Report any diff > 1e-5 (accounting for float vs double precision) */
    if (diff > 1e-5) {
        return 1;
    }
    return 0;
}

static int count_diffs(float *c_arr, const double *py_arr, int n,
                       const char *label, double *max_diff, int *max_idx,
                       const char **max_label) {
    int diffs = 0;
    for (int i = 0; i < n; i++) {
        if (compare_float(label, c_arr[i], py_arr[i], i, max_diff, max_idx, max_label))
            diffs++;
    }
    return diffs;
}

static void print_header(const char *title) {
    printf("\n============================================================\n");
    printf("  %s\n", title);
    printf("============================================================\n");
}

/* ================================================================
 * Test 1: Feature extraction — per-module comparison
 * ================================================================ */
static int test_feature_extraction(void) {
    print_header("MODULE 1-6: Feature Extraction");
    
    int total_errors = 0;
    double max_diff = 0.0;
    int max_idx = 0;
    const char *max_label = "";
    
    for (int c = 0; c < 5; c++) {  /* first 5: standing, sitting, lying, fall, empty */
        printf("\n--- Case %d: %s ---\n", c, test_names[c]);
        
        /* Build landmarks from test data */
        PoseLandmarks lm;
        for (int k = 0; k < 33; k++) {
            lm.kp[k].x = test_landmarks[c][k][0];
            lm.kp[k].y = test_landmarks[c][k][1];
            lm.kp[k].z = test_landmarks[c][k][2];
            lm.kp[k].visibility = test_landmarks[c][k][3];
        }
        
        /* Extract features */
        FeatureVector fv;
        huling_extract_features(&lm, &fv);
        
        /* Compare torso (6 dims) */
        {
            float c_torso[6];
            memcpy(c_torso, fv.f, 6 * sizeof(float));
            int errs = count_diffs(c_torso, py_torso[c], 6, "torso", &max_diff, &max_idx, &max_label);
            if (errs > 0) {
                printf("  torso: %d diffs\n", errs);
                for (int i = 0; i < 6; i++) {
                    double d = fabs((double)c_torso[i] - py_torso[c][i]);
                    if (d > 1e-5)
                        printf("    [%d] C=%.6f  Py=%.6f  diff=%.2e\n", i, c_torso[i], py_torso[c][i], d);
                }
            } else {
                printf("  torso: OK (6/6 match)\n");
            }
            total_errors += errs;
        }
        
        /* Compare joints (66 dims) */
        {
            float c_joints[66];
            memcpy(c_joints, fv.f + 6, 66 * sizeof(float));
            int errs = 0;
            for (int i = 0; i < 66; i++) {
                if (compare_float("joints", c_joints[i], py_joints[c][i], i, &max_diff, &max_idx, &max_label))
                    errs++;
            }
            if (errs > 0) {
                printf("  joints: %d diffs out of 66\n", errs);
                /* Show first 5 diffs */
                int shown = 0;
                for (int i = 0; i < 66 && shown < 5; i++) {
                    double d = fabs((double)c_joints[i] - py_joints[c][i]);
                    if (d > 1e-5) {
                        printf("    [%d] C=%.6f  Py=%.6f  diff=%.2e\n", i, c_joints[i], py_joints[c][i], d);
                        shown++;
                    }
                }
            } else {
                printf("  joints: OK (66/66 match)\n");
            }
            total_errors += errs;
        }
        
        /* Compare angles (8 dims) */
        {
            float c_angles[8];
            memcpy(c_angles, fv.f + 72, 8 * sizeof(float));
            int errs = count_diffs(c_angles, py_angles[c], 8, "angles", &max_diff, &max_idx, &max_label);
            if (errs > 0) {
                printf("  angles: %d diffs\n", errs);
                for (int i = 0; i < 8; i++) {
                    double d = fabs((double)c_angles[i] - py_angles[c][i]);
                    if (d > 1e-5)
                        printf("    [%d] C=%.4f  Py=%.4f  diff=%.2e\n", i, c_angles[i], py_angles[c][i], d);
                }
            } else {
                printf("  angles: OK (8/8 match)\n");
            }
            total_errors += errs;
        }
        
        /* Compare structure (8 dims) */
        {
            float c_struct[8];
            memcpy(c_struct, fv.f + 80, 8 * sizeof(float));
            int errs = count_diffs(c_struct, py_structure[c], 8, "structure", &max_diff, &max_idx, &max_label);
            if (errs > 0) {
                printf("  structure: %d diffs\n", errs);
                for (int i = 0; i < 8; i++) {
                    double d = fabs((double)c_struct[i] - py_structure[c][i]);
                    if (d > 1e-5)
                        printf("    [%d] C=%.6f  Py=%.6f  diff=%.2e\n", i, c_struct[i], py_structure[c][i], d);
                }
            } else {
                printf("  structure: OK (8/8 match)\n");
            }
            total_errors += errs;
        }
        
        /* Compare motion (4 dims, should be 0 for static extraction) */
        {
            float c_motion[4];
            memcpy(c_motion, fv.f + 88, 4 * sizeof(float));
            int errs = count_diffs(c_motion, py_motion[c], 4, "motion", &max_diff, &max_idx, &max_label);
            if (errs > 0) {
                printf("  motion: %d diffs\n", errs);
                for (int i = 0; i < 4; i++) {
                    double d = fabs((double)c_motion[i] - py_motion[c][i]);
                    if (d > 1e-5)
                        printf("    [%d] C=%.6f  Py=%.6f  diff=%.2e\n", i, c_motion[i], py_motion[c][i], d);
                }
            } else {
                printf("  motion: OK (4/4 match, all zeros)\n");
            }
            total_errors += errs;
        }
        
        /* Compare sensor (6 dims, should be 0) */
        {
            float c_sensor[6];
            memcpy(c_sensor, fv.f + 92, 6 * sizeof(float));
            int errs = count_diffs(c_sensor, py_sensor[c], 6, "sensor", &max_diff, &max_idx, &max_label);
            if (errs > 0) {
                printf("  sensor: %d diffs\n", errs);
            } else {
                printf("  sensor: OK (6/6 match, all zeros)\n");
            }
            total_errors += errs;
        }
    }
    
    printf("\n--- Feature extraction summary ---\n");
    printf("  Total diffs (>1e-5): %d\n", total_errors);
    printf("  Max diff: %.2e at %s[%d]\n", max_diff, max_label, max_idx);
    
    return total_errors;
}

/* ================================================================
 * Test 2: Full feature vector comparison (98-dim)
 * ================================================================ */
static int test_full_features(void) {
    print_header("Full 98-dim Feature Vector Comparison");
    
    int total_errors = 0;
    double max_diff = 0.0;
    int max_idx = 0, max_case = 0;
    
    for (int c = 0; c < 5; c++) {
        PoseLandmarks lm;
        for (int k = 0; k < 33; k++) {
            lm.kp[k].x = test_landmarks[c][k][0];
            lm.kp[k].y = test_landmarks[c][k][1];
            lm.kp[k].z = test_landmarks[c][k][2];
            lm.kp[k].visibility = test_landmarks[c][k][3];
        }
        
        FeatureVector fv;
        huling_extract_features(&lm, &fv);
        
        int case_errs = 0;
        for (int i = 0; i < 98; i++) {
            double d = fabs((double)fv.f[i] - py_features_raw[c][i]);
            if (d > 1e-5) {
                case_errs++;
                if (d > max_diff) {
                    max_diff = d;
                    max_idx = i;
                    max_case = c;
                }
            }
        }
        
        if (case_errs > 0) {
            printf("  %s: %d/98 diffs\n", test_names[c], case_errs);
            /* Show worst 5 diffs for this case */
            printf("    Worst diffs:\n");
            for (int show = 0; show < 5; show++) {
                double worst = 0;
                int worst_i = -1;
                for (int i = 0; i < 98; i++) {
                    double d = fabs((double)fv.f[i] - py_features_raw[c][i]);
                    if (d > worst) { worst = d; worst_i = i; }
                }
                if (worst_i >= 0) {
                    printf("      [%2d] C=%.8f  Py=%.8f  diff=%.2e\n",
                           worst_i, fv.f[worst_i], py_features_raw[c][worst_i], worst);
                    /* Null out so we find next-worst */
                    ((double*)py_features_raw)[c * 98 + worst_i] = fv.f[worst_i];
                }
            }
        } else {
            printf("  %s: OK (98/98 match)\n", test_names[c]);
        }
        total_errors += case_errs;
    }
    
    printf("\n  Total feature diffs: %d\n", total_errors);
    printf("  Worst: case=%s [%d] diff=%.2e\n", test_names[max_case], max_idx, max_diff);
    return total_errors > 0;
}

/* ================================================================
 * Test 3: Standardization comparison
 * ================================================================ */
static int test_standardization(void) {
    print_header("StandardScaler Comparison");
    
    int total_errors = 0;
    double max_diff = 0.0;
    int max_idx = 0, max_case = 0;
    
    /* standardize() is static inline from scaler_params.h, directly available */
    
    for (int c = 0; c < 5; c++) {
        PoseLandmarks lm;
        for (int k = 0; k < 33; k++) {
            lm.kp[k].x = test_landmarks[c][k][0];
            lm.kp[k].y = test_landmarks[c][k][1];
            lm.kp[k].z = test_landmarks[c][k][2];
            lm.kp[k].visibility = test_landmarks[c][k][3];
        }
        
        FeatureVector fv;
        huling_extract_features(&lm, &fv);
        standardize(fv.f);
        
        int case_errs = 0;
        for (int i = 0; i < 98; i++) {
            double d = fabs((double)fv.f[i] - py_features_norm[c][i]);
            if (d > 1e-5) {
                case_errs++;
                if (d > max_diff) {
                    max_diff = d;
                    max_idx = i;
                    max_case = c;
                }
            }
        }
        
        if (case_errs > 0) {
            printf("  %s: %d/98 diffs\n", test_names[c], case_errs);
        } else {
            printf("  %s: OK (98/98 match)\n", test_names[c]);
        }
        total_errors += case_errs;
    }
    
    printf("\n  Total norm diffs: %d\n", total_errors);
    printf("  Worst: case=%s [%d] diff=%.2e\n", test_names[max_case], max_idx, max_diff);
    return total_errors > 0;
}

/* ================================================================
 * Test 4: RandomForest inference comparison
 * ================================================================ */
static int test_inference(void) {
    print_header("RandomForest Inference Comparison");
    
    int mismatches = 0;
    double max_score_diff = 0.0;
    int max_case = 0;
    
    for (int c = 0; c < 5; c++) {
        PoseLandmarks lm;
        for (int k = 0; k < 33; k++) {
            lm.kp[k].x = test_landmarks[c][k][0];
            lm.kp[k].y = test_landmarks[c][k][1];
            lm.kp[k].z = test_landmarks[c][k][2];
            lm.kp[k].visibility = test_landmarks[c][k][3];
        }
        
        FeatureVector fv;
        huling_extract_features(&lm, &fv);
        standardize(fv.f);
        
        /* Get raw 3-class scores from m2cgen */
        double input_dbl[98];
        for (int i = 0; i < 98; i++) {
            input_dbl[i] = (double)fv.f[i];
        }
        double scores_dbl[3];
        predict_pose(scores_dbl, input_dbl);
        
        /* Get class prediction */
        int c_class = predict_pose_class(fv.f);
        
        printf("  %s:\n", test_names[c]);
        printf("    C raw scores: [%.6f, %.6f, %.6f]\n", scores_dbl[0], scores_dbl[1], scores_dbl[2]);
        printf("    Py scores:    [%.6f, %.6f, %.6f] (classes %d,%d,%d)\n",
               py_scores[c][0], py_scores[c][1], py_scores[c][5],
               g_model_class_map[0], g_model_class_map[1], g_model_class_map[2]);
        printf("    C pred: %d (%s)  Py pred: %d (%s)\n",
               c_class, class_name(c_class),
               py_pred_class[c],
               py_pred_class[c] >= 0 && py_pred_class[c] < 6 ?
                   (const char*[]){"walking","sitting","lying","long_sit","abnormal","fall"}[py_pred_class[c]]
                   : "?");
        
        /* Compare class prediction */
        if (c_class != py_pred_class[c]) {
            printf("    *** MISMATCH! ***\n");
            mismatches++;
        }
        
        /* Compare raw scores (map C indices 0,1,2 → classes 0,1,5) */
        for (int i = 0; i < 3; i++) {
            int cls = g_model_class_map[i];
            double d = fabs(scores_dbl[i] - py_scores[c][cls]);
            if (d > max_score_diff) {
                max_score_diff = d;
                max_case = c;
            }
        }
    }
    
    printf("\n  Prediction mismatches: %d\n", mismatches);
    printf("  Max score diff: %.2e (case=%s)\n", max_score_diff, test_names[max_case]);
    return mismatches;
}

/* ================================================================
 * Test 5: Python scaler params match C scaler params
 * ================================================================ */
static int test_scaler_params(void) {
    print_header("Scaler Parameter Verification");
    
    /* Note: scaler params in scaler_params.h are static.
       We verify at compile time by comparing the generated values. */
    
    /* Check that N_FEATURES matches */
    printf("  N_FEATURES = %d (expected 98)\n", N_FEATURES);
    if (N_FEATURES != 98) {
        printf("  *** MISMATCH! ***\n");
        return 1;
    }
    
    /* We can't easily access static variables from another TU,
       so just verify that the standardize function produces
       results matching Python (this is covered in test 3). */
    printf("  Scaler params verified via standardization test (Test 3).\n");
    return 0;
}

/* ================================================================
 * Main
 * ================================================================ */
int main(void) {
    printf("============================================================\n");
    printf("  HuLing C Code Verification\n");
    printf("  Python reference generated by generate_test_data.py\n");
    printf("============================================================\n");
    printf("  Test cases: %d\n", N_TEST_CASES);
    printf("  Features: %d\n", N_FEATURES);
    printf("  Model classes: %d (indices 0,1,5)\n", N_MODEL_CLASSES);
    
    int errors = 0;
    
    errors += test_feature_extraction();
    errors += test_full_features();
    errors += test_standardization();
    errors += test_inference();
    errors += test_scaler_params();
    
    printf("\n============================================================\n");
    if (errors == 0) {
        printf("  ALL TESTS PASSED\n");
    } else {
        printf("  %d TEST(S) HAD ERRORS\n", errors);
    }
    printf("============================================================\n");
    
    return errors ? 1 : 0;
}
